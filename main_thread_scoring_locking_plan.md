# Main-thread scoring without the time-slice mutex

**Status: implemented.** Both parts are in the working tree. Verification results are at the bottom.
Outstanding items are perf-only, not correctness: the main-thread per-key cost, and how coarse tag locking
should be.

## Context

Main-thread score recompute (`response_generator.cc VerifyFilter` → `SingleDocumentScorer::Score`) did two
things it shouldn't:

1. **Corpus-level work on the main thread.** `Search()` pre-builds `parameters.recompute_scorer` on a
   background thread, but in CME the coordinating `SearchParameters` never runs `Search()` — the local shard
   runs a separate `LocalResponderSearch` ([fanout.cc:410-418](src/query/fanout.cc#L410-L418)).

   For FT.SEARCH **with** content this is harmless: the local responder resolves its own content, so its
   neighbors reach the coordinator already populated and `ProcessNeighborsForReply` skips them
   ([response_generator.cc:466-469](src/query/response_generator.cc#L466-L469)). The recompute runs on the
   responder, off the scorer it pre-built.

   The gap is FT.AGGREGATE whose LOAD resolves to no record attributes
   ([ft_aggregate.cc:91](src/commands/ft_aggregate.cc#L91) sets `no_content`). That propagates to every
   responder, so nothing carries content and the coordinator fetches it for the keys it owns — while the
   responder, being `no_content`, skipped the pre-build. The coordinator then hit the lazy `kAcquireLock`
   fallback and did `ResolveLeaves` — rax lookups, stem expansion, per-term IDF — on the main thread. That
   also snapshots a *later* corpus than the one the carried scores came from, so recomputed scores land on a
   slightly different scale.

2. **A time-slice mutex acquisition on the main thread.** `Score()` took `ReaderMutexLock` on
   `time_sliced_mutex_`, which can stall behind an active writer phase.

Goal: the scorer is **always** built on a background thread and holds all corpus-level state; the main
thread reads only key-level state, under fine-grained locks, and never touches the time-slice mutex.

### Why fine-grained locks are needed (not just "no mutation in flight")

`PerformKeyContentionCheck` ([index_schema.cc:1918-1942](src/index_schema.cc#L1918-L1942)) runs once before
the neighbor loop and only guarantees that *the result keys* have no queued mutation. Mutation-pool threads
keep draining **other** keys throughout the loop, and `CommitKeyData` `emplace`s into the *shared* `Postings`
btree ([text_index.cc:264-275](src/indexes/text/text_index.cc#L264-L275) — the per-key tree and the main tree
point at the same object). So `key_to_positions_.find()` races a concurrent `emplace()` for an unrelated key.

Also note `GetContentProcessing()` only returns `kContentionCheckRequired` for queries with a text
predicate; tag/numeric-only filters skip the check entirely.

Every key-level read already has a lock **the writer takes**, so the time-slice mutex is replaceable:

| Read | Lives in | Writer's lock | Branch |
|---|---|---|---|
| `tf`, `doc_len` | `PostingValue` in shared `Postings::key_to_positions_` | `rax_target_mutex_pool_.Get(word)` ([text_index.cc:254](src/indexes/text/text_index.cc#L254), [:337](src/indexes/text/text_index.cc#L337)) | `kText` |
| tag membership | `Tag::tree_` rax | `Tag::index_mutex_` (already taken by `GetTagValueDocCount`) | `kTag` |
| `doc_len` | `per_key_scoring_info_` (`flat_hash_map`) | `per_key_text_indexes_mutex_` | `kTag` only |
| document score | `index_key_info_` (`flat_hash_map`) | `mutated_records_mutex_` | composition, only if `has_score_field` |

Position maps are never read by scoring — `tf`/`doc_len` are mirrored into `PostingValue` precisely to avoid
that ([posting.h:73-81](src/indexes/text/posting.h#L73-L81)).

### Why `Score()` must reach postings via the per-key tree

`SingleDocumentScorer`'s constructor pins `InvasivePtr<Postings>` via `ResolveLeaves`. Those pins can go
**stale**: if the re-evaluated key was the word's last holder, `RemoveKeyFromPostings` clears the tree's ref
([text_index.cc:55-67](src/indexes/text/text_index.cc#L55-L67)) and `DeleteKeyData` erases the word from the
rax ([:349-352](src/indexes/text/text_index.cc#L349-L352)); the re-add then finds no word and
`AddKeyToPostings` builds a **fresh** `Postings`. The pinned object stays alive (refcount) but orphaned and
empty, so `LookupKey` → nullopt → `tf == 0` → `Score()` → nullopt → `VerifyFilter` degrades to `0.0f`. Safe,
but a silent mis-ranking in exactly the case recompute exists for. This bug predates the change on the
`Search()` pre-built path; Part 1 would have made it universal.

So `Score()` resolves per-key data through the key's own tree, which `CommitKeyData` rebuilds wholesale
([text_index.cc:298-302](src/indexes/text/text_index.cc#L298-L302)) and which therefore always points at the
*current* shared `Postings`. It is stable for the duration because the key is quiesced. Corpus-level state
(IDF, `avg_doc_len`, `total_docs`) still comes from the background snapshot — that is what keeps the
recomputed score on the same scale as the carried neighbor scores. Bonus: membership becomes correct too
(key lost the word → tree lacks it → true non-match instead of a stale hit).

Background `ScoreTextQuery` keeps the pinned-postings route: it runs inside the read phase against the same
snapshot, so it has no staleness window and no reason to pay for a rax walk.

**The quiescence premise holds exactly where it is needed.** Walking the per-key tree without a lock relies on
the key having no in-flight mutation, which `PerformKeyContentionCheck` only guarantees for
`kContentionCheckRequired` queries. That is sufficient because the two conditions coincide: a query reaches
the `kText` branch only if it has a text predicate, and `QueryHasTextPredicate` is precisely what makes
`GetContentProcessing()` return `kContentionCheckRequired`. Tag/numeric-only queries skip the contention
check but never touch a per-key tree — their reads go through `Tag::index_mutex_`,
`per_key_text_indexes_mutex_`, and `mutated_records_mutex_`, none of which depend on quiescence.

---

## Review ordering

The two parts are independent and worth reviewing separately. Part 2 (per-key-tree routing + fine-grained
locks) is the correctness-critical, TSAN-sensitive half and stands alone. Part 1 is a behavioral change
touching FT.AGGREGATE and fanout. Part 2 alone does not reach the stated goal — the lazy `VerifyFilter`
path's `kAcquireLock` **constructor** still took the reader lock; Part 1 is what removed the last
acquisition.

---

## Part 1 — Always build the scorer on a background thread

**[search.h:282](src/query/search.h#L282)** — new virtual on `SearchParameters`:

```cpp
// Gates the background pre-build of recompute_scorer. The default matches
// GetContentProcessing() != kNoContent; overridden by operations that fetch
// content anyway (FT.AGGREGATE) and by the CME local responder.
virtual bool WillFetchContentOnMainThread() const { return !no_content; }
```

**[search.cc:1534](src/query/search.cc#L1534)** — pre-build gate switched from
`GetContentProcessing() != kNoContent` to `WillFetchContentOnMainThread()`.

The base-class default is **behavior-identical**, not a widening: `GetContentProcessing()` returns
`kNoContent` iff `no_content`, so `!no_content` ≡ `GetContentProcessing() != kNoContent`. Only the two
overrides change anything, so no existing non-CME path starts building a scorer it previously skipped.

**[ft_aggregate_parser.h:79](src/commands/ft_aggregate_parser.h#L79)** — override → `true` on
`AggregateParameters`. FT.AGGREGATE sets `no_content = !content` but still calls
`ProcessNeighborsForReply` unconditionally ([ft_aggregate.cc:240](src/commands/ft_aggregate.cc#L240)).

**[fanout.cc:248](src/query/fanout.cc#L248)** — override → `true` on `LocalResponderSearch`, so the shard
that actually runs `Search()` builds the scorer under its own reader lock.

**[response_generator.cc:203-214](src/query/response_generator.cc#L203-L214)** — the lazy construction is
replaced by a borrow chain:

```cpp
const query::SingleDocumentScorer *scorer = parameters.recompute_scorer.get();
if (scorer == nullptr && parameters.local_responder_ != nullptr) {
  scorer = parameters.local_responder_->recompute_scorer.get();
}
if (scorer == nullptr) {
  recompute_scorer_missing.Increment();   // dev counter, expected to stay 0
  return {true, std::nullopt};            // keep the carried score
}
```

A `DCHECK` was considered and rejected: it would abort debug builds on a path that cannot be proven
unreachable, and it makes the fall-through untestable. The counter is observable from `INFO SEARCH` as
`search_recompute_scorer_missing` ([response_generator.cc:158](src/query/response_generator.cc#L158)) and is
what the integration test asserts on.

Ordering is safe: `local_responder_` is stashed at
[fanout.cc:277-280](src/query/fanout.cc#L277-L280) *before* `tracker_copy` drops, and the tracker dtor is
what generates the reply. Semantics are right too: only locally-owned keys are recomputed, and each shard
scores against its own slot-local corpus.

The `document_scorer` out-parameter is deleted from `VerifyFilter` / `GetContentNoReturnJson` /
`GetContent` / `ProcessNeighborsForReply`. `LockPolicy::kAcquireLock` now has no production caller —
audited: the only construction sites are the ctor itself, and the sole production user of the default
argument was the lazy `VerifyFilter` path. The enum is retained for tests.

## Part 2 — Fine-grained locks in `Score()`

**`src/query/search.cc`**

- [`ScoreContext`:820-821](src/query/search.cc#L820-L821) — new `per_key_text_index` pointer and
  `lock_per_key_reads` flag. Unset on the background path, so `ScoreTextQuery` keeps its hot loop unchanged.
- [`ResolvedLeaf`](src/query/search.cc#L647) — carries the resolved `words` (original plus each stem variant,
  populated by the existing `add_word` lambda) and the `TextIndexSchema*`. `postings` is kept for the
  background path.
- [`Score()`:1139](src/query/search.cc#L1139) — fetches the per-key tree once per key via the existing
  `TextIndexSchema::GetPerKeyTextIndex(key, /*lock=*/true)`. Null (key gone) → no text leaf matches.
- [`ScoreNode` `kText`:878-897](src/query/search.cc#L878-L897) — in main-thread mode, resolves each word via
  `per_key_prefix.FindPostingsTarget(word)` and takes that word's bucket mutex around `LookupKey`.
- [`ScoreNode` `kTag`:936-950](src/query/search.cc#L936-L950) — `GetDocumentLengthLocked` and the locking
  `ContainsKey` overload.
- [Composition:1164](src/query/search.cc#L1164) — `GetDocumentScoreLocked`.
- The `ReaderMutexLock` is **removed** from `Score()`.

**`src/indexes/text/text_index.h`**

- [`GetWordMutex(word)`:162](src/indexes/text/text_index.h#L162) — exposes `rax_target_mutex_pool_.Get(word)`.
- `GetKeyDocLen(key, bool lock)` overload, mirroring the existing `bool lock` pattern on
  `GetPerKeyTextIndex` and `GetTrackedKeyCount(bool)`.
- `GetPerKeyTextIndex(key, lock)` reused as-is — unchanged.

**`src/indexes/tag.h` / `tag.cc`** — locking `ContainsKey(value, key, bool lock)` overload taking
`index_mutex_` (`GetTagValueDocCount` is the pattern).

**[index_schema.h:208-220](src/index_schema.h#L208-L220)** — `GetDocumentScoreLocked` (takes
`mutated_records_mutex_`) and `GetDocumentLengthLocked` (passes `lock=true` through). The
`ABSL_SHARED_LOCKS_REQUIRED(time_sliced_mutex_)` originals are retained for background callers so the
compiler still enforces which path may skip the lock.

---

## Verification results

**Unit tests — all 1365 pass across 24 binaries, no failures.** New/changed:

| Test | File | What it pins |
|---|---|---|
| `RecomputeSurvivesPostingsRecreation` | [search_test.cc:1867](testing/search_test.cc#L1867) | Staleness fix. **Fails without the per-key-tree routing** (`Score()` returns nullopt off the orphaned pin). |
| `ScoreIsSafeAgainstCommitsOnTheSameWord` | [search_test.cc:1901](testing/search_test.cc#L1901) | Word-bucket lock, under TSAN. |
| `ScoreDoesNotTakeTimeSlicedMutex` | [search_test.cc:1845](testing/search_test.cc#L1845) | `Score()` is callable with the reader lock held (would deadlock before). |
| `RecomputeScorerGateTest.DefaultsToContentFetchingQueries` | [search_test.cc:1946](testing/search_test.cc#L1946) | Base gate ≡ `GetContentProcessing() != kNoContent`. |
| `AggregateTest.AlwaysFetchesContentOnMainThread` | [ft_aggregate_parser_test.cc:314](testing/ft_aggregate_parser_test.cc#L314) | The FT.AGGREGATE override. |
| `RecomputeBorrowsLocalResponderScorer` | [response_generator_test.cc:395](testing/query/response_generator_test.cc#L395) | Borrow chain replaces the carried score. |
| `RecomputeKeepsCarriedScoreWithoutScorer` | [response_generator_test.cc:428](testing/query/response_generator_test.cc#L428) | Fall-through keeps the score, no crash. |

Stale lock comments in `RecomputePathMatchesExtraStepAtNonZero` and
`RecomputeScorerConstructibleUnderHeldLock` were corrected; both still assert scale equality with the
extra-step path.

**TSAN — clean, and the race test provably fails without the fix.** `./build.sh --tsan`, then all 157
`query_test` cases: 0 warnings. With the word-bucket lock removed, TSAN reports exactly the predicted race —
`btree::internal_emplace` (mutation thread) against `btree_node::finish()` (main thread) on
`btree_map<InternedStringPtr, PostingValue>`, plus an `operator delete` race. This was verified before
merging: a race test that cannot fail without the fix proves nothing.

**Integration — [test_scoring_recompute_cluster.py](integration/test_scoring_recompute_cluster.py), 2 pass.**
Uses the existing `block_mutation_queue` pausepoint (as `test_postfilter.py` does) to park a mutation
deterministically, and a numeric/tag index so revalidation runs synchronously instead of parking the query.
The mutation touches a field *outside* the filter, so revalidation still matches and actually reaches the
recompute.

- `test_aggregate_no_content_borrows_local_responder_scorer` — **fails without Part 1's overrides**
  (`search_recompute_scorer_missing` reaches 1). This is what established that the CME gap is
  FT.AGGREGATE-specific.
- `test_search_recomputes_on_local_responder` — passes with or without the overrides, confirming FT.SEARCH
  with content is handled entirely on the responder.

**Build.** `ninja -C .build-release` clean; `clang-format` clean on all changed files (the single violation
in `ft_aggregate_parser.h` is pre-existing).

## Outstanding

**Tag locking from the main thread is a new paradigm and needs optimization thought.** `Tag::ContainsKey`
was *deliberately* lock-free before this change, relying on the read-phase invariant. Now that `Score()` runs
outside the read phase it takes `Tag::index_mutex_` — and that is **one mutex for the entire tag index**,
guarding `tree_`, `tracked_tags_by_keys_` and `untracked_keys_` together, with no per-value or per-key
sharding. Consequences:

- It is the same lock tag writers take (`AddRecord` / `RemoveRecord` / `ModifyRecord`), so main-thread tag
  scoring now contends with tag ingestion across the *whole* index, not just the queried value.
- It is acquired once **per tag value in the query**, so `@color:{red|blue|green}` is three acquire/release
  cycles for one document.
- `Normalize(value)` (a string allocation, plus `AsciiStrToLower` on case-insensitive indexes) currently runs
  *inside* the lock, because the locking overload wraps the existing function wholesale.

This is structurally coarser than the text path, which shards via `RaxTargetMutexPool` so two different words
never contend. Cheap improvements, deliberately not applied without measurement: hoist the lock around the
whole `tag_values` loop (one acquisition instead of N, and a consistent snapshot), and move `Normalize` out of
the critical section. The structural fix, if it proves to matter, is to shard `Tag::index_mutex_` the way text
shards its word locks.

Note the tag path gains nothing from the quiescence argument and needs nothing from it — it never walks a
per-key structure, so `kContentRequired` tag/numeric queries skipping `PerformKeyContentionCheck` is harmless.
The question here is purely throughput, not correctness.

**Main-thread per-key perf measurement.** Part 2 trades a stale-but-O(1) pinned lookup for a correct
`GetPerKeyTextIndex` (mutex + hash lookup, hoisted once per key) plus a rax walk per word, so per-document
main-thread cost goes up even though time-slice contention goes away. Benchmark a reply where many neighbors
are mutated (worst case: every neighbor mutated, multi-term query with stem variants) and confirm the removed
contention dominates the added walk.

If it does not, the pinned postings can serve as a fast path with the per-key tree consulted only when
`LookupKey` misses — sound because an orphaned `Postings` is always empty, so a hit implies the object is
still live. Measure before optimizing.

The background `ScoreTextQuery` loop needs no measurement beyond confirming it still takes the unlocked
pinned-postings path, which `lock_per_key_reads == false` guarantees structurally.
