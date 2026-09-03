"""Cluster-mode coverage for the main-thread score recompute path.

Two shapes reach the recompute in CME and get their scorer differently:

  FT.SEARCH with content — the local responder resolves its own content, so its
  neighbors reach the coordinator populated and the coordinator skips them. The
  recompute runs on the responder, off the scorer it pre-built.

  FT.AGGREGATE whose LOAD resolves to no record attributes — no_content
  propagates to every responder, so nothing carries content and the coordinator
  fetches it for the keys it owns. The coordinating AggregateParameters never ran
  Search(), so it borrows the responder's scorer. Without the
  WillFetchContentOnMainThread overrides neither object has one, and the mutated
  document silently keeps its stale score. search_recompute_scorer_missing
  counts that fall-through and must stay 0.

The index is numeric/tag deliberately: those queries resolve to
kContentRequired, which skips PerformKeyContentionCheck, so revalidation runs
synchronously instead of parking the query (as test_postfilter.py relies on).
"""

import math
import threading

import pytest
from valkey.client import Valkey
from valkey.cluster import ValkeyCluster
from valkey_search_test_case import ValkeySearchClusterTestCaseDebugMode
from valkeytestframework.conftest import resource_port_tracker
from valkeytestframework.util import waiters
from utils import IndexingTestHelper

from test_scoring_cluster import _pairs

IDX_NAME = "idxRecompute"
IDX = [
    "FT.CREATE", IDX_NAME, "ON", "HASH", "PREFIX", "1", "r:",
    "SCHEMA", "n", "NUMERIC", "t", "TAG",
]

DOCS = {f"r:{i}": {"n": str(i), "t": "red"} for i in range(12)}


class TestScoringRecomputeCluster(ValkeySearchClusterTestCaseDebugMode):

    def _counter(self, client: Valkey, name: str) -> int:
        return int(client.info("SEARCH")[name])

    def _load(self, coordinator: Valkey) -> str:
        """Creates the index, spreads the documents, and returns a key the
        coordinator owns. KEYS is node-local, so it lists exactly this primary's
        documents; the recompute only runs for locally-owned keys (remote ones
        are skipped by CheckSlotOwnership), so the mutated key must live here."""
        cluster: ValkeyCluster = self.new_cluster_client()
        cluster.execute_command(*IDX)
        for key, mapping in DOCS.items():
            cluster.hset(key, mapping=mapping)
        for client in self.get_all_primary_clients():
            IndexingTestHelper.wait_for_backfill_complete_on_node(client, IDX_NAME)

        local = sorted(k.decode() for k in coordinator.keys("r:*"))
        if not local:
            pytest.fail("coordinator owns none of the documents")
        return local[0]

    def _run_with_mutation_in_flight(self, coordinator: Valkey, key: str, *cmd):
        """Parks a mutation for `key` in the queue so its db sequence number
        diverges from the one carried on the neighbor — what drives VerifyFilter
        into the recompute — then runs `cmd` on the coordinator.

        The mutation touches `n`, not the queried `t`: revalidation must still
        MATCH, otherwise VerifyFilter returns on the non-match branch and never
        reaches the recompute."""
        coordinator.execute_command("ft._debug PAUSEPOINT SET block_mutation_queue")
        writer = threading.Thread(
            target=lambda: self.new_client_for_primary(0).hset(key, "n", "999")
        )
        writer.start()
        try:
            waiters.wait_for_true(
                lambda: int(
                    coordinator.execute_command(
                        "ft._debug PAUSEPOINT test block_mutation_queue"
                    )
                )
                > 0
            )
            return coordinator.execute_command(*cmd)
        finally:
            coordinator.execute_command(
                "ft._debug PAUSEPOINT RESET block_mutation_queue"
            )
            writer.join()

    def test_search_recomputes_on_local_responder(self):
        coordinator: Valkey = self.new_client_for_primary(0)
        key = self._load(coordinator)
        before = self._counter(coordinator, "search_predicate_revalidation")

        res = self._run_with_mutation_in_flight(
            coordinator, key,
            "FT.SEARCH", IDX_NAME, "@t:{red}", "LIMIT", "0", "100", "WITHSCORES",
        )

        assert self._counter(coordinator, "search_predicate_revalidation") > before
        assert self._counter(coordinator, "search_recompute_scorer_missing") == 0
        scores = [s for _, s in _pairs(res)]
        assert scores and all(math.isfinite(s) for s in scores)

    def test_aggregate_no_content_borrows_local_responder_scorer(self):
        coordinator: Valkey = self.new_client_for_primary(0)
        key = self._load(coordinator)
        before = self._counter(coordinator, "search_predicate_revalidation")

        # LOAD __key alone resolves to no record attributes, so no_content is set
        # and nothing carries content back from any responder.
        res = self._run_with_mutation_in_flight(
            coordinator, key,
            "FT.AGGREGATE", IDX_NAME, "@t:{red}", "LOAD", "1", "__key",
        )

        assert self._counter(coordinator, "search_predicate_revalidation") > before
        assert self._counter(coordinator, "search_recompute_scorer_missing") == 0
        assert res
