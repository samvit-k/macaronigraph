"""Neo4j graph store (optional). Every Reel is written in ONE transaction that replaces only what the Reel owns
(its Passages and Assertions) and garbage-collects only its previous concepts that nothing references any
more — concepts shared with other Reels survive. A failed transaction rolls back: the previous graph stays.

Schema
  (:Reel {reel_id})-[:HAS_PASSAGE]->(:Passage {chunk_id = Chroma id})-[:MENTIONS {surface, method}]->(:Concept {key})
  (:Assertion {id})-[:SUBJECT]->(:Concept), -[:OBJECT]->(:Concept), -[:EVIDENCE]->(:Passage)
Assertion.status is "proposed" (model-extracted, quote verified) unless a human review (kept in SQLite and
re-applied by the sync layer on every rewrite) set "verified" / "rejected". "corroborated" (same
subject-predicate-object proposed in ≥2 Reels) is computed at query time. Co-occurrence is never stored.
"""
from __future__ import annotations

import threading

from .. import config

SCHEMA = [
    "CREATE CONSTRAINT reel_id IF NOT EXISTS FOR (r:Reel) REQUIRE r.reel_id IS UNIQUE",
    "CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (p:Passage) REQUIRE p.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT concept_key IF NOT EXISTS FOR (c:Concept) REQUIRE c.key IS UNIQUE",
    "CREATE CONSTRAINT assertion_id IF NOT EXISTS FOR (a:Assertion) REQUIRE a.id IS UNIQUE",
    "CREATE INDEX passage_reel IF NOT EXISTS FOR (p:Passage) ON (p.reel_id)",
    "CREATE INDEX assertion_reel IF NOT EXISTS FOR (a:Assertion) ON (a.reel_id)",
    "CREATE INDEX concept_type IF NOT EXISTS FOR (c:Concept) ON (c.type)",
    "CREATE FULLTEXT INDEX concept_names IF NOT EXISTS FOR (c:Concept) ON EACH [c.name, c.aliases_text]",
]


class GraphOffline(RuntimeError):
    pass


def _prev_keys(tx, rid: str) -> list[str]:
    rows = tx.run("""MATCH (p:Passage {reel_id: $rid})-[:MENTIONS]->(c:Concept) RETURN DISTINCT c.key AS k
                     UNION
                     MATCH (a:Assertion {reel_id: $rid})-[:SUBJECT|OBJECT]->(c:Concept) RETURN DISTINCT c.key AS k""",
                  rid=rid)
    return [r["k"] for r in rows]


def _gc_and_refresh(tx, keys: list[str]) -> None:
    """Delete the given concepts if nothing references them; refresh aliases of the rest from MENTIONS."""
    tx.run("""UNWIND $keys AS k MATCH (c:Concept {key: k})
              WHERE NOT (c)<-[:MENTIONS]-() AND NOT (c)<-[:SUBJECT|OBJECT]-()
              DETACH DELETE c""", keys=keys)
    tx.run("""UNWIND $keys AS k MATCH (c:Concept {key: k})
              OPTIONAL MATCH (c)<-[m:MENTIONS]-()
              WITH c, [s IN collect(DISTINCT m.surface) WHERE s IS NOT NULL] AS surfaces
              SET c.aliases = surfaces, c.aliases_text = reduce(t = '', s IN surfaces | t + ' ' + s)""", keys=keys)


def _delete_owned(tx, rid: str) -> None:
    tx.run("MATCH (a:Assertion {reel_id: $rid}) DETACH DELETE a", rid=rid)
    tx.run("MATCH (p:Passage {reel_id: $rid}) DETACH DELETE p", rid=rid)


def write_reel_tx(tx, doc: dict) -> None:
    """doc: {reel, passages, concepts, mentions, assertions, graph_hash, retrieval_ready}"""
    rid = doc["reel"]["reel_id"]
    prev = _prev_keys(tx, rid)
    _delete_owned(tx, rid)
    tx.run("MERGE (r:Reel {reel_id: $rid}) SET r += $props", rid=rid, props=doc["reel"])
    tx.run("""UNWIND $concepts AS c MERGE (k:Concept {key: c.key})
              ON CREATE SET k.name = c.name, k.type = c.type
              SET k.name = CASE WHEN c.curated THEN c.name ELSE k.name END,
                  k.type = CASE WHEN c.curated THEN c.type ELSE coalesce(k.type, c.type) END""",
           concepts=doc["concepts"])
    tx.run("""MATCH (r:Reel {reel_id: $rid}) UNWIND $passages AS p
              CREATE (x:Passage) SET x = p CREATE (r)-[:HAS_PASSAGE]->(x)""", rid=rid, passages=doc["passages"])
    tx.run("""UNWIND $mentions AS m MATCH (p:Passage {chunk_id: m.chunk_id}), (k:Concept {key: m.key})
              CREATE (p)-[:MENTIONS {surface: m.surface, method: m.method, extractor_version: m.extractor_version}]->(k)""",
           mentions=doc["mentions"])
    tx.run("""UNWIND $assertions AS a
              MATCH (s:Concept {key: a.subject}), (o:Concept {key: a.object}), (p:Passage {chunk_id: a.chunk_id})
              CREATE (x:Assertion {id: a.id, reel_id: a.reel_id, predicate: a.predicate, quote: a.quote,
                                   method: a.method, confidence: a.confidence, provenance: a.provenance,
                                   status: a.status, review_note: a.review_note, extractor_version: a.extractor_version,
                                   subject_key: a.subject, object_key: a.object, chunk_id: a.chunk_id})
              CREATE (x)-[:SUBJECT]->(s) CREATE (x)-[:OBJECT]->(o) CREATE (x)-[:EVIDENCE]->(p)""",
           assertions=doc["assertions"])
    _gc_and_refresh(tx, sorted(set(prev) | {c["key"] for c in doc["concepts"]}))
    tx.run("MATCH (r:Reel {reel_id: $rid}) SET r.graph_hash = $h, r.retrieval_ready = $ready",   # last
           rid=rid, h=doc["graph_hash"], ready=doc["retrieval_ready"])


def remove_reel_tx(tx, rid: str) -> None:
    prev = _prev_keys(tx, rid)
    _delete_owned(tx, rid)
    tx.run("MATCH (r:Reel {reel_id: $rid}) DETACH DELETE r", rid=rid)
    _gc_and_refresh(tx, prev)


class Neo4jStore:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None,
                 database: str | None = None):
        self.uri, self.user = uri or config.NEO4J_URI, user or config.NEO4J_USER
        self.password, self.database = password or config.NEO4J_PASSWORD, database or config.NEO4J_DATABASE
        self._driver = None
        self._lock = threading.Lock()

    def driver(self):
        with self._lock:
            if self._driver is None:
                if not self.password:
                    raise GraphOffline("NEO4J_PASSWORD is not set (run scripts/neo4j.sh install)")
                try:
                    from neo4j import GraphDatabase
                except Exception as e:  # noqa: BLE001
                    raise GraphOffline(f"neo4j driver not installed: {e}") from e
                d = None
                try:
                    d = GraphDatabase.driver(self.uri, auth=(self.user, self.password), connection_timeout=5)
                    d.verify_connectivity()
                except Exception as e:  # noqa: BLE001 — server down, bad credentials, ...
                    if d is not None:
                        try:
                            d.close()                       # never leak a half-initialised driver
                        except Exception:  # noqa: BLE001
                            pass
                    raise GraphOffline(f"Neo4j unavailable at {self.uri}: {type(e).__name__}: {e}") from e
                self._driver = d
            return self._driver

    def close(self) -> None:
        with self._lock:
            if self._driver is not None:
                self._driver.close()
                self._driver = None

    @staticmethod
    def _offline(e: Exception) -> bool:
        return type(e).__name__ in ("ServiceUnavailable", "SessionExpired")

    def run(self, cypher: str, **params) -> list[dict]:
        try:
            recs, _, _ = self.driver().execute_query(cypher, params, database_=self.database)
        except GraphOffline:
            raise
        except Exception as e:  # noqa: BLE001
            if self._offline(e):
                self.close()
                raise GraphOffline(str(e)) from e
            raise
        return [r.data() for r in recs]

    def _write(self, fn, *args) -> None:
        try:
            with self.driver().session(database=self.database) as s:
                s.execute_write(fn, *args)
        except GraphOffline:
            raise
        except Exception as e:  # noqa: BLE001
            if self._offline(e):
                self.close()
                raise GraphOffline(str(e)) from e
            raise

    # ---- interface used by sync ----
    def ensure_schema(self) -> None:
        for stmt in SCHEMA:
            self.run(stmt)

    def write_reel(self, doc: dict) -> None:
        self._write(write_reel_tx, doc)

    def remove_reel(self, rid: str) -> None:
        self._write(remove_reel_tx, rid)

    def reel_hashes(self) -> dict[str, str | None]:
        return {r["id"]: r["h"] for r in self.run("MATCH (r:Reel) RETURN r.reel_id AS id, r.graph_hash AS h")}

    def set_ready(self, rid: str, ready: bool) -> None:
        self.run("MATCH (r:Reel {reel_id: $rid}) SET r.retrieval_ready = $ready", rid=rid, ready=ready)

    def ping(self) -> bool:
        self.run("RETURN 1 AS ok")
        return True
