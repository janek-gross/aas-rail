"""Shared Neo4j connection settings for ICL graph reads and writes."""

from __future__ import annotations

import os


DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "aas-rail")
DEFAULT_NEO4J_IMPORT_DIR = os.getenv("NEO4J_IMPORT_DIR", "")
DEFAULT_NEO4J_IMPORT_URI_ROOT = os.getenv("NEO4J_IMPORT_URI_ROOT", "file:///var/lib/neo4j/import")


def connect_neo4j(graph_database, uri: str, user: str, password: str):
    """Connect to Neo4j, including compose-friendly URI alternatives."""
    candidates = [(uri, user)]

    if user != "neo4j":
        candidates.append((uri, "neo4j"))

    if "localhost" in uri or "127.0.0.1" in uri:
        candidates.append(("bolt://neo4j:7687", user))
        if user != "neo4j":
            candidates.append(("bolt://neo4j:7687", "neo4j"))

    seen = set()
    last_error = None
    for candidate_uri, candidate_user in candidates:
        key = (candidate_uri, candidate_user)
        if key in seen:
            continue
        seen.add(key)

        driver = graph_database.driver(candidate_uri, auth=(candidate_user, password))
        try:
            driver.verify_connectivity()
            return driver, candidate_uri, candidate_user
        except Exception as exc:
            driver.close()
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ConnectionError("No Neo4j connection candidates were available.")
