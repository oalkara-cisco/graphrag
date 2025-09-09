"""
Neo4j Integration for GraphRAG
==============================

This module provides Neo4j database backend support for GraphRAG,
enabling graph-based search operations as an alternative to parquet files.
"""

from .neo4j_backend_implementation import Neo4jConfig, Neo4jGraphBackend, Neo4jVectorGraphStore
from .neo4j_context_builders import (
    Neo4jGlobalContextBuilder,
    Neo4jLocalContextBuilder, 
    Neo4jBasicContextBuilder,
    Neo4jDRIFTContextBuilder
)
from .drift_json_patch import apply_drift_json_patch, remove_drift_json_patch
from .cli_integration import (
    handle_global_search_with_neo4j,
    handle_local_search_with_neo4j,
    handle_basic_search_with_neo4j,
    handle_drift_search_with_neo4j
)
from .neo4j_query_api import (
    neo4j_global_search,
    neo4j_local_search,
    neo4j_basic_search,
    neo4j_drift_search,
    is_neo4j_available
)

__all__ = [
    "Neo4jConfig",
    "Neo4jGraphBackend", 
    "Neo4jVectorGraphStore",
    "Neo4jGlobalContextBuilder",
    "Neo4jLocalContextBuilder",
    "Neo4jBasicContextBuilder", 
    "Neo4jDRIFTContextBuilder",
    "apply_drift_json_patch",
    "remove_drift_json_patch",
    "handle_global_search_with_neo4j",
    "handle_local_search_with_neo4j",
    "handle_basic_search_with_neo4j",
    "handle_drift_search_with_neo4j",
    "neo4j_global_search",
    "neo4j_local_search",
    "neo4j_basic_search",
    "neo4j_drift_search",
    "is_neo4j_available"
]
