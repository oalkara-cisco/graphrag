"""Neo4j Configuration Model for GraphRAG."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class Neo4jConfig(BaseModel):
    """Neo4j database configuration."""
    enabled: bool = Field(default=False, description="Enable Neo4j backend")
    uri: str = Field(..., description="Neo4j connection URI")
    username: str = Field(..., description="Neo4j username") 
    password: str = Field(..., description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")
    
    # Search configuration
    global_search: Dict[str, Any] = Field(default_factory=lambda: {
        "community_level": 2,
        "max_data_tokens": 8000
    })
    
    local_search: Dict[str, Any] = Field(default_factory=lambda: {
        "top_k_entities": 10,
        "top_k_text_units": 3,
        "top_k_relationships": 10, 
        "top_k_communities": 3,
        "max_context_tokens": 8000
    })
    
    basic_search: Dict[str, Any] = Field(default_factory=lambda: {
        "top_k_text_units": 20,
        "max_context_tokens": 8000
    })
    
    drift_search: Dict[str, Any] = Field(default_factory=lambda: {
        "n_depth": 3,
        "drift_k_followups": 5,
        "local_search_max_data_tokens": 8000,
        "local_search_top_k_mapped_entities": 10,
        "local_search_top_k_relationships": 10,
        "local_search_temperature": 0.1,
        "local_search_text_unit_prop": 0.5,
        "local_search_community_prop": 0.1
    })
