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
    
    # Enhanced Global Search Configuration (Microsoft GraphRAG Compatible)
    global_search: Dict[str, Any] = Field(default_factory=lambda: {
        # Core parameters
        "community_level": 2,
        "max_data_tokens": 8000,
        
        # Microsoft GraphRAG Enhanced Parameters
        "use_community_summary": True,          # Use summary vs full content
        "column_delimiter": "|",                # CSV delimiter
        "shuffle_data": True,                   # Randomize community order
        "include_community_rank": True,         # Include rank scoring
        "min_community_rank": 0,                # Minimum rank threshold
        "community_rank_name": "rank",          # Rank column name
        "include_community_weight": True,       # Include occurrence weights
        "community_weight_name": "occurrence weight",  # Weight column name
        "normalize_community_weight": True,     # Normalize weights 0-1
        "single_batch": False,                  # Multi-batch processing
        "context_name": "Reports",              # Context section name
        "random_state": 86                      # Random seed for shuffling
    })
    
    # Enhanced Local Search Configuration (Microsoft GraphRAG Compatible)
    local_search: Dict[str, Any] = Field(default_factory=lambda: {
        # Core parameters
        "max_context_tokens": 8000,
        "top_k_mapped_entities": 10,           # Number of entities to map from query
        "top_k_relationships": 10,
        "top_k_text_units": 3,
        "top_k_communities": 3,
        
        # Microsoft GraphRAG Enhanced Parameters
        "text_unit_prop": 0.5,                # 50% of tokens for text units
        "community_prop": 0.25,               # 25% of tokens for community reports  
        "include_community_rank": False,       # Include community ranking
        "include_entity_rank": False,          # Include entity ranking
        "rank_description": "number of relationships",  # Description for ranking
        "include_relationship_weight": False,  # Include relationship weights
        "relationship_ranking_attribute": "rank",  # Attribute for relationship ranking
        "return_candidate_context": False,     # Return candidate context data
        "use_community_summary": False,        # Use summary instead of full content
        "min_community_rank": 0,               # Minimum community rank threshold
        "community_context_name": "Reports",   # Name for community context section
        "column_delimiter": "|",               # Delimiter for CSV formatting
        "embedding_vectorstore_key": "id",     # Key for embedding vector store
        "conversation_history_max_turns": 5,   # Max conversation history turns
        "conversation_history_user_turns_only": True  # Only user turns in history
    })
    
    # Enhanced Basic Search Configuration (Microsoft GraphRAG Compatible)  
    basic_search: Dict[str, Any] = Field(default_factory=lambda: {
        # Core parameters
        "top_k_text_units": 20,
        "max_context_tokens": 12000,           # Increased for better context
        
        # Microsoft GraphRAG Enhanced Parameters
        "context_name": "Sources",             # Name for context section
        "column_delimiter": "|",               # Delimiter for CSV formatting
        "text_id_col": "source_id",            # Column name for text ID
        "text_col": "text",                    # Column name for text content
        "embedding_vectorstore_key": "id"      # Key for embedding vector store
    })
    
    # Enhanced DRIFT Search Configuration (Microsoft GraphRAG Compatible)
    drift_search: Dict[str, Any] = Field(default_factory=lambda: {
        # Core DRIFT parameters
        "n_depth": 3,
        "drift_k_followups": 5,
        
        # Microsoft GraphRAG Enhanced Parameters
        "data_max_tokens": 8000,
        "reduce_max_tokens": 2000,
        "reduce_temperature": 0.1,
        "concurrency": 5,
        "primer_folds": 3,
        "primer_llm_max_tokens": 8000,
        "embedding_vectorstore_key": "id",
        
        # Local search integration parameters
        "local_search_max_data_tokens": 8000,
        "local_search_top_k_mapped_entities": 10,
        "local_search_top_k_relationships": 10,
        "local_search_temperature": 0.1,
        "local_search_text_unit_prop": 0.5,
        "local_search_community_prop": 0.1,
        "local_search_top_p": 1.0,
        "local_search_n": 1,
        "local_search_llm_max_gen_tokens": 2000,
        "local_search_llm_max_gen_completion_tokens": 2000
    })
