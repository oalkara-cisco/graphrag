"""
Neo4j Query API for GraphRAG
=============================

This module provides GraphRAG-style API functions for Neo4j-based queries,
following the same patterns as the existing graphrag.api.query module.
"""

from typing import Any, Dict, Tuple
import logging

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.basic_search.search import BasicSearch
from graphrag.query.structured_search.drift_search.search import DRIFTSearch

from .cli_integration import (
    validate_neo4j_config,
    create_neo4j_components,
    should_use_neo4j
)
from .token_utils import get_token_encoder_from_config

logger = logging.getLogger(__name__)


async def neo4j_global_search(
    config: GraphRagConfig,
    query: str,
    community_level: int | None = None,
    response_type: str = "multiple paragraphs",
    verbose: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Perform global search using Neo4j backend.
    
    Args:
        config: GraphRAG configuration containing Neo4j settings
        query: The search query
        community_level: Community level to use (overrides config default)
        response_type: Type of response format
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (response_text, context_data)
    """
    
    # Validate Neo4j configuration
    validate_neo4j_config(config)
    
    # Create Neo4j components from GraphRAG config
    neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
    
    try:
        from ..neo4j_context_builders import Neo4jGlobalContextBuilder
        
        # Create context builder using config settings
        context_builder = Neo4jGlobalContextBuilder(
            neo4j_backend=neo4j_backend,
            max_data_tokens=config.neo4j.global_search.get("max_data_tokens", 8000),
            community_level=community_level or config.neo4j.global_search.get("community_level", 2)
        )
        
        # Create search engine
        token_encoder = get_token_encoder_from_config(config)
        search_engine = GlobalSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder
        )
        
        # Execute search
        result = await search_engine.search(query)
        
        if verbose:
            logger.info("Neo4j Global Search completed successfully")
        
        # Return in same format as existing API
        context_data = {
            "community_level": community_level,
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens
        }
        
        return result.response, context_data
        
    finally:
        # Clean up resources
        if 'vector_graph_store' in locals():
            vector_graph_store.close()


async def neo4j_local_search(
    config: GraphRagConfig,
    query: str,
    community_level: int = 2,
    response_type: str = "multiple paragraphs",
    verbose: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Perform local search using Neo4j backend.
    
    Args:
        config: GraphRAG configuration containing Neo4j settings
        query: The search query
        community_level: Community level (for compatibility)
        response_type: Type of response format
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (response_text, context_data)
    """
    
    # Validate Neo4j configuration
    validate_neo4j_config(config)
    
    # Create Neo4j components from GraphRAG config
    neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
    
    try:
        from ..neo4j_context_builders import Neo4jLocalContextBuilder
        
        # Create context builder using config settings
        local_config = config.neo4j.local_search
        context_builder = Neo4jLocalContextBuilder(
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            max_context_tokens=local_config.get("max_context_tokens", 8000),
            top_k_entities=local_config.get("top_k_entities", 10),
            top_k_text_units=local_config.get("top_k_text_units", 3),
            top_k_relationships=local_config.get("top_k_relationships", 10),
            top_k_communities=local_config.get("top_k_communities", 3)
        )
        
        # Create search engine
        token_encoder = get_token_encoder_from_config(config)
        search_engine = LocalSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder,
            response_type=response_type
        )
        
        # Execute search
        result = await search_engine.search(query)
        
        if verbose:
            logger.info("Neo4j Local Search completed successfully")
        
        # Return in same format as existing API
        context_data = {
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens
        }
        
        return result.response, context_data
        
    finally:
        # Clean up resources
        if 'vector_graph_store' in locals():
            vector_graph_store.close()


async def neo4j_basic_search(
    config: GraphRagConfig,
    query: str,
    response_type: str = "multiple paragraphs",
    verbose: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Perform basic search using Neo4j backend.
    
    Args:
        config: GraphRAG configuration containing Neo4j settings
        query: The search query
        response_type: Type of response format
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (response_text, context_data)
    """
    
    # Validate Neo4j configuration
    validate_neo4j_config(config)
    
    # Create Neo4j components from GraphRAG config
    neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
    
    try:
        from ..neo4j_context_builders import Neo4jBasicContextBuilder
        
        # Create context builder using config settings
        basic_config = config.neo4j.basic_search
        context_builder = Neo4jBasicContextBuilder(
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            max_context_tokens=basic_config.get("max_context_tokens", 8000),
            top_k_text_units=basic_config.get("top_k_text_units", 20)
        )
        
        # Create search engine
        token_encoder = get_token_encoder_from_config(config)
        search_engine = BasicSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder,
            response_type=response_type
        )
        
        # Execute search
        result = await search_engine.search(query)
        
        if verbose:
            logger.info("Neo4j Basic Search completed successfully")
        
        # Return in same format as existing API
        context_data = {
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens
        }
        
        return result.response, context_data
        
    finally:
        # Clean up resources
        if 'vector_graph_store' in locals():
            vector_graph_store.close()


async def neo4j_drift_search(
    config: GraphRagConfig,
    query: str,
    community_level: int = 2,
    response_type: str = "multiple paragraphs",
    verbose: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Perform DRIFT search using Neo4j backend.
    
    Args:
        config: GraphRAG configuration containing Neo4j settings
        query: The search query
        community_level: Community level (for compatibility)
        response_type: Type of response format
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (response_text, context_data)
    """
    
    # Validate Neo4j configuration
    validate_neo4j_config(config)
    
    # Create Neo4j components from GraphRAG config
    neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
    
    try:
        from ..neo4j_context_builders import Neo4jDRIFTContextBuilder
        from ..drift_json_patch import apply_drift_json_patch, remove_drift_json_patch
        
        # Apply DRIFT JSON patch
        apply_drift_json_patch()
        
        # Create context builder using config settings
        drift_config = config.neo4j.drift_search
        context_builder = Neo4jDRIFTContextBuilder(
            neo4j_backend=neo4j_backend,
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            chat_model=chat_model,
            config=drift_config
        )
        
        # Create search engine
        token_encoder = get_token_encoder_from_config(config)
        search_engine = DRIFTSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder
        )
        
        # Execute search
        result = await search_engine.search(query)
        
        if verbose:
            logger.info("Neo4j DRIFT Search completed successfully")
        
        # Return in same format as existing API
        context_data = {
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens
        }
        
        return result.response, context_data
        
    finally:
        # Clean up resources
        remove_drift_json_patch()
        if 'vector_graph_store' in locals():
            vector_graph_store.close()


def is_neo4j_available(config: GraphRagConfig) -> bool:
    """
    Check if Neo4j integration is available and properly configured.
    
    Args:
        config: GraphRAG configuration
        
    Returns:
        True if Neo4j is available and configured, False otherwise
    """
    try:
        return should_use_neo4j(None, config)
    except Exception:
        return False
