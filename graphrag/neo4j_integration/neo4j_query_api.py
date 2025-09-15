"""
Neo4j Query API for GraphRAG
=============================

This module provides GraphRAG-style API functions for Neo4j-based queries,
following the same patterns as the existing graphrag.api.query module.
"""

import asyncio
import logging
from typing import Any, Dict, Tuple

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.basic_search.search import BasicSearch
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.utils.api import load_search_prompt

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
    
    logger.info("Starting Neo4j Global Search")
    logger.info(f"Query: '{query}' (community_level: {community_level}, response_type: {response_type})")
    
    # Validate Neo4j configuration
    logger.info("Validating Neo4j configuration...")
    validate_neo4j_config(config)
    logger.info("Neo4j configuration validated successfully")
    
    # Create Neo4j components from GraphRAG config
    logger.info("Initializing Neo4j components for global search...")
    neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
    logger.info("Neo4j components initialized successfully")
    
    try:
        from .neo4j_context_builders import Neo4jGlobalContextBuilder
        
        # Create context builder using config settings
        logger.info("Building Neo4j Global Context Builder...")
        final_community_level = community_level or config.neo4j.global_search.get("community_level", 2)
        max_data_tokens = config.neo4j.global_search.get("max_data_tokens", 8000)
        logger.info(f"Global search parameters: community_level={final_community_level}, "
                   f"max_data_tokens={max_data_tokens}")
        
        # Get token encoder for accurate token counting
        token_encoder = get_token_encoder_from_config(config)
        
        # Get all Microsoft GraphRAG compatible parameters from config
        global_search_config = config.neo4j.global_search
        
        context_builder = Neo4jGlobalContextBuilder(
            neo4j_backend=neo4j_backend,
            token_encoder=token_encoder,
            max_data_tokens=max_data_tokens,
            community_level=final_community_level,
            # Microsoft GraphRAG Enhanced Parameters (configurable)
            use_community_summary=global_search_config.get("use_community_summary", True),
            column_delimiter=global_search_config.get("column_delimiter", "|"),
            shuffle_data=global_search_config.get("shuffle_data", True),
            include_community_rank=global_search_config.get("include_community_rank", True),
            min_community_rank=global_search_config.get("min_community_rank", 0),
            community_rank_name=global_search_config.get("community_rank_name", "rank"),
            include_community_weight=global_search_config.get("include_community_weight", True),
            community_weight_name=global_search_config.get("community_weight_name", "occurrence weight"),
            normalize_community_weight=global_search_config.get("normalize_community_weight", True),
            single_batch=global_search_config.get("single_batch", False),
            context_name=global_search_config.get("context_name", "Reports"),
            random_state=global_search_config.get("random_state", 86)
        )
        logger.info("Neo4j Global Context Builder created with Microsoft GraphRAG compatibility")
        
        # Create search engine
        logger.info("Initializing Neo4j Global Search Engine...")
        search_engine = GlobalSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder
        )
        logger.info("Neo4j Global Search Engine initialized successfully")
        
        # Execute search
        logger.info("Executing Neo4j Global Search query...")
        result = await search_engine.search(query)
        
        logger.info("Neo4j Global Search query completed successfully")
        logger.info(f"Search results: {len(result.response)} characters, "
                   f"{result.llm_calls} LLM calls, "
                   f"{result.prompt_tokens} prompt tokens, "
                   f"{result.output_tokens} output tokens")
        
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
    
    logger.info("Starting Neo4j Local Search")
    logger.info(f"Query: '{query}' (response_type: {response_type})")
    
    # Validate Neo4j configuration
    logger.info("Validating Neo4j configuration...")
    validate_neo4j_config(config)
    logger.info("Neo4j configuration validated successfully")
    
    # Create Neo4j components from GraphRAG config
    logger.info("Initializing Neo4j components for local search...")
    neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
    logger.info("Neo4j components initialized successfully")
    
    try:
        from .neo4j_context_builders import Neo4jLocalContextBuilder
        
        # Create context builder using config settings
        logger.info("Building Neo4j Local Context Builder...")
        local_config = config.neo4j.local_search
        logger.info(f"Local search parameters: top_k_entities={local_config.get('top_k_entities', 10)}, "
                   f"top_k_text_units={local_config.get('top_k_text_units', 3)}, "
                   f"max_context_tokens={local_config.get('max_context_tokens', 8000)}")
        
        # Get token encoder for accurate token counting
        token_encoder = get_token_encoder_from_config(config)
        
        context_builder = Neo4jLocalContextBuilder(
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            token_encoder=token_encoder,
            max_context_tokens=local_config.get("max_context_tokens", 8000),
            # Microsoft GraphRAG Enhanced Parameters (configurable)
            top_k_mapped_entities=local_config.get("top_k_mapped_entities", local_config.get("top_k_entities", 10)),
            top_k_text_units=local_config.get("top_k_text_units", 3),
            top_k_relationships=local_config.get("top_k_relationships", 10),
            top_k_communities=local_config.get("top_k_communities", 3),
            text_unit_prop=local_config.get("text_unit_prop", 0.5),
            community_prop=local_config.get("community_prop", 0.25),
            include_community_rank=local_config.get("include_community_rank", True),
            include_entity_rank=local_config.get("include_entity_rank", True),
            rank_description=local_config.get("rank_description", "number of relationships"),
            include_relationship_weight=local_config.get("include_relationship_weight", True),
            relationship_ranking_attribute=local_config.get("relationship_ranking_attribute", "rank"),
            return_candidate_context=local_config.get("return_candidate_context", False),
            use_community_summary=local_config.get("use_community_summary", False),
            min_community_rank=local_config.get("min_community_rank", 0),
            community_context_name=local_config.get("community_context_name", "Reports"),
            column_delimiter=local_config.get("column_delimiter", "|"),
            embedding_vectorstore_key=local_config.get("embedding_vectorstore_key", "id")
        )
        logger.info("Neo4j Local Context Builder created successfully")
        
        # Load system prompt using GraphRAG standard pattern
        prompt = load_search_prompt(config.root_dir, config.local_search.prompt)
        
        if prompt:
            logger.info(f"USING CUSTOM PROMPT: {config.local_search.prompt}")
            logger.info(f"Custom prompt size: {len(prompt)} characters")
            logger.info(f"Prompt loaded from: {config.root_dir}/{config.local_search.prompt}")
        else:
            logger.info("USING DEFAULT HARDCODED PROMPT (LOCAL_SEARCH_SYSTEM_PROMPT)")
            logger.info(f"Prompt config: {config.local_search.prompt}")
            logger.info(f"Root dir: {config.root_dir}")
            
        # Create search engine
        logger.info("Initializing Neo4j Local Search Engine...")
        token_encoder = get_token_encoder_from_config(config)
        search_engine = LocalSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder,
            response_type=response_type,
            system_prompt=prompt  # Pass loaded prompt (None defaults to hardcoded prompt)
        )
        logger.info("Neo4j Local Search Engine initialized successfully")
        
        # Execute search
        logger.info("Executing Neo4j Local Search query...")
        result = await search_engine.search(query)
        
        logger.info("Neo4j Local Search query completed successfully")
        logger.info(f"Search results: {len(result.response)} characters, "
                   f"{result.llm_calls} LLM calls, "
                   f"{result.prompt_tokens} prompt tokens, "
                   f"{result.output_tokens} output tokens")
        
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
    
    logger.info("Starting Neo4j Basic Search")
    logger.info(f"Query: '{query}' (response_type: {response_type})")
    
    # Validate Neo4j configuration
    logger.info("Validating Neo4j configuration...")
    validate_neo4j_config(config)
    logger.info("Neo4j configuration validated successfully")
    
    # Create Neo4j components from GraphRAG config
    logger.info("Initializing Neo4j components for basic search...")
    neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
    logger.info("Neo4j components initialized successfully")
    
    try:
        from .neo4j_context_builders import Neo4jBasicContextBuilder
        
        # Create context builder using config settings
        logger.info("Building Neo4j Basic Context Builder...")
        basic_config = config.neo4j.basic_search
        logger.info(f"Basic search parameters: top_k_text_units={basic_config.get('top_k_text_units', 20)}, "
                   f"max_context_tokens={basic_config.get('max_context_tokens', 8000)}")
        
        # Get token encoder for accurate token counting
        token_encoder = get_token_encoder_from_config(config)
        
        context_builder = Neo4jBasicContextBuilder(
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            token_encoder=token_encoder,
            max_context_tokens=basic_config.get("max_context_tokens", 12000),
            top_k_text_units=basic_config.get("top_k_text_units", 20),
            # Microsoft GraphRAG Enhanced Parameters (configurable)
            context_name=basic_config.get("context_name", "Sources"),
            column_delimiter=basic_config.get("column_delimiter", "|"),
            text_id_col=basic_config.get("text_id_col", "source_id"),
            text_col=basic_config.get("text_col", "text"),
            embedding_vectorstore_key=basic_config.get("embedding_vectorstore_key", "id")
        )
        logger.info("Neo4j Basic Context Builder created successfully")
        
        # Create search engine
        logger.info("Initializing Neo4j Basic Search Engine...")
        search_engine = BasicSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder,
            response_type=response_type
        )
        logger.info("Neo4j Basic Search Engine initialized successfully")
        
        # Execute search
        logger.info("Executing Neo4j Basic Search query...")
        result = await search_engine.search(query)
        
        logger.info("Neo4j Basic Search query completed successfully")
        logger.info(f"Search results: {len(result.response)} characters, "
                   f"{result.llm_calls} LLM calls, "
                   f"{result.prompt_tokens} prompt tokens, "
                   f"{result.output_tokens} output tokens")
        
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
    
    logger.info("Starting Neo4j DRIFT Search")
    logger.info(f"Query: '{query}' (community_level: {community_level}, response_type: {response_type})")
    
    # Validate Neo4j configuration
    logger.info("Validating Neo4j configuration...")
    validate_neo4j_config(config)
    logger.info("Neo4j configuration validated successfully")
    
    # Create Neo4j components from GraphRAG config
    logger.info("Initializing Neo4j components for DRIFT search...")
    neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
    logger.info("Neo4j components initialized successfully")
    
    try:
        from .neo4j_context_builders import Neo4jDRIFTContextBuilder
        from .drift_json_patch import apply_drift_json_patch, remove_drift_json_patch
        
        logger.info("Applying DRIFT JSON patch for robust parsing...")
        
        # Apply DRIFT JSON patch
        apply_drift_json_patch()
        logger.info("DRIFT JSON patch applied successfully")
        
        # Create context builder using config settings
        logger.info("Building Neo4j DRIFT Context Builder...")
        drift_config = config.neo4j.drift_search
        logger.info(f"DRIFT search parameters: n_depth={drift_config.get('n_depth', 3)}, "
                   f"drift_k_followups={drift_config.get('drift_k_followups', 5)}")
        
        # Get token encoder for accurate token counting
        token_encoder = get_token_encoder_from_config(config)
        
        context_builder = Neo4jDRIFTContextBuilder(
            neo4j_backend=neo4j_backend,
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            chat_model=chat_model,
            token_encoder=token_encoder,
            config=drift_config,
            # Microsoft GraphRAG Enhanced Parameters (configurable)
            embedding_vectorstore_key=drift_config.get("embedding_vectorstore_key", "id"),
            local_system_prompt=drift_config.get("local_system_prompt"),
            reduce_system_prompt=drift_config.get("reduce_system_prompt"),
            response_type=drift_config.get("response_type", "multiple paragraphs")
        )
        logger.info("Neo4j DRIFT Context Builder created successfully")
        
        # Create search engine
        logger.info("Initializing Neo4j DRIFT Search Engine...")
        search_engine = DRIFTSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder
        )
        logger.info("Neo4j DRIFT Search Engine initialized successfully")
        
        # Execute search with DRIFT context
        logger.info("Executing Neo4j DRIFT Search query...")
        from .drift_json_patch import set_drift_context
        try:
            set_drift_context(True)  # Enable DRIFT-specific JSON parsing
            result = await search_engine.search(query)
        finally:
            set_drift_context(False)  # Always reset context
        
        logger.info("Neo4j DRIFT Search query completed successfully")
        logger.info(f"Search results: {len(result.response)} characters, "
                   f"{result.llm_calls} LLM calls, "
                   f"{result.prompt_tokens} prompt tokens, "
                   f"{result.output_tokens} output tokens")
        
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
