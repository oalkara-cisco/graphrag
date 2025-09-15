"""
Neo4j CLI Integration for GraphRAG
==================================

This module provides seamless integration between GraphRAG CLI and Neo4j backend,
handling all the routing logic and configuration management.
"""

import asyncio
import logging
from .token_utils import get_token_encoder_from_config
from typing import Dict, Any, Tuple, Optional

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.language_model.providers.bedrock.models import BedrockChatModel, BedrockEmbeddingModel
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.search import LocalSearch  
from graphrag.query.structured_search.basic_search.search import BasicSearch
from graphrag.query.structured_search.drift_search.search import DRIFTSearch

from .neo4j_backend_implementation import Neo4jConfig, Neo4jGraphBackend, Neo4jVectorGraphStore
from .neo4j_context_builders import (
    Neo4jGlobalContextBuilder,
    Neo4jLocalContextBuilder,
    Neo4jBasicContextBuilder,
    Neo4jDRIFTContextBuilder
)
from .drift_json_patch import apply_drift_json_patch, remove_drift_json_patch
from .token_utils import get_token_encoder_from_model

logger = logging.getLogger(__name__)


class BedrockEmbeddingWrapper:
    """Wrapper to provide embed_query compatibility for BedrockEmbeddingModel."""
    
    def __init__(self, bedrock_model: BedrockEmbeddingModel):
        self.bedrock_model = bedrock_model
    
    def embed_query(self, text: str) -> list[float]:
        return self.bedrock_model.embed(text)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.bedrock_model.embed_batch(texts)


def should_use_neo4j(graph_db: str | None, config: GraphRagConfig) -> bool:
    """Determine if Neo4j backend should be used."""
    return (graph_db == "neo4j" or 
            (config.neo4j and config.neo4j.enabled))


def validate_neo4j_config(config: GraphRagConfig) -> None:
    """Validate Neo4j configuration."""
    if not config.neo4j:
        raise ValueError(
            "Neo4j configuration not found in settings.yaml. "
            "Please add a 'neo4j:' section with uri, username, password, and database."
        )
    
    required_fields = ['uri', 'username', 'password']
    missing_fields = [field for field in required_fields if not getattr(config.neo4j, field, None)]
    
    if missing_fields:
        raise ValueError(
            f"Missing required Neo4j configuration fields: {', '.join(missing_fields)}. "
            "Please check your settings.yaml file."
        )


def create_neo4j_components(config: GraphRagConfig) -> Tuple[Neo4jGraphBackend, Neo4jVectorGraphStore, BedrockChatModel, BedrockEmbeddingWrapper]:
    """Create and initialize Neo4j components from GraphRAG configuration."""
    
    # Initialize Neo4j backend using config from settings.yaml
    logger.info("Creating Neo4j backend connection...")
    logger.info(f"Neo4j URI: {config.neo4j.uri}")
    logger.info(f"Neo4j Database: {config.neo4j.database}")
    
    neo4j_backend_config = Neo4jConfig(
        uri=config.neo4j.uri,
        username=config.neo4j.username,
        password=config.neo4j.password,
        database=config.neo4j.database
    )
    neo4j_backend = Neo4jGraphBackend(neo4j_backend_config)
    logger.info("Neo4j backend connection established successfully")
    
    # Get LanceDB path from vector store configuration
    vector_store_config = config.get_vector_store_config("default_vector_store")
    lancedb_path = vector_store_config.db_uri
    logger.info(f"LanceDB path: {lancedb_path}")
    
    # Create vector-graph store  
    logger.info("Creating Neo4j vector-graph store...")
    vector_graph_store = Neo4jVectorGraphStore(neo4j_backend, lancedb_path)
    logger.info("Neo4j vector-graph store created successfully")
    
    # Initialize AI models from GraphRAG config
    llm_config = config.models.get("default_chat_model", None)
    if not llm_config:
        raise ValueError("Chat model configuration not found. Please check your settings.yaml.")
    
    chat_model = BedrockChatModel(name="bedrock_chat", config=llm_config)
    
    # Initialize embedding model
    embeddings_config = config.models.get("default_embedding_model", None)
    if not embeddings_config:
        raise ValueError("Embedding model configuration not found. Please check your settings.yaml.")
        
    bedrock_embedding_model = BedrockEmbeddingModel(name="bedrock_embedding", config=embeddings_config)
    embeddings_model = BedrockEmbeddingWrapper(bedrock_embedding_model)
    
    return neo4j_backend, vector_graph_store, chat_model, embeddings_model


async def run_neo4j_global_search(
    config: GraphRagConfig,
    query: str,
    community_level: int | None = None,
    response_type: str = "multiple paragraphs",
    streaming: bool = False,
    verbose: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """Run global search using Neo4j backend."""
    
    if streaming:
        logger.warning("Streaming not yet supported with Neo4j backend")
    
    neo4j_backend = None
    try:
        # Apply DRIFT JSON patch for compatibility
        apply_drift_json_patch()
        
        # Validate and create components
        validate_neo4j_config(config)
        neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
        
        # Create token encoder from configuration
        token_encoder = get_token_encoder_from_config(config)
        
        # Create context builder with full Microsoft GraphRAG compatibility
        global_search_config = config.neo4j.global_search
        final_community_level = community_level or global_search_config.get("community_level", 2)
        
        context_builder = Neo4jGlobalContextBuilder(
            neo4j_backend=neo4j_backend,
            token_encoder=token_encoder,
            max_data_tokens=global_search_config.get("max_data_tokens", 8000),
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
        
        # Create search engine
        search_engine = GlobalSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder
        )
        
        # Run search
        result = await search_engine.search(query)
        
        if verbose:
            logger.info("Neo4j Global Search Response:\n%s", result.response)
        
        context_data = {
            "community_level": community_level,
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens
        }
        
        return result.response, context_data
        
    finally:
        remove_drift_json_patch()
        if neo4j_backend:
            neo4j_backend.close()


async def run_neo4j_local_search(
    config: GraphRagConfig,
    query: str,
    community_level: int = 2,
    response_type: str = "multiple paragraphs",
    streaming: bool = False,
    verbose: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """Run local search using Neo4j backend."""
    
    if streaming:
        logger.warning("Streaming not yet supported with Neo4j backend")
    
    vector_graph_store = None
    try:
        # Apply DRIFT JSON patch for compatibility
        apply_drift_json_patch()
        
        # Validate and create components
        validate_neo4j_config(config)
        neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
        
        # Create token encoder from configuration
        token_encoder = get_token_encoder_from_config(config)
        
        # Create context builder
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
        search_engine = LocalSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder,
            response_type=response_type
        )
        
        # Run search
        result = await search_engine.search(query)
        
        if verbose:
            logger.info("Neo4j Local Search Response:\n%s", result.response)
        
        context_data = {
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens
        }
        
        return result.response, context_data
        
    finally:
        remove_drift_json_patch()
        if vector_graph_store:
            vector_graph_store.close()


async def run_neo4j_basic_search(
    config: GraphRagConfig,
    query: str,
    streaming: bool = False,
    verbose: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """Run basic search using Neo4j backend."""
    
    if streaming:
        logger.warning("Streaming not yet supported with Neo4j backend")
    
    vector_graph_store = None
    try:
        # Apply DRIFT JSON patch for compatibility
        apply_drift_json_patch()
        
        # Validate and create components
        validate_neo4j_config(config)
        neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
        
        # Create token encoder from configuration
        token_encoder = get_token_encoder_from_config(config)
        
        # Create context builder
        basic_config = config.neo4j.basic_search
        context_builder = Neo4jBasicContextBuilder(
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            max_context_tokens=basic_config.get("max_context_tokens", 8000),
            top_k_text_units=basic_config.get("top_k_text_units", 20)
        )
        
        # Create search engine
        search_engine = BasicSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder,
            response_type="multiple paragraphs"
        )
        
        # Run search
        result = await search_engine.search(query)
        
        if verbose:
            logger.info("Neo4j Basic Search Response:\n%s", result.response)
        
        context_data = {
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens
        }
        
        return result.response, context_data
        
    finally:
        remove_drift_json_patch()
        if vector_graph_store:
            vector_graph_store.close()


async def run_neo4j_drift_search(
    config: GraphRagConfig,
    query: str,
    community_level: int = 2,
    response_type: str = "multiple paragraphs",
    streaming: bool = False,
    verbose: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """Run DRIFT search using Neo4j backend."""
    
    if streaming:
        logger.warning("Streaming not yet supported with Neo4j backend")
    
    vector_graph_store = None
    try:
        # Apply DRIFT JSON patch for compatibility
        apply_drift_json_patch()
        
        # Validate and create components
        validate_neo4j_config(config)
        neo4j_backend, vector_graph_store, chat_model, embeddings_model = create_neo4j_components(config)
        
        # Create token encoder from configuration
        token_encoder = get_token_encoder_from_config(config)
        
        # Create context builder
        drift_config = config.neo4j.drift_search
        context_builder = Neo4jDRIFTContextBuilder(
            neo4j_backend=neo4j_backend,
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            chat_model=chat_model,
            config=drift_config
        )
        
        # Create search engine
        search_engine = DRIFTSearch(
            model=chat_model,
            context_builder=context_builder,
            token_encoder=token_encoder
        )
        
        # Run search
        result = await search_engine.search(query)
        
        if verbose:
            logger.info("Neo4j DRIFT Search Response:\n%s", result.response)
        
        context_data = {
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens
        }
        
        return result.response, context_data
        
    finally:
        remove_drift_json_patch()
        if vector_graph_store:
            vector_graph_store.close()


# Unified interface functions for CLI integration
def handle_global_search_with_neo4j(
    config: GraphRagConfig,
    query: str,
    graph_db: str | None,
    community_level: int | None = None,
    response_type: str = "multiple paragraphs",
    streaming: bool = False,
    verbose: bool = False
) -> Tuple[str, Dict[str, Any]] | None:
    """Handle global search with Neo4j backend if enabled."""
    
    if should_use_neo4j(graph_db, config):
        logger.info("Using Neo4j backend for global search")
        return asyncio.run(run_neo4j_global_search(
            config, query, community_level, response_type, streaming, verbose
        ))
    return None


def handle_local_search_with_neo4j(
    config: GraphRagConfig,
    query: str,
    graph_db: str | None,
    community_level: int = 2,
    response_type: str = "multiple paragraphs",
    streaming: bool = False,
    verbose: bool = False
) -> Tuple[str, Dict[str, Any]] | None:
    """Handle local search with Neo4j backend if enabled."""
    
    if should_use_neo4j(graph_db, config):
        logger.info("Using Neo4j backend for local search")
        return asyncio.run(run_neo4j_local_search(
            config, query, community_level, response_type, streaming, verbose
        ))
    return None


def handle_basic_search_with_neo4j(
    config: GraphRagConfig,
    query: str,
    graph_db: str | None,
    streaming: bool = False,
    verbose: bool = False
) -> Tuple[str, Dict[str, Any]] | None:
    """Handle basic search with Neo4j backend if enabled."""
    
    if should_use_neo4j(graph_db, config):
        logger.info("Using Neo4j backend for basic search")
        return asyncio.run(run_neo4j_basic_search(
            config, query, streaming, verbose
        ))
    return None


def handle_drift_search_with_neo4j(
    config: GraphRagConfig,
    query: str,
    graph_db: str | None,
    community_level: int = 2,
    response_type: str = "multiple paragraphs",
    streaming: bool = False,
    verbose: bool = False
) -> Tuple[str, Dict[str, Any]] | None:
    """Handle DRIFT search with Neo4j backend if enabled."""
    
    if should_use_neo4j(graph_db, config):
        logger.info("Using Neo4j backend for DRIFT search")
        return asyncio.run(run_neo4j_drift_search(
            config, query, community_level, response_type, streaming, verbose
        ))
    return None
