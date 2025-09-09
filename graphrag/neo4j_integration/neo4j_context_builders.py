"""
Neo4j Context Builders for GraphRAG
===================================

This module provides Neo4j-based context builders that replace the original 
parquet file-based implementations while maintaining full API compatibility.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import logging
from dataclasses import dataclass

from graphrag.query.context_builder.builders import (
    GlobalContextBuilder,
    LocalContextBuilder,
    BasicContextBuilder,
    DRIFTContextBuilder,
    ContextBuilderResult
)
from graphrag.query.context_builder.conversation_history import ConversationHistory

from .neo4j_backend_implementation import Neo4jGraphBackend, Neo4jVectorGraphStore

logger = logging.getLogger(__name__)


class Neo4jGlobalContextBuilder(GlobalContextBuilder):
    """
    Neo4j-based implementation of GlobalContextBuilder.
    
    This builder retrieves community reports from Neo4j instead of parquet files,
    maintaining the same output format for the global search map-reduce pattern.
    """
    
    def __init__(self, 
                 neo4j_backend: Neo4jGraphBackend,
                 max_data_tokens: int = 8000,
                 community_level: int = 2,
                 min_community_rank: float = 0.0,
                 min_community_weight: int = 1):
        """
        Initialize the Neo4j Global Context Builder.
        
        Args:
            neo4j_backend: Neo4j backend instance
            max_data_tokens: Maximum tokens for context data
            community_level: Which hierarchical level to use for communities
            min_community_rank: Minimum rank threshold for communities
            min_community_weight: Minimum weight threshold for communities
        """
        self.neo4j = neo4j_backend
        self.max_data_tokens = max_data_tokens
        self.community_level = community_level
        self.min_community_rank = min_community_rank
        self.min_community_weight = min_community_weight
    
    async def build_context(self,
                           query: str,
                           conversation_history: ConversationHistory | None = None,
                           **kwargs) -> ContextBuilderResult:
        """
        Build context for global search using Neo4j community data.
        
        Returns context chunks formatted as CSV-like strings for the LLM,
        exactly matching the original implementation's format.
        """
        try:
            # Allow runtime override of community level
            level = kwargs.get("community_level", self.community_level)
            
            # Get community reports from Neo4j
            community_reports_df = self.neo4j.get_communities_by_level(level)
            
            if community_reports_df.empty:
                logger.warning(f"No community reports found for level {level}")
                return ContextBuilderResult(
                    context_chunks=[],
                    context_records={"community_reports": pd.DataFrame()}
                )
            
            # Apply filtering if specified
            filtered_df = community_reports_df
            if self.min_community_rank > 0:
                filtered_df = filtered_df[filtered_df["rank"] >= self.min_community_rank]
            if self.min_community_weight > 1:
                filtered_df = filtered_df[filtered_df["community_weight"] >= self.min_community_weight]
            
            # Format community reports as context chunks
            # This mirrors the original community_context.py formatting
            context_chunks = []
            current_batch = []
            current_tokens = 0
            estimated_tokens_per_char = 0.25  # Rough estimation
            
            for _, row in filtered_df.iterrows():
                # Format community report as CSV-like string
                report_text = self._format_community_report(row)
                report_tokens = len(report_text) * estimated_tokens_per_char
                
                if current_tokens + report_tokens > self.max_data_tokens and current_batch:
                    # Finalize current batch
                    context_chunks.append("\n".join(current_batch))
                    current_batch = [report_text]
                    current_tokens = report_tokens
                else:
                    current_batch.append(report_text)
                    current_tokens += report_tokens
            
            # Add final batch
            if current_batch:
                context_chunks.append("\n".join(current_batch))
            
            return ContextBuilderResult(
                context_chunks=context_chunks,
                context_records={"community_reports": filtered_df},
                llm_calls=0,  # No LLM calls in context building
                prompt_tokens=0,
                output_tokens=0
            )
            
        except Exception as e:
            logger.error(f"Failed to build global context: {e}")
            return ContextBuilderResult(
                context_chunks=[],
                context_records={"community_reports": pd.DataFrame()}
            )
    
    def _format_community_report(self, report: pd.Series) -> str:
        """Format a community report as CSV-like string for LLM consumption."""
        # Mirror the original formatting from community_context.py
        formatted_lines = [
            f"id,community,level,title,summary,findings,rank,rank_explanation",
            f"{report.get('id', '')},{report.get('community', '')},{report.get('level', '')},"
            f'"{report.get('title', '').replace('"', '""')}",'
            f'"{report.get('summary', '').replace('"', '""')}",'
            f'"{self._format_findings(report.get('findings', []))}",'
            f"{report.get('rank', '')},{report.get('rank_explanation', '')}"
        ]
        return "\n".join(formatted_lines)
    
    def _format_findings(self, findings: Any) -> str:
        """Format findings array as string."""
        if not findings:
            return ""
        if isinstance(findings, list):
            return "; ".join([str(f) for f in findings])
        return str(findings)


class Neo4jLocalContextBuilder(LocalContextBuilder):
    """
    Neo4j-based implementation of LocalContextBuilder.
    
    This builder uses hybrid vector-graph search to find relevant entities
    and then traverses the Neo4j graph to collect related information.
    """
    
    def __init__(self,
                 vector_graph_store: Neo4jVectorGraphStore,
                 embeddings_model,
                 max_context_tokens: int = 8000,
                 top_k_entities: int = 10,
                 top_k_text_units: int = 3,
                 top_k_relationships: int = 10,
                 top_k_communities: int = 3,
                 include_entity_rank: bool = True,
                 include_relationship_weight: bool = True):
        """
        Initialize the Neo4j Local Context Builder.
        
        Args:
            vector_graph_store: Hybrid vector-graph store
            embeddings_model: Model for generating query embeddings
            max_context_tokens: Maximum context tokens
            top_k_entities: Number of top entities to retrieve
            top_k_text_units: Number of text units per entity
            top_k_relationships: Number of relationships to include
            top_k_communities: Number of community reports to include
        """
        self.vector_graph_store = vector_graph_store
        self.embeddings_model = embeddings_model
        self.max_context_tokens = max_context_tokens
        self.top_k_entities = top_k_entities
        self.top_k_text_units = top_k_text_units
        self.top_k_relationships = top_k_relationships
        self.top_k_communities = top_k_communities
        self.include_entity_rank = include_entity_rank
        self.include_relationship_weight = include_relationship_weight
    
    def build_context(self,
                     query: str,
                     conversation_history: ConversationHistory | None = None,
                     **kwargs) -> ContextBuilderResult:
        """
        Build context for local search using hybrid vector-graph approach.
        
        This mirrors the retrieval pattern from ms_graphrag_retriever.ipynb.
        """
        try:
            # Override parameters if provided
            top_k_entities = kwargs.get("top_k_entities", self.top_k_entities)
            top_k_text_units = kwargs.get("top_k_text_units", self.top_k_text_units)
            top_k_relationships = kwargs.get("top_k_relationships", self.top_k_relationships)
            top_k_communities = kwargs.get("top_k_communities", self.top_k_communities)
            
            # 1. Generate query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            
            # 2. Get comprehensive context using hybrid approach
            context_data = self.vector_graph_store.get_local_search_context(
                query_embedding=query_embedding,
                top_k_entities=top_k_entities,
                top_k_text_units=top_k_text_units,
                top_k_relationships=top_k_relationships,
                top_k_communities=top_k_communities
            )
            
            # 3. Format context for LLM
            context_chunks = self._format_local_context(context_data)
            
            return ContextBuilderResult(
                context_chunks=context_chunks,
                context_records=context_data,
                llm_calls=0,
                prompt_tokens=0,
                output_tokens=0
            )
            
        except Exception as e:
            logger.error(f"Failed to build local context: {e}")
            return ContextBuilderResult(
                context_chunks="",
                context_records={}
            )
    
    def _format_local_context(self, context_data: Dict[str, pd.DataFrame]) -> str:
        """
        Format local context data as structured text for LLM.
        
        This mirrors the format used in the ms_graphrag_retriever.ipynb notebook.
        """
        sections = []
        
        # Text Units Section
        if not context_data["text_units"].empty:
            text_units = context_data["text_units"]["text"].tolist()
            sections.append("Chunks: " + " | ".join(text_units))
        
        # Community Reports Section  
        if not context_data["communities"].empty:
            summaries = context_data["communities"]["summary"].tolist()
            sections.append("Reports: " + " | ".join(summaries))
        
        # Relationships Section
        if not context_data["relationships"].empty:
            relationships = context_data["relationships"]["description"].tolist()
            sections.append("Relationships: " + " | ".join(relationships))
        
        # Entities Section
        if not context_data["entities"].empty:
            entities = [
                row["description"] for _, row in context_data["entities"].iterrows()
                if row["description"]
            ]
            sections.append("Entities: " + " | ".join(entities))
        
        return "\n".join(sections)


class Neo4jBasicContextBuilder(BasicContextBuilder):
    """
    Neo4j-based implementation of BasicContextBuilder.
    
    This builder performs traditional RAG by using vector search on text units
    without any graph traversal.
    """
    
    def __init__(self,
                 vector_graph_store: Neo4jVectorGraphStore,
                 embeddings_model,
                 max_context_tokens: int = 8000,
                 top_k_text_units: int = 20):
        """
        Initialize the Neo4j Basic Context Builder.
        
        Args:
            vector_graph_store: Hybrid vector-graph store
            embeddings_model: Model for generating query embeddings
            max_context_tokens: Maximum context tokens
            top_k_text_units: Number of text units to retrieve
        """
        self.vector_graph_store = vector_graph_store
        self.embeddings_model = embeddings_model
        self.max_context_tokens = max_context_tokens
        self.top_k_text_units = top_k_text_units
    
    def build_context(self,
                     query: str,
                     conversation_history: ConversationHistory | None = None,
                     **kwargs) -> ContextBuilderResult:
        """Build context for basic search using only text unit vectors."""
        try:
            # Override parameters if provided
            top_k = kwargs.get("top_k_text_units", self.top_k_text_units)
            
            # 1. Generate query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            
            # 2. Vector search on text units only
            relevant_text_units = self.vector_graph_store.similarity_search_text_units(
                query_embedding, k=top_k
            )
            
            # 3. Format as simple concatenated text
            context_texts = [unit["text"] for unit in relevant_text_units]
            context_chunks = "\n\n".join(context_texts)
            
            return ContextBuilderResult(
                context_chunks=context_chunks,
                context_records={"text_units": pd.DataFrame(relevant_text_units)},
                llm_calls=0,
                prompt_tokens=0,
                output_tokens=0
            )
            
        except Exception as e:
            logger.error(f"Failed to build basic context: {e}")
            return ContextBuilderResult(
                context_chunks="",
                context_records={}
            )


class Neo4jDRIFTContextBuilder(DRIFTContextBuilder):
    """
    Neo4j-based implementation of DRIFTContextBuilder.
    
    This builder provides context for the DRIFT search primer phase,
    which then uses local search internally. It's fully compatible with
    the existing DRIFT search framework.
    """
    
    def __init__(self,
                 neo4j_backend: Neo4jGraphBackend,
                 vector_graph_store: Neo4jVectorGraphStore,
                 embeddings_model,
                 chat_model,
                 config: Dict[str, Any] = None):
        """
        Initialize the Neo4j DRIFT Context Builder.
        
        Args:
            neo4j_backend: Neo4j graph backend
            vector_graph_store: Hybrid vector-graph store
            embeddings_model: Model for generating query embeddings
            chat_model: Chat model for DRIFT search
            config: DRIFT configuration parameters
        """
        from graphrag.config.models.drift_search_config import DRIFTSearchConfig
        
        self.neo4j_backend = neo4j_backend
        self.vector_graph_store = vector_graph_store
        self.embeddings_model = embeddings_model
        self.chat_model = chat_model
        
        # Create proper DRIFTSearchConfig object
        config = config or {}
        self.config = DRIFTSearchConfig(
            chat_model_id=config.get("chat_model_id", "bedrock_chat"),
            embedding_model_id=config.get("embedding_model_id", "bedrock_embedding"),
            data_max_tokens=config.get("data_max_tokens", 8000),
            reduce_max_tokens=config.get("reduce_max_tokens", 2000),
            reduce_temperature=config.get("reduce_temperature", 0.1),
            concurrency=config.get("concurrency", 5),
            drift_k_followups=config.get("drift_k_followups", 5),
            primer_folds=config.get("primer_folds", 3),
            primer_llm_max_tokens=config.get("primer_llm_max_tokens", 8000),
            n_depth=config.get("n_depth", 3),
            local_search_text_unit_prop=config.get("local_search_text_unit_prop", 0.5),
            local_search_community_prop=config.get("local_search_community_prop", 0.1),
            local_search_top_k_mapped_entities=config.get("local_search_top_k_mapped_entities", 10),
            local_search_top_k_relationships=config.get("local_search_top_k_relationships", 10),
            local_search_max_data_tokens=config.get("local_search_max_data_tokens", 8000),
            local_search_temperature=config.get("local_search_temperature", 0.1),
            local_search_top_p=config.get("local_search_top_p", 1.0),
            local_search_n=config.get("local_search_n", 1),
            local_search_llm_max_gen_tokens=config.get("local_search_llm_max_gen_tokens", 2000),
            local_search_llm_max_gen_completion_tokens=config.get("local_search_llm_max_gen_completion_tokens", 2000)
        )
        
        # Add required attributes for DRIFT search compatibility
        self.local_system_prompt = config.get("local_system_prompt", "You are a helpful assistant.")
        self.reduce_system_prompt = config.get("reduce_system_prompt", "You are a helpful assistant that reduces and summarizes information.")
        self.response_type = config.get("response_type", "multiple paragraphs")
        
        # Initialize local context builder for DRIFT internal use
        self.local_context_builder = Neo4jLocalContextBuilder(
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            max_context_tokens=self.config.local_search_max_data_tokens,
            top_k_entities=self.config.local_search_top_k_mapped_entities,
            top_k_relationships=self.config.local_search_top_k_relationships
        )
        
        # Create a simple mixed context that uses our Neo4j local context builder
        # This is required by the DRIFTSearch for its internal LocalSearch
        self.local_mixed_context = self._create_mixed_context_adapter()
    
    def _create_mixed_context_adapter(self):
        """
        Create a mixed context adapter that wraps our Neo4j local context builder
        to be compatible with DRIFT search expectations.
        """
        class Neo4jMixedContextAdapter:
            """Adapter to make Neo4j context builder compatible with DRIFT search."""
            
            def __init__(self, local_context_builder):
                self.local_context_builder = local_context_builder
            
            def build_context(self, query: str, **kwargs):
                """Build context using our Neo4j local context builder."""
                # Use our Neo4j local context builder - this returns a ContextBuilderResult
                context_result = self.local_context_builder.build_context(query, **kwargs)
                
                # Return the full ContextBuilderResult object, not just the text
                # The LocalSearch expects this to have llm_calls, prompt_tokens, etc.
                return context_result
        
        return Neo4jMixedContextAdapter(self.local_context_builder)
    
    async def build_context(self,
                           query: str,
                           **kwargs) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Build context for DRIFT primer search.
        
        Returns community reports as DataFrame for primer phase.
        """
        try:
            # Get community reports for primer (use higher level communities for broader context)
            primer_level = kwargs.get("primer_community_level", 1)
            community_reports_df = self.neo4j_backend.get_communities_by_level(primer_level)
            
            # If no reports at level 1, try level 2
            if community_reports_df.empty:
                logger.warning(f"No community reports found at level {primer_level}, trying level 2")
                community_reports_df = self.neo4j_backend.get_communities_by_level(2)
            
            # Estimate token usage (rough approximation)
            total_text_length = 0
            if not community_reports_df.empty:
                for col in ['title', 'summary', 'full_content']:
                    if col in community_reports_df.columns:
                        total_text_length += community_reports_df[col].fillna('').str.len().sum()
            
            estimated_tokens = int(total_text_length * 0.25)  # Rough token estimation
            
            token_counts = {
                "llm_calls": 0,
                "prompt_tokens": estimated_tokens,
                "output_tokens": 0
            }
            
            logger.info(f"DRIFT primer context: {len(community_reports_df)} community reports, ~{estimated_tokens} tokens")
            
            return community_reports_df, token_counts
            
        except Exception as e:
            logger.error(f"Failed to build DRIFT context: {e}")
            return pd.DataFrame(), {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0}


# ============================================================================
# Factory Functions for Easy Integration
# ============================================================================

def create_neo4j_context_builders(neo4j_config: Dict[str, str],
                                 lancedb_path: str,
                                 embeddings_model,
                                 chat_model) -> Dict[str, Any]:
    """
    Factory function to create all Neo4j context builders.
    
    Args:
        neo4j_config: Neo4j connection configuration
        lancedb_path: Path to LanceDB vector stores
        embeddings_model: Embeddings model instance
        chat_model: Chat model instance (needed for DRIFT)
        
    Returns:
        Dictionary containing all context builders
    """
    from neo4j_backend_implementation import Neo4jConfig, Neo4jGraphBackend, Neo4jVectorGraphStore
    
    # Initialize backend components
    config = Neo4jConfig(**neo4j_config)
    neo4j_backend = Neo4jGraphBackend(config)
    vector_graph_store = Neo4jVectorGraphStore(neo4j_backend, lancedb_path)
    
    # Create context builders
    builders = {
        "global": Neo4jGlobalContextBuilder(neo4j_backend),
        "local": Neo4jLocalContextBuilder(vector_graph_store, embeddings_model),
        "basic": Neo4jBasicContextBuilder(vector_graph_store, embeddings_model),
        "drift": Neo4jDRIFTContextBuilder(
            neo4j_backend=neo4j_backend,
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            chat_model=chat_model
        )
    }
    
    return builders


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # This would be used in practice to replace the existing context builders
    
    # Example configuration - replace with your actual Neo4j settings in settings.yaml
    neo4j_config = {
        "uri": "neo4j://localhost:7687",
        "username": "neo4j", 
        "password": "your-password",
        "database": "neo4j"
    }
    
    # Get LanceDB path from GraphRAG configuration
    # In production, this would be passed from the calling function via config.get_vector_store_config("default_vector_store").db_uri
    # For demo purposes, using environment variable with sensible default
    import os
    lancedb_path = os.getenv("LANCEDB_PATH", "./output/lancedb")
    
    # Note: You would need to initialize your embeddings model here
    # from graphrag.language_model.providers.bedrock.models import BedrockEmbeddingModel
    # from graphrag.config.models.language_model_config import LanguageModelConfig
    # from graphrag.config.enums import ModelType
    # 
    # config = LanguageModelConfig(
    #     type=ModelType.BedrockEmbedding,
    #     model="amazon.titan-embed-text-v2:0",
    #     aws_region="us-east-1"
    # )
    # embeddings_model = BedrockEmbeddingModel(name="bedrock_embedding", config=config)
    
    # Create builders
    # builders = create_neo4j_context_builders(neo4j_config, lancedb_path, embeddings_model)
    
    print("Neo4j context builders ready for integration with GraphRAG search engines")

