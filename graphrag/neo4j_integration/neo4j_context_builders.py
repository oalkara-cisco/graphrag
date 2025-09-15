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
    maintaining the same output format and advanced features as Microsoft's original
    parquet-based implementation for optimal LLM consumption.
    """
    
    def __init__(self, 
                 neo4j_backend: Neo4jGraphBackend,
                 token_encoder = None,
                 max_data_tokens: int = 8000,
                 community_level: int = 2,
                 min_community_rank: float = 0.0,
                 min_community_weight: int = 1,
                 use_community_summary: bool = True,
                 column_delimiter: str = "|",
                 shuffle_data: bool = True,
                 include_community_rank: bool = False,
                 community_rank_name: str = "rank",
                 include_community_weight: bool = True,
                 community_weight_name: str = "occurrence weight",
                 normalize_community_weight: bool = True,
                 single_batch: bool = True,
                 context_name: str = "Reports",
                 random_state: int = 86):
        """
        Initialize the Neo4j Global Context Builder with full Microsoft GraphRAG compatibility.
        
        Args:
            neo4j_backend: Neo4j backend instance
            token_encoder: Token encoder for accurate token counting (tiktoken)
            max_data_tokens: Maximum tokens for context data
            community_level: Which hierarchical level to use for communities
            min_community_rank: Minimum rank threshold for communities
            min_community_weight: Minimum weight threshold for communities
            use_community_summary: Use summary instead of full content
            column_delimiter: Delimiter for CSV formatting
            shuffle_data: Whether to shuffle community reports
            include_community_rank: Include rank in output
            community_rank_name: Name of rank column
            include_community_weight: Include weight calculations
            community_weight_name: Name of weight column
            normalize_community_weight: Normalize weights to 0-1 range
            single_batch: Whether to use single batch mode
            context_name: Name for context sections
            random_state: Random seed for shuffling
        """
        self.neo4j = neo4j_backend
        self.token_encoder = token_encoder
        self.max_data_tokens = max_data_tokens
        self.community_level = community_level
        self.min_community_rank = min_community_rank
        self.min_community_weight = min_community_weight
        self.use_community_summary = use_community_summary
        self.column_delimiter = column_delimiter
        self.shuffle_data = shuffle_data
        self.include_community_rank = include_community_rank
        self.community_rank_name = community_rank_name
        self.include_community_weight = include_community_weight
        self.community_weight_name = community_weight_name
        self.normalize_community_weight = normalize_community_weight
        self.single_batch = single_batch
        self.context_name = context_name
        self.random_state = random_state
    
    async def build_context(self,
                           query: str,
                           conversation_history: ConversationHistory | None = None,
                           **kwargs) -> ContextBuilderResult:
        """
        Build context for global search using Neo4j community data.
        
        This implementation mirrors Microsoft's build_community_context function
        with full parameter support, community weight calculation, proper batching,
        and advanced CSV formatting for optimal LLM consumption.
        """
        try:
            from graphrag.query.llm.text_utils import num_tokens
            import random
            
            # Allow runtime parameter overrides
            level = kwargs.get("community_level", self.community_level)
            use_community_summary = kwargs.get("use_community_summary", self.use_community_summary)
            column_delimiter = kwargs.get("column_delimiter", self.column_delimiter)
            shuffle_data = kwargs.get("shuffle_data", self.shuffle_data)
            include_community_rank = kwargs.get("include_community_rank", self.include_community_rank)
            min_community_rank = kwargs.get("min_community_rank", self.min_community_rank)
            community_rank_name = kwargs.get("community_rank_name", self.community_rank_name)
            include_community_weight = kwargs.get("include_community_weight", self.include_community_weight)
            community_weight_name = kwargs.get("community_weight_name", self.community_weight_name)
            normalize_community_weight = kwargs.get("normalize_community_weight", self.normalize_community_weight)
            max_context_tokens = kwargs.get("max_context_tokens", self.max_data_tokens)
            single_batch = kwargs.get("single_batch", self.single_batch)
            context_name = kwargs.get("context_name", self.context_name)
            random_state = kwargs.get("random_state", self.random_state)
            
            logger.info(f"Building Neo4j global context (level: {level}, summary: {use_community_summary}, "
                       f"rank: {include_community_rank}, weight: {include_community_weight})")
            
            # 1. Get community reports from Neo4j
            community_reports_df = self.neo4j.get_communities_by_level(level)
            logger.info(f"Retrieved {len(community_reports_df)} community reports from Neo4j")
            
            if community_reports_df.empty:
                logger.warning(f"No community reports found for level {level}")
                return ContextBuilderResult(
                    context_chunks=[],
                    context_records={"community_reports": pd.DataFrame()}
                )
            
            # 2. Convert DataFrame rows to CommunityReport-like objects for processing
            community_reports = self._convert_df_to_reports(community_reports_df)
            
            # 3. Compute community weights if needed (enhanced version)
            if include_community_weight and self._should_compute_weights(community_reports, community_weight_name):
                logger.info("Computing community weights from Neo4j entity data...")
                community_reports = await self._compute_neo4j_community_weights(
                    community_reports, community_weight_name, normalize_community_weight
                )
            
            # 4. Filter reports by rank
            selected_reports = [
                report for report in community_reports
                if report.get("rank") is not None and report.get("rank", 0) >= min_community_rank
            ]
            
            if not selected_reports:
                logger.warning("No community reports passed rank filtering")
                return ContextBuilderResult(context_chunks=[], context_records={})
            
            # 5. Shuffle if requested
            if shuffle_data:
                random.seed(random_state)
                random.shuffle(selected_reports)
                logger.info(f"Shuffled {len(selected_reports)} reports with seed {random_state}")
            
            # 6. Build CSV context using Microsoft's algorithm
            context_chunks, context_records = self._build_csv_context(
                selected_reports=selected_reports,
                use_community_summary=use_community_summary,
                column_delimiter=column_delimiter,
                include_community_rank=include_community_rank,
                community_rank_name=community_rank_name,
                include_community_weight=include_community_weight,
                community_weight_name=community_weight_name,
                max_context_tokens=max_context_tokens,
                single_batch=single_batch,
                context_name=context_name
            )
            
            logger.info(f"Built {len(context_chunks)} context chunks for global search")
            
            return ContextBuilderResult(
                context_chunks=context_chunks,
                context_records=context_records,
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
    
    def _convert_df_to_reports(self, community_reports_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert Neo4j DataFrame to CommunityReport-like objects for processing."""
        reports = []
        for _, row in community_reports_df.iterrows():
            report = {
                "id": row.get("id", ""),
                "short_id": row.get("id", ""),  # Use id as short_id
                "title": row.get("title", ""),
                "summary": row.get("summary", ""),
                "full_content": row.get("full_content", row.get("summary", "")),
                "rank": row.get("rank", 0.0),
                "rank_explanation": row.get("rank_explanation", ""),
                "community_id": row.get("community", row.get("id", "")),
                "sources": row.get("sources", []),
                "attributes": {}  # Will be populated with weights if needed
            }
            reports.append(report)
        return reports
    
    def _should_compute_weights(self, community_reports: List[Dict[str, Any]], weight_attribute: str) -> bool:
        """Check if we need to compute community weights."""
        if not community_reports:
            return False
        
        # Check if any report is missing the weight attribute
        return any(
            not report.get("attributes") or 
            weight_attribute not in report.get("attributes", {})
            for report in community_reports
        )
    
    async def _compute_neo4j_community_weights(self, 
                                             community_reports: List[Dict[str, Any]], 
                                             weight_attribute: str = "occurrence weight",
                                             normalize: bool = True) -> List[Dict[str, Any]]:
        """
        Compute community weights based on Neo4j entity and text unit data.
        
        This mirrors Microsoft's _compute_community_weights but uses Neo4j queries.
        """
        try:
            # Get entity-to-text-unit mappings from Neo4j
            entity_text_units_cypher = """
            MATCH (e:__Entity__)-[:HAS_ENTITY]-(c:__Chunk__)
            OPTIONAL MATCH (e)-[:MEMBER_OF]->(com:__Community__)
            RETURN 
                e.id as entity_id,
                collect(DISTINCT c.id) as text_unit_ids,
                collect(DISTINCT com.id) as community_ids
            """
            
            entity_data = self.neo4j.execute_query(entity_text_units_cypher)
            
            # Build community -> text units mapping
            community_text_units = {}
            for _, row in entity_data.iterrows():
                entity_communities = row.get("community_ids", []) or []
                entity_text_units = row.get("text_unit_ids", []) or []
                
                for community_id in entity_communities:
                    if community_id and community_id not in community_text_units:
                        community_text_units[community_id] = set()
                    if community_id:
                        community_text_units[community_id].update(entity_text_units)
            
            # Calculate weights for each community report
            for report in community_reports:
                if "attributes" not in report:
                    report["attributes"] = {}
                
                community_id = report.get("community_id", report.get("id", ""))
                text_units = community_text_units.get(community_id, set())
                report["attributes"][weight_attribute] = len(text_units)
            
            # Normalize weights if requested
            if normalize and community_reports:
                all_weights = [
                    report["attributes"].get(weight_attribute, 0)
                    for report in community_reports
                ]
                max_weight = max(all_weights) if all_weights else 1
                
                if max_weight > 0:
                    for report in community_reports:
                        current_weight = report["attributes"].get(weight_attribute, 0)
                        report["attributes"][weight_attribute] = current_weight / max_weight
            
            logger.info(f"Computed weights for {len(community_reports)} communities")
            return community_reports
            
        except Exception as e:
            logger.error(f"Failed to compute community weights: {e}")
            # Return reports with default weights
            for report in community_reports:
                if "attributes" not in report:
                    report["attributes"] = {}
                report["attributes"][weight_attribute] = 1.0
            return community_reports
    
    def _build_csv_context(self,
                          selected_reports: List[Dict[str, Any]],
                          use_community_summary: bool,
                          column_delimiter: str,
                          include_community_rank: bool,
                          community_rank_name: str,
                          include_community_weight: bool,
                          community_weight_name: str,
                          max_context_tokens: int,
                          single_batch: bool,
                          context_name: str) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
        """
        Build CSV context using Microsoft's advanced algorithm.
        
        This mirrors the batching and formatting logic from build_community_context.
        """
        from graphrag.query.llm.text_utils import num_tokens
        
        if not selected_reports:
            return [], {}
        
        # Build header (Microsoft's _get_header logic)
        header = ["id", "title"]
        attributes = list(selected_reports[0].get("attributes", {}).keys()) if selected_reports[0].get("attributes") else []
        attributes = [col for col in attributes if col not in header]
        
        if not include_community_weight:
            attributes = [col for col in attributes if col != community_weight_name]
        header.extend(attributes)
        header.append("summary" if use_community_summary else "content")
        if include_community_rank:
            header.append(community_rank_name)
        
        # Prepare batching variables
        all_context_text: List[str] = []
        all_context_records: List[pd.DataFrame] = []
        
        # Initialize first batch
        batch_text = f"-----{context_name}-----\n" + column_delimiter.join(header) + "\n"
        batch_tokens = num_tokens(batch_text, self.token_encoder)
        batch_records: List[List[str]] = []
        
        def _cut_batch():
            """Convert current batch to DataFrame and add to results."""
            nonlocal all_context_text, all_context_records, batch_records
            
            if not batch_records:
                return
            
            # Create DataFrame from batch records
            record_df = pd.DataFrame(batch_records, columns=header)
            
            # Sort by weight and rank (Microsoft's _rank_report_context logic)
            rank_attributes = []
            if include_community_weight and community_weight_name in record_df.columns:
                rank_attributes.append(community_weight_name)
                record_df[community_weight_name] = pd.to_numeric(record_df[community_weight_name], errors='coerce').fillna(0)
            if include_community_rank and community_rank_name in record_df.columns:
                rank_attributes.append(community_rank_name)
                record_df[community_rank_name] = pd.to_numeric(record_df[community_rank_name], errors='coerce').fillna(0)
            
            if rank_attributes:
                record_df.sort_values(by=rank_attributes, ascending=False, inplace=True)
            
            # Convert to CSV text
            current_context_text = record_df.to_csv(index=False, sep=column_delimiter)
            if not all_context_text and single_batch:
                current_context_text = f"-----{context_name}-----\n{current_context_text}"
            
            all_context_text.append(current_context_text)
            all_context_records.append(record_df)
        
        # Process each report
        for report in selected_reports:
            # Build context row (Microsoft's _report_context_text logic)
            context_row = [
                report.get("short_id", report.get("id", "")),
                report.get("title", "")
            ]
            
            # Add attributes
            for field in attributes:
                if field == community_weight_name and include_community_weight:
                    weight = report.get("attributes", {}).get(field, 0)
                    context_row.append(str(weight))
                else:
                    attr_value = report.get("attributes", {}).get(field, "")
                    context_row.append(str(attr_value))
            
            # Add content (summary or full content)
            content = report.get("summary", "") if use_community_summary else report.get("full_content", "")
            context_row.append(content)
            
            # Add rank if requested
            if include_community_rank:
                context_row.append(str(report.get("rank", "")))
            
            # Calculate tokens for this row
            row_text = column_delimiter.join(context_row) + "\n"
            row_tokens = num_tokens(row_text, self.token_encoder)
            
            # Check if we need to start a new batch
            if batch_tokens + row_tokens > max_context_tokens and batch_records:
                _cut_batch()
                if single_batch:
                    break
                
                # Start new batch
                batch_text = f"-----{context_name}-----\n" + column_delimiter.join(header) + "\n"
                batch_tokens = num_tokens(batch_text, self.token_encoder)
                batch_records = []
            
            # Add to current batch
            batch_tokens += row_tokens
            batch_records.append(context_row)
        
        # Process final batch
        if batch_records:
            # Check for duplicate batches (Microsoft's duplicate prevention)
            current_batch_ids = {record[0] for record in batch_records}
            existing_ids_sets = [set(record["id"].tolist()) for record in all_context_records]
            
            if current_batch_ids not in existing_ids_sets:
                _cut_batch()
        
        if not all_context_records:
            logger.warning("No community records added when building context")
            return [], {}
        
        # Combine all records
        final_records = pd.concat(all_context_records, ignore_index=True)
        
        return all_context_text, {context_name.lower(): final_records}


class Neo4jLocalContextBuilder(LocalContextBuilder):
    """
    Neo4j-based implementation of LocalContextBuilder.
    
    Enhanced to match Microsoft's LocalSearchMixedContext with proportional
    token allocation, conversation history support, and advanced parameters
    for optimal local search performance.
    """
    
    def __init__(self,
                 vector_graph_store: Neo4jVectorGraphStore,
                 embeddings_model,
                 token_encoder = None,
                 max_context_tokens: int = 8000,
                 text_unit_prop: float = 0.5,
                 community_prop: float = 0.25,
                 top_k_mapped_entities: int = 10,
                 top_k_relationships: int = 10,
                 top_k_text_units: int = 3,
                 top_k_communities: int = 3,
                 include_community_rank: bool = False,
                 include_entity_rank: bool = False,
                 rank_description: str = "number of relationships",
                 include_relationship_weight: bool = False,
                 relationship_ranking_attribute: str = "rank",
                 return_candidate_context: bool = False,
                 use_community_summary: bool = False,
                 min_community_rank: int = 0,
                 community_context_name: str = "Reports",
                 column_delimiter: str = "|",
                 embedding_vectorstore_key: str = "id"):
        """
        Initialize the Neo4j Local Context Builder with Microsoft GraphRAG compatibility.
        
        Args:
            vector_graph_store: Hybrid vector-graph store
            embeddings_model: Model for generating query embeddings
            token_encoder: Token encoder for accurate token counting
            max_context_tokens: Maximum context tokens
            text_unit_prop: Proportion of tokens for text units (0.5 = 50%)
            community_prop: Proportion of tokens for community reports (0.25 = 25%)
            top_k_mapped_entities: Number of top entities to map from query
            top_k_relationships: Number of relationships to include
            top_k_text_units: Number of text units per entity
            top_k_communities: Number of community reports to include
            include_community_rank: Include community ranking
            include_entity_rank: Include entity ranking
            rank_description: Description for ranking
            include_relationship_weight: Include relationship weights
            relationship_ranking_attribute: Attribute for relationship ranking
            return_candidate_context: Return candidate context data
            use_community_summary: Use summary instead of full content
            min_community_rank: Minimum community rank threshold
            community_context_name: Name for community context section
            column_delimiter: Delimiter for CSV formatting
            embedding_vectorstore_key: Key for embedding vector store
        """
        self.vector_graph_store = vector_graph_store
        self.embeddings_model = embeddings_model
        self.token_encoder = token_encoder
        self.max_context_tokens = max_context_tokens
        self.text_unit_prop = text_unit_prop
        self.community_prop = community_prop
        self.top_k_mapped_entities = top_k_mapped_entities
        self.top_k_relationships = top_k_relationships
        self.top_k_text_units = top_k_text_units
        self.top_k_communities = top_k_communities
        self.include_community_rank = include_community_rank
        self.include_entity_rank = include_entity_rank
        self.rank_description = rank_description
        self.include_relationship_weight = include_relationship_weight
        self.relationship_ranking_attribute = relationship_ranking_attribute
        self.return_candidate_context = return_candidate_context
        self.use_community_summary = use_community_summary
        self.min_community_rank = min_community_rank
        self.community_context_name = community_context_name
        self.column_delimiter = column_delimiter
        self.embedding_vectorstore_key = embedding_vectorstore_key
    
    def build_context(self,
                     query: str,
                     conversation_history: ConversationHistory | None = None,
                     **kwargs) -> ContextBuilderResult:
        """
        Build context for local search using Microsoft's advanced proportional approach.
        
        This implementation mirrors LocalSearchMixedContext with:
        - Proportional token allocation between components
        - Entity mapping using embeddings
        - Community matching based on selected entities
        - Conversation history support
        - Advanced sorting and ranking
        """
        try:
            from graphrag.query.llm.text_utils import num_tokens
            
            # Override parameters from kwargs
            max_context_tokens = kwargs.get("max_context_tokens", self.max_context_tokens)
            text_unit_prop = kwargs.get("text_unit_prop", self.text_unit_prop)
            community_prop = kwargs.get("community_prop", self.community_prop)
            top_k_mapped_entities = kwargs.get("top_k_mapped_entities", self.top_k_mapped_entities)
            top_k_relationships = kwargs.get("top_k_relationships", self.top_k_relationships)
            include_community_rank = kwargs.get("include_community_rank", self.include_community_rank)
            include_entity_rank = kwargs.get("include_entity_rank", self.include_entity_rank)
            rank_description = kwargs.get("rank_description", self.rank_description)
            include_relationship_weight = kwargs.get("include_relationship_weight", self.include_relationship_weight)
            relationship_ranking_attribute = kwargs.get("relationship_ranking_attribute", self.relationship_ranking_attribute)
            return_candidate_context = kwargs.get("return_candidate_context", self.return_candidate_context)
            use_community_summary = kwargs.get("use_community_summary", self.use_community_summary)
            min_community_rank = kwargs.get("min_community_rank", self.min_community_rank)
            community_context_name = kwargs.get("community_context_name", self.community_context_name)
            column_delimiter = kwargs.get("column_delimiter", self.column_delimiter)
            
            # Validate proportions
            if community_prop + text_unit_prop > 1:
                raise ValueError("The sum of community_prop and text_unit_prop should not exceed 1.")
            
            logger.info(f"Building Neo4j Local Search context with proportional allocation: "
                       f"community={community_prop:.1%}, text_units={text_unit_prop:.1%}, "
                       f"local_entities={1-community_prop-text_unit_prop:.1%}")
            
            # Handle conversation history and adjust available tokens
            final_context = []
            final_context_data = {}
            available_tokens = max_context_tokens
            
            if conversation_history:
                logger.info("Building conversation history context...")
                conversation_context, conversation_context_data = conversation_history.build_context(
                    include_user_turns_only=kwargs.get("conversation_history_user_turns_only", True),
                    max_qa_turns=kwargs.get("conversation_history_max_turns", 5),
                    column_delimiter=column_delimiter,
                    max_context_tokens=max_context_tokens,
                    recency_bias=False,
                )
                if conversation_context.strip():
                    final_context.append(conversation_context)
                    final_context_data.update(conversation_context_data)
                    available_tokens -= num_tokens(conversation_context, self.token_encoder)
                    logger.info(f"Added conversation context, remaining tokens: {available_tokens}")
            
            # 1. Map query to entities using embeddings (Microsoft's approach)
            logger.info(f"Mapping query to top {top_k_mapped_entities} entities using embeddings...")
            selected_entities = self._map_query_to_entities(
                query=query,
                conversation_history=conversation_history,
                k=top_k_mapped_entities,
                **kwargs
            )
            logger.info(f"Selected {len(selected_entities)} entities from query mapping")
            
            # 2. Build community context (proportional allocation)
            community_tokens = max(int(available_tokens * community_prop), 0)
            logger.info(f"Building community context with {community_tokens} tokens ({community_prop:.1%})")
            community_context, community_context_data = self._build_community_context_proportional(
                selected_entities=selected_entities,
                max_context_tokens=community_tokens,
                use_community_summary=use_community_summary,
                column_delimiter=column_delimiter,
                include_community_rank=include_community_rank,
                min_community_rank=min_community_rank,
                return_candidate_context=return_candidate_context,
                context_name=community_context_name,
            )
            if community_context.strip():
                final_context.append(community_context)
                final_context_data.update(community_context_data)
            
            # 3. Build local entity-relationship context (remaining proportion)
            local_prop = 1 - community_prop - text_unit_prop
            local_tokens = max(int(available_tokens * local_prop), 0)
            logger.info(f"Building local entity-relationship context with {local_tokens} tokens ({local_prop:.1%})")
            local_context, local_context_data = self._build_local_entity_relationship_context(
                selected_entities=selected_entities,
                max_context_tokens=local_tokens,
                include_entity_rank=include_entity_rank,
                rank_description=rank_description,
                include_relationship_weight=include_relationship_weight,
                top_k_relationships=top_k_relationships,
                relationship_ranking_attribute=relationship_ranking_attribute,
                return_candidate_context=return_candidate_context,
                column_delimiter=column_delimiter,
            )
            if local_context.strip():
                final_context.append(local_context)
                final_context_data.update(local_context_data)
            
            # 4. Build text unit context (proportional allocation)
            text_unit_tokens = max(int(available_tokens * text_unit_prop), 0)
            logger.info(f"Building text unit context with {text_unit_tokens} tokens ({text_unit_prop:.1%})")
            text_unit_context, text_unit_context_data = self._build_text_unit_context_proportional(
                selected_entities=selected_entities,
                max_context_tokens=text_unit_tokens,
                return_candidate_context=return_candidate_context,
                column_delimiter=column_delimiter,
            )
            if text_unit_context.strip():
                final_context.append(text_unit_context)
                final_context_data.update(text_unit_context_data)
            
            # Combine all context components
            final_context_text = "\n\n".join(final_context)
            
            logger.info(f"Built comprehensive local context: {len(final_context_text)} characters, "
                       f"{len(final_context)} components")
            
            return ContextBuilderResult(
                context_chunks=final_context_text,
                context_records=final_context_data,
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
    
    def _map_query_to_entities(self,
                              query: str,
                              conversation_history: ConversationHistory | None = None,
                              k: int = 10,
                              include_entity_names: List[str] | None = None,
                              exclude_entity_names: List[str] | None = None,
                              **kwargs) -> List[Dict[str, Any]]:
        """
        Map query to entities using embeddings (Microsoft's approach).
        
        This mirrors the map_query_to_entities function from Microsoft's implementation.
        """
        try:
            if include_entity_names is None:
                include_entity_names = []
            if exclude_entity_names is None:
                exclude_entity_names = []
            
            # Enhance query with conversation history if provided
            enhanced_query = query
            if conversation_history:
                max_turns = kwargs.get("conversation_history_max_turns", 5)
                pre_user_questions = "\n".join(
                    conversation_history.get_user_turns(max_turns)
                )
                enhanced_query = f"{query}\n{pre_user_questions}"
            
            # Generate query embedding
            query_embedding = self.embeddings_model.embed_query(enhanced_query)
            
            # Get top-k entities using vector similarity
            similar_entities = self.vector_graph_store.similarity_search_entities(
                query_embedding=query_embedding,
                k=k * 2  # Oversample for filtering
            )
            
            # Filter by include/exclude lists
            filtered_entities = []
            for entity in similar_entities:
                entity_name = entity.get("name", "")
                
                # Apply include filter
                if include_entity_names and entity_name not in include_entity_names:
                    continue
                    
                # Apply exclude filter
                if exclude_entity_names and entity_name in exclude_entity_names:
                    continue
                    
                filtered_entities.append(entity)
                
                # Stop when we have enough
                if len(filtered_entities) >= k:
                    break
            
            logger.info(f"Mapped query to {len(filtered_entities)} entities (from {len(similar_entities)} candidates)")
            return filtered_entities
            
        except Exception as e:
            logger.error(f"Failed to map query to entities: {e}")
            return []
    
    def _build_community_context_proportional(self,
                                             selected_entities: List[Dict[str, Any]],
                                             max_context_tokens: int = 2000,
                                             use_community_summary: bool = False,
                                             column_delimiter: str = "|",
                                             include_community_rank: bool = False,
                                             min_community_rank: int = 0,
                                             return_candidate_context: bool = False,
                                             context_name: str = "Reports") -> Tuple[str, Dict[str, pd.DataFrame]]:
        """
        Build community context using Microsoft's proportional approach.
        
        This mirrors the _build_community_context method from LocalSearchMixedContext.
        """
        try:
            if not selected_entities:
                return ("", {context_name.lower(): pd.DataFrame()})
            
            # Find communities that contain the selected entities
            entity_ids = [entity.get("id", "") for entity in selected_entities]
            
            # Get community reports for entities using Neo4j
            community_cypher = """
            UNWIND $entity_ids as entity_id
            MATCH (e:__Entity__ {id: entity_id})-[:MEMBER_OF]->(c:__Community__)
            MATCH (cr:CommunityReport)-[:ANALYZES]->(c)
            WITH cr, c, count(DISTINCT e) as entity_matches
            RETURN 
                cr.id as id,
                cr.community as community_id,
                cr.title as title,
                cr.summary as summary,
                cr.full_content as full_content,
                cr.rank as rank,
                c.weight as weight,
                entity_matches
            ORDER BY entity_matches DESC, cr.rank DESC
            """
            
            community_data = self.vector_graph_store.neo4j.execute_query(
                community_cypher, {"entity_ids": entity_ids}
            )
            
            if community_data.empty:
                return ("", {context_name.lower(): pd.DataFrame()})
            
            # Apply rank filtering
            if min_community_rank > 0:
                community_data = community_data[
                    community_data["rank"].fillna(0) >= min_community_rank
                ]
            
            # Convert to Microsoft's community report format for processing
            community_reports = []
            for _, row in community_data.iterrows():
                report = {
                    "short_id": row.get("id", ""),
                    "community_id": row.get("community_id", ""),
                    "title": row.get("title", ""),
                    "summary": row.get("summary", ""),
                    "full_content": row.get("full_content", ""),
                    "rank": row.get("rank", 0.0),
                    "attributes": {"matches": row.get("entity_matches", 0)}
                }
                community_reports.append(report)
            
            # Use Microsoft's build_community_context approach
            from graphrag.query.context_builder.community_context import build_community_context
            from graphrag.data_model.community_report import CommunityReport
            
            # Convert to CommunityReport objects
            report_objects = []
            for report_data in community_reports:
                report = CommunityReport(
                    id=report_data["short_id"],  # Required 'id' parameter from Named base class
                    short_id=report_data["short_id"],
                    title=report_data["title"],
                    community_id=report_data["community_id"],
                    summary=report_data["summary"],
                    full_content=report_data["full_content"],
                    rank=report_data["rank"],
                    attributes=report_data["attributes"]
                )
                report_objects.append(report)
            
            # Build context using Microsoft's function
            context_text, context_data = build_community_context(
                community_reports=report_objects,
                token_encoder=self.token_encoder,
                use_community_summary=use_community_summary,
                column_delimiter=column_delimiter,
                shuffle_data=False,  # Don't shuffle for local search
                include_community_rank=include_community_rank,
                min_community_rank=min_community_rank,
                max_context_tokens=max_context_tokens,
                single_batch=True,
                context_name=context_name,
            )
            
            # Handle list vs string return
            if isinstance(context_text, list) and context_text:
                context_text = "\n\n".join(context_text)
            
            return (str(context_text), context_data)
            
        except Exception as e:
            logger.error(f"Failed to build community context: {e}")
            return ("", {context_name.lower(): pd.DataFrame()})
    
    def _build_local_entity_relationship_context(self,
                                                selected_entities: List[Dict[str, Any]],
                                                max_context_tokens: int = 2000,
                                                include_entity_rank: bool = False,
                                                rank_description: str = "relationship count",
                                                include_relationship_weight: bool = False,
                                                top_k_relationships: int = 10,
                                                relationship_ranking_attribute: str = "rank",
                                                return_candidate_context: bool = False,
                                                column_delimiter: str = "|") -> Tuple[str, Dict[str, pd.DataFrame]]:
        """
        Build local entity-relationship context using Microsoft's approach.
        
        This mirrors the _build_local_context method from LocalSearchMixedContext.
        """
        try:
            from graphrag.query.llm.text_utils import num_tokens
            
            if not selected_entities:
                return ("", {})
            
            entity_ids = [entity.get("id", "") for entity in selected_entities]
            
            # Build entity context
            entity_context_data = []
            for entity in selected_entities:
                entity_context_data.append({
                    "id": entity.get("id", ""),
                    "name": entity.get("name", ""),
                    "type": entity.get("type", ""),
                    "description": entity.get("description", ""),
                    "degree": entity.get("degree", 0),
                    "rank": entity.get("degree", 0) if include_entity_rank else None
                })
            
            entity_df = pd.DataFrame(entity_context_data)
            
            # Format entity context as CSV
            if include_entity_rank:
                entity_context = entity_df[["name", "type", "description", "rank"]].to_csv(
                    index=False, sep=column_delimiter
                )
                entity_context = f"-----Entities-----\n{entity_context}"
            else:
                entity_context = entity_df[["name", "type", "description"]].to_csv(
                    index=False, sep=column_delimiter
                )
                entity_context = f"-----Entities-----\n{entity_context}"
            
            entity_tokens = num_tokens(entity_context, self.token_encoder)
            
            # Build relationship context gradually
            final_context = [entity_context]
            final_context_data = {"entities": entity_df}
            current_tokens = entity_tokens
            
            # Get relationships for selected entities
            relationship_cypher = """
            UNWIND $entity_ids as entity_id
            MATCH (source:__Entity__ {id: entity_id})-[r:RELATED]->(target:__Entity__)
            RETURN 
                r.id as id,
                source.name as source,
                target.name as target,
                r.description as description,
                r.weight as weight,
                r.rank as rank
            ORDER BY r.rank DESC, r.weight DESC
            LIMIT $limit
            """
            
            relationship_data = self.vector_graph_store.neo4j.execute_query(
                relationship_cypher, {"entity_ids": entity_ids, "limit": top_k_relationships}
            )
            
            if not relationship_data.empty:
                # Format relationship context
                if include_relationship_weight:
                    rel_columns = ["source", "target", "description", "weight"]
                else:
                    rel_columns = ["source", "target", "description"]
                
                relationship_context = relationship_data[rel_columns].to_csv(
                    index=False, sep=column_delimiter
                )
                relationship_context = f"-----Relationships-----\n{relationship_context}"
                
                rel_tokens = num_tokens(relationship_context, self.token_encoder)
                
                if current_tokens + rel_tokens <= max_context_tokens:
                    final_context.append(relationship_context)
                    final_context_data["relationships"] = relationship_data
                    current_tokens += rel_tokens
                else:
                    logger.warning("Relationship context exceeds token limit, skipping")
            
            final_context_text = "\n\n".join(final_context)
            
            return (final_context_text, final_context_data)
            
        except Exception as e:
            logger.error(f"Failed to build local entity-relationship context: {e}")
            return ("", {})
    
    def _build_text_unit_context_proportional(self,
                                             selected_entities: List[Dict[str, Any]],
                                             max_context_tokens: int = 4000,
                                             return_candidate_context: bool = False,
                                             column_delimiter: str = "|",
                                             context_name: str = "Sources") -> Tuple[str, Dict[str, pd.DataFrame]]:
        """
        Build text unit context using Microsoft's proportional approach.
        
        This mirrors the _build_text_unit_context method from LocalSearchMixedContext.
        """
        try:
            from graphrag.query.llm.text_utils import num_tokens
            
            if not selected_entities:
                return ("", {context_name.lower(): pd.DataFrame()})
            
            entity_ids = [entity.get("id", "") for entity in selected_entities]
            
            # Get text units for selected entities using Neo4j
            text_unit_cypher = """
            UNWIND $entity_ids as entity_id
            MATCH (e:__Entity__ {id: entity_id})-[:HAS_ENTITY]-(c:__Chunk__)
            WITH c, count(DISTINCT e) as entity_frequency
            RETURN 
                c.id as id,
                c.text as text,
                c.n_tokens as n_tokens,
                c.human_readable_id as short_id,
                entity_frequency
            ORDER BY entity_frequency DESC, c.id
            """
            
            text_unit_data = self.vector_graph_store.neo4j.execute_query(
                text_unit_cypher, {"entity_ids": entity_ids}
            )
            
            if text_unit_data.empty:
                return ("", {context_name.lower(): pd.DataFrame()})
            
            # Build context incrementally with token management
            current_tokens = 0
            selected_units = []
            header_text = f"-----{context_name}-----\n" + column_delimiter.join(["id", "text"]) + "\n"
            current_tokens = num_tokens(header_text, self.token_encoder)
            
            for _, row in text_unit_data.iterrows():
                unit_text = f"{row['short_id']}{column_delimiter}{row['text']}\n"
                unit_tokens = num_tokens(unit_text, self.token_encoder)
                
                if current_tokens + unit_tokens > max_context_tokens:
                    logger.info(f"Reached token limit for text units: {current_tokens + unit_tokens}")
                    break
                
                selected_units.append({
                    "id": row["id"],
                    "short_id": row["short_id"],
                    "text": row["text"],
                    "n_tokens": row["n_tokens"]
                })
                current_tokens += unit_tokens
            
            if not selected_units:
                return ("", {context_name.lower(): pd.DataFrame()})
            
            # Format as CSV
            selected_df = pd.DataFrame(selected_units)
            context_text = selected_df[["short_id", "text"]].to_csv(
                index=False, sep=column_delimiter
            )
            context_text = f"-----{context_name}-----\n{context_text}"
            
            return (context_text, {context_name.lower(): selected_df})
            
        except Exception as e:
            logger.error(f"Failed to build text unit context: {e}")
            return ("", {context_name.lower(): pd.DataFrame()})


class Neo4jBasicContextBuilder(BasicContextBuilder):
    """
    Neo4j-based implementation of BasicContextBuilder.
    
    Enhanced to match Microsoft's BasicSearchContext with conversation history
    support, token-aware context building, and proper CSV formatting for
    traditional RAG operations without graph traversal.
    """
    
    def __init__(self,
                 vector_graph_store: Neo4jVectorGraphStore,
                 embeddings_model,
                 token_encoder = None,
                 max_context_tokens: int = 12000,
                 top_k_text_units: int = 20,
                 context_name: str = "Sources",
                 column_delimiter: str = "|",
                 text_id_col: str = "source_id",
                 text_col: str = "text",
                 embedding_vectorstore_key: str = "id"):
        """
        Initialize the Neo4j Basic Context Builder with Microsoft GraphRAG compatibility.
        
        Args:
            vector_graph_store: Hybrid vector-graph store
            embeddings_model: Model for generating query embeddings
            token_encoder: Token encoder for accurate token counting
            max_context_tokens: Maximum context tokens
            top_k_text_units: Number of text units to retrieve
            context_name: Name for context section
            column_delimiter: Delimiter for CSV formatting
            text_id_col: Column name for text ID
            text_col: Column name for text content
            embedding_vectorstore_key: Key for embedding vector store
        """
        self.vector_graph_store = vector_graph_store
        self.embeddings_model = embeddings_model
        self.token_encoder = token_encoder
        self.max_context_tokens = max_context_tokens
        self.top_k_text_units = top_k_text_units
        self.context_name = context_name
        self.column_delimiter = column_delimiter
        self.text_id_col = text_id_col
        self.text_col = text_col
        self.embedding_vectorstore_key = embedding_vectorstore_key
        
        # Initialize text unit ID mapping (Microsoft's approach)
        self.text_id_map = self._build_text_id_map()
    
    def build_context(self,
                     query: str,
                     conversation_history: ConversationHistory | None = None,
                     **kwargs) -> ContextBuilderResult:
        """
        Build context for basic search using Microsoft's approach.
        
        This implementation mirrors BasicSearchContext with conversation history
        support, token-aware context building, and proper CSV formatting.
        """
        try:
            from graphrag.query.llm.text_utils import num_tokens
            
            # Override parameters from kwargs
            k = kwargs.get("k", self.top_k_text_units)
            max_context_tokens = kwargs.get("max_context_tokens", self.max_context_tokens)
            context_name = kwargs.get("context_name", self.context_name)
            column_delimiter = kwargs.get("column_delimiter", self.column_delimiter)
            text_id_col = kwargs.get("text_id_col", self.text_id_col)
            text_col = kwargs.get("text_col", self.text_col)
            
            logger.info(f"Building Neo4j Basic Search context (k={k}, max_tokens={max_context_tokens})")
            
            # Handle empty query case (Microsoft's approach)
            if not query.strip():
                logger.info("Empty query provided, returning empty context")
                return ContextBuilderResult(
                    context_chunks="",
                    context_records={context_name.lower(): pd.DataFrame({
                        text_id_col: [],
                        text_col: [],
                    })},
                    llm_calls=0,
                    prompt_tokens=0,
                    output_tokens=0
                )
            
            # Perform vector similarity search on text units
            logger.info("Performing vector similarity search on text units...")
            query_embedding = self.embeddings_model.embed_query(query)
            related_text_units = self.vector_graph_store.similarity_search_text_units(
                query_embedding=query_embedding, 
                k=k
            )
            
            logger.info(f"Retrieved {len(related_text_units)} related text units from vector search")
            
            # Debug: Log first unit structure
            if related_text_units:
                logger.info(f"Sample unit structure: {list(related_text_units[0].keys())}")
                logger.info(f"Sample unit data types: {[(k, type(v)) for k, v in related_text_units[0].items()]}")
            
            # Convert to Microsoft's format with ID mapping
            related_text_list = []
            for unit in related_text_units:
                unit_id = unit.get("id", unit.get("human_readable_id", ""))
                mapped_id = self.text_id_map.get(unit_id, unit_id)
                # Ensure all values are strings
                id_str = str(mapped_id) if mapped_id is not None else ""
                text_str = str(unit.get("text", "")) if unit.get("text") is not None else ""
                
                related_text_list.append({
                    text_id_col: id_str,
                    text_col: text_str,
                })
                
                # Debug: Log first few entries
                if len(related_text_list) <= 2:
                    logger.info(f"Unit {len(related_text_list)}: id='{id_str}', text_length={len(text_str)}")
            
            related_text_df = pd.DataFrame(related_text_list)
            
            # Add text chunks to context until we hit the token limit (Microsoft's approach)
            current_tokens = 0
            selected_indices = []
            
            # Account for header tokens
            header_text = text_id_col + column_delimiter + text_col + "\n"
            current_tokens = num_tokens(header_text, self.token_encoder)
            
            for i, row in related_text_df.iterrows():
                # Ensure all values are strings before concatenation
                id_value = str(row[text_id_col]) if row[text_id_col] is not None else ""
                text_value = str(row[text_col]) if row[text_col] is not None else ""
                text = id_value + column_delimiter + text_value + "\n"
                text_tokens = num_tokens(text, self.token_encoder)
                
                if current_tokens + text_tokens > max_context_tokens:
                    logger.info(f"Reached token limit: {current_tokens + text_tokens}. "
                               f"Stopping at {len(selected_indices)} text units")
                    break
                
                current_tokens += text_tokens
                selected_indices.append(i)
            
            # Create final DataFrame with selected text units
            if selected_indices:
                final_text_df = related_text_df[related_text_df.index.isin(selected_indices)].reset_index(drop=True)
            else:
                logger.warning("No text units fit within token limit")
                final_text_df = pd.DataFrame({
                    text_id_col: [],
                    text_col: [],
                })
            
            # Format as CSV (Microsoft's approach)
            if not final_text_df.empty:
                final_text = final_text_df.to_csv(
                    index=False, 
                    escapechar="\\", 
                    sep=column_delimiter
                )
            else:
                final_text = ""
            
            logger.info(f"Built basic context: {len(final_text)} characters, "
                       f"{len(final_text_df)} text units, {current_tokens} tokens")
            logger.info(f"Context preview: {final_text[:200]}..." if final_text else "Context is empty!")
            
            # Validate that we have actual content
            if not final_text.strip():
                logger.warning("Basic context is empty - this will cause LLM to respond with 'no data provided'")
            
            return ContextBuilderResult(
                context_chunks=final_text,
                context_records={context_name.lower(): final_text_df},
                llm_calls=0,
                prompt_tokens=0,
                output_tokens=0
            )
            
        except Exception as e:
            logger.error(f"Failed to build basic context: {e}")
            return ContextBuilderResult(
                context_chunks="",
                context_records={self.context_name.lower(): pd.DataFrame()}
            )
    
    def _build_text_id_map(self) -> Dict[str, str]:
        """
        Build ID mapping for text units (Microsoft's approach).
        
        This mirrors the _map_ids method from BasicSearchContext.
        """
        try:
            # Get all text units from Neo4j to build ID mapping
            text_units_cypher = """
            MATCH (c:__Chunk__)
            RETURN 
                c.id as id,
                c.human_readable_id as short_id
            ORDER BY c.id
            """
            
            text_units_data = self.vector_graph_store.neo4j.execute_query(text_units_cypher)
            
            id_map = {}
            for _, row in text_units_data.iterrows():
                unit_id = str(row.get("id", "")) if row.get("id") is not None else ""
                short_id = str(row.get("short_id", unit_id)) if row.get("short_id") is not None else unit_id
                if unit_id:
                    id_map[unit_id] = short_id
            
            logger.info(f"Built text unit ID mapping for {len(id_map)} units")
            return id_map
            
        except Exception as e:
            logger.error(f"Failed to build text ID map: {e}")
            return {}


class Neo4jDRIFTContextBuilder(DRIFTContextBuilder):
    """
    Neo4j-based implementation of DRIFTContextBuilder.
    
    Enhanced to match Microsoft's DRIFTSearchContextBuilder with sophisticated
    primer query processing, embedding compatibility checking, vectorized
    similarity computation, and advanced local mixed context integration.
    """
    
    def __init__(self,
                 neo4j_backend: Neo4jGraphBackend,
                 vector_graph_store: Neo4jVectorGraphStore,
                 embeddings_model,
                 chat_model,
                 token_encoder = None,
                 config: Dict[str, Any] = None,
                 local_system_prompt: str | None = None,
                 reduce_system_prompt: str | None = None,
                 response_type: str | None = None,
                 embedding_vectorstore_key: str = "id"):
        """
        Initialize the Neo4j DRIFT Context Builder with Microsoft GraphRAG compatibility.
        
        Args:
            neo4j_backend: Neo4j graph backend
            vector_graph_store: Hybrid vector-graph store
            embeddings_model: Model for generating query embeddings
            chat_model: Chat model for DRIFT search
            token_encoder: Token encoder for accurate token counting
            config: DRIFT configuration parameters
            local_system_prompt: System prompt for local search
            reduce_system_prompt: System prompt for result reduction
            response_type: Target response format type
            embedding_vectorstore_key: Key for embedding vector store
        """
        import numpy as np
        from graphrag.config.models.drift_search_config import DRIFTSearchConfig
        from graphrag.prompts.query.drift_search_system_prompt import (
            DRIFT_LOCAL_SYSTEM_PROMPT,
            DRIFT_REDUCE_PROMPT,
        )
        
        self.neo4j_backend = neo4j_backend
        self.vector_graph_store = vector_graph_store
        self.embeddings_model = embeddings_model
        self.chat_model = chat_model
        self.token_encoder = token_encoder
        self.embedding_vectorstore_key = embedding_vectorstore_key
        
        # Create proper DRIFTSearchConfig object (enhanced)
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
        
        # Microsoft GraphRAG prompts (enhanced)
        self.local_system_prompt = local_system_prompt or DRIFT_LOCAL_SYSTEM_PROMPT
        self.reduce_system_prompt = reduce_system_prompt or DRIFT_REDUCE_PROMPT
        self.response_type = response_type or "multiple paragraphs"
        
        # Initialize enhanced local context builder for DRIFT internal use
        self.local_context_builder = Neo4jLocalContextBuilder(
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            token_encoder=token_encoder,
            max_context_tokens=self.config.local_search_max_data_tokens,
            top_k_mapped_entities=self.config.local_search_top_k_mapped_entities,
            top_k_relationships=self.config.local_search_top_k_relationships,
            text_unit_prop=self.config.local_search_text_unit_prop,
            community_prop=self.config.local_search_community_prop
        )
        
        # Create mixed context adapter with Microsoft GraphRAG compatibility
        self.local_mixed_context = self._create_mixed_context_adapter()
        
        # Store community reports for DRIFT processing (will be populated)
        self.reports = None
    
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
        Build DRIFT search context using Microsoft's sophisticated approach.
        
        This implementation mirrors DRIFTSearchContextBuilder with:
        - Sophisticated primer query processing
        - Embedding compatibility checking
        - Vectorized cosine similarity computation
        - Report DataFrame validation
        - Top-k similarity selection
        """
        try:
            import numpy as np
            from dataclasses import asdict
            from graphrag.data_model.community_report import CommunityReport
            
            logger.info("Building DRIFT search context with advanced primer processing...")
            
            # Get community reports from Neo4j for DRIFT processing
            primer_level = kwargs.get("primer_community_level", 1)
            community_reports_df = self.neo4j_backend.get_communities_by_level(primer_level)
            
            # If no reports at level 1, try level 2
            if community_reports_df.empty:
                logger.warning(f"No community reports found at level {primer_level}, trying level 2")
                community_reports_df = self.neo4j_backend.get_communities_by_level(2)
            
            if community_reports_df.empty:
                raise ValueError("No community reports available. Please provide a list of reports.")
            
            # Convert Neo4j data to CommunityReport objects (Microsoft's approach)
            community_reports = []
            for _, row in community_reports_df.iterrows():
                report = CommunityReport(
                    id=row.get("id", ""),  # Required 'id' parameter from Named base class
                    short_id=row.get("id", ""),
                    title=row.get("title", ""),
                    community_id=row.get("community", row.get("id", "")),
                    summary=row.get("summary", ""),
                    full_content=row.get("full_content", row.get("summary", "")),
                    rank=row.get("rank", 0.0),
                    sources=row.get("sources", []),
                    # Add full_content_embedding - we'll generate this
                    full_content_embedding=None
                )
                community_reports.append(report)
            
            logger.info(f"Converted {len(community_reports)} Neo4j reports to CommunityReport objects")
            
            # Store reports for processing
            self.reports = community_reports
            
            # Sophisticated primer query processing (Microsoft's approach)
            logger.info("Running sophisticated primer query processor...")
            query_processor = self._create_primer_query_processor()
            query_embedding, token_ct = await query_processor(query)
            
            # Convert reports to DataFrame with validation (Microsoft's approach)
            logger.info("Converting reports to DataFrame with validation...")
            report_df = self._convert_reports_to_df_with_embeddings(community_reports, query_embedding)
            
            # Check embedding compatibility (Microsoft's approach)
            if not self._check_query_doc_encodings(query_embedding, report_df["full_content_embedding"].iloc[0]):
                raise ValueError(
                    "Query and document embeddings are not compatible. "
                    "Please ensure that the embeddings are of the same type and length."
                )
            
            # Vectorized cosine similarity computation (Microsoft's approach)
            logger.info("Computing vectorized cosine similarity...")
            query_norm = np.linalg.norm(query_embedding)
            document_norms = np.linalg.norm(
                report_df["full_content_embedding"].to_list(), axis=1
            )
            dot_products = np.dot(
                np.vstack(report_df["full_content_embedding"].to_list()), query_embedding
            )
            report_df["similarity"] = dot_products / (document_norms * query_norm)
            
            # Sort by similarity and select top-k (Microsoft's approach)
            top_k = report_df.nlargest(self.config.drift_k_followups, "similarity")
            
            logger.info(f"Selected top {len(top_k)} community reports based on similarity")
            logger.info(f"Similarity range: {top_k['similarity'].min():.3f} - {top_k['similarity'].max():.3f}")
            
            # Return in Microsoft's format
            result_df = top_k.loc[:, ["short_id", "community_id", "full_content"]]
            
            return result_df, token_ct
            
        except Exception as e:
            logger.error(f"Failed to build DRIFT context: {e}")
            return pd.DataFrame(), {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0}
    
    def _create_primer_query_processor(self):
        """
        Create primer query processor (Microsoft's approach).
        
        This mirrors the PrimerQueryProcessor from Microsoft's implementation.
        """
        class Neo4jPrimerQueryProcessor:
            """Neo4j-based primer query processor."""
            
            def __init__(self, chat_model, text_embedder, token_encoder, reports):
                self.chat_model = chat_model
                self.text_embedder = text_embedder
                self.token_encoder = token_encoder
                self.reports = reports
            
            async def __call__(self, query: str) -> Tuple[List[float], Dict[str, int]]:
                """Process query and return embedding with token counts."""
                from graphrag.query.llm.text_utils import num_tokens
                
                # Generate query embedding
                query_embedding = self.text_embedder.embed_query(query)
                
                # Estimate token usage
                token_ct = {
                    "llm_calls": 0,  # No LLM calls for embedding
                    "prompt_tokens": num_tokens(query, self.token_encoder) if self.token_encoder else len(query) // 4,
                    "output_tokens": 0
                }
                
                return query_embedding, token_ct
        
        return Neo4jPrimerQueryProcessor(
            self.chat_model,
            self.embeddings_model,
            self.token_encoder,
            self.reports
        )
    
    def _convert_reports_to_df_with_embeddings(self, reports: List[Any], query_embedding: List[float]) -> pd.DataFrame:
        """
        Convert community reports to DataFrame with embeddings (Microsoft's approach).
        
        This mirrors the convert_reports_to_df method from Microsoft's implementation.
        """
        try:
            from dataclasses import asdict
            
            # Convert reports to DataFrame
            report_data = []
            for report in reports:
                if hasattr(report, '__dict__'):
                    report_dict = asdict(report) if hasattr(report, '__dataclass_fields__') else vars(report)
                else:
                    report_dict = report
                
                # Generate embedding for full content if not present
                full_content = report_dict.get("full_content", "")
                if not report_dict.get("full_content_embedding"):
                    report_dict["full_content_embedding"] = self.embeddings_model.embed_query(full_content)
                
                report_data.append(report_dict)
            
            report_df = pd.DataFrame(report_data)
            
            # Validate required columns (Microsoft's validation)
            if "full_content" not in report_df.columns or report_df["full_content"].isna().sum() > 0:
                raise ValueError("Some reports are missing full content.")
            
            if "full_content_embedding" not in report_df.columns or report_df["full_content_embedding"].isna().sum() > 0:
                missing = report_df["full_content_embedding"].isna().sum()
                total = len(report_df)
                raise ValueError(f"Some reports are missing full content embeddings. {missing} out of {total}")
            
            return report_df
            
        except Exception as e:
            logger.error(f"Failed to convert reports to DataFrame: {e}")
            raise
    
    def _check_query_doc_encodings(self, query_embedding: Any, embedding: Any) -> bool:
        """
        Check if embeddings are compatible (Microsoft's approach).
        
        This mirrors the check_query_doc_encodings method from Microsoft's implementation.
        """
        try:
            return (
                query_embedding is not None
                and embedding is not None
                and isinstance(query_embedding, type(embedding))
                and len(query_embedding) == len(embedding)
                and isinstance(query_embedding[0], type(embedding[0]))
            )
        except Exception:
            return False


# ============================================================================
# Factory Functions for Easy Integration
# ============================================================================

def create_neo4j_context_builders(neo4j_config: Dict[str, str],
                                 lancedb_path: str,
                                 embeddings_model,
                                 chat_model,
                                 token_encoder = None) -> Dict[str, Any]:
    """
    Factory function to create all Neo4j context builders.
    
    Args:
        neo4j_config: Neo4j connection configuration
        lancedb_path: Path to LanceDB vector stores
        embeddings_model: Embeddings model instance
        chat_model: Chat model instance (needed for DRIFT)
        token_encoder: Token encoder for accurate token counting (tiktoken)
        
    Returns:
        Dictionary containing all context builders
    """
    from neo4j_backend_implementation import Neo4jConfig, Neo4jGraphBackend, Neo4jVectorGraphStore
    
    # Initialize backend components
    config = Neo4jConfig(**neo4j_config)
    neo4j_backend = Neo4jGraphBackend(config)
    vector_graph_store = Neo4jVectorGraphStore(neo4j_backend, lancedb_path)
    
    # Create context builders with enhanced Microsoft GraphRAG compatibility
    builders = {
        "global": Neo4jGlobalContextBuilder(
            neo4j_backend=neo4j_backend,
            token_encoder=token_encoder,
            # Enable Microsoft GraphRAG advanced features by default
            use_community_summary=True,
            include_community_rank=True,
            include_community_weight=True,
            normalize_community_weight=True,
            shuffle_data=True,
            single_batch=False
        ),
        "local": Neo4jLocalContextBuilder(
            vector_graph_store=vector_graph_store, 
            embeddings_model=embeddings_model,
            token_encoder=token_encoder,
            # Enable Microsoft GraphRAG proportional allocation
            text_unit_prop=0.5,
            community_prop=0.25,
            include_community_rank=True,
            include_entity_rank=True,
            include_relationship_weight=True
        ),
        "basic": Neo4jBasicContextBuilder(
            vector_graph_store=vector_graph_store, 
            embeddings_model=embeddings_model,
            token_encoder=token_encoder,
            # Microsoft GraphRAG compatibility settings
            max_context_tokens=12000,
            context_name="Sources",
            column_delimiter="|"
        ),
        "drift": Neo4jDRIFTContextBuilder(
            neo4j_backend=neo4j_backend,
            vector_graph_store=vector_graph_store,
            embeddings_model=embeddings_model,
            chat_model=chat_model,
            token_encoder=token_encoder,
            # Microsoft GraphRAG DRIFT configuration
            embedding_vectorstore_key="id"
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

