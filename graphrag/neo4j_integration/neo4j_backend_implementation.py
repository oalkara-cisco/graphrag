"""
Neo4j Backend Implementation for GraphRAG
=========================================

This module provides Neo4j-based implementations for GraphRAG search operations,
replacing parquet file access with direct graph database queries.
"""

import pandas as pd
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""
    uri: str
    username: str
    password: str
    database: str = "neo4j"


class Neo4jGraphBackend:
    """Neo4j backend for GraphRAG operations."""
    
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.uri, 
            auth=(config.username, config.password)
        )
    
    def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
    
    def execute_query(self, cypher: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute a Cypher query and return results as DataFrame."""
        with self.driver.session(database=self.config.database) as session:
            result = session.run(cypher, params or {})
            records = [record.data() for record in result]
            return pd.DataFrame(records)
    
    # ============================================================================
    # Community Operations (Global Search)
    # ============================================================================
    
    def get_communities_by_level(self, level: int) -> pd.DataFrame:
        """Get all community reports for a specific hierarchical level."""
        cypher = """
        MATCH (cr:CommunityReport)
        WHERE cr.level = $level
        RETURN 
            cr.id as id,
            cr.community as community,
            cr.level as level,
            cr.title as title,
            cr.summary as summary,
            cr.full_content as full_content,
            cr.rank as rank,
            cr.rank_explanation as rank_explanation,
            cr.sources as sources
        ORDER BY cr.rank DESC
        """
        return self.execute_query(cypher, {"level": level})
    
    def get_all_community_levels(self) -> List[int]:
        """Get all available community levels."""
        cypher = """
        MATCH (c:__Community__)
        RETURN DISTINCT c.level as level
        ORDER BY level
        """
        result = self.execute_query(cypher)
        return result['level'].tolist()
    
    def get_community_hierarchy_stats(self) -> pd.DataFrame:
        """Get statistics about community hierarchy."""
        cypher = """
        MATCH (c:__Community__)
        RETURN 
            c.level as level,
            count(c) as community_count,
            avg(c.size) as avg_size,
            max(c.size) as max_size,
            min(c.size) as min_size
        ORDER BY level
        """
        return self.execute_query(cypher)
    
    # ============================================================================
    # Entity Operations (Local Search)  
    # ============================================================================
    
    def get_entities_by_ids(self, entity_ids: List[str]) -> pd.DataFrame:
        """Get entity details by IDs."""
        cypher = """
        UNWIND $entity_ids as entity_id
        MATCH (e:__Entity__ {id: entity_id})
        RETURN 
            e.id as id,
            e.name as name,
            e.type as type,
            e.description as description,
            e.frequency as frequency,
            e.degree as degree,
            e.x as x,
            e.y as y
        """
        return self.execute_query(cypher, {"entity_ids": entity_ids})
    
    def get_text_units_for_entities(self, entity_ids: List[str], limit: int = 10) -> pd.DataFrame:
        """Get text units that contain the specified entities."""
        cypher = """
        UNWIND $entity_ids as entity_id
        MATCH (e:__Entity__ {id: entity_id})<-[:HAS_ENTITY]-(c:__Chunk__)
        WITH c, count(DISTINCT e) as entity_frequency
        RETURN 
            c.id as id,
            c.text as text,
            c.n_tokens as n_tokens,
            entity_frequency
        ORDER BY entity_frequency DESC
        LIMIT $limit
        """
        return self.execute_query(cypher, {"entity_ids": entity_ids, "limit": limit})
    
    def get_relationships_for_entities(self, 
                                     entity_ids: List[str], 
                                     include_internal: bool = True,
                                     include_external: bool = True,
                                     limit: int = 20) -> pd.DataFrame:
        """Get relationships involving the specified entities."""
        conditions = []
        if include_internal and include_external:
            # Get all relationships
            where_clause = ""
        elif include_internal:
            # Only relationships between entities in the set
            conditions.append("target.id IN $entity_ids")
        elif include_external:
            # Only relationships to entities outside the set
            conditions.append("NOT target.id IN $entity_ids")
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        cypher = f"""
        UNWIND $entity_ids as entity_id
        MATCH (source:__Entity__ {{id: entity_id}})-[r:RELATED]->(target:__Entity__)
        {where_clause}
        RETURN 
            r.id as id,
            source.name as source,
            target.name as target,
            r.description as description,
            r.weight as weight,
            r.rank as rank,
            r.human_readable_id as human_readable_id
        ORDER BY r.rank DESC, r.weight DESC
        LIMIT $limit
        """
        return self.execute_query(cypher, {"entity_ids": entity_ids, "limit": limit})
    
    def get_community_reports_for_entities(self, entity_ids: List[str], limit: int = 5) -> pd.DataFrame:
        """Get community reports for communities containing the specified entities."""
        cypher = """
        UNWIND $entity_ids as entity_id
        MATCH (e:__Entity__ {id: entity_id})-[:MEMBER_OF]->(c:__Community__)<-[:ANALYZES]-(cr:CommunityReport)
        WITH cr, c, count(DISTINCT e) as entity_overlap
        RETURN 
            cr.id as id,
            cr.community as community,
            cr.level as level,
            cr.title as title,
            cr.summary as summary,
            cr.full_content as full_content,
            cr.rank as rank,
            c.weight as weight,
            entity_overlap
        ORDER BY cr.rank DESC, c.weight DESC, entity_overlap DESC
        LIMIT $limit
        """
        return self.execute_query(cypher, {"entity_ids": entity_ids, "limit": limit})
    
    # ============================================================================
    # Text Unit Operations (Basic Search)
    # ============================================================================
    
    def get_text_units_by_ids(self, text_unit_ids: List[str]) -> pd.DataFrame:
        """Get text units by their IDs."""
        cypher = """
        UNWIND $text_unit_ids as unit_id
        MATCH (c:__Chunk__ {id: unit_id})
        RETURN 
            c.id as id,
            c.text as text,
            c.n_tokens as n_tokens,
            c.human_readable_id as human_readable_id
        """
        return self.execute_query(cypher, {"text_unit_ids": text_unit_ids})
    
    def get_all_text_units(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get all text units (for basic search initialization)."""
        cypher = """
        MATCH (c:__Chunk__)
        RETURN 
            c.id as id,
            c.text as text,
            c.n_tokens as n_tokens,
            c.human_readable_id as human_readable_id
        ORDER BY c.id
        """
        if limit:
            cypher += f" LIMIT {limit}"
        
        return self.execute_query(cypher)
    
    # ============================================================================
    # Graph Analytics & Statistics
    # ============================================================================
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        stats = {}
        
        # Node counts
        node_count_cypher = """
        MATCH (d:__Document__) WITH count(d) as docs
        MATCH (c:__Chunk__) WITH docs, count(c) as chunks  
        MATCH (e:__Entity__) WITH docs, chunks, count(e) as entities
        MATCH (com:__Community__) WITH docs, chunks, entities, count(com) as communities
        MATCH (cr:CommunityReport) WITH docs, chunks, entities, communities, count(cr) as reports
        RETURN docs, chunks, entities, communities, reports
        """
        node_stats = self.execute_query(node_count_cypher).iloc[0].to_dict()
        stats.update(node_stats)
        
        # Relationship count
        rel_count_cypher = "MATCH ()-[r:RELATED]->() RETURN count(r) as relationships"
        rel_stats = self.execute_query(rel_count_cypher).iloc[0].to_dict()
        stats.update(rel_stats)
        
        # Entity degree distribution
        degree_cypher = """
        MATCH (e:__Entity__)
        RETURN 
            avg(e.degree) as avg_entity_degree,
            max(e.degree) as max_entity_degree,
            min(e.degree) as min_entity_degree
        """
        degree_stats = self.execute_query(degree_cypher).iloc[0].to_dict()
        stats.update(degree_stats)
        
        return stats
    
    def get_top_entities_by_degree(self, limit: int = 10) -> pd.DataFrame:
        """Get the most connected entities."""
        cypher = """
        MATCH (e:__Entity__)
        RETURN 
            e.name as name,
            e.type as type,
            e.degree as degree,
            e.frequency as frequency
        ORDER BY e.degree DESC
        LIMIT $limit
        """
        return self.execute_query(cypher, {"limit": limit})


class Neo4jVectorGraphStore:
    """
    Hybrid store combining Neo4j graph data with LanceDB vector search.
    This class provides the bridge between vector similarity search and graph traversal.
    """
    
    def __init__(self, neo4j_backend: Neo4jGraphBackend, lancedb_path: str):
        self.neo4j = neo4j_backend
        self.lancedb_path = lancedb_path
        self._vector_stores = {}
        self._init_vector_stores()
    
    def _init_vector_stores(self):
        """Initialize LanceDB vector stores."""
        import lancedb
        
        # Connect to LanceDB
        db = lancedb.connect(self.lancedb_path)
        
        # Initialize vector tables
        try:
            self._vector_stores = {
                "entities": db.open_table("default-entity-description"),
                "text_units": db.open_table("default-text_unit-text"), 
                "communities": db.open_table("default-community-full_content")
            }
            logger.info("Successfully initialized LanceDB vector stores")
        except Exception as e:
            logger.error(f"Failed to initialize vector stores: {e}")
            raise
    
    def similarity_search_entities(self, 
                                 query_embedding: List[float], 
                                 k: int = 10) -> List[Dict[str, Any]]:
        """Perform vector similarity search on entity descriptions."""
        try:
            # Search in LanceDB
            results = self._vector_stores["entities"].search(query_embedding).limit(k).to_list()
            
            # Extract entity IDs
            entity_ids = [result.get("id", result.get("human_readable_id")) for result in results]
            
            # Enrich with Neo4j data
            neo4j_entities = self.neo4j.get_entities_by_ids(entity_ids)
            
            # Combine vector scores with graph data
            enriched_results = []
            for result in results:
                entity_id = result.get("id", result.get("human_readable_id"))
                neo4j_row = neo4j_entities[neo4j_entities["id"] == entity_id]
                
                if not neo4j_row.empty:
                    entity_data = neo4j_row.iloc[0].to_dict()
                    entity_data["_distance"] = result.get("_distance", 0.0)
                    enriched_results.append(entity_data)
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Entity similarity search failed: {e}")
            return []
    
    def similarity_search_text_units(self, 
                                   query_embedding: List[float], 
                                   k: int = 10) -> List[Dict[str, Any]]:
        """Perform vector similarity search on text units."""
        try:
            # Search in LanceDB  
            results = self._vector_stores["text_units"].search(query_embedding).limit(k).to_list()
            
            # Extract text unit IDs
            text_unit_ids = [result.get("id", result.get("human_readable_id")) for result in results]
            
            # Enrich with Neo4j data
            neo4j_text_units = self.neo4j.get_text_units_by_ids(text_unit_ids)
            
            # Combine vector scores with graph data
            enriched_results = []
            for result in results:
                unit_id = result.get("id", result.get("human_readable_id"))
                neo4j_row = neo4j_text_units[neo4j_text_units["id"] == unit_id]
                
                if not neo4j_row.empty:
                    unit_data = neo4j_row.iloc[0].to_dict()
                    unit_data["_distance"] = result.get("_distance", 0.0)
                    enriched_results.append(unit_data)
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Text unit similarity search failed: {e}")
            return []
    
    def similarity_search_communities(self, 
                                    query_embedding: List[float], 
                                    k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector similarity search on community reports."""
        try:
            # Search in LanceDB
            results = self._vector_stores["communities"].search(query_embedding).limit(k).to_list()
            
            # Extract community IDs and enrich with Neo4j data
            enriched_results = []
            for result in results:
                community_id = result.get("id", result.get("community"))
                
                # Get community data from Neo4j
                cypher = """
                MATCH (cr:CommunityReport {community: $community_id})
                MATCH (cr)-[:ANALYZES]->(c:__Community__)
                RETURN 
                    cr.id as id,
                    cr.community as community,
                    cr.level as level,
                    cr.title as title,
                    cr.summary as summary,
                    cr.full_content as full_content,
                    cr.rank as rank,
                    c.weight as weight
                """
                neo4j_result = self.neo4j.execute_query(cypher, {"community_id": community_id})
                
                if not neo4j_result.empty:
                    community_data = neo4j_result.iloc[0].to_dict()
                    community_data["_distance"] = result.get("_distance", 0.0)
                    enriched_results.append(community_data)
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Community similarity search failed: {e}")
            return []
    
    def get_local_search_context(self, 
                               query_embedding: List[float],
                               top_k_entities: int = 10,
                               top_k_text_units: int = 3,
                               top_k_relationships: int = 10,
                               top_k_communities: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive local search context using hybrid vector-graph approach.
        This mirrors the retrieval logic from the ms_graphrag_retriever.ipynb notebook.
        """
        
        # 1. Find relevant entities using vector search
        relevant_entities = self.similarity_search_entities(query_embedding, top_k_entities)
        entity_ids = [e["id"] for e in relevant_entities]
        
        # 2. Get related data from Neo4j graph
        context_data = {
            "entities": pd.DataFrame(relevant_entities),
            "text_units": self.neo4j.get_text_units_for_entities(entity_ids, top_k_text_units),
            "relationships": self.neo4j.get_relationships_for_entities(entity_ids, limit=top_k_relationships),
            "communities": self.neo4j.get_community_reports_for_entities(entity_ids, top_k_communities)
        }
        
        return context_data
    
    def close(self):
        """Close connections."""
        self.neo4j.close()


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    import os
    
    # Configuration from environment variables - DO NOT hardcode credentials
    config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),  # Must be provided via env var
        database=os.getenv("NEO4J_DATABASE", "neo4j")
    )
    
    if not config.password:
        print("Error: NEO4J_PASSWORD environment variable must be set")
        print("Usage: NEO4J_PASSWORD=your_password python neo4j_backend_implementation.py")
        exit(1)
    
    # Initialize backend
    backend = Neo4jGraphBackend(config)
    
    # Get LanceDB path from environment variable (for demo purposes)
    # In production, this should come from GraphRAG config: config.get_vector_store_config("default_vector_store").db_uri
    lancedb_path = os.getenv("LANCEDB_PATH", "./output/lancedb")
    print(f"Using LanceDB path: {lancedb_path}")
    print("In production, configure this via vector_store.default_vector_store.db_uri in settings.yaml")
    
    vector_store = Neo4jVectorGraphStore(backend, lancedb_path)
    
    try:
        # Example: Get graph statistics
        stats = backend.get_graph_statistics()
        print("Graph Statistics:", stats)
        
        # Example: Get communities for global search
        communities = backend.get_communities_by_level(level=2)
        print(f"Level 2 communities: {len(communities)}")
        
        print("Neo4j backend connection successful!")
        print("Tip: Use this backend in your GraphRAG application by configuring settings.yaml")
        
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        print("Please check your connection details and ensure Neo4j is running.")
        
    finally:
        vector_store.close()

