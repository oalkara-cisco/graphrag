# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Full graph module."""

import altair as alt
import pandas as pd
import streamlit as st
import gc
from state.session_variables import SessionVariables

# Enable VegaFusion to handle large datasets
try:
    alt.data_transformers.enable("vegafusion")
except Exception:
    # Fallback to default with increased limit
    alt.data_transformers.enable('json', urlpath='')
    alt.data_transformers.disable_max_rows()

def create_full_graph_ui(sv: SessionVariables):
    """Return graph UI object."""
    entities = sv.entities.value.copy()
    communities = sv.communities.value.copy()
    
    level = sv.graph_community_level.value

    if communities.empty or entities.empty:
        st.warning("⚠️ No data available for visualization.")
        return None

    # CRITICAL: Pre-filter communities to prevent memory explosion
    max_communities = 500  # Limit number of communities to process 
    level_communities = communities[communities["level"] == level]
    
    st.info(f"🔍 Found {len(level_communities):,} communities at level {level}")
    
    # Add option for ultra-conservative mode for very large datasets
    if len(level_communities) > 1000:
        max_communities = 200  # Ultra-conservative for huge datasets
        st.warning(f"🔥 Extremely large dataset detected! Using ultra-conservative limit of {max_communities} communities.")
    elif len(level_communities) > max_communities:
        st.warning(f"⚠️ Large community set detected ({len(level_communities):,} communities). Processing top {max_communities} by size to prevent memory issues.")
    
    if len(level_communities) > max_communities:
        level_communities = (
            level_communities
            .nlargest(max_communities, 'size')  # Use community size for selection
        )
    
    st.info(f"📊 Processing {len(level_communities)} communities...")
    
    # Now do the expensive operations on the limited dataset
    try:
        communities_entities = (
            level_communities.explode("entity_ids")
            .merge(
                entities,
                left_on="entity_ids",
                right_on="id",
                suffixes=("_entities", "_communities"),
            )
            .dropna(subset=["x", "y"])
        )
        
        # ULTRA-AGGRESSIVE: Browser-friendly limits to prevent WebSocket issues  
        max_viz_points = 2000  # Ultra-conservative for browser stability
        
        # Let user choose visualization density
        viz_mode = st.selectbox(
            "🎛️ Visualization Mode",
            options=["Ultra Fast (1000 points)", "Fast (2000 points)", "Detailed (3000 points)"],
            index=1,  # Default to Fast mode
            help="Lower point counts prevent browser crashes and WebSocket timeouts"
        )
        
        # Parse selected limit
        if "1000" in viz_mode:
            max_viz_points = 1000
        elif "2000" in viz_mode:
            max_viz_points = 2000
        else:  # 3000
            max_viz_points = 3000
            
        if len(communities_entities) > max_viz_points:
            st.info(f"📊 Showing top {max_viz_points:,} entities by degree ({viz_mode})")
            communities_entities = (
                communities_entities
                .nlargest(max_viz_points, 'degree')
            )
        else:
            st.info(f"📊 Showing all {len(communities_entities):,} entities")
        
        # Memory check before visualization
        estimated_memory_mb = len(communities_entities) * 0.02  # Rough estimate
        if estimated_memory_mb > 50:  # >50MB could cause browser issues
            st.warning(f"⚠️ Large visualization detected (~{estimated_memory_mb:.1f}MB). Consider using Ultra Fast mode if you experience issues.")
        
        communities_entities_filtered = communities_entities
        
    except Exception as e:
        st.error(f"❌ Error processing graph data: {str(e)}")
        st.info("💡 Try selecting a different community level or restart the application.")
        return None

    # Final safeguard: Add loading indicator and timeout handling
    chart_placeholder = st.empty()
    
    with st.spinner(f"🎨 Rendering graph with {len(communities_entities_filtered):,} entities..."):
        try:
            # Reduce chart complexity for browser stability
            height = 800 if len(communities_entities_filtered) > 2000 else 1000
            
            # Simplify tooltips for large datasets to reduce JSON size
            if len(communities_entities_filtered) > 1500:
                tooltip_fields = ["id_entities", "community"]  # Minimal tooltip
            else:
                tooltip_fields = ["id_entities", "type", "description", "community"]  # Full tooltip
            
            graph = (
                alt.Chart(communities_entities_filtered)
                .mark_circle(opacity=0.7, stroke='white', strokeWidth=0.5)  # Add stroke for better visibility
                .encode(
                    x=alt.X("x", scale=alt.Scale(padding=0.1)),
                    y=alt.Y("y", scale=alt.Scale(padding=0.1)),
                    color=alt.Color(
                        "community",
                        scale=alt.Scale(
                            domain=communities_entities_filtered["community"].unique(),
                            scheme="category10",
                        ),
                        legend=alt.Legend(orient="right", titleLimit=200, labelLimit=100)
                    ),
                    size=alt.Size("degree", scale=alt.Scale(range=[30, 500]), legend=None),  # Smaller circles
                    tooltip=tooltip_fields,
                )
                .properties(
                    height=height,
                    title=f"Entity Graph - Level {level} ({len(communities_entities_filtered):,} entities)"
                )
                .configure_axis(disable=True)
                .configure_view(strokeWidth=0)  # Remove border
            )
            
            # Display the chart
            chart_placeholder.altair_chart(graph, use_container_width=True)
            
            # Add performance info
            st.caption(f"💡 **Performance Info**: Displaying {len(communities_entities_filtered):,} entities. Use visualization mode selector above to adjust performance.")
            
            # Memory cleanup to prevent server bloat
            del communities_entities
            gc.collect()
            
            return graph
            
        except Exception as e:
            chart_placeholder.empty()
            st.error(f"❌ Graph visualization failed: {str(e)}")
            
            # Provide specific troubleshooting
            if "rows" in str(e).lower() or "limit" in str(e).lower():
                st.info("🔧 **Try**: Switch to 'Ultra Fast' mode or restart the browser.")
            elif "memory" in str(e).lower():
                st.info("🔧 **Try**: Close other browser tabs or use a different browser.")
            else:
                st.info("🔧 **Try**: Refresh the page or select a different community level.")
                
            # Fallback: Show data table instead
            if len(communities_entities_filtered) <= 100:
                st.info("📊 **Fallback**: Showing data table instead of graph:")
                st.dataframe(
                    communities_entities_filtered[['id_entities', 'type', 'community', 'degree']].head(50),
                    use_container_width=True
                )
            
            # Memory cleanup on error
            try:
                del communities_entities
                gc.collect()
            except:
                pass
            
            return None
