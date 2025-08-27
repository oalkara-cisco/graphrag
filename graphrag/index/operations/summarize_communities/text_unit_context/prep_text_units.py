# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Prepare text units for community reports."""

import logging
import re
from typing import Dict, Any

import pandas as pd

import graphrag.data_model.schemas as schemas

logger = logging.getLogger(__name__)


def extract_source_metadata(text_unit_text: str) -> Dict[str, Any]:
    """
    Extract source URL and citation metadata from text unit content.
    
    Args:
        text_unit_text: The text content of the text unit containing metadata
        
    Returns:
        Dictionary containing source_url and citation information
    """
    metadata = {}
    
    # Extract full_url using regex
    url_match = re.search(r'full_url:\s*([^\n]+)', text_unit_text)
    if url_match:
        metadata['source_url'] = url_match.group(1).strip().rstrip('.')
        
    # Extract title for citation
    title_match = re.search(r'title:\s*([^\n]+)', text_unit_text)
    if title_match:
        metadata['source_title'] = title_match.group(1).strip().rstrip('.')
    
    # Extract page_id for reference  
    page_id_match = re.search(r'page_id:\s*([^\n]+)', text_unit_text)
    if page_id_match:
        metadata['page_id'] = page_id_match.group(1).strip().rstrip('.')
        
    # Create a citation string if we have the necessary information
    if 'source_url' in metadata:
        if 'source_title' in metadata:
            metadata['citation'] = f"[{metadata['source_title']}]({metadata['source_url']})"
        else:
            metadata['citation'] = metadata['source_url']
    
    return metadata


def prep_text_units(
    text_unit_df: pd.DataFrame,
    node_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate text unit degree  and concatenate text unit details.

    Returns : dataframe with columns [COMMUNITY_ID, TEXT_UNIT_ID, ALL_DETAILS]
    """
    node_df.drop(columns=["id"], inplace=True)
    node_to_text_ids = node_df.explode(schemas.TEXT_UNIT_IDS).rename(
        columns={schemas.TEXT_UNIT_IDS: schemas.ID}
    )
    node_to_text_ids = node_to_text_ids[
        [schemas.TITLE, schemas.COMMUNITY_ID, schemas.NODE_DEGREE, schemas.ID]
    ]
    text_unit_degrees = (
        node_to_text_ids.groupby([schemas.COMMUNITY_ID, schemas.ID])
        .agg({schemas.NODE_DEGREE: "sum"})
        .reset_index()
    )
    result_df = text_unit_df.merge(text_unit_degrees, on=schemas.ID, how="left")
    
    def create_all_details(row):
        """Create ALL_DETAILS dictionary with source metadata."""
        # Extract source metadata from text unit content
        source_metadata = extract_source_metadata(row[schemas.TEXT]) if row[schemas.TEXT] else {}
        
        details = {
            schemas.SHORT_ID: row[schemas.SHORT_ID],
            schemas.TEXT: row[schemas.TEXT],
            schemas.ENTITY_DEGREE: row[schemas.NODE_DEGREE],
        }
        
        # Add source metadata if available
        if source_metadata:
            details.update({
                'source_citation': source_metadata.get('citation'),
                'source_metadata': source_metadata
            })
            logger.debug(f"Added source metadata for text unit {row[schemas.SHORT_ID]}: {source_metadata.get('source_url', 'No URL')}")
        
        return details
    
    result_df[schemas.ALL_DETAILS] = result_df.apply(create_all_details, axis=1)
    return result_df.loc[:, [schemas.COMMUNITY_ID, schemas.ID, schemas.ALL_DETAILS]]
