# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to transform final entities."""

from uuid import uuid4

import pandas as pd

from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS


def finalize_community_reports(
    reports: pd.DataFrame,
    communities: pd.DataFrame,
) -> pd.DataFrame:
    """All the steps to transform final community reports."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Debug: Log DataFrame info
    logger.debug(f"Reports DataFrame shape: {reports.shape}")
    logger.debug(f"Reports DataFrame columns: {list(reports.columns)}")
    logger.debug(f"Communities DataFrame columns: {list(communities.columns)}")
    
    # Check if reports DataFrame is empty or missing community column
    if reports.empty:
        logger.warning("Reports DataFrame is empty, returning empty DataFrame")
        return pd.DataFrame(columns=[
            'id', 'human_readable_id', 'community', 'level', 'parent', 'children',
            'title', 'summary', 'full_content', 'rank', 'rating_explanation',
            'findings', 'full_content_json', 'period', 'size', 'sources'
        ])
    
    if 'community' not in reports.columns:
        logger.error(f"Missing 'community' column in reports. Available columns: {list(reports.columns)}")
        raise KeyError("Reports DataFrame is missing the 'community' column")
    
    # Merge with communities to add shared fields
    community_reports = reports.merge(
        communities.loc[:, ["community", "parent", "children", "size", "period"]],
        on="community",
        how="left",
        copy=False,
    )

    community_reports["community"] = community_reports["community"].astype(int)
    community_reports["human_readable_id"] = community_reports["community"]
    community_reports["id"] = [uuid4().hex for _ in range(len(community_reports))]

    return community_reports.loc[
        :,
        COMMUNITY_REPORTS_FINAL_COLUMNS,
    ]
