# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Local source module."""

import logging
import os
from pathlib import Path

import pandas as pd
from knowledge_loader.data_sources.typing import Datasource

from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig

# Import our Bedrock configuration extension
try:
    from bedrock_config_extension import (
        load_extended_config, 
        is_bedrock_config, 
        get_config_info
    )
    BEDROCK_SUPPORT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Bedrock config extension not available: {e}")
    BEDROCK_SUPPORT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logging.getLogger("azure").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def load_local_prompt_config(base_path="") -> dict[str, str]:
    """Load local prompt configuration."""
    # for each file inside folder base_path
    prompts = {}

    for path in os.listdir(base_path):  # noqa: PTH208
        with open(os.path.join(base_path, path), "r") as f:  # noqa: UP015, PTH123, PTH118
            map_name = path.split(".")[0]
            prompts[map_name] = f.read()
    return prompts


class LocalDatasource(Datasource):
    """Datasource that reads from a local parquet file."""

    _base_path: str

    def __init__(self, base_path: str):
        """Init method definition."""
        self._base_path = base_path

    def read(
        self,
        table: str,
        throw_on_missing: bool = False,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read file from local source."""
        table = os.path.join(self._base_path, f"{table}.parquet")  # noqa: PTH118

        if not os.path.exists(table):  # noqa: PTH110
            if throw_on_missing:
                error_msg = f"Table {table} does not exist"
                raise FileNotFoundError(error_msg)

            print(f"Table {table} does not exist")  # noqa T201
            return (
                pd.DataFrame(data=[], columns=columns)
                if columns is not None
                else pd.DataFrame()
            )
        return pd.read_parquet(table, columns=columns)

    def read_settings(
        self,
        file: str,
        throw_on_missing: bool = False,
    ) -> GraphRagConfig | None:
        """Read settings file from local source with Bedrock support."""
        cwd = Path(__file__).parent
        root_dir = (cwd / self._base_path).resolve()
        
        # Try enhanced loading first if Bedrock support is available
        if BEDROCK_SUPPORT_AVAILABLE:
            try:
                logger.info("Attempting to load configuration with Bedrock support...")
                config = load_extended_config(root_dir=root_dir)
                
                if config is not None:
                    # Log configuration information
                    config_info = get_config_info(config)
                    if config_info['is_bedrock']:
                        logger.info(f"✅ Successfully loaded Bedrock configuration:")
                        logger.info(f"   - Chat Model: {config_info['chat_model_type']} ({config_info['chat_model_name']})")
                        logger.info(f"   - Embedding Model: {config_info['embedding_model_type']} ({config_info['embedding_model_name']})")
                    else:
                        logger.info("✅ Successfully loaded standard GraphRAG configuration")
                    
                    return config
                else:
                    logger.warning("Enhanced config loading returned None, falling back to standard loading")
                    
            except Exception as e:
                logger.warning(f"Enhanced config loading failed: {e}, falling back to standard loading")
        
        # Fallback to standard loading
        try:
            logger.info("Attempting standard GraphRAG config loading...")
            config = load_config(root_dir=root_dir)
            logger.info("✅ Successfully loaded standard GraphRAG configuration")
            return config
        except Exception as e:
            logger.error(f"❌ Standard config loading failed: {e}")
            if throw_on_missing:
                raise
            return None
