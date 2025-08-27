# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""App logic module."""

import asyncio
import logging
from typing import TYPE_CHECKING

import streamlit as st
from knowledge_loader.data_sources.loader import (
    create_datasource,
    load_dataset_listing,
)
from knowledge_loader.model import load_model
from rag.typing import SearchResult, SearchType
from state.session_variables import SessionVariables
from ui.search import display_search_result

import graphrag.api as api

# Import Bedrock support
try:
    from bedrock_config_extension import is_bedrock_config, get_config_info
    from bedrock_model_factory import (
        initialize_bedrock_for_search_app, 
        get_bedrock_search_engines,
        bedrock_manager
    )
    BEDROCK_SUPPORT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Bedrock support not available: {e}")
    BEDROCK_SUPPORT_AVAILABLE = False

if TYPE_CHECKING:
    import pandas as pd

logging.basicConfig(level=logging.INFO)
logging.getLogger("azure").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def initialize() -> SessionVariables:
    """Initialize app logic with Bedrock support."""
    if "session_variables" not in st.session_state:
        st.set_page_config(
            layout="wide",
            initial_sidebar_state="collapsed",
            page_title="GraphRAG",
        )
        
        # Initialize Bedrock support if available
        if BEDROCK_SUPPORT_AVAILABLE:
            try:
                logger.info("🚀 Initializing Bedrock support for GraphRAG...")
                bedrock_initialized = initialize_bedrock_for_search_app()
                if bedrock_initialized:
                    logger.info("✅ Bedrock support initialized successfully")
                else:
                    logger.warning("⚠️ Bedrock support initialization had issues")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Bedrock support: {e}")
        
        sv = SessionVariables()
        datasets = load_dataset_listing()
        sv.datasets.value = datasets
        sv.dataset.value = (
            st.query_params["dataset"].lower()
            if "dataset" in st.query_params
            else datasets[0].key
        )
        load_dataset(sv.dataset.value, sv)
        st.session_state["session_variables"] = sv
    return st.session_state["session_variables"]


def load_dataset(dataset: str, sv: SessionVariables):
    """Load dataset from the dropdown with Bedrock support."""
    sv.dataset.value = dataset
    sv.dataset_config.value = next(
        (d for d in sv.datasets.value if d.key == dataset), None
    )
    if sv.dataset_config.value is not None:
        sv.datasource.value = create_datasource(f"{sv.dataset_config.value.path}")  # type: ignore
        sv.graphrag_config.value = sv.datasource.value.read_settings("settings.yaml")
        
        # Check if this is a Bedrock configuration and handle accordingly
        if sv.graphrag_config.value and BEDROCK_SUPPORT_AVAILABLE:
            if is_bedrock_config(sv.graphrag_config.value):
                logger.info("🔥 Detected Bedrock configuration!")
                
                # Get configuration details
                config_info = get_config_info(sv.graphrag_config.value)
                logger.info(f"📋 Configuration Details:")
                logger.info(f"   - Chat Model: {config_info['chat_model_type']} ({config_info['chat_model_name']})")
                logger.info(f"   - Embedding Model: {config_info['embedding_model_type']} ({config_info['embedding_model_name']})")
                
                # Check search capabilities
                search_info = get_bedrock_search_engines(sv.graphrag_config.value)
                if search_info:
                    logger.info(f"🔍 Available Search Types: {', '.join(search_info['available_searches'])}")
                    if search_info['recommendations']:
                        for rec in search_info['recommendations']:
                            logger.warning(f"💡 Recommendation: {rec}")
                
                # Prepare Bedrock models if possible
                try:
                    bedrock_manager.create_models_for_config(sv.graphrag_config.value)
                    logger.info("✅ Bedrock models prepared successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Could not prepare Bedrock models: {e}")
                    logger.warning("Search functionality may be limited")
            else:
                logger.info("📊 Standard GraphRAG configuration detected")
        
        load_knowledge_model(sv)


def dataset_name(key: str, sv: SessionVariables) -> str:
    """Get dataset name."""
    return next((d for d in sv.datasets.value if d.key == key), None).name  # type: ignore


async def run_all_searches(query: str, sv: SessionVariables) -> list[SearchResult]:
    """Run all search engines and return the results."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    if sv.include_drift_search.value:
        tasks.append(
            run_drift_search(
                query=query,
                sv=sv,
            )
        )

    if sv.include_basic_rag.value:
        tasks.append(
            run_basic_search(
                query=query,
                sv=sv,
            )
        )
    if sv.include_local_search.value:
        tasks.append(
            run_local_search(
                query=query,
                sv=sv,
            )
        )
    if sv.include_global_search.value:
        tasks.append(
            run_global_search(
                query=query,
                sv=sv,
            )
        )

    return await asyncio.gather(*tasks)


async def run_generate_questions(query: str, sv: SessionVariables):
    """Run global search to generate questions for the dataset."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []

    tasks.append(
        run_global_search_question_generation(
            query=query,
            sv=sv,
        )
    )

    return await asyncio.gather(*tasks)


async def run_global_search_question_generation(
    query: str,
    sv: SessionVariables,
) -> SearchResult:
    """Run global search question generation process."""
    empty_context_data: dict[str, pd.DataFrame] = {}

    response, context_data = await api.global_search(
        config=sv.graphrag_config.value,
        entities=sv.entities.value,
        communities=sv.communities.value,
        community_reports=sv.community_reports.value,
        dynamic_community_selection=True,
        response_type="Single paragraph",
        community_level=sv.dataset_config.value.community_level,
        query=query,
    )

    # display response and reference context to UI
    return SearchResult(
        search_type=SearchType.Global,
        response=str(response),
        context=context_data if isinstance(context_data, dict) else empty_context_data,
    )


async def run_local_search(
    query: str,
    sv: SessionVariables,
) -> SearchResult:
    """Run local search."""
    print(f"Local search query: {query}")  # noqa T201

    # build local search engine
    response_placeholder = st.session_state[
        f"{SearchType.Local.value.lower()}_response_placeholder"
    ]
    response_container = st.session_state[f"{SearchType.Local.value.lower()}_container"]

    with response_placeholder, st.spinner("Generating answer using local search..."):
        empty_context_data: dict[str, pd.DataFrame] = {}

        response, context_data = await api.local_search(
            config=sv.graphrag_config.value,
            communities=sv.communities.value,
            entities=sv.entities.value,
            community_reports=sv.community_reports.value,
            text_units=sv.text_units.value,
            relationships=sv.relationships.value,
            covariates=sv.covariates.value,
            community_level=sv.dataset_config.value.community_level,
            response_type="Multiple Paragraphs",
            query=query,
        )

        print(f"Local Response: {response}")  # noqa T201
        print(f"Context data: {context_data}")  # noqa T201

    # display response and reference context to UI
    search_result = SearchResult(
        search_type=SearchType.Local,
        response=str(response),
        context=context_data if isinstance(context_data, dict) else empty_context_data,
    )

    display_search_result(
        container=response_container, result=search_result, stats=None
    )

    if "response_lengths" not in st.session_state:
        st.session_state.response_lengths = []

    st.session_state["response_lengths"].append({
        "result": search_result,
        "search": SearchType.Local.value.lower(),
    })

    return search_result


async def run_global_search(query: str, sv: SessionVariables) -> SearchResult:
    """Run global search."""
    print(f"Global search query: {query}")  # noqa T201

    # build global search engine
    response_placeholder = st.session_state[
        f"{SearchType.Global.value.lower()}_response_placeholder"
    ]
    response_container = st.session_state[
        f"{SearchType.Global.value.lower()}_container"
    ]

    response_placeholder.empty()
    with response_placeholder, st.spinner("Generating answer using global search..."):
        empty_context_data: dict[str, pd.DataFrame] = {}

        response, context_data = await api.global_search(
            config=sv.graphrag_config.value,
            entities=sv.entities.value,
            communities=sv.communities.value,
            community_reports=sv.community_reports.value,
            dynamic_community_selection=False,
            response_type="Multiple Paragraphs",
            community_level=sv.dataset_config.value.community_level,
            query=query,
        )

        print(f"Context data: {context_data}")  # noqa T201
        print(f"Global Response: {response}")  # noqa T201

    # display response and reference context to UI
    search_result = SearchResult(
        search_type=SearchType.Global,
        response=str(response),
        context=context_data if isinstance(context_data, dict) else empty_context_data,
    )

    display_search_result(
        container=response_container, result=search_result, stats=None
    )

    if "response_lengths" not in st.session_state:
        st.session_state.response_lengths = []

    st.session_state["response_lengths"].append({
        "result": search_result,
        "search": SearchType.Global.value.lower(),
    })

    return search_result


async def run_drift_search(
    query: str,
    sv: SessionVariables,
) -> SearchResult:
    """Run drift search."""
    print(f"Drift search query: {query}")  # noqa T201

    # build drift search engine
    response_placeholder = st.session_state[
        f"{SearchType.Drift.value.lower()}_response_placeholder"
    ]
    response_container = st.session_state[f"{SearchType.Drift.value.lower()}_container"]

    with response_placeholder, st.spinner("Generating answer using drift search..."):
        empty_context_data: dict[str, pd.DataFrame] = {}

        response, context_data = await api.drift_search(
            config=sv.graphrag_config.value,
            entities=sv.entities.value,
            communities=sv.communities.value,
            community_reports=sv.community_reports.value,
            text_units=sv.text_units.value,
            relationships=sv.relationships.value,
            community_level=sv.dataset_config.value.community_level,
            response_type="Multiple Paragraphs",
            query=query,
        )

        print(f"Drift Response: {response}")  # noqa T201
        print(f"Context data: {context_data}")  # noqa T201

    # display response and reference context to UI
    search_result = SearchResult(
        search_type=SearchType.Drift,
        response=str(response),
        context=context_data if isinstance(context_data, dict) else empty_context_data,
    )

    display_search_result(
        container=response_container, result=search_result, stats=None
    )

    if "response_lengths" not in st.session_state:
        st.session_state.response_lengths = []

    st.session_state["response_lengths"].append({
        "result": None,
        "search": SearchType.Drift.value.lower(),
    })

    return search_result


async def run_basic_search(
    query: str,
    sv: SessionVariables,
) -> SearchResult:
    """Run basic search."""
    print(f"Basic search query: {query}")  # noqa T201

    # build local search engine
    response_placeholder = st.session_state[
        f"{SearchType.Basic.value.lower()}_response_placeholder"
    ]
    response_container = st.session_state[f"{SearchType.Basic.value.lower()}_container"]

    with response_placeholder, st.spinner("Generating answer using basic RAG..."):
        empty_context_data: dict[str, pd.DataFrame] = {}

        response, context_data = await api.basic_search(
            config=sv.graphrag_config.value,
            text_units=sv.text_units.value,
            query=query,
        )

        print(f"Basic Response: {response}")  # noqa T201
        print(f"Context data: {context_data}")  # noqa T201

    # display response and reference context to UI
    search_result = SearchResult(
        search_type=SearchType.Basic,
        response=str(response),
        context=context_data if isinstance(context_data, dict) else empty_context_data,
    )

    display_search_result(
        container=response_container, result=search_result, stats=None
    )

    if "response_lengths" not in st.session_state:
        st.session_state.response_lengths = []

    st.session_state["response_lengths"].append({
        "search": SearchType.Basic.value.lower(),
        "result": search_result,
    })

    return search_result


def load_knowledge_model(sv: SessionVariables):
    """Load knowledge model from the datasource."""
    print("Loading knowledge model...", sv.dataset.value, sv.dataset_config.value)  # noqa T201
    model = load_model(sv.dataset.value, sv.datasource.value)

    sv.generated_questions.value = []
    sv.selected_question.value = ""
    sv.entities.value = model.entities
    sv.relationships.value = model.relationships
    sv.covariates.value = model.covariates
    sv.community_reports.value = model.community_reports
    sv.communities.value = model.communities
    sv.text_units.value = model.text_units

    return sv
