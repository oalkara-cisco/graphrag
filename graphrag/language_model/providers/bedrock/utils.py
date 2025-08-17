# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""AWS Bedrock utility functions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from graphrag.config.models.language_model_config import LanguageModelConfig

logger = logging.getLogger(__name__)


def create_bedrock_client(config: LanguageModelConfig) -> Any:
    """
    Create a Bedrock runtime client.
    
    Args:
        config: Language model configuration.
        
    Returns
    -------
        Bedrock runtime client.
    """
    try:
        import boto3
    except ImportError as e:
        msg = "boto3 is required for AWS Bedrock support. Install with: pip install graphrag[bedrock]"
        raise ImportError(msg) from e
    
    # Extract AWS configuration
    aws_region = config.aws_region or os.environ.get("AWS_REGION", "us-east-1")
    aws_profile = config.aws_profile or os.environ.get("AWS_PROFILE", None)
    
    # Create session with profile if specified, otherwise use default credential chain
    if aws_profile:
        logger.info("Using AWS profile: %s", aws_profile)
        session = boto3.Session(profile_name=aws_profile)
    else:
        logger.info("No AWS profile specified, using default credential chain (env vars, instance profile, etc.)")
        session = boto3.Session()
    
    try:
        # Create Bedrock runtime client
        client = session.client(
            "bedrock-runtime",
            region_name=aws_region,
        )
        
        # Test credentials by getting caller identity (optional verification)
        sts_client = session.client("sts", region_name=aws_region)
        caller_identity = sts_client.get_caller_identity()
        logger.info("Successfully authenticated with AWS. Account: %s, ARN: %s", 
                   caller_identity.get("Account", "unknown"),
                   caller_identity.get("Arn", "unknown"))
        
    except Exception as e:
        error_msg = f"Failed to create Bedrock client or authenticate with AWS: {e}"
        if not aws_profile:
            error_msg += "\nWhen running on EC2, ensure the instance has an IAM role with Bedrock permissions."
        else:
            error_msg += f"\nCheck that AWS profile '{aws_profile}' exists and has valid credentials."
        logger.error(error_msg)
        raise
    
    return client


def map_model_parameters(
    model_id: str,
    config: LanguageModelConfig,
    prompt: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Map GraphRAG parameters to model-specific Bedrock parameters.
    
    Args:
        model_id: Bedrock model ID.
        config: Language model configuration.
        prompt: The prompt text.
        **kwargs: Additional parameters.
        
    Returns
    -------
        Model-specific parameters dictionary.
    """
    # Base parameters from config
    temperature = kwargs.get("temperature", config.temperature)
    max_tokens = kwargs.get("max_tokens", config.max_tokens or 1000)
    top_p = kwargs.get("top_p", config.top_p)
    
    # Debug logging
    logger.debug("Mapping parameters for model_id: %s", model_id)
    
    # Model-specific parameter mapping
    if model_id.startswith("anthropic.claude") or ".anthropic.claude" in model_id:
        # Check if this is a cross-region inference profile
        if not model_id.startswith("anthropic."):  # Cross-region profiles don't start with anthropic directly
            # Cross-region inference profiles use the newer Messages API format
            params = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,  # Cross-region profiles use max_tokens
                "temperature": temperature,
                "top_p": top_p,
            }
            logger.debug("Using cross-region Claude parameters: %s", params)
            return params
        else:
            # Regular Claude models use the legacy format
            params = {
                "anthropic_version": "bedrock-2023-05-31", 
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens_to_sample": max_tokens,  # Regular Claude uses max_tokens_to_sample
                "temperature": temperature,
                "top_p": top_p,
            }
            logger.debug("Using regular Claude parameters: %s", params)
            return params
    
    elif model_id.startswith("meta.llama") or ".meta.llama" in model_id:
        # Llama models (regular and cross-region inference profiles)
        return {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    
    elif model_id.startswith("mistral") or ".mistral" in model_id:
        # Mistral models (regular and cross-region inference profiles)
        return {
            "prompt": f"<s>[INST] {prompt} [/INST]",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    
    elif model_id.startswith("cohere.command") or ".cohere.command" in model_id:
        # Cohere Command models (regular and cross-region inference profiles)
        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "p": top_p,
        }
    
    elif model_id.startswith("amazon.titan-text") or ".amazon.titan-text" in model_id:
        # Amazon Titan Text models (regular and cross-region inference profiles)
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
        }
    
    elif model_id.startswith("ai21.jamba") or model_id.startswith("ai21.j2") or ".ai21.jamba" in model_id or ".ai21.j2" in model_id:
        # AI21 models (regular and cross-region inference profiles)
        return {
            "prompt": prompt,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        }
    
    else:
        # Default/generic format
        logger.warning("Unknown model ID %s, using generic parameters", model_id)
        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }


def _extract_json_from_markdown(text: str) -> str:
    """
    Extract JSON content from markdown code blocks.
    
    Args:
        text: Text that may contain JSON wrapped in markdown code blocks.
        
    Returns
    -------
        Cleaned JSON text.
    """
    import re
    
    # Remove markdown code block markers
    # Matches ```json\n{...}\n``` or ```\n{...}\n```
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no code blocks found, return original text
    return text.strip()


def parse_bedrock_response(
    model_id: str, response: dict[str, Any], is_stream: bool = False
) -> str:
    """
    Parse response from different Bedrock models.
    
    Args:
        model_id: Bedrock model ID.
        response: Response from Bedrock API.
        is_stream: Whether this is a streaming response chunk.
        
    Returns
    -------
        Extracted text content.
    """
    try:
        if model_id.startswith("anthropic.claude") or ".anthropic.claude" in model_id:
            # Claude models (regular and cross-region inference profiles)
            if is_stream:
                if response.get("type") == "content_block_delta":
                    return response.get("delta", {}).get("text", "")
                return ""
            else:
                content = response.get("content", [])
                if content and isinstance(content, list):
                    raw_text = content[0].get("text", "")
                    # Extract JSON from markdown code blocks if present
                    return _extract_json_from_markdown(raw_text)
                return ""
        
        elif model_id.startswith("meta.llama") or ".meta.llama" in model_id:
            # Llama models (regular and cross-region inference profiles)
            if is_stream:
                return response.get("generation", "")
            else:
                return response.get("generation", "")
        
        elif model_id.startswith("mistral") or ".mistral" in model_id:
            # Mistral models (regular and cross-region inference profiles)
            if is_stream:
                return response.get("outputs", [{}])[0].get("text", "")
            else:
                outputs = response.get("outputs", [])
                if outputs:
                    return outputs[0].get("text", "")
                return ""
        
        elif model_id.startswith("cohere.command") or ".cohere.command" in model_id:
            # Cohere Command models (regular and cross-region inference profiles)
            if is_stream:
                return response.get("text", "")
            else:
                generations = response.get("generations", [])
                if generations:
                    return generations[0].get("text", "")
                return ""
        
        elif model_id.startswith("amazon.titan-text") or ".amazon.titan-text" in model_id:
            # Amazon Titan Text models (regular and cross-region inference profiles)
            if is_stream:
                return response.get("outputText", "")
            else:
                results = response.get("results", [])
                if results:
                    return results[0].get("outputText", "")
                return ""
        
        elif model_id.startswith("ai21.jamba") or model_id.startswith("ai21.j2") or ".ai21.jamba" in model_id or ".ai21.j2" in model_id:
            # AI21 models (regular and cross-region inference profiles)
            if is_stream:
                return response.get("completions", [{}])[0].get("data", {}).get("text", "")
            else:
                completions = response.get("completions", [])
                if completions:
                    return completions[0].get("data", {}).get("text", "")
                return ""
        
        else:
            # Try common response structures
            logger.warning("Unknown model ID %s, trying generic parsing", model_id)
            
            # Try different common response formats
            if "completion" in response:
                return response["completion"]
            elif "text" in response:
                return response["text"]
            elif "content" in response:
                return response["content"]
            elif "generation" in response:
                return response["generation"]
            else:
                logger.error("Could not parse response: %s", response)
                return ""
                
    except Exception as e:
        logger.error("Error parsing Bedrock response: %s", e)
        return ""


# Async execution helper (similar to fnllm utils)
T = TypeVar("T")

_loop = asyncio.new_event_loop()
_thr = threading.Thread(target=_loop.run_forever, name="Bedrock Async Runner", daemon=True)


def run_coroutine_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    """
    Run a coroutine synchronously.

    Args:
        coroutine: The coroutine to run.

    Returns
    -------
        The result of the coroutine.
    """
    if not _thr.is_alive():
        _thr.start()
    future = asyncio.run_coroutine_threadsafe(coroutine, _loop)
    return future.result()
