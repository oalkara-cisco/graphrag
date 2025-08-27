# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Enhanced AWS Bedrock model implementations with robust error handling and retry logic."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
    from graphrag.config.models.language_model_config import LanguageModelConfig

from graphrag.language_model.providers.bedrock.utils import (
    create_bedrock_client,
    map_model_parameters,
    parse_bedrock_response,
    run_coroutine_sync,
)
from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class BedrockRetryConfig:
    """Configuration for Bedrock retry logic."""
    
    def __init__(self, max_retries: int = 10):
        self.max_retries = max_retries
        
        # Different retry strategies for different error types
        self.retry_strategies = {
            'throttling': {
                'base_delay': 10.0,  # Start with 10s for throttling
                'max_delay': 300.0,  # Cap at 5 minutes
                'backoff_factor': 2.0,
                'jitter': True,
            },
            'timeout': {
                'base_delay': 5.0,   # Start with 5s for timeouts
                'max_delay': 120.0,  # Cap at 2 minutes
                'backoff_factor': 1.5,
                'jitter': True,
            },
            'connection': {
                'base_delay': 2.0,   # Start with 2s for connection issues
                'max_delay': 60.0,   # Cap at 1 minute
                'backoff_factor': 2.0,
                'jitter': True,
            },
            'server_error': {
                'base_delay': 15.0,  # Start with 15s for server errors
                'max_delay': 300.0,  # Cap at 5 minutes
                'backoff_factor': 2.0,
                'jitter': True,
            }
        }


def _classify_bedrock_error(exception: Exception) -> tuple[str, bool]:
    """
    Classify Bedrock errors to determine retry strategy.
    
    Returns:
        (error_type, should_retry): tuple with error classification and retry decision
    """
    error_str = str(exception).lower()
    exception_type = type(exception).__name__.lower()
    
    # Check for throttling errors
    throttling_indicators = [
        'throttlingexception',
        'throttling',
        'too many tokens',
        'rate limit',
        'quota exceeded',
        'service unavailable'
    ]
    
    # Check for timeout errors  
    timeout_indicators = [
        'timeout',
        'readtimeouterror',
        'connecttimeouterror',
        'read timeout',
        'connect timeout',
        'timed out',
        'operation timed out'
    ]
    
    # Check for connection errors
    connection_indicators = [
        'connectionerror',
        'connection',
        'network',
        'dns',
        'unable to connect',
        'connection refused',
        'connection reset'
    ]
    
    # Check for server errors (5xx)
    server_error_indicators = [
        'internal server error',
        'internal error',
        'server error',
        'service error',
        'bad gateway',
        '500',
        '502',
        '503',
        '504'
    ]
    
    # Check for non-retryable errors
    non_retryable_indicators = [
        'validation',
        'invalid',
        'malformed',
        'bad request',
        'unauthorized',
        'forbidden',
        'not found',
        '400',
        '401',
        '403',
        '404'
    ]
    
    # First check if it's non-retryable
    for indicator in non_retryable_indicators:
        if indicator in error_str or indicator in exception_type:
            return ('non_retryable', False)
    
    # Check for specific error types that should be retried
    # Check throttling first (highest priority)
    for indicator in throttling_indicators:
        if indicator in error_str or indicator in exception_type:
            return ('throttling', True)
    
    # Check timeout before connection (timeout is more specific)
    for indicator in timeout_indicators:
        if indicator in error_str or indicator in exception_type:
            return ('timeout', True)
    
    # Check connection errors last (most general)
    for indicator in connection_indicators:
        if indicator in error_str or indicator in exception_type:
            return ('connection', True)
    
    for indicator in server_error_indicators:
        if indicator in error_str or indicator in exception_type:
            return ('server_error', True)
    
    # Check for specific exception response attributes (safely)
    try:
        if hasattr(exception, 'response') and exception.response is not None:
            if hasattr(exception.response, 'get'):
                error_code = exception.response.get('Error', {}).get('Code', '')
                if 'throttling' in error_code.lower():
                    return ('throttling', True)
            elif hasattr(exception.response, 'status_code'):
                status_code = exception.response.status_code
                if status_code >= 500:
                    return ('server_error', True)
                elif status_code == 429:
                    return ('throttling', True)
    except (AttributeError, TypeError):
        # Safely ignore if we can't access response attributes
        pass
    
    # Default to connection error for unknown exceptions (conservative retry)
    logger.warning(f"Unknown error type, defaulting to connection retry: {exception_type}: {error_str}")
    return ('connection', True)


def _calculate_retry_delay(error_type: str, attempt: int, config: BedrockRetryConfig) -> float:
    """Calculate retry delay based on error type and attempt number."""
    strategy = config.retry_strategies.get(error_type, config.retry_strategies['connection'])
    
    # Calculate exponential backoff
    delay = strategy['base_delay'] * (strategy['backoff_factor'] ** attempt)
    
    # Apply maximum delay cap
    delay = min(delay, strategy['max_delay'])
    
    # Add jitter to avoid thundering herd
    if strategy['jitter']:
        jitter_factor = random.uniform(0.5, 1.5)
        delay *= jitter_factor
    
    return delay


def _enhanced_retry_decorator(config, retry_config: BedrockRetryConfig = None):
    """Enhanced decorator with robust error handling and multiple retry strategies."""
    if retry_config is None:
        retry_config = BedrockRetryConfig(max_retries=getattr(config, 'max_retries', 10))
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Classify the error
                    error_type, should_retry = _classify_bedrock_error(e)
                    
                    if not should_retry:
                        logger.error(f"Non-retryable Bedrock error: {error_type}: {str(e)}")
                        raise e
                    
                    if attempt < retry_config.max_retries:
                        # Calculate delay for this error type
                        delay = _calculate_retry_delay(error_type, attempt, retry_config)
                        
                        logger.warning(
                            f"Bedrock {error_type} error detected (attempt {attempt + 1}/{retry_config.max_retries + 1}). "
                            f"Waiting {delay:.1f} seconds before retry. "
                            f"Error: {str(e)[:200]}"
                        )
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Bedrock {error_type} error failed after {retry_config.max_retries + 1} attempts. "
                            f"Final error: {str(e)}"
                        )
                        raise e
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class EnhancedBedrockChatModel:
    """Enhanced AWS Bedrock Chat Model with robust error handling."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        """Initialize Enhanced Bedrock Chat Model."""
        self.name = name
        self.config = config
        self.callbacks = callbacks
        self.cache = cache
        
        # Initialize retry configuration
        max_retries = getattr(config, 'max_retries', 10)
        self.retry_config = BedrockRetryConfig(max_retries=max_retries)
        
        # Create Bedrock client with error handling
        try:
            self.client = create_bedrock_client(config)
        except Exception as e:
            logger.error(f"Failed to create Bedrock client: {e}")
            raise
        
        # Extract model ID from config
        self.model_id = config.bedrock_model_id or config.model
        if not self.model_id:
            msg = "bedrock_model_id or model must be specified for Bedrock models"
            raise ValueError(msg)
        
        logger.info(f"Initialized Enhanced Bedrock Chat Model: {self.model_id} with {max_retries} max retries")

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt with enhanced error handling.
        """
        # Format the prompt with history if provided
        formatted_prompt = self._format_prompt(prompt, history)
        
        # Extract JSON mode settings
        json_mode = kwargs.get("json", False)
        json_model = kwargs.get("json_model", None)
        
        # Debug: Log JSON mode parameters
        logger.debug(f"JSON mode: {json_mode}, JSON model: {json_model}")
        
        # Map parameters to model-specific format
        try:
            model_params = map_model_parameters(
                self.model_id,
                self.config,
                formatted_prompt,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to map model parameters: {e}")
            raise
        
        try:
            # Check cache first - use hash to avoid filename length issues
            prompt_hash = hashlib.sha256(formatted_prompt.encode('utf-8')).hexdigest()[:16]
            params_hash = hashlib.sha256(json.dumps(model_params, sort_keys=True).encode('utf-8')).hexdigest()[:16]
            
            # Include JSON mode parameters in cache key
            json_mode_suffix = f":json={json_mode}:json_model={json_model.__name__ if json_model else None}"
            cache_key = self._build_cache_key(prompt_hash, params_hash, json_mode_suffix)
            
            if self.cache:
                try:
                    cached_data = await self.cache.get(cache_key)
                    if cached_data:
                        logger.debug("Cache hit for Bedrock request")
                        return BaseModelResponse(**cached_data)
                except Exception as e:
                    logger.warning(f"Cache lookup failed: {e}")
            
            # Call Bedrock API with enhanced error handling
            response = await self._invoke_model_async(model_params)
            
            # Parse response
            content = parse_bedrock_response(self.model_id, response)
            
            # Handle JSON parsing if json mode is enabled
            parsed_response = None
            if json_mode and json_model:
                parsed_response = self._robust_json_parse(content, json_model)
            
            # Create response object
            model_response = BaseModelResponse(
                output=BaseModelOutput(
                    content=content,
                    full_response=response,
                ),
                parsed_response=parsed_response,
                history=history or [],
                cache_hit=False,
                tool_calls=[],
                metrics={
                    "model_id": self.model_id,
                    "prompt_tokens": len(formatted_prompt.split()),
                    "completion_tokens": len(content.split()),
                },
            )
            
            # Cache the response
            if self.cache:
                try:
                    await self.cache.set(cache_key, model_response.model_dump())
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
            
            return model_response
            
        except Exception as e:
            logger.error("Error invoking Enhanced Bedrock model: %s", e)
            raise

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Stream Chat with enhanced error handling."""
        formatted_prompt = self._format_prompt(prompt, history)
        
        try:
            model_params = map_model_parameters(
                self.model_id,
                self.config,
                formatted_prompt,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to map model parameters for streaming: {e}")
            raise
        
        try:
            async for chunk in self._invoke_model_stream_async(model_params):
                content = parse_bedrock_response(self.model_id, chunk, is_stream=True)
                if content:
                    yield content
                    
        except Exception as e:
            logger.error("Error streaming from Enhanced Bedrock model: %s", e)
            raise

    def chat(self, prompt: str, history: list | None = None, **kwargs: Any) -> ModelResponse:
        """Synchronous chat with enhanced error handling."""
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> Generator[str, None]:
        """Synchronous streaming not supported."""
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)

    def _format_prompt(self, prompt: str, history: list | None) -> str:
        """Format prompt with conversation history."""
        if not history:
            return prompt
        
        conversation = ""
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conversation += f"{role}: {content}\n"
        
        conversation += f"assistant: {prompt}"
        return conversation

    def _get_model_family(self, model_id: str) -> str:
        """Get model family for cache grouping."""
        if "claude" in model_id.lower():
            return "claude"
        elif "llama" in model_id.lower():
            return "llama"
        elif "mistral" in model_id.lower():
            return "mistral"
        elif "titan" in model_id.lower():
            return "titan"
        elif "cohere" in model_id.lower():
            return "cohere"
        elif "ai21" in model_id.lower():
            return "ai21"
        else:
            return "unknown"

    def _build_cache_key(self, prompt_hash: str, params_hash: str, json_mode_suffix: str) -> str:
        """
        Build cache key based on configuration strategy.
        
        Supports multiple cache strategies:
        - 'shared': Share cache across all models (maximum cache hits)
        - 'family': Share cache within model families (e.g., all Claude models)
        - 'model_specific': Separate cache per model (default, most conservative)
        """
        # Check configuration for cache strategy
        cache_strategy = getattr(self.config, 'cache_strategy', 'model_specific')
        enable_cross_model_cache = getattr(self.config, 'enable_cross_model_cache', False)
        
        # For backward compatibility, check enable_cross_model_cache flag
        if enable_cross_model_cache:
            cache_strategy = 'shared'
        
        if cache_strategy == 'shared':
            # Share cache across all models - maximum cache reuse
            cache_key = f"{prompt_hash}:{params_hash}{json_mode_suffix}"
            logger.debug(f"Using shared cache key (cross-model): {cache_key[:50]}...")
            
        elif cache_strategy == 'family':
            # Share cache within model families (e.g., claude-3-sonnet, claude-3-haiku)
            model_family = self._get_model_family(self.model_id)
            cache_key = f"{model_family}:{prompt_hash}:{params_hash}{json_mode_suffix}"
            logger.debug(f"Using family cache key ({model_family}): {cache_key[:50]}...")
            
        else:  # 'model_specific' (default)
            # Separate cache per model - most conservative
            cache_key = f"{self.model_id}:{prompt_hash}:{params_hash}{json_mode_suffix}"
            logger.debug(f"Using model-specific cache key: {cache_key[:50]}...")
        
        return cache_key

    def _robust_json_parse(self, content: str, json_model: type) -> Any:
        """
        Robust JSON parsing with multiple repair strategies for truncated responses.
        
        Args:
            content: Raw response content that may contain truncated JSON
            json_model: Pydantic model class to parse with
            
        Returns:
            Parsed model instance or None if parsing fails
        """
        import json as json_lib
        import re
        
        try:
            # Strategy 1: Try direct parsing first
            parsed_json = json_lib.loads(content)
            parsed_response = json_model(**parsed_json)
            logger.debug(f"Successfully parsed JSON response with {json_model.__name__}")
            return parsed_response
            
        except json_lib.JSONDecodeError as json_error:
            logger.warning(f"Initial JSON parsing failed: {json_error}")
            logger.debug(f"Content preview: {content[:200]}...")
            
            # Strategy 2: Try to repair truncated JSON
            try:
                repaired_json = self._repair_truncated_json(content)
                if repaired_json != content:
                    parsed_json = json_lib.loads(repaired_json)
                    parsed_response = json_model(**parsed_json)
                    logger.info(f"Successfully repaired and parsed truncated JSON with {json_model.__name__}")
                    return parsed_response
            except Exception as repair_error:
                logger.debug(f"JSON repair failed: {repair_error}")
            
            # Strategy 3: Extract partial data and create minimal valid model
            try:
                partial_data = self._extract_partial_json_data(content)
                if partial_data:
                    parsed_response = json_model(**partial_data)
                    logger.info(f"Successfully created model from partial data with {json_model.__name__}")
                    return parsed_response
            except Exception as partial_error:
                logger.debug(f"Partial data extraction failed: {partial_error}")
            
            # Strategy 4: Create model with minimal required fields
            try:
                minimal_data = self._create_minimal_model_data(json_model)
                parsed_response = json_model(**minimal_data)
                logger.warning(f"Using minimal fallback model for {json_model.__name__}")
                return parsed_response
            except Exception as minimal_error:
                logger.error(f"Minimal model creation failed: {minimal_error}")
            
        except Exception as model_error:
            logger.error(f"Model instantiation failed: {model_error}")
            
        logger.error("All JSON parsing strategies failed")
        logger.error(f"Content: {content[:500]}")
        return None
    
    def _repair_truncated_json(self, content: str) -> str:
        """Attempt to repair truncated JSON by adding missing closing brackets/quotes."""
        import json as json_lib
        
        # Remove any trailing incomplete text after the last valid JSON structure
        content = content.strip()
        
        # Count open/close brackets and quotes to determine what's missing
        brace_count = content.count('{') - content.count('}')
        bracket_count = content.count('[') - content.count(']')
        
        # Try to complete the JSON structure
        repaired = content
        
        # Handle incomplete string values (common truncation point)
        if repaired.endswith('"'):
            # String was complete, no action needed
            pass
        elif '"' in repaired and not repaired.endswith('"'):
            # Likely truncated in the middle of a string value
            if repaired.count('"') % 2 == 1:  # Odd number of quotes = unclosed string
                repaired += '"'
        
        # Add missing closing brackets
        repaired += ']' * bracket_count
        repaired += '}' * brace_count
        
        # Validate the repair
        try:
            json_lib.loads(repaired)
            return repaired
        except json_lib.JSONDecodeError:
            # If repair didn't work, return original
            return content
    
    def _extract_partial_json_data(self, content: str) -> dict:
        """Extract whatever valid JSON data we can from truncated content."""
        import re
        import json as json_lib
        
        partial_data = {}
        
        # Try to extract title field
        title_match = re.search(r'"title"\s*:\s*"([^"]*)"', content)
        if title_match:
            partial_data['title'] = title_match.group(1)
        
        # Try to extract summary field
        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', content, re.DOTALL)
        if summary_match:
            partial_data['summary'] = summary_match.group(1)
        
        # Try to extract rating field
        rating_match = re.search(r'"rating"\s*:\s*([0-9.]+)', content)
        if rating_match:
            try:
                partial_data['rating'] = float(rating_match.group(1))
            except ValueError:
                pass
        
        # Try to extract rating_explanation
        rating_exp_match = re.search(r'"rating_explanation"\s*:\s*"([^"]*)"', content)
        if rating_exp_match:
            partial_data['rating_explanation'] = rating_exp_match.group(1)
        
        # Try to extract findings array (even if incomplete)
        findings_match = re.search(r'"findings"\s*:\s*\[(.*?)\]', content, re.DOTALL)
        if findings_match:
            try:
                findings_str = f'[{findings_match.group(1)}]'
                findings_data = json_lib.loads(findings_str)
                partial_data['findings'] = findings_data
            except json_lib.JSONDecodeError:
                # If findings array is malformed, create empty array
                partial_data['findings'] = []
        else:
            partial_data['findings'] = []
        
        return partial_data
    
    def _create_minimal_model_data(self, json_model: type) -> dict:
        """Create minimal data structure that satisfies the model's required fields."""
        minimal_data = {
            'title': 'Community Report',
            'summary': 'Report generation was incomplete due to response truncation.',
            'rating': 1.0,
            'rating_explanation': 'Low rating due to incomplete data generation.',
            'findings': []
        }
        return minimal_data

    async def _invoke_model_async(self, model_params: dict[str, Any]) -> dict[str, Any]:
        """Invoke Bedrock model asynchronously with enhanced error handling."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._invoke_model_sync,
            model_params
        )

    def _invoke_model_sync(self, model_params: dict[str, Any]) -> dict[str, Any]:
        """Invoke Bedrock model synchronously with enhanced retry logic."""
        @_enhanced_retry_decorator(self.config, self.retry_config)
        def _invoke_with_enhanced_retry():
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(model_params),
                    contentType="application/json",
                    accept="application/json",
                )
                
                response_body = json.loads(response["body"].read())
                return response_body
                
            except Exception as e:
                # Add context to the error for better debugging
                logger.debug(f"Bedrock API call failed for model {self.model_id}: {str(e)[:200]}")
                raise
        
        return _invoke_with_enhanced_retry()

    async def _invoke_model_stream_async(
        self, model_params: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Invoke Bedrock model with streaming and enhanced error handling."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        
        # Get the streaming response with error handling
        response = await loop.run_in_executor(
            None,
            self._get_stream_response,
            model_params
        )
        
        # Yield chunks asynchronously with error handling
        try:
            for event in response["body"]:
                try:
                    chunk = json.loads(event["chunk"]["bytes"])
                    yield chunk
                except Exception as e:
                    logger.warning(f"Failed to parse streaming chunk: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error processing streaming response: {e}")
            raise

    def _get_stream_response(self, model_params: dict[str, Any]) -> dict[str, Any]:
        """Get streaming response with enhanced retry logic."""
        @_enhanced_retry_decorator(self.config, self.retry_config)
        def _get_stream_with_enhanced_retry():
            try:
                return self.client.invoke_model_with_response_stream(
                    modelId=self.model_id,
                    body=json.dumps(model_params),
                    contentType="application/json",
                    accept="application/json",
                )
            except Exception as e:
                logger.debug(f"Bedrock streaming API call failed for model {self.model_id}: {str(e)[:200]}")
                raise
        
        return _get_stream_with_enhanced_retry()


class EnhancedBedrockEmbeddingModel:
    """Enhanced AWS Bedrock Embedding Model with robust error handling."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        """Initialize Enhanced Bedrock Embedding Model."""
        self.name = name
        self.config = config
        self.callbacks = callbacks
        self.cache = cache
        
        # Initialize retry configuration
        max_retries = getattr(config, 'max_retries', 10)
        self.retry_config = BedrockRetryConfig(max_retries=max_retries)
        
        try:
            self.client = create_bedrock_client(config)
        except Exception as e:
            logger.error(f"Failed to create Bedrock client for embeddings: {e}")
            raise
        
        # Extract model ID from config
        self.model_id = config.bedrock_model_id or config.model
        if not self.model_id:
            msg = "bedrock_model_id or model must be specified for Bedrock models"
            raise ValueError(msg)
        
        # Validate embedding model
        if not self._is_embedding_model(self.model_id):
            msg = f"Model {self.model_id} is not a supported embedding model"
            raise ValueError(msg)
        
        logger.info(f"Initialized Enhanced Bedrock Embedding Model: {self.model_id}")

    async def aembed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """Embed text batch with enhanced error handling."""
        embeddings = []
        batch_size = self._get_batch_size()
        
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            try:
                batch_embeddings = await self._embed_batch_async(batch, **kwargs)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to embed batch {i//batch_size + 1}: {e}")
                raise
        
        return embeddings

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """Embed single text with enhanced error handling."""
        embeddings = await self.aembed_batch([text], **kwargs)
        return embeddings[0]

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """Synchronous batch embedding."""
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """Synchronous single embedding."""
        return run_coroutine_sync(self.aembed(text, **kwargs))

    def _is_embedding_model(self, model_id: str) -> bool:
        """Check if the model is a supported embedding model."""
        embedding_models = [
            "amazon.titan-embed-text-v1",
            "amazon.titan-embed-text-v2:0",
            "amazon.titan-embed-g1-text-02",
            "cohere.embed-english-v3",
            "cohere.embed-multilingual-v3",
        ]
        return any(model_id.startswith(model) for model in embedding_models)

    def _get_batch_size(self) -> int:
        """Get the batch size for the model."""
        if self.model_id.startswith("amazon.titan"):
            return 25
        elif self.model_id.startswith("cohere"):
            return 96
        return 1

    def _build_embedding_cache_key(self, text_hash: str) -> str:
        """
        Build embedding cache key based on configuration strategy.
        
        Supports the same cache strategies as chat models:
        - 'shared': Share cache across all embedding models
        - 'family': Share cache within embedding model families  
        - 'model_specific': Separate cache per model (default)
        """
        # Check configuration for cache strategy
        cache_strategy = getattr(self.config, 'cache_strategy', 'model_specific')
        enable_cross_model_cache = getattr(self.config, 'enable_cross_model_cache', False)
        
        # For backward compatibility
        if enable_cross_model_cache:
            cache_strategy = 'shared'
        
        if cache_strategy == 'shared':
            # Share embedding cache across all models
            cache_key = f"embed:{text_hash}"
            logger.debug(f"Using shared embedding cache key: {cache_key}")
            
        elif cache_strategy == 'family':
            # Share cache within embedding model families (e.g., all Titan embeddings)
            if self.model_id.startswith("amazon.titan"):
                model_family = "titan-embed"
            elif self.model_id.startswith("cohere"):
                model_family = "cohere-embed"
            else:
                model_family = "unknown-embed"
            cache_key = f"{model_family}:embed:{text_hash}"
            logger.debug(f"Using family embedding cache key ({model_family}): {cache_key}")
            
        else:  # 'model_specific' (default)
            # Separate cache per embedding model
            cache_key = f"{self.model_id}:embed:{text_hash}"
            logger.debug(f"Using model-specific embedding cache key: {cache_key}")
        
        return cache_key

    async def _embed_batch_async(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Embed a batch of texts asynchronously with enhanced error handling."""
        import asyncio
        
        # Check cache first
        embeddings = []
        texts_to_embed = []
        cache_keys = []
        
        for text in texts:
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
            cache_key = self._build_embedding_cache_key(text_hash)
            
            if self.cache:
                try:
                    cached_embedding = await self.cache.get(cache_key)
                    if cached_embedding:
                        embeddings.append(cached_embedding)
                        continue
                except Exception as e:
                    logger.warning(f"Cache lookup failed for embedding: {e}")
            
            texts_to_embed.append(text)
            cache_keys.append(cache_key)
            embeddings.append(None)  # Placeholder
        
        if not texts_to_embed:
            return [e for e in embeddings if e is not None]
        
        # Prepare request based on model type with error handling
        try:
            if self.model_id.startswith("amazon.titan"):
                new_embeddings = []
                for text in texts_to_embed:
                    request_body = {"inputText": text}
                    response = await self._invoke_embedding_model_async(request_body)
                    embedding = response.get("embedding", [])
                    new_embeddings.append(embedding)
                    
            elif self.model_id.startswith("cohere"):
                request_body = {
                    "texts": texts_to_embed,
                    "input_type": "search_document",
                }
                response = await self._invoke_embedding_model_async(request_body)
                new_embeddings = response.get("embeddings", [])
            else:
                msg = f"Unsupported embedding model: {self.model_id}"
                raise ValueError(msg)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
        
        # Fill in the embeddings and cache them
        j = 0
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                embeddings[i] = new_embeddings[j]
                if self.cache and j < len(cache_keys):
                    try:
                        await self.cache.set(cache_keys[j], new_embeddings[j])
                    except Exception as e:
                        logger.warning(f"Failed to cache embedding: {e}")
                j += 1
        
        return embeddings

    async def _invoke_embedding_model_async(
        self, request_body: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke embedding model asynchronously."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._invoke_embedding_model_sync,
            request_body
        )

    def _invoke_embedding_model_sync(
        self, request_body: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke embedding model synchronously with enhanced retry logic."""
        @_enhanced_retry_decorator(self.config, self.retry_config)
        def _invoke_embedding_with_enhanced_retry():
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json",
                )
                
                response_body = json.loads(response["body"].read())
                return response_body
            except Exception as e:
                logger.debug(f"Bedrock embedding API call failed for model {self.model_id}: {str(e)[:200]}")
                raise
        
        return _invoke_embedding_with_enhanced_retry()


# Alias classes for backward compatibility
BedrockChatModel = EnhancedBedrockChatModel
BedrockEmbeddingModel = EnhancedBedrockEmbeddingModel
