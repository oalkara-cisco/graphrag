# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""AWS Bedrock model implementations."""

from __future__ import annotations

import hashlib
import json
import logging
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


class BedrockChatModel:
    """AWS Bedrock Chat Model implementation."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        """Initialize Bedrock Chat Model."""
        self.name = name
        self.config = config
        self.callbacks = callbacks
        self.cache = cache
        self.client = create_bedrock_client(config)
        
        # Extract model ID from config
        self.model_id = config.bedrock_model_id or config.model
        if not self.model_id:
            msg = "bedrock_model_id or model must be specified for Bedrock models"
            raise ValueError(msg)

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        # Format the prompt with history if provided
        formatted_prompt = self._format_prompt(prompt, history)
        
        # Extract JSON mode settings
        json_mode = kwargs.get("json", False)
        json_model = kwargs.get("json_model", None)
        
        # Debug: Log JSON mode parameters
        logger.debug(f"JSON mode: {json_mode}, JSON model: {json_model}")
        
        # Map parameters to model-specific format
        model_params = map_model_parameters(
            self.model_id,
            self.config,
            formatted_prompt,
            **kwargs
        )
        
        try:
            # Check cache first - use hash to avoid filename length issues
            # SHA256 hash truncated to 16 chars provides good uniqueness while keeping filenames short
            prompt_hash = hashlib.sha256(formatted_prompt.encode('utf-8')).hexdigest()[:16]
            params_hash = hashlib.sha256(json.dumps(model_params, sort_keys=True).encode('utf-8')).hexdigest()[:16]
            
            # Include JSON mode parameters in cache key to ensure proper cache isolation
            json_mode_suffix = f":json={json_mode}:json_model={json_model.__name__ if json_model else None}"
            cache_key = f"{self.model_id}:{prompt_hash}:{params_hash}{json_mode_suffix}"
            if self.cache:
                cached_data = await self.cache.get(cache_key)
                if cached_data:
                    # Reconstruct BaseModelResponse from cached data  
                    return BaseModelResponse(**cached_data)
            
            # Call Bedrock API
            response = await self._invoke_model_async(model_params)
            
            # Parse response
            content = parse_bedrock_response(self.model_id, response)
            
            # Handle JSON parsing if json mode is enabled
            parsed_response = None
            if json_mode and json_model:
                try:
                    import json as json_lib
                    parsed_json = json_lib.loads(content)
                    # Validate and parse with the provided Pydantic model
                    parsed_response = json_model(**parsed_json)
                    logger.debug(f"Successfully parsed JSON response with {json_model.__name__}")
                except Exception as json_error:
                    logger.error("Failed to parse JSON response: %s", json_error)
                    logger.error("Content: %s", content[:500])
                    # Don't raise - let GraphRAG handle the None parsed_response
            
            # Create response object
            model_response = BaseModelResponse(
                output=BaseModelOutput(
                    content=content,
                    full_response=response,
                ),
                parsed_response=parsed_response,
                history=history or [],  # Ensure it's always a list
                cache_hit=False,
                tool_calls=[],  # Ensure it's always a list
                metrics={
                    "model_id": self.model_id,
                    "prompt_tokens": len(formatted_prompt.split()),
                    "completion_tokens": len(content.split()),
                },
            )
            
            # Cache the response (serialize to dict for JSON compatibility)
            if self.cache:
                await self.cache.set(cache_key, model_response.model_dump())
            
            return model_response
            
        except Exception as e:
            logger.error("Error invoking Bedrock model: %s", e)
            raise

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        # Format the prompt with history if provided
        formatted_prompt = self._format_prompt(prompt, history)
        
        # Map parameters to model-specific format
        model_params = map_model_parameters(
            self.model_id,
            self.config,
            formatted_prompt,
            **kwargs
        )
        
        try:
            # Call Bedrock API with streaming
            async for chunk in self._invoke_model_stream_async(model_params):
                content = parse_bedrock_response(self.model_id, chunk, is_stream=True)
                if content:
                    yield content
                    
        except Exception as e:
            logger.error("Error streaming from Bedrock model: %s", e)
            raise

    def chat(self, prompt: str, history: list | None = None, **kwargs: Any) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> Generator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)

    def _format_prompt(self, prompt: str, history: list | None) -> str:
        """Format prompt with conversation history."""
        if not history:
            return prompt
        
        # Build conversation string
        conversation = ""
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conversation += f"{role}: {content}\n"
        
        conversation += f"assistant: {prompt}"
        return conversation

    async def _invoke_model_async(self, model_params: dict[str, Any]) -> dict[str, Any]:
        """Invoke Bedrock model asynchronously."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._invoke_model_sync,
            model_params
        )

    def _invoke_model_sync(self, model_params: dict[str, Any]) -> dict[str, Any]:
        """Invoke Bedrock model synchronously."""
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(model_params),
            contentType="application/json",
            accept="application/json",
        )
        
        response_body = json.loads(response["body"].read())
        return response_body

    async def _invoke_model_stream_async(
        self, model_params: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Invoke Bedrock model with streaming asynchronously."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        
        # Get the streaming response
        response = await loop.run_in_executor(
            None,
            self._get_stream_response,
            model_params
        )
        
        # Yield chunks asynchronously
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            yield chunk

    def _get_stream_response(self, model_params: dict[str, Any]) -> dict[str, Any]:
        """Get streaming response from Bedrock."""
        return self.client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(model_params),
            contentType="application/json",
            accept="application/json",
        )


class BedrockEmbeddingModel:
    """AWS Bedrock Embedding Model implementation."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        """Initialize Bedrock Embedding Model."""
        self.name = name
        self.config = config
        self.callbacks = callbacks
        self.cache = cache
        self.client = create_bedrock_client(config)
        
        # Extract model ID from config
        self.model_id = config.bedrock_model_id or config.model
        if not self.model_id:
            msg = "bedrock_model_id or model must be specified for Bedrock models"
            raise ValueError(msg)
        
        # Validate embedding model
        if not self._is_embedding_model(self.model_id):
            msg = f"Model {self.model_id} is not a supported embedding model"
            raise ValueError(msg)

    async def aembed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """
        Embed the given text batch using the Model.

        Args:
            text_list: The list of texts to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the texts.
        """
        embeddings = []
        
        # Process in batches if needed
        batch_size = self._get_batch_size()
        
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            batch_embeddings = await self._embed_batch_async(batch, **kwargs)
            embeddings.extend(batch_embeddings)
        
        return embeddings

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        embeddings = await self.aembed_batch([text], **kwargs)
        return embeddings[0]

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """
        Embed the given text batch using the Model.

        Args:
            text_list: The list of texts to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the texts.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
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
        # Titan models support different batch sizes
        if self.model_id.startswith("amazon.titan"):
            return 25
        elif self.model_id.startswith("cohere"):
            return 96
        return 1

    async def _embed_batch_async(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Embed a batch of texts asynchronously."""
        import asyncio
        
        # Check cache first
        embeddings = []
        texts_to_embed = []
        cache_keys = []
        
        for text in texts:
            # Use hash to avoid filename length issues with long texts
            # SHA256 hash truncated to 16 chars provides good uniqueness while keeping filenames short
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
            cache_key = f"{self.model_id}:embed:{text_hash}"
            if self.cache:
                cached_embedding = await self.cache.get(cache_key)
                if cached_embedding:
                    embeddings.append(cached_embedding)
                    continue
            
            texts_to_embed.append(text)
            cache_keys.append(cache_key)
            embeddings.append(None)  # Placeholder
        
        if not texts_to_embed:
            return [e for e in embeddings if e is not None]
        
        # Prepare request based on model type
        if self.model_id.startswith("amazon.titan"):
            request_body = {"inputText": texts_to_embed[0]}  # Titan doesn't support batch
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
        
        # Fill in the embeddings and cache them
        j = 0
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                embeddings[i] = new_embeddings[j]
                if self.cache and j < len(cache_keys):
                    await self.cache.set(cache_keys[j], new_embeddings[j])
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
        """Invoke embedding model synchronously."""
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )
        
        response_body = json.loads(response["body"].read())
        return response_body
