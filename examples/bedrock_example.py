#!/usr/bin/env python3
"""Example of using AWS Bedrock models with GraphRAG."""

import asyncio
import os

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.providers.bedrock.models import (
    BedrockChatModel,
    BedrockEmbeddingModel,
)


async def main():
    """Example usage of Bedrock models."""
    
    # Configure Claude chat model
    chat_config = LanguageModelConfig(
        type="bedrock_chat",
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        bedrock_model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        aws_profile=os.getenv("AWS_PROFILE"),  # Uses default chain if not set
        max_tokens=1000,
        temperature=0.7,
    )
    
    # Configure Titan embedding model
    embedding_config = LanguageModelConfig(
        type="bedrock_embedding",
        model="amazon.titan-embed-text-v2:0",
        bedrock_model_id="amazon.titan-embed-text-v2:0",
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        aws_profile=os.getenv("AWS_PROFILE"),
    )
    
    # Create model instances
    chat_model = BedrockChatModel(name="example_chat", config=chat_config)
    embedding_model = BedrockEmbeddingModel(name="example_embed", config=embedding_config)
    
    # Example 1: Simple chat
    print("=== Chat Example ===")
    response = await chat_model.achat("What are the benefits of graph-based RAG systems?")
    print(f"Response: {response.output.content[:200]}...")
    
    # Example 2: Chat with history
    print("\n=== Chat with History ===")
    history = [
        {"role": "user", "content": "What is GraphRAG?"},
        {"role": "assistant", "content": "GraphRAG is a graph-based retrieval augmented generation system."},
    ]
    response = await chat_model.achat("How does it differ from traditional RAG?", history=history)
    print(f"Response: {response.output.content[:200]}...")
    
    # Example 3: Generate embeddings
    print("\n=== Embedding Example ===")
    texts = [
        "GraphRAG uses knowledge graphs for better context understanding.",
        "Traditional RAG relies on vector similarity search.",
        "Graph structures capture relationships between entities.",
    ]
    
    # Single embedding
    embedding = await embedding_model.aembed(texts[0])
    print(f"Single embedding dimension: {len(embedding)}")
    
    # Batch embeddings
    embeddings = await embedding_model.aembed_batch(texts)
    print(f"Batch embeddings: {len(embeddings)} texts, each with {len(embeddings[0])} dimensions")
    
    # Example 4: Streaming (if you want to demonstrate)
    print("\n=== Streaming Example ===")
    print("Streaming response: ", end="", flush=True)
    async for chunk in chat_model.achat_stream("Write a haiku about knowledge graphs"):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
