# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""AWS Bedrock Language Model Provider."""

from graphrag.language_model.providers.bedrock.models import (
    BedrockChatModel,
    BedrockEmbeddingModel,
)

__all__ = ["BedrockChatModel", "BedrockEmbeddingModel"]
