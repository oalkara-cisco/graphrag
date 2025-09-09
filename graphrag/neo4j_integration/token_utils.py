"""
Token encoding utilities for Neo4j integration.

Provides proper token encoding using GraphRAG configuration settings.
"""

import logging
import tiktoken
from typing import Optional

import graphrag.config.defaults as defs

logger = logging.getLogger(__name__)


def get_token_encoder_from_config(config, model_key: str = "default_chat_model") -> tiktoken.Encoding:
    """
    Get the appropriate token encoder from GraphRAG configuration.
    
    Args:
        config: GraphRagConfig object
        model_key: Which model configuration to use for encoding ("default_chat_model" or "default_embedding_model")
        
    Returns:
        tiktoken.Encoding: The token encoder to use
        
    Notes:
        This is the preferred method as it uses the encoding_model specified in settings.yaml
    """
    
    try:
        # Get the encoding model from the configuration
        model_config = config.models.get(model_key)
        if model_config and hasattr(model_config, 'encoding_model'):
            encoding_name = model_config.encoding_model
            if encoding_name and encoding_name.strip():
                logger.debug("Using encoding from config: %s (model: %s)", encoding_name, model_key)
                return tiktoken.get_encoding(encoding_name)
        
        # Fallback to default if not found
        logger.debug("No encoding_model found in config for %s, using default: %s", model_key, defs.ENCODING_MODEL)
        return tiktoken.get_encoding(defs.ENCODING_MODEL)
        
    except Exception as e:
        logger.warning("Failed to get encoding from config: %s. Using default: %s", e, defs.ENCODING_MODEL)
        return tiktoken.get_encoding(defs.ENCODING_MODEL)


def get_token_encoder_for_model(model_name: Optional[str] = None) -> tiktoken.Encoding:
    """
    Get the appropriate token encoder for a given model.
    
    Args:
        model_name: The name of the model (e.g., "gpt-4", "claude-3-sonnet")
        
    Returns:
        tiktoken.Encoding: The token encoder to use
        
    Notes:
        This is a fallback method. Prefer get_token_encoder_from_config() when possible.
    """
    
    # If no model specified, use default
    if not model_name:
        logger.debug("No model name specified, using default encoding: %s", defs.ENCODING_MODEL)
        return tiktoken.get_encoding(defs.ENCODING_MODEL)
    
    # Try to get model-specific encoding first
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        logger.debug("Using model-specific encoding for: %s", model_name)
        return encoding
    except KeyError:
        # Model not supported by tiktoken (e.g., Bedrock models)
        logger.debug(
            "Model '%s' not supported by tiktoken, falling back to default encoding: %s", 
            model_name, defs.ENCODING_MODEL
        )
        return tiktoken.get_encoding(defs.ENCODING_MODEL)


def get_model_name_from_config(chat_model) -> Optional[str]:
    """
    Extract model name from various model objects.
    
    Args:
        chat_model: The chat model object (BedrockChatModel, OpenAI, etc.)
        
    Returns:
        Optional[str]: The model name if extractable, None otherwise
    """
    
    # Try various attributes where model name might be stored
    model_name_attrs = ['model', 'model_name', 'model_id', '_model_id', 'bedrock_model_id']
    
    for attr in model_name_attrs:
        if hasattr(chat_model, attr):
            model_name = getattr(chat_model, attr)
            if model_name:
                logger.debug("Extracted model name '%s' from attribute '%s'", model_name, attr)
                return model_name
    
    # Try to get from class name as fallback
    if hasattr(chat_model, '__class__'):
        class_name = chat_model.__class__.__name__
        logger.debug("Could not extract model name, using class name: %s", class_name)
        
        # Map class names to reasonable tokenizer choices
        if 'bedrock' in class_name.lower():
            return None  # Will fall back to cl100k_base
        elif 'openai' in class_name.lower():
            return 'gpt-4'  # Reasonable default for OpenAI
    
    logger.warning("Could not determine model name from chat_model: %s", type(chat_model))
    return None


def get_token_encoder_from_model(chat_model) -> tiktoken.Encoding:
    """
    Get token encoder from a model object (fallback method).
    
    Args:
        chat_model: The chat model object
        
    Returns:
        tiktoken.Encoding: The appropriate token encoder
        
    Notes:
        This is a fallback method. Prefer get_token_encoder_from_config() when possible.
    """
    model_name = get_model_name_from_config(chat_model)
    return get_token_encoder_for_model(model_name)
