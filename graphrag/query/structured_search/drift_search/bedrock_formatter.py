compl"""
Bedrock-Compatible Prompt Formatter for DRIFT Search
====================================================

Elegant solution to handle OpenAI vs AWS Bedrock API differences in DRIFT search.
"""

from typing import Dict, List, Any, Optional


class BedrockPromptFormatter:
    """
    Elegant prompt formatter that handles AWS Bedrock API compatibility 
    for DRIFT search reduce operations.
    """
    
    @staticmethod
    def format_reduce_prompt(
        system_prompt_template: str,
        context_data: List[str],
        response_type: str,
        query: str,
        **template_kwargs
    ) -> str:
        """
        Format a reduce prompt for AWS Bedrock compatibility.
        
        Args:
            system_prompt_template: The prompt template (e.g., DRIFT_REDUCE_PROMPT)
            context_data: List of response data to include
            response_type: Expected response format
            query: User query
            **template_kwargs: Additional template variables
            
        Returns:
            Complete formatted prompt suitable for AWS Bedrock
        """
        try:
            # First try to use the provided template
            formatted_prompt = system_prompt_template.format(
                context_data=context_data,
                response_type=response_type,
                **template_kwargs
            )
            
            # Add query at the end for Bedrock compatibility
            complete_prompt = f"{formatted_prompt}\n\nQuery: {query}"
            
            return complete_prompt
            
        except (KeyError, ValueError) as e:
            # Graceful fallback if template formatting fails
            return BedrockPromptFormatter._create_fallback_prompt(
                context_data, response_type, query
            )
    
    @staticmethod
    def _create_fallback_prompt(
        context_data: List[str], 
        response_type: str, 
        query: str
    ) -> str:
        """Create a fallback prompt if template formatting fails."""
        
        context_text = "\n\n".join(str(resp) for resp in context_data) if context_data else "No relevant information found in the available data."
        
        return f"""You are a helpful assistant responding to questions about data in the reports provided.

You must ONLY use information from the provided data reports. Do NOT use external knowledge or general knowledge.

---Data Reports---

{context_text}

---Target response length and format---

{response_type}

Generate a response that answers the user's question using ONLY the information provided above. If no relevant information is found, state clearly 'No relevant information found in the available data about this topic.'

Query: {query}"""

    @staticmethod
    def format_system_message_prompt(
        system_prompt: str,
        query: str,
        history_messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Convert OpenAI-style system messages to Bedrock-compatible format.
        
        Args:
            system_prompt: System instruction prompt
            query: User query  
            history_messages: Optional conversation history
            
        Returns:
            Complete prompt formatted for AWS Bedrock
        """
        if not history_messages:
            # Simple case: just system prompt + query
            return f"{system_prompt}\n\nQuery: {query}"
        
        # Handle history messages
        conversation_text = ""
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Use system content as the main prompt
                conversation_text = content
            else:
                conversation_text += f"\n\n{role}: {content}"
        
        return f"{conversation_text}\n\nQuery: {query}"
    
    @staticmethod 
    def is_bedrock_model(model_config) -> bool:
        """
        Check if the model is AWS Bedrock to determine if formatting is needed.
        """
        if hasattr(model_config, 'model_id'):
            return 'bedrock' in str(model_config.model_id).lower() or 'anthropic' in str(model_config.model_id).lower()
        return False
