"""
DRIFT Response Adapter - Elegant Solution
==========================================

Unified response interface that adapts different search response formats
for seamless integration with DRIFT workflow.
"""

import json
import logging
from typing import Dict, Any, Optional, Union
from graphrag.query.llm.text_utils import try_parse_json_object

logger = logging.getLogger(__name__)


class DriftResponseAdapter:
    """
    Elegant adapter that normalizes responses from different search components
    into the format expected by DRIFT workflow.
    """
    
    @staticmethod
    def adapt_search_response(
        raw_response: str, 
        component_name: str = "Unknown",
        default_score: int = 80
    ) -> Dict[str, Any]:
        """
        Adapt any search response into DRIFT-compatible format.
        
        Args:
            raw_response: Raw response from search component
            component_name: Name of the component for debugging
            default_score: Default score to assign if not found
            
        Returns:
            Normalized response dict with response, score, follow_up_queries
        """
        if not raw_response or not raw_response.strip():
            return DriftResponseAdapter._create_empty_response(
                "No response received from search component"
            )
        
        # Try JSON parsing first
        success, parsed = try_parse_json_object(raw_response)
        
        if success and isinstance(parsed, dict):
            return DriftResponseAdapter._normalize_json_response(parsed, default_score)
        
        # Handle plain text response (LocalSearch standard output)
        return DriftResponseAdapter._adapt_text_response(
            raw_response, component_name, default_score
        )
    
    @staticmethod
    def _normalize_json_response(parsed: Dict[str, Any], default_score: int) -> Dict[str, Any]:
        """Normalize a parsed JSON response to DRIFT format."""
        
        # Extract response content
        response_text = (
            parsed.get("response") or 
            parsed.get("intermediate_answer") or 
            parsed.get("content") or 
            str(parsed)
        )
        
        # Extract score
        score = parsed.get("score", default_score)
        if not isinstance(score, (int, float)):
            score = default_score
        
        # Extract follow-up queries
        follow_ups = (
            parsed.get("follow_up_queries") or 
            parsed.get("followup_queries") or 
            parsed.get("next_questions") or 
            []
        )
        
        return {
            "response": response_text,
            "score": float(score),
            "follow_up_queries": list(follow_ups) if follow_ups else []
        }
    
    @staticmethod
    def _adapt_text_response(
        text: str, 
        component_name: str, 
        default_score: int
    ) -> Dict[str, Any]:
        """Adapt plain text response to DRIFT format."""
        
        # For substantial text responses, generate intelligent follow-ups
        follow_ups = []
        if len(text.strip()) > 100:
            # Parse text to generate relevant follow-up questions
            follow_ups = DriftResponseAdapter._generate_followups_from_text(text)
        
        return {
            "response": text.strip(),
            "score": float(default_score),
            "follow_up_queries": follow_ups
        }
    
    @staticmethod
    def _generate_followups_from_text(text: str) -> list[str]:
        """Generate intelligent follow-up queries from text content."""
        
        # Simple heuristic-based follow-up generation
        followups = []
        
        # Extract key entities/concepts for follow-ups
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ["system", "platform", "service", "product"]):
            followups.extend([
                "What specific products does this organization develop?",
                "How is this organization structured?",
                "What services does this platform provide?"
            ])
        
        if "team" in text_lower or "organization" in text_lower:
            followups.extend([
                "What teams are part of this organization?",
                "How is the team structured?",
                "What are the team's responsibilities?"
            ])
        
        if "security" in text_lower:
            followups.extend([
                "What security technologies are involved?",
                "How does this integrate with other security systems?",
                "What threats does this address?"
            ])
        
        # Return up to 5 diverse follow-ups
        return list(dict.fromkeys(followups))[:5] if followups else [
            "What additional details are available?",
            "How does this relate to other systems?",
            "What are the key components involved?"
        ]
    
    @staticmethod
    def _create_empty_response(reason: str) -> Dict[str, Any]:
        """Create empty response with explanation."""
        
        return {
            "response": f"No information found: {reason}",
            "score": 0,
            "follow_up_queries": ["Could you provide more specific information?"]
        }
    
    @staticmethod
    def is_valid_drift_response(response: Dict[str, Any]) -> bool:
        """Validate if response meets DRIFT requirements."""
        
        required_keys = {"response", "score", "follow_up_queries"}
        
        if not isinstance(response, dict):
            return False
        
        if not all(key in response for key in required_keys):
            return False
        
        if not isinstance(response["follow_up_queries"], list):
            return False
        
        return True
