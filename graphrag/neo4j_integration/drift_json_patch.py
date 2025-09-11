"""
Monkey patch for DRIFT search to use robust JSON parsing.
Apply this before running DRIFT search to fix JSON parsing issues.
"""

import json
import re
import ast
from typing import Any, Dict

# Store original json.loads function
_original_json_loads = json.loads

# Track if we're in a DRIFT context (set by the neo4j_query_api when DRIFT search is running)
_drift_context_active = False


def set_drift_context(active: bool):
    """Set whether we're currently in a DRIFT search context."""
    global _drift_context_active
    _drift_context_active = active


def _looks_like_drift_context(s: str) -> bool:
    """
    Determine if this JSON parsing failure might be from a DRIFT response.
    We use multiple heuristics to catch potential DRIFT responses.
    """
    # If DRIFT context is explicitly set, always use robust parsing
    if _drift_context_active:
        return True
    
    # Check for DRIFT-specific keywords
    if any(keyword in s for keyword in ['intermediate_answer', 'follow_up_queries', 'score']):
        return True
    
    # Check for empty or very short responses (common DRIFT failure cases)
    stripped = s.strip()
    if len(stripped) < 10 or stripped in ['{}', '{ }', '', 'null', 'None']:
        return True
    
    # Check for dictionary-like structure with potential DRIFT patterns
    if stripped.startswith('{') and stripped.endswith('}'):
        # Look for common DRIFT response patterns even if malformed
        if any(pattern in stripped for pattern in ['answer', 'query', 'queries', 'response']):
            return True
    
    # Check for single-quoted dictionaries (common LLM output format)
    if "'" in s and '{' in s and '}' in s:
        return True
    
    return False


def robust_json_loads(s: str, **kwargs) -> Any:
    """
    Robust JSON parser that handles LLM responses with single quotes or formatting issues.
    Falls back to original json.loads for standard use cases.
    """
    # First try the original json.loads
    try:
        return _original_json_loads(s, **kwargs)
    except json.JSONDecodeError as e:
        # If original parsing fails, check if we should use DRIFT robust parsing
        # We'll be more aggressive about catching potential DRIFT responses
        if isinstance(s, str) and _looks_like_drift_context(s):
            print(f"DEBUG: DRIFT JSON parsing failed, trying robust parsing: {s[:100]}...")
            return _drift_robust_json_parse(s)
        else:
            # Re-raise the original error for non-DRIFT contexts
            raise e


def _drift_robust_json_parse(response: str) -> Dict[str, Any]:
    """
    Parse JSON from DRIFT LLM response that might have single quotes or other formatting issues.
    """
    # Handle empty or near-empty responses first
    response_stripped = response.strip()
    if not response_stripped or response_stripped in ['{}', '{ }', 'null', 'None']:
        print(f"DEBUG: Empty DRIFT response, returning fallback")
        return {
            "intermediate_answer": "No information found for this query.",
            "score": 0,
            "follow_up_queries": ["Could you provide more specific information?"]
        }
    
    # First, try original parsing
    try:
        return _original_json_loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to fix single quotes to double quotes
    try:
        # Simple replacement approach for DRIFT-style responses
        fixed_response = response.replace("'", '"')
        return _original_json_loads(fixed_response)
    except json.JSONDecodeError:
        pass
    
    # Try to use ast.literal_eval for Python dictionary syntax
    try:
        # Find the dictionary-like structure
        dict_match = re.search(r'\{.*\}', response, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group(0)
            # Use ast.literal_eval to safely evaluate Python dictionary syntax
            parsed_dict = ast.literal_eval(dict_str)
            if isinstance(parsed_dict, dict):
                return parsed_dict
    except (ValueError, SyntaxError):
        pass
    
    # Try more aggressive cleanup of malformed JSON
    try:
        # Remove extra whitespace, fix common JSON issues
        cleaned = re.sub(r'\s+', ' ', response.strip())
        cleaned = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', cleaned)  # Fix unquoted keys
        cleaned = cleaned.replace("'", '"')  # Fix single quotes
        return _original_json_loads(cleaned)
    except (json.JSONDecodeError, re.error):
        pass
    
    # Manual extraction as last resort
    try:
        # Extract intermediate_answer
        answer_match = re.search(r"[\"']?intermediate_answer[\"']?\s*:\s*[\"'](.*?)[\"']", response, re.DOTALL)
        intermediate_answer = answer_match.group(1) if answer_match else "Unable to parse response from this query."
        
        # Extract score
        score_match = re.search(r"[\"']?score[\"']?\s*:\s*(\d+)", response)
        score = int(score_match.group(1)) if score_match else 0
        
        # Extract follow_up_queries
        queries_match = re.search(r"[\"']?follow_up_queries[\"']?\s*:\s*\[(.*?)\]", response, re.DOTALL)
        if queries_match:
            queries_str = queries_match.group(1)
            # Extract individual queries
            query_matches = re.findall(r"[\"'](.*?)[\"']", queries_str)
            follow_up_queries = query_matches[:5] if query_matches else ["Could you provide more information?"]
        else:
            follow_up_queries = ["Could you provide more information?", "What additional details are available?"]
        
        return {
            "intermediate_answer": intermediate_answer,
            "score": score,
            "follow_up_queries": follow_up_queries
        }
    except Exception as e:
        print(f"DEBUG: Manual extraction failed: {e}")
        pass
    
    # Final fallback
    print(f"Warning: Could not parse DRIFT JSON response: {response[:200]}...")
    return {
        "intermediate_answer": f"Unable to parse response: {response[:100]}..." if len(response) > 100 else response,
        "score": 0,
        "follow_up_queries": ["Could you provide more information?", "What other details are available?"]
    }


def apply_drift_json_patch():
    """
    Apply the monkey patch to replace json.loads with robust version.
    Call this before initializing DRIFT search.
    """
    json.loads = robust_json_loads
    print("Applied DRIFT JSON parsing patch")


def remove_drift_json_patch():
    """
    Remove the monkey patch and restore original json.loads.
    """
    json.loads = _original_json_loads
    print("Removed DRIFT JSON parsing patch")


if __name__ == "__main__":
    # Test the patch
    print("Testing DRIFT JSON patch...")
    
    # Apply patch
    apply_drift_json_patch()
    
    # Test with problematic DRIFT response
    test_response = """{'intermediate_answer': "# Test Answer\\n\\nThis is a test", 'score': 85, 'follow_up_queries': ['What are the tools?', 'How does it work?']}"""
    
    try:
        result = json.loads(test_response)
        print(f"Patched JSON parsing worked: {result}")
    except Exception as e:
        print(f"Patch failed: {e}")
    
    # Test with normal JSON (should still work)
    normal_json = '{"test": "value"}'
    try:
        result = json.loads(normal_json)
        print(f"Normal JSON still works: {result}")
    except Exception as e:
        print(f"Normal JSON failed: {e}")
    
    # Remove patch
    remove_drift_json_patch()
