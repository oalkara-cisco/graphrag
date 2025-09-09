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


def robust_json_loads(s: str, **kwargs) -> Any:
    """
    Robust JSON parser that handles LLM responses with single quotes or formatting issues.
    Falls back to original json.loads for standard use cases.
    """
    # If it looks like a DRIFT response (contains our expected keys), use robust parsing
    if isinstance(s, str) and ('intermediate_answer' in s or 'follow_up_queries' in s):
        return _drift_robust_json_parse(s)
    
    # Otherwise, use the original json.loads
    return _original_json_loads(s, **kwargs)


def _drift_robust_json_parse(response: str) -> Dict[str, Any]:
    """
    Parse JSON from DRIFT LLM response that might have single quotes or other formatting issues.
    """
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
    
    # Manual extraction as last resort
    try:
        # Extract intermediate_answer
        answer_match = re.search(r"'intermediate_answer':\s*[\"'](.*?)[\"']", response, re.DOTALL)
        intermediate_answer = answer_match.group(1) if answer_match else "Unable to parse response"
        
        # Extract score
        score_match = re.search(r"'score':\s*(\d+)", response)
        score = int(score_match.group(1)) if score_match else 0
        
        # Extract follow_up_queries
        queries_match = re.search(r"'follow_up_queries':\s*\[(.*?)\]", response, re.DOTALL)
        if queries_match:
            queries_str = queries_match.group(1)
            # Extract individual queries
            query_matches = re.findall(r"[\"'](.*?)[\"']", queries_str)
            follow_up_queries = query_matches[:5]  # Limit to 5 queries
        else:
            follow_up_queries = ["Could you provide more information?"]
        
        return {
            "intermediate_answer": intermediate_answer,
            "score": score,
            "follow_up_queries": follow_up_queries
        }
    except Exception:
        pass
    
    # Final fallback
    print(f"Warning: Could not parse DRIFT JSON response: {response[:200]}...")
    return {
        "intermediate_answer": "Unable to parse response",
        "score": 0,
        "follow_up_queries": ["Could you provide more information?"]
    }


def apply_drift_json_patch():
    """
    Apply the monkey patch to replace json.loads with robust version.
    Call this before initializing DRIFT search.
    """
    json.loads = robust_json_loads
    print("✅ Applied DRIFT JSON parsing patch")


def remove_drift_json_patch():
    """
    Remove the monkey patch and restore original json.loads.
    """
    json.loads = _original_json_loads
    print("✅ Removed DRIFT JSON parsing patch")


if __name__ == "__main__":
    # Test the patch
    print("Testing DRIFT JSON patch...")
    
    # Apply patch
    apply_drift_json_patch()
    
    # Test with problematic DRIFT response
    test_response = """{'intermediate_answer': "# Test Answer\\n\\nThis is a test", 'score': 85, 'follow_up_queries': ['What are the tools?', 'How does it work?']}"""
    
    try:
        result = json.loads(test_response)
        print(f"✅ Patched JSON parsing worked: {result}")
    except Exception as e:
        print(f"❌ Patch failed: {e}")
    
    # Test with normal JSON (should still work)
    normal_json = '{"test": "value"}'
    try:
        result = json.loads(normal_json)
        print(f"✅ Normal JSON still works: {result}")
    except Exception as e:
        print(f"❌ Normal JSON failed: {e}")
    
    # Remove patch
    remove_drift_json_patch()
