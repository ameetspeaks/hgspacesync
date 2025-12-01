"""
Shared utilities for the Astrology App backend.
Provides common functions for error handling, JSON parsing, and response formatting.
"""
import json
import logging
from typing import Any, Dict, Optional
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def parse_ai_json(text: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Safely parses AI-generated JSON response.
    Handles markdown code blocks, control characters, and extracts JSON from mixed content.
    
    Args:
        text: Raw text response from AI
        fallback: Default dict to return if parsing fails
        
    Returns:
        Parsed JSON dictionary or fallback
    """
    if fallback is None:
        fallback = {}
    
    if not text or not isinstance(text, str):
        return fallback
    
    import re
    
    try:
        # Step 1: Remove markdown code blocks
        clean_text = text.replace("```json", "").replace("```", "").strip()
        
        # Step 2: Try to extract JSON object if text contains extra content
        # Find JSON object boundaries
        json_start = clean_text.find('{')
        json_end = clean_text.rfind('}')
        
        if json_start >= 0 and json_end > json_start:
            # Extract just the JSON part
            clean_text = clean_text[json_start:json_end + 1]
        
        # Step 3: Remove invalid control characters
        # Control characters that are invalid in JSON (except when escaped in strings)
        # We'll remove unescaped control characters, but keep valid whitespace
        # This regex removes control characters except \n, \r, \t (which can be valid)
        clean_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', clean_text)
        
        # Step 4: Try to parse
        return json.loads(clean_text)
        
    except json.JSONDecodeError as e:
        # If parsing fails, try more aggressive cleaning
        try:
            # Try to find JSON object boundaries again with original text
            json_start = text.find('{')
            json_end = text.rfind('}')
            
            if json_start >= 0 and json_end > json_start:
                json_text = text[json_start:json_end + 1]
                
                # Remove markdown
                json_text = json_text.replace("```json", "").replace("```", "").strip()
                
                # Remove all control characters (more aggressive)
                json_text = re.sub(r'[\x00-\x1F\x7F]', '', json_text)
                
                # Try parsing again
                return json.loads(json_text)
            
            logger.warning(f"JSON parse error: {e}. Text sample: {text[:200]}...")
        except Exception as e2:
            logger.warning(f"JSON parse error (fallback also failed): {e2}. Text sample: {text[:200]}...")
        
        return fallback
        
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}", exc_info=True)
        return fallback


def safe_db_operation(operation, error_message: str = "Database operation failed"):
    """
    Safely executes a database operation with error handling.
    
    Args:
        operation: Callable that performs the DB operation
        error_message: Custom error message
        
    Returns:
        Result of operation or None if it fails
    """
    try:
        return operation()
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        return None


def format_error_response(error: Exception, user_message: str = None) -> Dict[str, Any]:
    """
    Formats a consistent error response.
    
    Args:
        error: The exception that occurred
        user_message: User-friendly error message
        
    Returns:
        Formatted error response dictionary
    """
    if user_message is None:
        user_message = "An error occurred. Please try again."
    
    logger.error(f"Error: {type(error).__name__}: {str(error)}")
    
    return {
        "status": "error",
        "message": user_message,
        "error_type": type(error).__name__
    }


def validate_chart_result(result: Any) -> bool:
    """
    Validates that a chart calculation result is valid.
    
    Args:
        result: Result from calculate_birth_chart
        
    Returns:
        True if valid, False otherwise
    """
    if isinstance(result, str) and "error" in result.lower():
        return False
    if isinstance(result, dict) and "error" in result:
        return False
    if not isinstance(result, dict):
        return False
    return True


def get_chart_summary(result: Any) -> str:
    """
    Extracts AI summary from chart result.
    
    Args:
        result: Result from calculate_birth_chart
        
    Returns:
        Summary string or empty string
    """
    if isinstance(result, dict):
        return result.get("ai_summary", str(result))
    return str(result) if result else ""


def create_success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """
    Creates a consistent success response.
    
    Args:
        data: Response data
        message: Success message
        
    Returns:
        Formatted success response
    """
    response = {"status": "success", "message": message}
    if data is not None:
        response["data"] = data
    return response

