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
    
    Args:
        text: Raw text response from AI
        fallback: Default dict to return if parsing fails
        
    Returns:
        Parsed JSON dictionary or fallback
    """
    if fallback is None:
        fallback = {}
    
    try:
        clean_text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}. Text: {text[:100]}...")
        return fallback
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
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

