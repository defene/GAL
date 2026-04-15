import json
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, Optional
import tiktoken
import csv
import os
import math
from config import LLM_CONFIG
from utils.retry_utils import retry_with_backoff

load_dotenv()

total_tokens_used = 0
total_calls = 0

def get_token_stats():
    return total_calls, total_tokens_used

def count_tokens(prompt: str, model: str = None):
    """
    Calculate token count for a prompt.
    
    Args:
        prompt: Input text to count tokens for
        model: Model name, defaults to config default model
        
    Returns:
        Estimated token count
    """
    if model is None:
        model = LLM_CONFIG["default_model"]
    
    # Use simplified estimation for Gemini models
    if model and model.startswith("gemini-"):
        return math.ceil(len(prompt) / 4)  # Simplified estimate: ~1 token per 4 characters
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(prompt))

def save_token_log(timestamp: str, total_calls: int, total_tokens: int, cost: float, filename: str = "token_log.csv"):
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "total_calls", "total_tokens", "estimated_cost"])
        writer.writerow([timestamp, total_calls, total_tokens, f"{cost:.4f}"])

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Google Generative AI initialization
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


def call_model(model: str = None, system: str = "", user: str = "", temperature: float = None, max_retries: int = None):
    """
    Call LLM model with system and user messages, including retry mechanism.
    
    Supports both OpenAI and Google Gemini models with automatic model detection
    and configuration-based parameter handling.
    
    Args:
        model: Model name, defaults to config default model
        system: System prompt message
        user: User message
        temperature: Temperature parameter, defaults to config default value
        max_retries: Maximum retry attempts, defaults to config default value
    
    Returns:
        LLM response content, or None if all retries failed
    """
    # Use default values from config
    if model is None:
        model = LLM_CONFIG["default_model"]
    if max_retries is None:
        max_retries = LLM_CONFIG["default_max_retries"]
    
    # Get model-specific temperature configuration
    model_config = LLM_CONFIG["supported_models"].get(model, {})
    if temperature is None:
        # Prefer model-specific default temperature, otherwise use global default
        temperature = model_config.get("default_temperature", LLM_CONFIG["default_temperature"])
    else:
        # Check if temperature is supported by model
        supported_temps = model_config.get("supported_temperature", None)
        if supported_temps is not None and temperature not in supported_temps:
            print(f"Warning: Model {model} does not support temperature={temperature}. Using model default: {model_config.get('default_temperature', 1.0)}")
            temperature = model_config.get("default_temperature", 1.0)
    
    # Check if Gemini model
    if model and model.startswith("gemini-"):
        return retry_with_backoff(
            _call_gemini_model_core,
            max_retries,
            model,
            model, system, user, temperature
        )
    
    # OpenAI model path
    return retry_with_backoff(
        _call_openai_model_core,
        max_retries,
        model,
        model, system, user, temperature
    )


def _call_openai_model_core(model: str, system: str, user: str, temperature: float):
    """
    Core OpenAI model call logic without retry handling.
    
    Args:
        model: Model name
        system: System prompt
        user: User message
        temperature: Temperature parameter
    
    Returns:
        Model response content
    
    Raises:
        Exception: If API call fails
    """
    global total_tokens_used, total_calls
    
    if not client:
        raise RuntimeError("OpenAI client not initialized. Please check OPENAI_API_KEY.")
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if user:
        messages.append({"role": "user", "content": user})
    
    if not messages:
        raise ValueError("No messages provided")
    
    input_tokens = sum(count_tokens(msg["content"], model) for msg in messages)
    
    # Get model-specific configuration
    model_config = LLM_CONFIG["supported_models"].get(model, {})
    max_tokens = model_config.get("max_tokens", LLM_CONFIG["default_max_tokens"])
    token_param = model_config.get("token_param", "max_tokens")
    
    # Dynamically build parameters based on config token_param
    api_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        token_param: max_tokens
    }
    
    response = client.chat.completions.create(**api_params)
    
    output_tokens = response.usage.completion_tokens
    total = input_tokens + output_tokens

    total_calls += 1
    total_tokens_used += total

    print(f"[LLM] Model: {model}, Input: {input_tokens}, Output: {output_tokens}, Total so far: {total_tokens_used} tokens in {total_calls} calls")

    return response.choices[0].message.content

def _call_gemini_model_core(model: str, system: str, user: str, temperature: float):
    """
    Core Gemini model call logic without retry handling.
    
    Args:
        model: Gemini model name
        system: System instruction
        user: User message
        temperature: Temperature parameter
        
    Returns:
        Model response text
    
    Raises:
        Exception: If API call fails
    """
    global total_tokens_used, total_calls
    
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google Generative AI client not initialized. Please check GOOGLE_API_KEY.")
    
    if not user:
        raise ValueError("No user message provided")
    
    # Calculate input tokens (simplified estimation)
    input_tokens = count_tokens(system, model) + count_tokens(user, model)
    
    # Get model-specific configuration
    model_config = LLM_CONFIG["supported_models"].get(model, {})
    max_tokens = model_config.get("max_tokens", LLM_CONFIG["default_max_tokens"])
    
    # Create generation configuration
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens
    }
    
    # Create model instance
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system if system else None
    )
    
    # Generate content
    response = gemini_model.generate_content(
        user,
        generation_config=generation_config
    )
    
    # Get response text
    response_text = response.text
    
    # Calculate output tokens (simplified estimation)
    output_tokens = count_tokens(response_text, model)
    total = input_tokens + output_tokens
    
    total_calls += 1
    total_tokens_used += total
    
    print(f"[LLM] Model: {model}, Input: {input_tokens}, Output: {output_tokens}, Total so far: {total_tokens_used} tokens in {total_calls} calls")
    
    return response_text

def safe_extract_number(value, default=0, as_int=False):
    """
    Safely extract numeric value from various data types.
    
    Handles strings with formatting (commas, currency symbols), None values,
    and performs type conversions with fallback to default value.
    
    Args:
        value: Value to convert (may be string, number, None, etc.)
        default: Default value to return on conversion failure
        as_int: Whether to convert to integer (False converts to float)
    
    Returns:
        Converted numeric value or default value on failure
    """
    if value is None:
        return default
    
    # If already target numeric type, return directly
    if as_int and isinstance(value, int):
        return value
    elif not as_int and isinstance(value, (int, float)):
        return float(value)
    
    # Try conversion from string
    if isinstance(value, str):
        # Remove common non-numeric characters (commas, dollar signs, etc.)
        cleaned_value = value.strip().replace(',', '').replace('$', '').replace('_', '')
        
        # Handle empty string
        if not cleaned_value:
            return default
        
        try:
            if as_int:
                return int(float(cleaned_value))  # Convert to float first, handles "123.0" cases
            else:
                return float(cleaned_value)
        except (ValueError, TypeError):
            return default
    
    # Try direct conversion for other types
    try:
        if as_int:
            return int(float(value))
        else:
            return float(value)
    except (ValueError, TypeError):
        return default


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division, avoiding division by zero errors.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return when denominator is zero
        
    Returns:
        Division result, or default if denominator is zero
    """
    if denominator == 0 or denominator is None:
        return default
    try:
        return float(numerator) / float(denominator)
    except (ValueError, TypeError, ZeroDivisionError):
        return default