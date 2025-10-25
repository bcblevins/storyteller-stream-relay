import openai
from typing import List, Dict, Generator, Optional, Union, AsyncGenerator
import json
import logging
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIService:
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.initialized = False

    def _ensure_initialized(self):
        """Ensure the client is initialized before use"""
        if not self.initialized:
            raise ValueError("OpenAI client not initialized. Call initialize_with_config() first.")

    def _log_error_with_context(self, method_name: str, error: Exception, bot_config: Optional[Dict] = None):
        """
        Enhanced error logging with method name and bot configuration context

        Args:
            method_name: Name of the method where the error occurred
            error: The exception that was raised
            bot_config: Bot configuration dictionary (optional)
        """
        # Sanitize bot config for logging (remove sensitive data)
        sanitized_config = {}
        if bot_config:
            sanitized_config = {
                'model': bot_config.get('model'),
                'base_url': bot_config.get('access_path'),
                'temperature': bot_config.get('temperature'),
                'max_tokens': bot_config.get('max_tokens'),
                'context_size': bot_config.get('context_size')
            }

        logger.error(
            f"OpenAI Service Error - Method: {method_name}, "
            f"Error: {type(error).__name__}: {str(error)}, "
            "Bot Config:\n"
            f"'model': {sanitized_config.get('model')},\n"
            f"'base_url': {sanitized_config.get('access_path')},\n"
            f"'temperature': {sanitized_config.get('temperature')},\n"
            f"'max_tokens': {sanitized_config.get('max_tokens')},\n"
            f"'context_size': {sanitized_config.get('context_size')}\n"
        )

    async def initialize_with_config(self, api_key: str, base_url: Optional[str] = None):
        """Initialize client with specific configuration"""
        try:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url if base_url else "https://api.deepseek.com/v1"
            )
            self.initialized = True
            logger.info(
                "OpenAI client initialized with config:\n"
                f"api_key: {api_key}"
                f"base_url: {base_url}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client with custom config: {e}")
            raise

    async def create_chat_completion_stream(
            self,
            messages: List[ChatCompletionMessageParam],
            model: str = "deepseek-chat",
            temperature: float = 0.7,
            max_tokens: int = 1000,
            bot_config: Optional[Dict] = None,
            **kwargs
    ) -> AsyncGenerator[Dict, None]:
        """
        Create a streaming chat completion using OpenAI API

        Args:
            messages: List of message objects with role and content
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            bot_config: Bot configuration dictionary for error context (optional)
            **kwargs: Additional parameters for the API call

        Yields:
            Dictionary with content chunks or error information
        """
        self._ensure_initialized()
        try:
            async with self.client.chat.completions.stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield {"content": chunk.choices[0].delta.content, "error": None}

        except RateLimitError as e:
            yield {"content": None, "error": f"Rate limit exceeded: {e}"}
        except AuthenticationError as e:
            yield {"content": None, "error": f"Authentication failed: {e}"}
        except (APIConnectionError, APIError) as e:
            yield {"content": None, "error": f"API error: {e}"}
        except Exception as e:
            yield {"content": None, "error": f"Unexpected error: {e}"}

    def create_chat_completion(
            self,
            messages: List[ChatCompletionMessageParam],
            model: str = "deepseek-chat",
            temperature: float = 0.7,
            max_tokens: int = 1000,
            bot_config: Optional[Dict] = None,
            **kwargs
    ) -> Dict:
        """
        Create a non-streaming chat completion

        Args:
            messages: List of message objects with role and content
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            bot_config: Bot configuration dictionary for error context (optional)
            **kwargs: Additional parameters for the API call

        Returns:
            Dictionary with the completion response
        """
        self._ensure_initialized()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                **kwargs
            )

            return {
                "content": response.choices[0].message.content,
                "error": None,
                "usage": response.usage.dict() if response.usage else None
            }

        except RateLimitError as e:
            self._log_error_with_context("create_chat_completion", e, bot_config)
            return {
                "content": None,
                "error": f"Rate limit exceeded. Please try again later. {e}"
            }
        except AuthenticationError as e:
            self._log_error_with_context("create_chat_completion", e, bot_config)
            return {
                "content": None,
                "error": f"Authentication failed. Please check your API key. {e}"
            }
        except APIConnectionError as e:
            self._log_error_with_context("create_chat_completion", e, bot_config)
            return {
                "content": None,
                "error": f"Connection error. Please check your network connection. {e}"
            }
        except APIError as e:
            self._log_error_with_context("create_chat_completion", e, bot_config)
            return {
                "content": None,
                "error": f"API error: {e}"
            }
        except Exception as e:
            self._log_error_with_context("create_chat_completion", e, bot_config)
            return {
                "content": None,
                "error": f"Unexpected error: {e}"
            }

    def validate_api_key(self, api_key: str, base_url: Optional[str] = None) -> bool:
        """
        Validate an API key by making a simple test request

        Args:
            api_key: API key to validate
            base_url: Base URL for the API (optional)

        Returns:
            True if valid, False otherwise
        """
        try:
            test_client = OpenAI(
                api_key=api_key,
                base_url=base_url if base_url else "https://api.deepseek.com/v1"
            )

            # Make a minimal test request
            test_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )

            return True
        except Exception as e:
            # Create a minimal bot config for error context
            bot_config = {
                'model': 'deepseek-chat',
                'access_path': base_url if base_url else "https://api.deepseek.com/v1"
            }
            self._log_error_with_context("validate_api_key", e, bot_config)
            return False


# Global instance for easy access
openai_service = OpenAIService()
