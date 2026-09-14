import os
import time
import logging
from typing import List, Union, Optional, Type
from google import genai
from google.genai import types
from google.genai import errors
from pydantic import BaseModel
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# NOTE: gemini-2.0-flash was shut down June 1, 2026, and the entire
# gemini-2.5-* generation is now restricted from new API keys/projects
# ("no longer available to new users"), even though the models still
# appear in the list-models response. gemini-flash-latest is a
# Google-maintained alias that always resolves to whichever flash model is
# currently active for new accounts -- confirmed working via debug_step6.py.
# Using an alias here avoids re-chasing this exact problem the next time
# Google reshuffles model availability.
DEFAULT_MODEL = "gemini-flash-latest"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ZMETLLMClient")

def is_rate_limit_error(exception: Exception) -> bool:
    """Helper to check if exception is a 429 rate limit error."""
    if isinstance(exception, errors.APIError):
        # A 429 error code or containing "429" or "quota" in message
        if exception.code == 429 or "429" in str(exception) or "quota" in str(exception).lower():
            logger.warning(f"Rate limit hit: {exception}. Backing off...")
            return True
    return False

class ZMETLLMClient:
    """Client wrapper for Gemini Developer API utilizing official google-genai SDK."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        # Resolve API Key
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            
    def set_api_key(self, api_key: str):
        """Dynamically update the client's API Key."""
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)

    @retry(
        retry=retry_if_exception(is_rate_limit_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate_content(
        self,
        prompt: Union[str, List[any]],
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.2
    ) -> str:
        """
        Sends content to Gemini API. Automatically retries if 429 Rate Limit error is hit.
        
        Args:
            prompt: Text prompt or list of contents
            image: Optional PIL Image, or list of PIL Images, to pass inline for
                   multimodal queries. A list is required for triad/multi-image
                   comparisons (e.g. Step 6 laddering) since Gemini needs all
                   images present in the same call to compare them.
            response_schema: Optional Pydantic model for structured JSON output
            temperature: Prompt temperature parameter (defaults to 0.2 for analytical tasks)
        """
        if not self.client:
            raise ValueError("Gemini API key is not set. Please set the GEMINI_API_KEY environment variable or input it in the app settings.")

        # Prepare content list
        contents = []
        if image:
            if isinstance(image, list):
                contents.extend(image)
            else:
                contents.append(image)
        
        if isinstance(prompt, list):
            contents.extend(prompt)
        else:
            contents.append(prompt)

        image_count = sum(1 for c in contents if isinstance(c, Image.Image))
        logger.info(f"Calling {self.model_name} with {image_count} image(s) attached, schema={getattr(response_schema, '__name__', None)}")

        # Prepare config
        config_params = {"temperature": temperature}
        if response_schema:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_params)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            return response.text
        except errors.APIError as e:
            msg = str(e)
            if "404" in msg or "not found" in msg.lower() or "not available" in msg.lower():
                logger.error(
                    f"API Error during Gemini call (model='{self.model_name}'): {e} "
                    f"-- this usually means the model ID is deprecated/shut down. "
                    f"Check https://ai.google.dev/gemini-api/docs/deprecations"
                )
            else:
                logger.error(f"API Error during Gemini call: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during Gemini call: {e}")
            raise e
