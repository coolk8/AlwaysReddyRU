from openai import OpenAI
import os
import config
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=config.LOGGING_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')

class OpenAIClient:
    """Client for interacting with OpenAI API."""
    def __init__(self, verbose=False):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.verbose = verbose

    def stream_completion(self, messages, model, temperature=0.7, max_tokens=2048, **kwargs):
        """Get completion from OpenAI API.

        Args:
            messages (list): List of messages.
            model (str): Model for completion.
            temperature (float): Temperature for sampling.
            max_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional keyword arguments.

        Yields:
            str: Text generated by the OpenAI API.
        """
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content != None:
                    yield content
        except Exception as e:
            logging.exception(f"An error occurred streaming completion from OpenAI: {e}")
            raise RuntimeError(f"An error occurred streaming completion from OpenAI: {e}")