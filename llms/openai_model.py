from io import BytesIO
from PIL import Image
import base64
import os  # Import the os module

from openai import OpenAI
import json
from config_loader import load_config

OPENAI_CLIENT_TIMEOUT_SECONDS = 3600

class OpenAIModel:
    def __init__(self, name, config_key='openai'):
        config = load_config()
        provider_config = config.get('llms', {}).get(config_key, {})
        api_key = (provider_config.get('api_key') or os.getenv('OPENAI_API_KEY') or 'none').strip()
        
        # Prioritize environment variables, fallback to config.yaml
        # Support both OPENAI_API_BASE (vLLM/common convention) and OPENAI_BASE_URL (OpenAI SDK convention)
        # For eval endpoint, use OPENAI_EVAL_API_BASE / OPENAI_EVAL_BASE_URL
        if config_key == 'openai_eval':
            api_base = (os.getenv('OPENAI_EVAL_API_BASE') or
                        os.getenv('OPENAI_EVAL_BASE_URL') or
                        provider_config.get('api_base'))
        else:
            api_base = (os.getenv('OPENAI_API_BASE') or
                        os.getenv('OPENAI_BASE_URL') or
                        provider_config.get('api_base'))
        
        # Configure client with connection limits to prevent leaks and hanging
        import httpx
        self.client = OpenAI(
            api_key=api_key, 
            base_url=api_base,
            timeout=float(OPENAI_CLIENT_TIMEOUT_SECONDS),
            max_retries=2,   # Limit retries to prevent hanging
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
                timeout=httpx.Timeout(float(OPENAI_CLIENT_TIMEOUT_SECONDS), connect=30.0)
            )
        )
        self.name = name
        self.hparams = config.get('hparams', {}).copy()
        self.hparams.update(provider_config.get('hparams') or {})

    def make_request(self, conversation, add_image=None, max_tokens=None, json=False, stream=False):
        conversation = [{"role": "user" if i%2 == 0 else "assistant", "content": content} for i,content in enumerate(conversation)]
    
        if add_image:
            buffered = BytesIO()
            add_image.convert("RGB").save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_str = f"data:image/jpeg;base64,{img_str}"
            
            conversation[0]['content'] = [{"type": "text", "text": conversation[0]['content']},
                                          {
                                            "type": "image_url",
                                            "image_url": {
                                              "url": img_str
                                            }
                                          }
                                          ]
        kwargs = {
            "messages": conversation,
            "max_tokens": max_tokens,
        }
        kwargs.update(self.hparams)
    
        # Extract extra_body parameter before cleaning - OpenAI client needs it separate
        extra_body = kwargs.pop('extra_body', None)
    
        for k,v in list(kwargs.items()):
            if v is None:
                del kwargs[k]
        if json:
            kwargs['response_format'] = { "type": "json_object" }
        if self.name.startswith("o1"):
            kwargs.pop('temperature', None)  # Remove temperature if present, ignore if not

        # Log non-message kwargs for debugging, exclude potentially large messages
        debug_kwargs = {k: v for k, v in kwargs.items() if k != 'messages'}
        if extra_body:
            debug_kwargs['extra_body'] = extra_body
        import logging
        logging.debug(f"OpenAI request to {self.client.base_url} model={self.name}, stream={stream}, kwargs={debug_kwargs}")

        # Prepare API call parameters
        api_kwargs = {
            "model": self.name,
            "stream": stream,
            **kwargs
        }
        
        # Add extra_body if present (for thinking models and custom API servers)
        if extra_body is not None:
            api_kwargs["extra_body"] = extra_body

        # Pass stream=True to the API call
        stream_response = self.client.chat.completions.create(**api_kwargs)

        # Return the stream object if stream=True, otherwise process and return string
        if stream:
            return stream_response
        else:
            # Fallback for non-streaming (likely won't be used with new llm.py logic)
            return stream_response.choices[0].message.content

if __name__ == "__main__":
    import sys
    #q = sys.stdin.read().strip()
    q = "hello there"
    print(q+":", OpenAIModel("o1-mini").make_request([q]))
