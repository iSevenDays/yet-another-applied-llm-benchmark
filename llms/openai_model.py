from io import BytesIO
from PIL import Image
import base64
import os  # Import the os module

from openai import OpenAI
import json

class OpenAIModel:
    def __init__(self, name, config_key='openai'):
        config = json.load(open("config.json"))
        api_key = config['llms'][config_key]['api_key'].strip()
        
        # Prioritize environment variable, fallback to config.json
        api_base = os.getenv('OPENAI_BASE_URL') or config['llms'][config_key]['api_base']
        
        # Configure client with connection limits to prevent leaks and hanging
        import httpx
        self.client = OpenAI(
            api_key=api_key, 
            base_url=api_base,
            timeout=900.0,  # Set explicit timeout (15 minutes)
            max_retries=2,   # Limit retries to prevent hanging
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
                timeout=httpx.Timeout(900.0, connect=30.0)
            )
        )
        self.name = name
        self.hparams = config['hparams']
        self.hparams.update(config['llms'][config_key].get('hparams') or {})

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
    
        for k,v in list(kwargs.items()):
            if v is None:
                del kwargs[k]
        if json:
            kwargs['response_format'] = { "type": "json_object" }
        if self.name.startswith("o1"):
            kwargs.pop('temperature', None)  # Remove temperature if present, ignore if not

        # Log non-message kwargs for debugging, exclude potentially large messages
        debug_kwargs = {k: v for k, v in kwargs.items() if k != 'messages'}
        import logging
        logging.debug(f"OpenAI request to {self.client.base_url} model={self.name}, stream={stream}, kwargs={debug_kwargs}")

        # Pass stream=True to the API call
        stream_response = self.client.chat.completions.create(
            model=self.name,
            stream=stream, # Pass the stream flag
            **kwargs
        )

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
