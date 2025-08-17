import json
import requests
from io import BytesIO
import base64
from PIL import Image
import os
class OllamaModel:
    def __init__(self, name):
        with open("config.json") as f:
            config = json.load(f)
        self.name = name
        
        # Fix system prompt handling
        ollama_config = config['llms'].get('ollama', {})
        self.system = ollama_config.get('system_prompt', "You are a helpful assistant.")
        
        # Fix hparams handling - start with global, then overlay ollama-specific
        self.hparams = config.get('hparams', {}).copy()
        self.hparams.update(ollama_config.get('hparams', {}))
        
        # Get API base from config or environment variable, fallback to default
        self.api_base = ollama_config.get('api_base') or \
                       os.getenv('OLLAMA_API_BASE') or \
                       "http://localhost:11434"  # Default Ollama API endpoint

    def make_request(self, conversation, add_image=None, max_tokens=None, json=False, stream=False):
        messages = [{"role": "system", "content": self.system}] + [{"role": "user" if i % 2 == 0 else "assistant", "content": content} for i, content in enumerate(conversation)]

        if add_image:
            buffered = BytesIO()
            add_image.convert("RGB").save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            messages[0]['images'] = [img_str]

        payload = {
            "model": self.name,
            "messages": messages,
            "stream": stream
        }

        if json:
            payload["format"] = "json"

        # Handle options properly for Ollama API
        options = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
            
        # Add stop sequences if defined in hparams
        if "stop" in self.hparams:
            options["stop"] = self.hparams["stop"]
        
        # Add other numeric options
        for key in ["temperature", "top_p", "top_k", "repeat_penalty"]:
            if key in self.hparams:
                options[key] = self.hparams[key]
        
        if options:
            payload["options"] = options

        response = requests.post(f"{self.api_base}/api/chat", json=payload, stream=stream)
        response.raise_for_status()
        result = response.json()

        return result['message']['content']


if __name__ == "__main__":
    import sys
    q = "hello there"
    print(q + ":", OllamaModel("llama3.1:8b-instruct-q8_0").make_request([q]))