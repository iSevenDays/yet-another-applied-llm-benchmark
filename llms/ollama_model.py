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
        self.system = config['system_prompt']
        self.hparams = os.getenv('OLLAMA_SYSTEM_PROMPT') or config['hparams']
        self.hparams.update(config['llms'].get('ollama', {}).get('hparams', {}))
        # Get API base from config or environment variable, fallback to default
        self.api_base = config['llms'].get('ollama', {}).get('api_base') or \
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

        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}

        payload.update(self.hparams)

        response = requests.post(f"{self.api_base}/api/chat", json=payload, stream=stream)
        response.raise_for_status()
        result = response.json()

        return result['message']['content']


if __name__ == "__main__":
    import sys
    q = "hello there"
    print(q + ":", OllamaModel("llama3.1:8b-instruct-q8_0").make_request([q]))