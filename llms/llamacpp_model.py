import json
import requests
from config_loader import load_config

class LlamaCpp:
    def __init__(self):
        config = load_config()
        self.hparams = config['hparams']
        self.hparams.update(config['llms'].get('llama_cpp', {}).get('hparams', {}))
        self.api_base = "http://192.168.0.18:8080"  # LlamaCpp API endpoint

    def make_request(self, conversation, add_image=None, max_tokens=None, json=False, stream=False):
        # For LlamaCpp we just use the last message as the prompt
        # The conversation history isn't directly used in the standard /completion endpoint
        prompt = conversation[-1]

        data = {
            "prompt": prompt,
            "stream": stream # Pass stream parameter
        }

        if max_tokens:
            data["n_predict"] = max_tokens

        data.update(self.hparams)

        if json:
            # Assuming the llama.cpp server supports a 'format' parameter like Ollama
            data["format"] = "json"
            # Remove the incorrect grammar line from previous attempt
            # data["grammar"] = LlamaGrammar.from_json_schema(schema=DEFAULT_SCHEMA)

        response = requests.post(f"{self.api_base}/completion", 
                               headers={"Content-Type": "application/json"},
                               json=data,
                               stream=stream) # Pass stream to requests
        response.raise_for_status()

        # Handling response needs to change based on stream
        if stream:
            # Placeholder: Need to implement proper stream handling for llama.cpp
            # This likely involves iterating over response.iter_lines() or similar
            # and parsing the SSE format.
            # For now, return an indicator that streaming happened but content needs parsing.
            print("Warning: Stream handling for llama.cpp not fully implemented.")
            # Read the first chunk to avoid blocking? Or return the iterator?
            # Let's return the raw response for the caller to handle for now.
            return response # Or maybe response.iter_lines()? Needs testing.
        else:
            result = response.json()
            return result['content']


if __name__ == "__main__":
    import sys
    q = "hello there"
    print(q + ":", LlamaCpp().make_request([q]))