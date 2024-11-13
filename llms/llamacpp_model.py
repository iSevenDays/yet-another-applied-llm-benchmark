import json
import requests

class LlamaCpp:
    def __init__(self):
        with open("config.json") as f:
            config = json.load(f)
        self.hparams = config['hparams']
        self.hparams.update(config['llms'].get('llama_cpp', {}).get('hparams', {}))
        self.api_base = "http://192.168.0.18:8080"  # LlamaCpp API endpoint

    def make_request(self, conversation, add_image=None, max_tokens=None, json_format=False):
        # For LlamaCpp we just use the last message as the prompt
        prompt = conversation[-1]

        data = {
            "prompt": prompt,
        }

        if max_tokens:
            data["n_predict"] = max_tokens

        data.update(self.hparams)

        response = requests.post(f"{self.api_base}/completion", 
                               headers={"Content-Type": "application/json"},
                               json=data)
        response.raise_for_status()
        result = response.json()

        return result['content']


if __name__ == "__main__":
    import sys
    q = "hello there"
    print(q + ":", LlamaCpp().make_request([q]))