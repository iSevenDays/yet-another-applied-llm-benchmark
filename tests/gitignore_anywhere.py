from evaluator import *

TAGS = ['explain']

question = """
gitignore any file called .KEYFILE anywhere
"""

TestWhatIsAutoModel = question >> LLMRun() >> SubstringEvaluator("**/.KEYFILE")

if __name__ == "__main__":
    print(run_test(TestWhatIsAutoModel))