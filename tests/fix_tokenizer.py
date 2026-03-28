from evaluator import *
from pathlib import Path

DESCRIPTION = "Test if the model can identify and fix an issue with a tokenizer in a Python code snippet. Identifying the problem is in the regex, and fixing the regex, are both hard."

TAGS = ['code', 'fix', 'python']

_PROMPTS = Path(__file__).parent / 'prompts'
question = (_PROMPTS / 'fix_tokenizer_question.txt').read_text()

test_case, answer = make_python_test([("Tokenizer('sin(3+2*4)-cos(15)').tokens", "['sin', '(', '3', '+', '2', '*', '4', ')', '-', 'cos', '(', '15', ')']")])


TestSimpleFix = question >> LLMRun() >> ExtractCode() >> PythonRun(test_case) >> SubstringEvaluator(answer)


if __name__ == "__main__":
    print(run_test(TestSimpleFix))
