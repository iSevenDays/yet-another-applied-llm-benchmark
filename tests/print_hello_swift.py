from evaluator import *

DESCRIPTION = "Test if the model can generate a basic swift program that prints \"hello world\"."

TAGS = ['code', 'swift']

question = 'Write a swift program that prints the string "hello world" and tell me how it works in a sentence'

answer = "hello world"

TestPrintHelloSwift = question >> LLMRun() >> ExtractCode(keep_main=True) >> SwiftRun() >> SubstringEvaluator(answer)


if __name__ == "__main__":
    print(run_test(TestPrintHelloSwift))

