from evaluator import *

DESCRIPTION = "Test if the model can solve a rather hard dynamic programming problem"

TAGS = ['code', 'c']

question = "A binary 8x8 matrix is 'unholy' if there are no isolated 1s. A 1 is isolated if there isn't another 1 in any of the 4 direct cardinal directions (up, down, left, right). Positions outside the matrix boundaries are considered to contain 0s. Write a C program that counts the total number of unholy 8x8 matrices."


step = """To solve this question:
- Use bit arithmetic and a uint64_t to represent the matrix.
- Use dynamic programming to count the number of unholy matrices.
- Consider processing the matrix row by row.

Write out a plan for the program, and then implement the plan in C."""

answer = "1121231537486377866"

TestUnholyMatrix = question >> LLMRun() >> ExtractCode(keep_main=True) >> CRun() >> SubstringEvaluator(answer)
TestUnholyMatrixStep = (question + step) >> LLMRun() >> ExtractCode(keep_main=True) >> CRun() >> SubstringEvaluator(answer)


if __name__ == "__main__":
    print(run_test(TestUnholyMatrix))



