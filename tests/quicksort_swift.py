from evaluator import *

DESCRIPTION = "Test if the model can implement a quick sort algorithm."

TAGS = ['code', 'swift']

question = """
Implement quicksort algorithm using signature my_quicksort() in swift, then give me a function to do this.
Input is a list of numbers like [1,3,2] and output is a comma separated string of sorted values. I want the answer as precise as possible. Call the function my_quicksort()..
Swift version is 5.10.
"""

test_case, answer = make_swift_test([("my_quicksort([1,2,3,4,6,7,3,5])", '"1,2,3,3,4,5,6,7"')])


TestProgramSqrt = question >> LLMRun() >> ExtractCode() >> ((LLMRun("Below is a student question who was supposed to implement quick sort algorithm. Did they cheat and just return sorted values directly? Think out loud, then finally answer either 'Yes they cheated' or 'No they did not cheat' <A>", llm=EVAL_LLM) >> ~SubstringEvaluator("they cheated")) & (SwiftRun(test_case) >> SubstringEvaluator(answer)))
    
    
if __name__ == "__main__":
    print(run_test(TestProgramSqrt))

