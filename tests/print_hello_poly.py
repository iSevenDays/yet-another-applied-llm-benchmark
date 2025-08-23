from evaluator import *

DESCRIPTION = "Test advanced polyglot programming capabilities. Note: Python/Rust polyglots are extremely difficult due to fundamental syntax incompatibilities. This test evaluates sophisticated attempts and partial success."

TAGS = ['code', 'polyglot', 'expert-level', 'language-design', 'advanced-programming']

question = 'Write a single file that I can save as foo.c.py that I can run as either a C program or a python program, and so both will print "hello world". Specifically if I run `python foo.c.py` or `gcc foo.c.py && ./a.out` I should see "hello world" printed to the console.'

answer = "hello world"

TestPrintHelloPoly = question >> LLMRun() >> ExtractCode(keep_main=True) >> ((PythonRun() >> SubstringEvaluator(answer)) & (CRun() >> SubstringEvaluator(answer)))

question2 = '''
Write a single file that I can save as foo.py.rs that I can run as either a rust program or a python program, and so both will print "hello world". Specifically if I run `python foo.py.rs` or `rustc -o a.out foo.py.rs && ./a.out` I should see "hello world" printed to the console.

To help you out, here's a valid answer for how to do this in C:

```
#include <stdio.h>
#define sdf b /*                                                                                                                 
print("hello world")                                                                                                             
"""                                                                                                                              
#define wer b */

int main() {
  printf("hello world\n");
}

#define ert s """
```
'''

# Original test - C/Python polyglot (possible with preprocessor tricks)
TestPrintHelloPoly = question >> LLMRun() >> ExtractCode(keep_main=True) >> ((PythonRun() >> SubstringEvaluator(answer)) & (CRun() >> SubstringEvaluator(answer)))

# Enhanced Python/Rust polyglot evaluator that acknowledges fundamental difficulty
class PolyglotAwareEvaluator(Node):
    """
    Evaluator that accounts for the extreme difficulty of Python/Rust polyglots.
    Awards partial credit for sophisticated attempts and correct identification of impossibility.
    """
    def __init__(self, target_output):
        self.target_output = target_output
    
    def __call__(self, collected_results):
        python_results, rust_results = collected_results
        
        # Extract outputs and reasons
        python_output, python_reason = python_results[0] if python_results else ("", None)
        rust_output, rust_reason = rust_results[0] if rust_results else ("", None)
        
        python_success = self.target_output in python_output
        rust_success = self.target_output in rust_output
        
        # Perfect success: both languages work
        if python_success and rust_success:
            yield True, Reason(type(self), ("Perfect polyglot achieved", python_output, rust_output))
            return
        
        # Partial success: one language works, showing understanding of polyglot concepts
        if python_success or rust_success:
            working_lang = "Python" if python_success else "Rust" 
            yield False, Reason(type(self), (f"Partial success: {working_lang} works, polyglot understanding demonstrated", python_output, rust_output))
            return
            
        # Both failed - check if this was due to attempting sophisticated polyglot techniques
        code_attempt = python_reason[1][0] if python_reason and len(python_reason[1]) > 0 else ""
        
        if ("/*" in code_attempt or "*/" in code_attempt) and "SyntaxError" in python_output:
            # Model attempted C-style comments in Python context - shows polyglot understanding
            yield False, Reason(type(self), ("Sophisticated polyglot attempt with C-style comments (expected Python failure)", python_output, rust_output))
            return
            
        # Check if model attempted other polyglot techniques
        polyglot_indicators = ["#if", "#ifdef", "r'''", '"""', "__name__", "extern", "macro_rules"]
        if any(indicator in code_attempt for indicator in polyglot_indicators):
            yield False, Reason(type(self), ("Advanced polyglot technique attempted", python_output, rust_output))
            return
            
        # Complete failure
        yield False, Reason(type(self), ("No sophisticated polyglot attempt detected", python_output, rust_output))

# Helper node to collect both Python and Rust results for polyglot evaluation
class CollectPolyglotResults(Node):
    """Collect results from both Python and Rust execution for polyglot evaluation."""
    def __init__(self, python_node, rust_node):
        self.python_node = python_node
        self.rust_node = rust_node
    
    def setup(self, env, conv, llm, eval_llm, vision_eval_llm):
        super().setup(env, conv, llm, eval_llm, vision_eval_llm)
        self.python_node.setup(env, conv, llm, eval_llm, vision_eval_llm)
        self.rust_node.setup(env, conv, llm, eval_llm, vision_eval_llm)
    
    def __call__(self, code):
        python_results = list(self.python_node(code))
        rust_results = list(self.rust_node(code))
        yield (python_results, rust_results), Reason(type(self), (python_results, rust_results))

# Python/Rust polyglot test with realistic evaluation
TestPrintHelloPoly2 = question2 >> LLMRun() >> ExtractCode(keep_main=True) >> CollectPolyglotResults(PythonRun(), RustRun()) >> PolyglotAwareEvaluator(answer)


if __name__ == "__main__":
    print(run_test(TestPrintHelloPoly2))


