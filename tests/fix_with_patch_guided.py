from evaluator import *

DESCRIPTION = "Guided assessment of regex debugging and patch generation with clear problem statement"

TAGS = ['regex', 'patch', 'debugging', 'guided']

code = """import re

class Tokenizer:
    def __init__(self, input_str):
        '''
        input_str: a string of digits and operators
        '''
        self.position = 0
        self.tokens = re.findall(r'\d+|(\+|\-|\*|/|sin|cos|\(|\))', input_str)

    def get_next_token(self):
        if self.position >= len(self.tokens):
            return None
        token = self.tokens[self.position]
        self.position += 1
        return token

    def peek_next_token(self):
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]
"""

question_guided = f"""
This tokenizer has a bug in its regex pattern. When parsing "cos(1)+sin(23)*5", 
it currently produces: ['cos', '(', '', ')', '+', 'sin', '(', '', ')', '*', '']
but should produce: ['cos', '(', '1', ')', '+', 'sin', '(', '23', ')', '*', '5']

The issue is with the regex capturing groups in re.findall(). When you use capturing groups `()` 
in a regex with multiple alternatives, re.findall() returns tuples, and non-matching groups 
return empty strings.

Write a .patch file that fixes this regex bug by removing the unnecessary capturing group.

The patch should be in unified diff format:
```
--- filename.py
+++ filename.py  
@@ -line,count +line,count @@
 context line
-old line  
+new line
```

Here is the code to fix:
```python
{code}
```
"""

def setup_fn():
    code = """import re

class Tokenizer:
    def __init__(self, input_str):
        '''
        input_str: a string of digits and operators
        '''
        self.position = 0
        self.tokens = re.findall(r'\d+|(\+|\-|\*|/|sin|cos|\(|\))', input_str)

    def get_next_token(self):
        if self.position >= len(self.tokens):
            return None
        token = self.tokens[self.position]
        self.position += 1
        return token

    def peek_next_token(self):
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]"""
    import time
    open('run_tokenizer.py', 'w').write(code)

    time.sleep(.5)


def check_guided():
    import os
    import time
    result = os.system("patch < fix.patch")
    if result != 0:
        return False, "Patch application failed - check patch format"
        
    time.sleep(.5)
    
    try:
        import run_tokenizer
        actual = run_tokenizer.Tokenizer("cos(1)+sin(23)*5").tokens
        expected = ['cos', '(', '1', ')', '+', 'sin', '(', '23', ')', '*', '5']
        if actual == expected:
            return True, ""
        else:
            return False, f"Expected {expected}, got {actual}"
    except Exception as e:
        return False, f"Import or execution failed: {e}"

TestFixPatchGuided = Setup(setup_fn) >> question_guided >> LLMRun() >> ExtractCode(manual="Take the below answer to my question and return just the complete .patch in a single file so I can copy and paste it into an editor and apply it with the `patch` tool as-is. Include nothing else other than the patch. Here is the code: <A>") >> MakeFile("fix.patch") >> PyEvaluator(check_guided)


if __name__ == "__main__":
    print(run_test(TestFixPatchGuided))