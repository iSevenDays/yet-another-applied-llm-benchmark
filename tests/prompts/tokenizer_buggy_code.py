# Prompt snippet: buggy Tokenizer class with regex capturing group issue.
# Used by: fix_with_patch.py, fix_with_patch_guided.py (question f-strings)
# This file is read as text, not imported as a module.
import re

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
