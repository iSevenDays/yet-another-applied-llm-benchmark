#!/usr/bin/env python3
import json
import unittest

from evaluator import ExtractCode, ExtractJSON, SubstringEvaluator


class TestExtractionHeuristics(unittest.TestCase):
    def _setup_node(self, node):
        node.setup(None, None, lambda *args, **kwargs: "", lambda *args, **kwargs: "", None)
        return node

    def _setup_node_with_eval(self, node, eval_response):
        node.setup(None, None, lambda *args, **kwargs: "", lambda *args, **kwargs: eval_response, None)
        return node

    def test_extract_json_from_prose_wrapped_response_without_fallback(self):
        node = self._setup_node(ExtractJSON())

        response = """
Here is the JSON you asked for:

```json
{"name": "demo", "size": 7}
```

That should do it.
"""

        extracted, _ = next(node(response))
        self.assertEqual(json.loads(extracted), {"name": "demo", "size": 7})

    def test_extract_code_discards_leading_prose_for_assembly(self):
        node = self._setup_node(ExtractCode(lang="assembly"))

        response = """
Here is the assembly program:

start:
    SET R1 1
    STORE R1 0
    HCF

This writes 1 into memory cell 0.
"""

        extracted, _ = next(node(response))
        self.assertTrue(extracted.lstrip().startswith("start:"))
        self.assertIn("SET R1 1", extracted)
        self.assertNotIn("Here is the assembly program", extracted)

    def test_substring_evaluator_normalizes_unicode_punctuation(self):
        evaluator = SubstringEvaluator("DB-9")
        result, _ = next(evaluator("The connector is a DB‑9 serial port."))
        self.assertTrue(result)

    def test_extract_code_discards_markdown_table_before_function(self):
        node = self._setup_node(ExtractCode(lang="python"))

        response = """
Below is a stand-alone implementation.

| broken syntax | replacement |
| --- | --- |
| `(` | `[` |

def fix_json(text):
    return text
"""

        extracted, _ = next(node(response))
        self.assertTrue(extracted.lstrip().startswith("def fix_json"))
        self.assertNotIn("| broken syntax |", extracted)
        self.assertNotIn("Below is a stand-alone implementation.", extracted)

    def test_extract_code_stops_before_postscript_usage_notes(self):
        node = self._setup_node(ExtractCode(lang="python"))

        response = """
```python
#!/usr/bin/env python3
import sys

def main():
    print("ok")

if __name__ == "__main__":
    main()
```

1. Install Pillow if you haven't already:
pip install pillow
python code.py /tmp
"""

        extracted, _ = next(node(response))
        self.assertIn('print("ok")', extracted)
        self.assertNotIn("Install Pillow", extracted)
        self.assertNotIn("pip install pillow", extracted)

    def test_extract_code_preserves_python_indentation_from_fenced_block(self):
        node = self._setup_node(ExtractCode(lang="python"))

        response = """
```python
class Tokenizer:
    def __init__(self, input_str):
        self.input_str = input_str

    def get_next_token(self):
        if not self.input_str:
            return None
        return self.input_str[0]
```
"""

        extracted, _ = next(node(response))
        self.assertIn("    def __init__(self, input_str):", extracted)
        self.assertIn("        self.input_str = input_str", extracted)
        self.assertIn("    def get_next_token(self):", extracted)

    def test_extract_code_does_not_treat_markdown_bullets_as_code(self):
        node = self._setup_node(ExtractCode(lang="python"))

        response = """
* walks the input character-by-character
* keeps state outside strings

def fix_json(text):
    return text
"""

        extracted, _ = next(node(response))
        self.assertTrue(extracted.lstrip().startswith("def fix_json"))
        self.assertNotIn("* walks the input", extracted)

    def test_extract_code_falls_back_when_python_candidate_is_syntax_invalid(self):
        node = self._setup_node_with_eval(
            ExtractCode(lang="python"),
            "def one_hot(indexes, num_classes):\n    return indexes\n",
        )

        response = """
def one_hot(indexes, num_classes):
    \"\"\"
    Args:
"""

        extracted, _ = next(node(response))
        self.assertIn("def one_hot(indexes, num_classes):", extracted)
        self.assertIn("return indexes", extracted)
        self.assertNotIn('"""', extracted)

    def test_extract_code_strips_python_main_block_when_keep_main_is_false(self):
        node = self._setup_node(ExtractCode(lang="python"))

        response = """
```python
def evaluate(expr):
    return 1.0

if __name__ == "__main__":
    got = evaluate("1+2")
    print(f"{got:.12f}")
```
"""

        extracted, _ = next(node(response))
        self.assertIn("def evaluate(expr):", extracted)
        self.assertNotIn('__name__ == "__main__"', extracted)
        self.assertNotIn("print(f", extracted)

    def test_extract_code_rejects_main_only_python_candidate_and_uses_fallback(self):
        node = self._setup_node_with_eval(
            ExtractCode(lang="python"),
            "def move(x):\n    return {x}\n",
        )

        response = """
```python
if __name__ == "__main__":
    tests = [("ab", {"ba"})]
    for s, expected in tests:
        assert move(s) == expected
```
"""

        extracted, _ = next(node(response))
        self.assertIn("def move(x):", extracted)
        self.assertNotIn('__name__ == "__main__"', extracted)

    def test_extract_code_rejects_plain_output_text_and_uses_fallback(self):
        node = self._setup_node_with_eval(
            ExtractCode(lang="python", keep_main=True),
            "import traceback\n\ndef crashes():\n    x = 5\n    raise Exception()\n",
        )

        response = """
y: 6
x: 5
exc: This is a test exception
tb: <traceback object at 0x123>
"""

        extracted, _ = next(node(response))
        self.assertIn("import traceback", extracted)
        self.assertIn("def crashes():", extracted)
        self.assertNotIn("tb: <traceback object", extracted)


if __name__ == "__main__":
    unittest.main()
