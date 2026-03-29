#!/usr/bin/env python3
import json
import unittest

from evaluator import ExtractCode, ExtractJSON, SubstringEvaluator


class TestExtractionHeuristics(unittest.TestCase):
    def _setup_node(self, node):
        node.setup(None, None, lambda *args, **kwargs: "", lambda *args, **kwargs: "", None)
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


if __name__ == "__main__":
    unittest.main()
