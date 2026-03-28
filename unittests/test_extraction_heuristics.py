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


if __name__ == "__main__":
    unittest.main()
