#!/usr/bin/env python3
import unittest

from evaluator import EVAL_LLM, LLMRun


class TestLLMRunJudgePrompting(unittest.TestCase):
    def _setup_node(self, node, llm=None, eval_llm=None):
        node.setup(
            None,
            None,
            llm or (lambda *args, **kwargs: ""),
            eval_llm or (lambda *args, **kwargs: ""),
            None,
        )
        return node

    def test_eval_llm_judge_prompt_appends_structured_student_answer_context(self):
        prompts = []

        def fake_eval_llm(prompt, json=False):
            prompts.append(prompt)
            return "The student passes"

        node = self._setup_node(
            LLMRun(
                "Below is a student's answer: <A>\nDoes the student's final answer say *x+2?",
                llm=EVAL_LLM,
            ),
            eval_llm=fake_eval_llm,
        )

        output, _ = next(node("**Final answer:** `*x + 2`"))

        self.assertEqual(output, "The student passes")
        sent_prompt = prompts[0]
        self.assertIn("Below is a student's answer: **Final answer:** `*x + 2`", sent_prompt)
        self.assertIn("Student answer (verbatim):", sent_prompt)
        self.assertIn("Extracted final answer candidate:", sent_prompt)
        self.assertIn("*x + 2", sent_prompt)

    def test_non_eval_llm_prompt_is_not_augmented(self):
        prompts = []

        def fake_llm(prompt, json=False):
            prompts.append(prompt)
            return "ok"

        node = self._setup_node(
            LLMRun("Question: <A>"),
            llm=fake_llm,
        )

        output, _ = next(node("demo"))

        self.assertEqual(output, "ok")
        self.assertEqual(prompts, ["Question: demo"])


if __name__ == "__main__":
    unittest.main()
