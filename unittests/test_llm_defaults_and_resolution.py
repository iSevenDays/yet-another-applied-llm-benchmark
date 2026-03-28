#!/usr/bin/env python3
import importlib
import unittest
from types import SimpleNamespace
from unittest import mock

import llm


class TestLLMResolution(unittest.TestCase):
    def setUp(self):
        self.original_llm = llm.llm
        self.original_eval_llm = llm.eval_llm
        self.original_vision_eval_llm = llm.vision_eval_llm

    def tearDown(self):
        llm.llm = self.original_llm
        llm.eval_llm = self.original_eval_llm
        llm.vision_eval_llm = self.original_vision_eval_llm

    def test_module_import_has_no_eager_default_models(self):
        reloaded = importlib.reload(llm)
        self.assertIsNone(reloaded.llm)
        self.assertIsNone(reloaded.eval_llm)
        self.assertIsNone(reloaded.vision_eval_llm)

    def test_openai_local_prefix_uses_openai_local_config(self):
        fake_model = SimpleNamespace(name="demo-model", hparams={})

        with mock.patch.object(llm, "OpenAIModel", return_value=fake_model) as openai_model:
            model = llm.LLM("openai_local_demo-model", use_cache=False)

        openai_model.assert_called_once_with("demo-model", config_key="openai_local")
        self.assertEqual(model.name, "demo-model")

    def test_missing_optional_provider_dependency_raises_runtime_error(self):
        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("anthropic")):
            with self.assertRaises(RuntimeError) as context:
                llm.LLM("claude-3-5-sonnet", use_cache=False)

        self.assertIn("optional dependency", str(context.exception))

    def test_ensure_default_models_initializes_only_missing_entries(self):
        created = []

        def fake_llm(name, use_cache=True, override_hparams=None):
            created.append(name)
            return SimpleNamespace(original_name=name, name=name)

        with mock.patch.object(llm, "LLM", side_effect=fake_llm):
            llm.llm = None
            llm.eval_llm = None
            llm.vision_eval_llm = None

            llm.ensure_default_models(test_model_name="openai_demo")

        self.assertEqual(created[:2], ["openai_demo", llm.DEFAULT_EVAL_MODEL])
        self.assertEqual(llm.llm.original_name, "openai_demo")
        self.assertEqual(llm.eval_llm.original_name, llm.DEFAULT_EVAL_MODEL)


if __name__ == "__main__":
    unittest.main()
