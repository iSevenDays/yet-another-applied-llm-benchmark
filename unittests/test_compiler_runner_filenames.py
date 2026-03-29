#!/usr/bin/env python3
import unittest
from unittest import mock

from evaluator import CRun


class TestCompilerRunnerFilenames(unittest.TestCase):
    @mock.patch("evaluator.invoke_docker")
    def test_crun_uses_default_source_name(self, invoke_docker):
        invoke_docker.return_value = "ok"
        node = CRun()
        node.setup(None, None, None, None, None)

        output, _ = next(node("int main(void) { return 0; }"))

        self.assertEqual(output, "ok")
        files = invoke_docker.call_args.args[1]
        self.assertIn("main.c", files)
        self.assertNotIn("foo.c.py", files)
        self.assertIn("gcc -o a.out main.c -lm", files["main.sh"].decode())

    @mock.patch("evaluator.invoke_docker")
    def test_crun_uses_configured_source_name_in_compile_command(self, invoke_docker):
        invoke_docker.return_value = "ok"
        node = CRun(source_name="foo.c.py", gccflags="-x c")
        node.setup(None, None, None, None, None)

        output, _ = next(node("int main(void) { return 0; }"))

        self.assertEqual(output, "ok")
        files = invoke_docker.call_args.args[1]
        self.assertIn("foo.c.py", files)
        self.assertNotIn("main.c", files)
        self.assertIn("gcc -o a.out foo.c.py -lm -x c", files["main.sh"].decode())


if __name__ == "__main__":
    unittest.main()
