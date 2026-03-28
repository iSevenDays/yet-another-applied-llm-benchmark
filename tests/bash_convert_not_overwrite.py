from evaluator import *
from pathlib import Path

DESCRIPTION = "Test if a model can run an incremental update of a bash command without overwriting files that already exist"

TAGS = ['bash']

_PROMPTS = Path(__file__).parent / 'prompts'
question = (_PROMPTS / 'bash_convert_not_overwrite_question.txt').read_text()


def setup_fn():
    open("process.py", "w").write("import sys\nopen(sys.argv[2], 'w').write(open(sys.argv[1]).read())\n")
    open("a.pdf", "w").write("hello1")
    open("b.pdf", "w").write("hello2")
    open("c.pdf", "w").write("hello3")
    import os
    os.mkdir("txt")
    open("txt/a.txt", "w").write("done1")
    open("txt/c.txt", "w").write("done3")


def check_ok():
    import os
    return open("txt/a.txt").read() == "done1" and open("txt/b.txt").read() == "hello2" and open("txt/c.txt").read() == "done3" and len(os.listdir("txt")) == 3



TestBashIncrementalUpdate = Setup(setup_fn) >> question >> LLMRun() >> ExtractCode(keep_main=True) >> BashRun() >> PyEvaluator(check_ok)


if __name__ == "__main__":
    print(run_test(TestBashIncrementalUpdate))


