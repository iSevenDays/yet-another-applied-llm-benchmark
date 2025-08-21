from evaluator import *

DESCRIPTION = "Test if the model can write a Python script that merges a list of file paths into 16 files of approximately equal size."

TAGS = ['code', 'python']

question_easy = """
I have a list of several thousand file paths in /tmp/files. Write a python script that merges these all together and splits the content into 16 files called /tmp/merged_%d for i in 0..15 where each is approximately the same size. To do this, sort files by size (largest first) and add files in order always to the smallest one. "/tmp/files" is a list of file paths, one per line.

IMPORTANT: Use efficient algorithms. Pre-calculate all file sizes once at the start. Use a heap/priority queue or similar data structure to track bin sizes - do NOT repeatedly sort or search through bins for each file. The script should complete in seconds, not minutes.
"""

question = """
I have a list of several thousand file paths in /tmp/files. Write a python script that merges the content int 16 files called /tmp/merged_%d for i in 0..15 where each is approximately the same size. "/tmp/files" is a list of file paths, one per line.

IMPORTANT: Use efficient algorithms. Pre-calculate all file sizes once at the start. Use a heap/priority queue to track the smallest bin - do NOT repeatedly sort or search through bins for each file. The script should complete in seconds, not minutes.
"""


def setup_fn():
    import random
    import os
    os.mkdir("/tmp/a")
    with open("/tmp/files", "w") as f:
        for i in range(1000):
            f.write("/tmp/a/file_%d\n" % i)
            with open("/tmp/a/file_%d" % i, "w") as f2:
                if i < 5:
                    f2.write("x" * random.randint(10000, 30000))
                else:
                    f2.write("x" * random.randint(1, 1000))
    
def test():
    import os
    import numpy as np
    
    # Pre-calculate input file sizes once (major optimization)
    try:
        input_sizes = []
        for i in range(1000):
            path = f"/tmp/a/file_{i}"
            if os.path.exists(path):
                input_sizes.append(os.path.getsize(path))
            else:
                return False  # Missing input file
        size_before = sum(input_sizes)
    except OSError:
        return False
    
    # Check output files with early termination
    sz = []
    for i in range(16):
        filepath = f"/tmp/merged_{i}"
        try:
            if not os.path.exists(filepath):
                return False
            
            # Protect against huge files that would cause memory issues
            file_size = os.path.getsize(filepath)
            if file_size > 10_000_000:  # 10MB limit
                return False
            sz.append(file_size)
        except OSError:
            return False

    size_after = sum(sz)

    # I ddn't say exactly how to merge, adding newlines is okay
    if abs(size_before-size_after) > 2000:
        return False

    # easy to hit with proper merging, hard otherwise
    # 10 trials suggest the range is usualy [10, 40]
    return max(sz) - min(sz) < 200
            
                    
    
TestMake16FilesEasy = Setup(setup_fn) >> question_easy >> LLMRun() >> ExtractCode(keep_main=True) >> Echo() >> PythonRun() >> PyEvaluator(test)

TestMake16Files = Setup(setup_fn) >> question >> LLMRun() >> ExtractCode(keep_main=True) >> Echo() >> PythonRun() >> PyEvaluator(test)


if __name__ == "__main__":
    print(run_test(TestMake16FilesEasy))
