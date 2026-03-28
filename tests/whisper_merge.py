from evaluator import *
from pathlib import Path

DESCRIPTION = "Test if the model can implement some string logic given a fuzzy description."

TAGS = ['code', 'python']

_PROMPTS = Path(__file__).parent / 'prompts'
_parse_fn = (_PROMPTS / 'whisper_merge_parse_fn.py').read_text()

question = f"""
Given a parse function like

{_parse_fn}

for data like

[00:00:14.240 --> 00:00:14.420]   the
[00:00:14.420 --> 00:00:14.860]   robust
[00:00:14.860 --> 00:00:15.020]  ness
[00:00:15.020 --> 00:00:15.140]   of
[00:00:15.140 --> 00:00:15.500]   neural
[00:00:15.500 --> 00:00:15.870]   networks
[00:00:15.870 --> 00:00:16.200]  .
[00:00:16.200 --> 00:00:16.410]   And
[00:00:16.410 --> 00:00:16.700]   this
[00:00:16.700 --> 00:00:16.840]   is
[00:00:16.840 --> 00:00:17.200]   joint


write a merge() function that (1) calls parse and (2) merges words into sentences that end in period. Start with the first word and end with the period. Return a list of strings.
"""

test_case, answer = make_python_test([("""merge("[00:00:14.240 --> 00:00:14.420]   the\\n[00:00:14.420 --> 00:00:14.860]   robust\\n[00:00:14.860 --> 00:00:15.020]  ness\\n[00:00:15.020 --> 00:00:15.140]   of\\n[00:00:15.140 --> 00:00:15.500]   neural\\n[00:00:15.500 --> 00:00:15.870]   networks\\n[00:00:15.870 --> 00:00:16.200]  .\\n[00:00:16.200 --> 00:00:16.410]   And\\n[00:00:16.410 --> 00:00:16.700]   this\\n[00:00:16.700 --> 00:00:16.840]   is\\n[00:00:16.840 --> 00:00:17.200]   joint\\n")""", "[' the robustness of neural networks.', ' And this is joint']")], header=_parse_fn)


TestWhisperMerge = question >> LLMRun() >> ExtractCode() >> PythonRun(test_case) >> SubstringEvaluator(answer)

if __name__ == "__main__":
    print(run_test(TestWhisperMerge))
