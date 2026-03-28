## Copyright (C) 2024, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import subprocess
import pickle
import random
import json
import os
import time
import io
import docker
import inspect
import re
import unicodedata

import numpy as np

from PIL import Image

import docker_controller
from docker_controller import invoke_docker, DockerJob

def strip_thinking_tokens(text):
    """Remove thinking tokens from text output - used for both main and eval LLM responses.
    
    Handles multiple formats:
    - Simple tags: <think>...</think>, <seed:think>...</seed:think>
    - OpenAI Harmony format: GPT-OSS analysis/commentary channels
    """
    result = text
    
    # Handle simple thinking tags
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    result = re.sub(r'<seed:think>.*?</seed:think>', '', result, flags=re.DOTALL)
    
    # Handle GPT-OSS Harmony format - remove analysis and commentary channels
    # Pattern: <|start|>assistant<|channel|>analysis<|message|>...thinking...<|end|>
    result = re.sub(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>', '', result, flags=re.DOTALL)
    result = re.sub(r'<\|start\|>assistant<\|channel\|>commentary<\|message\|>.*?<\|end\|>', '', result, flags=re.DOTALL)
    
    # Alternative pattern without assistant role prefix
    result = re.sub(r'<\|channel\|>analysis<\|message\|>.*?<\|end\|>', '', result, flags=re.DOTALL)
    result = re.sub(r'<\|channel\|>commentary<\|message\|>.*?<\|end\|>', '', result, flags=re.DOTALL)
    
    return result

# Model role constants
LLM = "llm"                         # The LLM under evaluation
EVAL_LLM = "eval_llm"               # A good LLM that can act as a judge
VISION_EVAL_LLM = "vision_eval_llm" # A good judge for vision tasks
PYTHON_ENV = "python3"              # The version of python to use

# Code extraction constants
CODE_BLOCK_DELIMITER = "```"
CODE_EXTRACTION_TIMEOUT = 30
MAX_ITERATIONS = 100
SCREENSHOT_DELAY_SECONDS = 2

UNICODE_TEXT_TRANSLATION = str.maketrans({
    "\u00a0": " ",
    "\u2009": " ",
    "\u202f": " ",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
})

ASSEMBLY_OPS = {
    "SET", "ADD", "SUB", "MUL", "DIV", "MOD", "EQ", "NEQ", "LT", "LTE",
    "GT", "GTE", "INC", "DEC", "JMP", "JT", "JF", "LOAD", "STORE", "HCF",
}


def normalize_text_for_comparison(text):
    if not isinstance(text, str):
        return text
    text = strip_thinking_tokens(text)
    text = unicodedata.normalize("NFKC", text).translate(UNICODE_TEXT_TRANSLATION)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text)


def _fenced_code_blocks(text):
    return [match.group(1).strip("\n") for match in re.finditer(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", text, flags=re.DOTALL)]


def _extract_balanced_json_strings(text):
    decoder = json.JSONDecoder()
    candidates = []
    normalized = normalize_text_for_comparison(text)
    for match in re.finditer(r"[\[{]", normalized):
        snippet = normalized[match.start():].lstrip()
        try:
            parsed, end = decoder.raw_decode(snippet)
        except json.JSONDecodeError:
            continue
        candidates.append(json.dumps(parsed))
        candidates.append(snippet[:end])
    return candidates


def _looks_like_code_line(line):
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith(("#", "//", "/*", "*", "*/", "--")):
        return True
    if re.match(r"^[A-Za-z_][\w]*:\s*(?:;.*)?$", stripped):
        return True
    if re.match(r"^(?:from|import|def|class|return|if|for|while|try|except|fn|let|const|var|public|private|template|SELECT|INSERT|UPDATE|DELETE|CREATE|BEGIN|END|#!/)\b", stripped):
        return True
    first_token = stripped.replace(",", " ").split()[0]
    if first_token.upper() in ASSEMBLY_OPS:
        return True
    if any(token in stripped for token in ("{", "}", "();", " = ", "->", "::", "#include", "<html", "</", "println!", "printf(")):
        return True
    return False


def _looks_like_prose_line(line):
    stripped = line.strip()
    if not stripped:
        return False
    if _looks_like_code_line(stripped):
        return False
    words = re.findall(r"[A-Za-z]{3,}", stripped)
    return len(words) >= 4 and stripped.endswith((".", ":"))


def _extract_code_candidates(text):
    normalized = normalize_text_for_comparison(text)
    fenced_blocks = _fenced_code_blocks(normalized)
    candidates = []
    candidates.extend(block for block in fenced_blocks if block.strip())

    lines = normalized.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if _looks_like_code_line(line):
            start_idx = idx
            break

    if start_idx is not None:
        candidate_lines = []
        prose_streak = 0
        for line in lines[start_idx:]:
            if not line.strip():
                candidate_lines.append(line)
                prose_streak = 0
                continue
            if _looks_like_prose_line(line):
                prose_streak += 1
                if prose_streak >= 2:
                    break
                continue
            prose_streak = 0
            candidate_lines.append(line)

        candidate = "\n".join(candidate_lines).strip("\n")
        if candidate:
            candidates.append(candidate)

    raw = normalized.strip()
    if raw and (fenced_blocks or start_idx is not None):
        candidates.append(raw)

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped

class Env:
    """
    An environment that holds the local variables for each test case.
    """

    # The docker object we're running the test in
    docker = None

    # (Optionally, if in unsafe mode, the fake docker object)
    fake_docker_id = None

    # The docker container we're running the tests in
    container = None

    # A DockerJob object, if the test case requires it.
    # These objects allow the test to interact with stdin/out
    # of a process running in the docker container and must be
    # persistant across multiple classes in the test case.
    docker_job = None
    
class Reason:
    """
    A class to keep track of the solution path of a test.
    """
    def __init__(self, node, children):
        self.node = node
        self.children = children

    def __repr__(self):
        return repr((self.node, self.children))
    
    def describe_failure(self, max_length=400):
        """
        Describe this failure in a human-readable way.
        Following SPARC principles: Simple, focused error description.
        
        Args:
            max_length: Maximum length of description
            
        Returns:
            str: Human-readable failure description
        """
        node_name = getattr(self.node, '__name__', None) or self.node.__class__.__name__
        
        # Execution nodes (PythonRun, BashRun, etc.)
        if 'Run' in node_name:
            return self._describe_execution_failure(node_name, max_length)
        
        # Evaluator nodes (SubstringEvaluator, RegexEvaluator, etc.)  
        elif 'Evaluator' in node_name:
            return self._describe_evaluator_failure(node_name, max_length)
        
        # Pipeline nodes (ThenNode, AndNode, etc.)
        elif node_name in ['ThenNode', 'AndNode', 'OrNode']:
            return self._describe_pipeline_failure(node_name, max_length)
        
        # Fallback for unknown node types
        else:
            return f"{node_name} failed"
    
    def _describe_execution_failure(self, node_name, max_length):
        """Describe execution failures from Run nodes."""
        if not isinstance(self.children, (list, tuple)) or len(self.children) < 2:
            return f"{node_name} failed"
        
        code, output = self.children[0], self.children[1]
        
        if not isinstance(output, str):
            return f"{node_name} failed"
        
        # Extract key error information
        error_lines = []
        for line in output.strip().split('\n'):
            line_clean = line.strip()
            if any(indicator in line_clean.lower() for indicator in 
                   ['error:', 'exception:', 'traceback', 'timeout:', 'failed', 'syntax error']):
                error_lines.append(line_clean)
        
        if error_lines:
            error_summary = ' | '.join(error_lines[:2])  # First 2 error lines
            description = f"{node_name}: {error_summary}"
        else:
            # Show last lines as potential error context
            output_lines = output.strip().split('\n')
            if len(output_lines) > 1:
                description = f"{node_name}: {' | '.join(output_lines[-2:])}"
            else:
                description = f"{node_name}: {output.strip()}"
        
        return description[:max_length-3] + "..." if len(description) > max_length else description
    
    def _describe_evaluator_failure(self, node_name, max_length):
        """Describe evaluator failures."""
        if not isinstance(self.children, (list, tuple)) or len(self.children) < 2:
            return f"{node_name} failed"
        
        expected = self.children[0]
        result = self.children[1] if len(self.children) > 1 else None
        actual = self.children[2] if len(self.children) > 2 else "(no output)"
        
        # Only describe actual failures
        if result is not False:
            return None  # Not a failure
        
        # Truncate actual output if too long
        actual_str = str(actual)
        if len(actual_str) > 100:
            actual_str = actual_str[:97] + "..."
        
        description = f"{node_name}: Expected '{expected}', got '{actual_str}'"
        return description[:max_length-3] + "..." if len(description) > max_length else description
    
    def _describe_pipeline_failure(self, node_name, max_length):
        """Describe pipeline failures with step context."""
        if not isinstance(self.children, (list, tuple)) or len(self.children) < 2:
            return f"{node_name} failed"
        
        step1_reason, step2_reason = self.children[0], self.children[1]
        
        # Get step names
        step1_name = "unknown"
        step2_name = "unknown"
        
        if hasattr(step1_reason, 'node') and hasattr(step1_reason.node, '__name__'):
            step1_name = step1_reason.node.__name__
        if hasattr(step2_reason, 'node') and hasattr(step2_reason.node, '__name__'):
            step2_name = step2_reason.node.__name__
        
        # Try to get detailed error from steps
        error_detail = ""
        for step_name, step_reason in [(step1_name, step1_reason), (step2_name, step2_reason)]:
            if hasattr(step_reason, 'describe_failure'):
                step_error = step_reason.describe_failure(max_length=150)
                if step_error:
                    # Extract the error part (after the node name)
                    if ':' in step_error:
                        error_part = step_error.split(':', 1)[1].strip()
                        error_detail = f" ({step_name}: {error_part})"
                        break
        
        description = f"{node_name}: {step1_name} -> {step2_name} pipeline failed{error_detail}"
        return description[:max_length-3] + "..." if len(description) > max_length else description
        
    
class Node:
    """
    A node forms the operations in the computation graph for evaluating a test case;
    the most important object in this file. A test case might look like

        Node1 >> Node2 >> (Node3 & Node4)

    Each of these operators that connects nodes return a new node. So this graph
    would be equivalent to writing:

        ThenNode(ThenNode(Node1, Node2), AndNode(Node3, Node4))

    Once the computation graph has been constructed, evaluation is performed by
    calling __call__ on the root node, that then passes off the evalaution process
    as defined by each of the node types.
    """

    def __init__(self, runner):
        """
        Many sub-classes take a single argument, the runner, which is a function
        that should be executed for performing this node's computation.
        """
        self.runner = runner
    
    def setup(self, env, conv, llm, eval_llm, vision_eval_llm):
        """
        Once the graph has been constructed, before running __call__ to evaluate
        the test case, we run setup() on each of the nodes to pass all the
        necessary context. 
        """
        self.env = env
        self.conv = conv
        self.llm = llm
        self.eval_llm = eval_llm
        self.vision_eval_llm = vision_eval_llm

    def __call__(self, orig_output=""):
        """
        Evaluate the test case, starting at this node. This is the main entry
        point for the evaluation process.

        Returns two arguments:
        1. The output of the current node that should be passed to the next node.
        2. A Reason object that explains how the output was generated for debugging.
        
        """
        raise NotImplementedError()
        
    def __rshift__(self, other_node):
        """
        Add the >> operator, which creates a ThenNode.
        Wrap any strings in a StringNode first, to allow for code like

            SetupNode >> "command to run" >> LLMRunNode
        """
        
        if isinstance(other_node, str):
            other_node = StringNode(other_node)
        return ThenNode(self, other_node)
    
    def __rrshift__(self, other_node):
        """
        If a string is the first node, we need to special case the
        rrshift operator, since we can't override the string class.
        Allows the (very common) pattern of

            "command to run" >> LLMRunNode
        """
        if isinstance(other_node, str):
            other_node = StringNode(other_node)
        return ThenNode(other_node, self)
    
    def __and__(self, other_node):
        return AndNode(self, other_node)

    def __or__(self, other_node):
        return OrNode(self, other_node)

    def __invert__(self):
        return NotNode(self)

class StringNode(Node):
    def __init__(self, string):
        """
        A boring node, just returns the string.
        """
        self.string = string

    def __call__(self, orig_output=""):
        """
        Just pass whatever the provided constant string is to the next node.
        """
        yield self.string, Reason(type(self), self.string)
        

class ThenNode(Node):
    """
    Perform two operations in sequence. The output of node1 is passed to node2.
    """
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def setup(self, env, conv, llm, eval_llm, vision_eval_llm):
        super().setup(env, conv, llm, eval_llm, vision_eval_llm)
        self.node1.setup(env, conv, llm, eval_llm, vision_eval_llm)
        self.node2.setup(env=env, conv=conv, llm=llm, eval_llm=eval_llm, vision_eval_llm=vision_eval_llm)

    def __call__(self, orig_output=None):
        for output1, response1 in self.node1(orig_output):
            for output2, response2 in self.node2(output1):
                yield output2, Reason(type(self), (response1, response2))

class AndNode(ThenNode):
    """
    An evaluation node that returns true if both outputs are true.
    """
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def __call__(self, orig_output):
        for output1, txt1 in self.node1(orig_output):
            for output2, txt2 in self.node2(orig_output):
                yield output1 and output2, Reason(type(self), (txt1, txt2, output1 and output2))

class OrNode(ThenNode):
    """
    An evaluation node that returns true if either outputs are true.
    """
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def __call__(self, orig_output):
        for output1, txt1 in self.node1(orig_output):
            for output2, txt2 in self.node2(orig_output):
                yield output1 or output2, Reason(type(self), (txt1, txt2, output1 or output2))
                
class NotNode(Node):
    """
    An evaluation node that negates the prior answer.
    """
    def __init__(self, node1):
        self.node1 = node1

    def setup(self, env, conv, llm, eval_llm, vision_eval_llm):
        super().setup(env, conv, llm, eval_llm, vision_eval_llm)
        self.node1.setup(env, conv, llm, eval_llm, vision_eval_llm)
        
    def __call__(self, orig_output):
        for output1, txt1 in self.node1(orig_output):
            yield not output1, Reason(type(self), [txt1, not output1])

class PyFunc(Node):
    """
    A node that just runs a python function on the prior result.
    If the code crashes then just return an error.
    """
    def __call__(self, x):
        try:
            out = self.runner(x)
            if type(out) == tuple:
                ok, log = out
                return [(ok, Reason(type(self), (log, ok)))]
            else:
                return [(out, Reason(type(self), ("", out)))]
        except Exception as e:
            return [("", Reason(type(self), [f"Error: {e}", False]))]

class Echo(Node):
    """
    A no-op node that helps debug test cases by printing whatever's being
    passed along the pipe. Kind of like the Unix tee command.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        # More concise echo - show length and first few lines instead of full content
        if isinstance(x, str):
            lines = x.strip().split('\n')
            if len(lines) <= 3:
                summary = x.strip()
            else:
                first_lines = '\n'.join(lines[:2])
                summary = f"{first_lines}\n... ({len(lines)} total lines, {len(x)} chars)"
        else:
            summary = str(x)[:200] + ("..." if len(str(x)) > 200 else "")
        
        print(f'ECHO: {summary}')
        yield x, Reason(type(self), None)
    
class Setup(Node):
    """
    A node that starts up a new Docker environment with a specific setup file.

    Even though the argument is a method, this function needs to be able to
    extract the string representation of that function so it can be executed
    in the context of the docker environment.
    """
    def __call__(self, x):
        docker_controller.setup_docker(self.env)
        code = inspect.getsource(self.runner)
        to_invoke = self.runner.__name__

        code = code + f"\n\n{to_invoke}()"
        out = invoke_docker(self.env, {"setup.py": code.encode()}, [PYTHON_ENV, "setup.py"])

        return [(out, Reason(type(self), None))]

class PyEvaluator(Node):
    """
    A node that runs a python program within the docker environment to judge whether
    or not the test case is solved.

    Even though the argument is a method, this function needs to be able to
    extract the string representation of that function so it can be executed
    in the context of the docker environment.
    """
    def __call__(self, x):
        code = inspect.getsource(self.runner)
        to_invoke = self.runner.__name__

        code = code + f"\n\nprint('final: ' + str({to_invoke}()))"
        out = invoke_docker(self.env, {"check.py": code.encode()}, [PYTHON_ENV, "check.py"])

        return [("final: True" in out, Reason(type(self), [out, "final: True" in out]))]
    

class SubstringEvaluator(Node):
    """
    An evaluation node that checks if a substring is in the output.
    """
    def __init__(self, substr, lower=False):
        self.substr = substr
        self.lower = lower

    def __call__(self, output):
        try:
            output_normalized = normalize_text_for_comparison(output)
            substr_normalized = normalize_text_for_comparison(self.substr)
            if self.lower:
                cond = substr_normalized.lower() in output_normalized.lower()
            else:
                cond = self.substr in output or substr_normalized in output_normalized
        except Exception as exc:
            print(f'SubstringEvaluator error: {exc}, substr: {self.substr}, output type: {type(output)}')
            cond = False
            
        result = bool(cond)
        # Include expected, result, and actual output for better debugging
        yield result, Reason(type(self), [self.substr, result, output])

class RegexEvaluator(Node):
    """
    An evaluation node that checks if a regex pattern matches the output.
    """
    def __init__(self, pattern, ignore_case=False):
        self.pattern = pattern
        self.ignore_case = ignore_case

    def __call__(self, output):
        import re

        flags = re.IGNORECASE if self.ignore_case else 0
        output_normalized = normalize_text_for_comparison(output)
        match = re.search(self.pattern, output, flags) or re.search(self.pattern, output_normalized, flags)

        if match:
            yield True, Reason(type(self), [self.pattern, True, output])
        else:
            yield False, Reason(type(self), [self.pattern, False, output])
            
class ContainsIntEvaluator(Node):
    """
    An evaluation node that checks if a given integer is in the output.
    """
    def __init__(self, num):
        self.num = num

    def __call__(self, output):
        """Check if the specified integer appears in the output."""
        all_integers = re.findall(r'-?[\d,]*\d+\.?\d*', output)
        all_integers = [x.replace(",", "") for x in all_integers]
        
        found = str(self.num) in all_integers
        yield found, Reason(type(self), [self.num, found, output])
            
class EqualEvaluator(Node):
    """
    An evaluation node that checks if the output is equal to a given string.
    """
    def __init__(self, goal):
        self.goal = goal

    def __call__(self, output):
        """Check if output exactly matches the goal."""
        matches = (self.goal == output)
        if not matches and isinstance(self.goal, str) and isinstance(output, str):
            matches = normalize_text_for_comparison(self.goal).strip() == normalize_text_for_comparison(output).strip()
        yield matches, Reason(type(self), [self.goal, matches, output])

class UntilDone(Node):
    """
    A node that will loop a specific body node until the condition returns true and it's finished.

    This node is useful when you want a model to, e.g., iterative interact
    with a sqlite database until it's completed some task.
    """
    def __init__(self, cond, body, max_iters=100):
        self.cond = cond
        self.body = body
        self.max_iters = max_iters
        
    def setup(self, env, conv, llm, eval_llm, vision_eval_llm):
        super().setup(env, conv, llm, eval_llm, vision_eval_llm)
        self.cond.setup(env, conv, llm, eval_llm, vision_eval_llm)
        self.body.setup(env, conv, llm, eval_llm, vision_eval_llm)

    def __call__(self, orig_output=None):
        log = []
        for i in range(self.max_iters):
            for output, txt in self.cond(orig_output):
                if output:
                    yield orig_output, Reason(type(self), log)
                    return
            orig_output, partial = next(self.body(orig_output))
            log.append(partial)
        yield orig_output, Reason(type(self), log)
            
class ExtractJSON(Node):
    """
    A node that extracts a JSON object from the response.

    Usually you can just extract the json blob out of the response,
    but if the response contains multiple possible JSON blobs,
    then this node queries the model again asking it for just the JSON.
    """
    def __init__(self):
        pass

    def try_extract(self, output):
        """Extract JSON from code blocks or raw text."""
        normalized = normalize_text_for_comparison(output)
        for candidate in _extract_balanced_json_strings(normalized):
            yield candidate
        for block in _fenced_code_blocks(normalized):
            yield block
        if normalized:
            yield normalized
        
    def __call__(self, orig_output):
        candidates = list(self.try_extract(orig_output))
        for maybe in candidates:
            try:
                json.loads(maybe)
                yield maybe, Reason(type(self), [maybe])
                return
            except json.JSONDecodeError:
                continue

        prompt = (
            "Take the below answer to my question asking for a JSON output and just return "
            "the JSON object directly, with no other description, so I can copy it into an "
            "editor directly:\n" + orig_output
        )
        output = self.llm(prompt)
        for maybe in self.try_extract(output):
            yield maybe, Reason(type(self), [maybe])

class ExtractCode(Node):
    """
    A node that extracts code from the response

    Usually you can just extract the code out of the response,
    but if the response contains multiple possible code objects,
    then this node queries the model again asking it for just the code.
    """
    def __init__(self, keep_main=False, postfix="", manual=None, lang=None):
        self.keep_main = keep_main
        self.postfix = postfix
        self.manual = manual
        self.lang = lang

    def try_extract(self, output):
        """Extract code from markdown blocks or raw text."""
        for candidate in _extract_code_candidates(output):
            yield candidate + ("\n" + self.postfix if self.postfix else "")
        
    def __call__(self, orig_output):
        import logging
        logging.debug(f"ExtractCode input ({len(orig_output)} chars): {orig_output[:200]}{'...' if len(orig_output) > 200 else ''}")
        
        immediate_candidates = list(self.try_extract(orig_output))
        if immediate_candidates:
            logging.debug("ExtractCode: Using direct extraction candidates before eval_llm fallback")
            for maybe in immediate_candidates:
                yield maybe, Reason(type(self), maybe)
            return

        language = ""
        if self.lang is not None:
            language = f"(in {self.lang})"
        
        logging.debug(f"ExtractCode: No clear code blocks found, asking eval_llm for extraction {language}")
                
        if self.manual is not None:
            output = self.llm(self.manual.replace("<A>", orig_output))
        elif self.keep_main:
            assert self.postfix == ""
            # Use eval_llm, because it is smarter and will not hallucinate code
            output = self.eval_llm(f"Take the below answer to my programming question {language} and return just the complete code in a single file so I can copy and paste it into an editor and directly run it. Include any header and main necessary so I can run it by copying this one file. DO NOT MODIFY THE CODE OR WRITE NEW CODE. Here is the code: \n" + orig_output)
        else:
            output = self.eval_llm(f"Take the below answer to my programming question {language} and return just the complete code in a single file so I can copy and paste it into an editor and directly run it. Remove any test cases or example code after the function definition. Remove any main function. I will write those myself. Do include header imports. DO NOT MODIFY THE CODE OR WRITE NEW CODE. Here is the code: \n" + orig_output + ("\nI will be running this code with the following helper functions:\n" + self.postfix if self.postfix else ""))

        logging.debug(f"ExtractCode eval_llm output ({len(output)} chars): {output[:200]}{'...' if len(output) > 200 else ''}")

        # Strip thinking tokens from eval_llm output
        output = strip_thinking_tokens(output)

        for maybe in self.try_extract(output):
            yield maybe, Reason(type(self), maybe)

class MakeFile(Node):
    """
    A node that makes a new file within the docker environment.
    """
    def __init__(self, name):
        self.name = name

    def __call__(self, code):
        out = invoke_docker(self.env, {self.name: code.encode()}, ["echo"])
        yield out, Reason(type(self), (code, out))

class MakeFilesFromJSON(Node):
    """
    A node that makes a new file within the docker environment.
    """
    def __init__(self):
        pass

    def __call__(self, json_str):
        try:
            json_obj = json.loads(json_str)
        except json.JSONDecodeError:
            json_obj = {}
            
        for k in json_obj.keys():
            if not isinstance(json_obj[k], bytes):
                json_obj[k] = json_obj[k].encode()

        out = invoke_docker(self.env, json_obj, ["echo"])
        yield out, Reason(type(self), (json_str, out))
        

class PythonRun(Node):
    """
    A node that runs the output from the prior command as a python function.

    Optionally append a set of test cases to the code that's been provided.
    """
    def __init__(self, test_case="", out_bytes=False):
        self.test_case = test_case
        self.out_bytes = out_bytes

    def __call__(self, code):
        code = code + "\n\n" + self.test_case

        out = invoke_docker(self.env, {"main.py": code.encode()}, [PYTHON_ENV, "main.py"], out_bytes=self.out_bytes)
        yield out, Reason(type(self), (code, out))

class SwiftRun(Node):
    def __init__(self, test_case="", out_bytes=False):
        self.test_case = test_case
        self.out_bytes = out_bytes

    def __call__(self, code):
        code = code + "\n\n" + self.test_case
        
        files = {
            'main.swift': code.encode('utf-8')
        }

        run_cmd = ["swift", "main.swift"]
        
        output = invoke_docker(self.env, files, run_cmd, out_bytes=self.out_bytes)
        
        yield output, Reason(type(self), (code, output))
    
class SQLRun(Node):
    """
    A node that runs the output from the prior command as a sqlite function.
    """
    def __init__(self):
        pass

    def __call__(self, code):
        out = invoke_docker(self.env, {"run.sql": code.encode()}, ["sqlite3", "-init", "run.sql", "database.db", ".exit"])
        yield out, Reason(type(self), (code, out))
        
class BashRun(Node):
    """
    A node that runs the output from the prior command as a bash script.
    """
    def __init__(self, test_case="", args=[]):
        self.test_case = test_case
        self.args = args

    def __call__(self, code):
        code = code + "\n\n" + self.test_case

        out = invoke_docker(self.env, {"main.sh": code.encode()}, ["bash", "main.sh", *self.args])
        yield out, Reason(type(self), (code, out))

class TerminalRun(Node):
    """
    A node that directly runs a command line argument in the terminal.
    """
    def __init__(self):
        return

    def __call__(self, code):
        if code:
            out = invoke_docker(self.env, {"main.sh": code.encode()}, ["bash", "main.sh"])
        else:
            out = ""
        yield out, Reason(type(self), (code, out))

class RustRun(Node):
    """
    A node that compiles and runs the output Rust code from the prior command.

    Optionally append a set of test cases to the code that's been provided.
    """
    def __init__(self, test_case=""):
        self.test_case = test_case

    def __call__(self, code):
        if 'fn main' in code and 'fn main' in self.test_case:
            code = code.replace('fn main', 'fn __delete_this__main')

        code = code + "\n\n" + self.test_case
            
        out = invoke_docker(self.env, {"main.rs": code.encode(),
                                       "main.sh": "rustc -o a.out main.rs\n./a.out".encode()},
                            ["bash", "main.sh"])
        yield out, Reason(type(self), (code, out))

class CRun(Node):
    """
    A node that runs the output from the prior command as a c function.

    Optionally append a set of test cases to the code that's been provided.
    """
    def __init__(self, test_case="", out_bytes=False, gccflags="", argv=""):
        self.test_case = test_case
        self.out_bytes = out_bytes
        self.gccflags = gccflags
        self.argv = argv

    def __call__(self, code):
        if 'int main' in code and 'int main' in self.test_case:
            code = code.replace('int main', 'int __delete_this__main')

        code = code + "\n\n" + self.test_case
        
        out = invoke_docker(self.env, {"main.c": code.encode(),
                                       "main.sh": f"gcc -o a.out main.c -lm {self.gccflags}\n./a.out {self.argv}".encode()},
                            ["bash", "main.sh"], out_bytes=self.out_bytes)
        yield out, Reason(type(self), (code, out))


class CppRun(Node):
    """
    A node that runs the output from the prior command as a c++ function.

    Optionally append a set of test cases to the code that's been provided.
    """
    def __init__(self, test_case="", out_bytes=False):
        self.test_case = test_case
        self.out_bytes = out_bytes

    def __call__(self, code):
        if 'int main' in code and 'int main' in self.test_case:
            code = code.replace('int main', 'int __delete_this__main')

        code = code + "\n\n" + self.test_case
        
        out = invoke_docker(self.env, {"main.cpp": code.encode(),
                                       "main.sh": "g++ -o a.out main.cpp -lm\n./a.out".encode()},
                            ["bash", "main.sh"], out_bytes=self.out_bytes)
        yield out, Reason(type(self), (code, out))
        

class StartDockerJob(Node):
    """
    Start a new process within the docker container that's termainl interactive.

    This lets us test models that expect to be able to interface with other pieces
    of software by connecting the llm to stdin and stdout, sending data to the
    program and then reading the output back.
    """
    def __init__(self, command, eos_string):
        self.command = command
        self.eos_string = eos_string

    def __call__(self, text):
        self.env.docker_job = DockerJob(self.env.container.id if 'id' in dir(self.env.container) else self.env.container, self.eos_string)
        out = self.env.docker_job(self.command)

        yield out, Reason(type(self), (text, out))

class SendStdoutReceiveStdin(Node):
    """
    This node takes a given piece of text and sends it to the stdin of whatever
    the current running DockerJob is. It then waits for the running process to handle
    this input, and returns the output that the DockerJob returned from stdout.
    """
    def __init__(self):
        pass

    def __call__(self, text):
        out = self.env.docker_job(text)
        yield out, Reason(type(self), (out,))

            
class LLMRun(Node):
    """
    A node to invoke a language model on any given text.

    This is the core function that allows us to evaluate the capabilities of any model.
    By default, it strips the <think>...</think> tags from the output.
    """
    def __init__(self, check_prompt="<A>", llm=LLM, json=False, strip_think: bool = True):
        self.check_prompt = check_prompt
        self.which_llm = llm
        self.json = json
        self.strip_think = strip_think

    def __call__(self, output):
        import logging
        import time
        import os
        
        start_time = time.time()
        pid = os.getpid()
        
        logging.debug(f"LLMRUN_TRACE[{pid}]: Starting LLMRun.__call__")
        
        llm = getattr(self, self.which_llm)
        to_send = self.check_prompt.replace("<A>", output)
        
        logging.debug(f"LLMRUN_TRACE[{pid}]: LLMRun prompt ({len(to_send)} chars): {to_send[:200]}{'...' if len(to_send) > 200 else ''}")
        
        logging.debug(f"LLMRUN_TRACE[{pid}]: About to call LLM.{self.which_llm}()")
        llm_call_start = time.time()
        out = llm(to_send, json=self.json)
        llm_call_duration = time.time() - llm_call_start
        
        logging.debug(f"LLMRUN_TRACE[{pid}]: LLM call completed in {llm_call_duration:.3f}s")
        logging.debug(f"LLMRUN_TRACE[{pid}]: LLMRun response ({len(out) if out else 0} chars): {out[:200] if out else '(empty)'}{'...' if out and len(out) > 200 else ''}")
        
        if self.strip_think:
            logging.debug(f"LLMRUN_TRACE[{pid}]: Stripping thinking tokens")
            out = strip_thinking_tokens(out)
            
        total_duration = time.time() - start_time
        logging.debug(f"LLMRUN_TRACE[{pid}]: LLMRun complete in {total_duration:.3f}s total")
        
        yield out, Reason(type(self), (to_send, out))

class LLMConversation(Node):
    """
    A node to invoke a language model on any given text, but keeps state.

    This node allows us to send messages that refer to prior messages, whereas
    LLMRun is just a stateless operation.
    """
    def __init__(self, check_prompt="<A>"):
        self.check_prompt = check_prompt

    def __call__(self, output):
        to_send = self.check_prompt.replace("<A>", output)
        out = self.conv(to_send)
        yield out, Reason(type(self), (to_send, out))

class SeleniumDraw(Node):
    """
    A node that creates a new HTML page, renders it in chrome, and then
    captures the output with Selenium.
    """
    def __init__(self):
        pass

    def __call__(self, code):
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-search-engine-choice-screen")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            chrome_options.add_argument("--ignore-certificate-errors")

            r = random.randint(0, 1000000)

            with open("/tmp/a%d.html"%r, "w") as f:
                f.write(code)
    
            url = 'file:///tmp/a%d.html'%r
    
            browser = webdriver.Chrome(options=chrome_options)
            browser.get(url)
    
            time.sleep(2)
    
            screenshot_path = '/tmp/a%d.png'%r
            browser.save_screenshot(screenshot_path)
    
            browser.quit()
    
            time.sleep(1)
    
            img = Image.open(screenshot_path).convert('RGB')
    
            # get png data
            img_data = io.BytesIO()
            img.save(img_data, format="PNG")
            img_data.seek(0)
            img_data = img_data.read()
            
            
            yield img_data, Reason(type(self), img_data)

    
        except Exception as e:
            import logging
            error_msg = str(e)
            if "ChromeDriver" in error_msg and "supports Chrome version" in error_msg:
                logging.error("ChromeDriver version mismatch. Please update ChromeDriver to match your Chrome browser version.")
                logging.error("Run: brew upgrade chromedriver")
            else:
                logging.error(f"Error during SeleniumDraw execution: {e}")
            yield b"", Reason(type(self), b"")
        

class JSONSubsetEvaluator(Node):
    def __init__(self, goal):
        self.goal = goal
        
    def check(self, goal, output):
        if isinstance(goal, dict) and isinstance(output, dict):
            # Iterate over all key-value pairs in the goal dictionary
            for key, value in goal.items():
                # Check if the key is present in the output
                if key not in output:
                    return False
                # If the value is a dict or list, recursively check
                if isinstance(value, (dict, list)):
                    if not self.check(value, output[key]):
                        return False
                # Otherwise, check if the value matches
                elif output[key] != value:
                    return False
        elif isinstance(goal, list) and isinstance(output, list):
            # Check each element in the goal list
            for item in goal:
                if item not in output:
                    return False, Reason(self, ["Item not present", item])
        else:
            # Not a dict or list, so check if the values are equal
            if goal == output:
                return True
            else:
                return False
    
        # todo better error message
        return True
        
    def __call__(self, output):
        original_output = output  # Keep original for debugging
        try:
            parsed_output = json.loads(output)
        except json.JSONDecodeError:
            yield False, Reason(type(self), [self.goal, False, original_output])
            return

        ok = self.check(self.goal, parsed_output)
        yield ok, Reason(type(self), [self.goal, ok, original_output])

class LLMVisionRun(Node):
    """
    A node to evalaute an image output from a prior operation. Invokes the
    vision evaluation model.
    """
    def __init__(self, check_prompt="<A>", llm=VISION_EVAL_LLM):
        self.check_prompt = check_prompt
        self.which_llm = llm

    def __call__(self, output):
        llm = getattr(self, self.which_llm)
        if llm is None:
            out = "Vision LLM not configured - skipping vision test"
            yield False, Reason(type(self), (self.check_prompt, out))
            return
            
        try:
            if isinstance(output, bytes):
                img = Image.open(io.BytesIO(output))
            else:
                img = output
            out = llm(self.check_prompt, add_image=img, max_tokens=512)
        except Exception as e:
            out = str(e)
        yield out, Reason(type(self), (self.check_prompt, out))

class Conversation:
    """
    An object that keeps track of the conversation history between the
    model and the test case prior questions/steps.
    """
    def __init__(self, llm,preample = ''):
        self.llm = llm
        self.history = []
        self.preample = preample

    def __call__(self, msg):
        if len(self.history)==0:
            msg = self.preample + msg        
        self.history.append(msg)
        output = self.llm(self.history)
        self.history.append(output)
        return output

    def __repr__(self):
        return "Conversation(" + repr(self.history) + ")"

def run_test(test):
    """
    A helper function to run just one specific test case.
    Used to debug tests by running each file directly.
    """
    import llm as llm_module
    llm_module.ensure_default_models()
    from llm import llm, eval_llm, vision_eval_llm
    env = Env()
    test.setup(env, Conversation(llm), llm, eval_llm, vision_eval_llm)

    ok = False
    for success, output in test():
        if success:
            ok = True
            break

    import create_results_html
    fmt = create_results_html.format_markdown(output)
    while '\n\n' in fmt:
        fmt = fmt.replace('\n\n', '\n')
    fmt = fmt.replace("\n#", "\n\n#")
    print(fmt)
        
    if env.container:
        if hasattr(docker_controller, 'return_container_to_pool'):
            docker_controller.return_container_to_pool(env)
        else:
            # Fallback for compatibility
            docker_controller.async_kill_container(env.docker, env.container)

    return ok
    

def make_python_test(q_and_a, header=""):
    qs = [header]
    
    for q, a in q_and_a:
        qs.append(f"""
answer = {q}
expected = {a}
assert answer == expected, f'Wrong answer; got {{answer}} instead of {{expected}}'""")
    qs.append("print('All tests passed')")

    return "\n".join(qs), "All tests passed"

def make_swift_test(q_and_a, header="", extra_methods=""):
    qs = [header, extra_methods]
    
    qs.append("""
import Foundation

func assert(_ condition: Bool, _ message: String) {
    guard condition else {
        print(message)
        exit(1)
    }
}

""")

    for q, a in q_and_a:
        qs.append(f"""
let answer = {q}
let expected = {a}
assert(answer == expected, "Wrong answer; got \\(answer) instead of \\(expected)")
""")
    
    qs.append('print("All tests passed")')

    return "\n".join(qs), "All tests passed"

def make_c_test(q_and_a, header="", extra_methods=""):
    qs = []

    qs.append("#include<stdio.h>\n#include<stdlib.h>\n" + extra_methods + "\nint main() {")
    qs.append(header)
    for q, a in q_and_a:
        qs.append(f"""
int answer = {q};
int expected = {a};
if (answer != expected) {{
    printf("Wrong answer; got %d instead of %d.\\n", answer, expected);
    exit(1);
}}""")
    qs.append('printf("All tests passed\\n");')

    qs.append("}");
    
    return "\n".join(qs), "All tests passed"
        
    
