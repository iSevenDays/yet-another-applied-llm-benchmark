#!/usr/bin/env python3
import unittest
import sys
import os

# Add parent directory to path to import evaluator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import strip_thinking_tokens

class TestStripThink(unittest.TestCase):
    """Test the centralized strip_thinking_tokens function"""
    
    def test_basic_think_tags(self):
        """Test basic <think> tag removal"""
        input_text = "Hello <think>I need to think about this</think> World"
        expected = "Hello  World"
        result = strip_thinking_tokens(input_text)
        self.assertEqual(result, expected)
    
    def test_multiline_thinking(self):
        """Test multi-line thinking blocks"""
        input_text = "Start <think>\nLine 1\nLine 2\n</think> End"
        expected = "Start  End"
        result = strip_thinking_tokens(input_text)
        self.assertEqual(result, expected)
    
    def test_seed_think_tags(self):
        """Test <seed:think> tag removal"""
        input_text = "Hello <seed:think>some reasoning</seed:think> World"
        expected = "Hello  World"
        result = strip_thinking_tokens(input_text)
        self.assertEqual(result, expected)
    
    def test_multiple_think_blocks(self):
        """Test multiple thinking blocks"""
        input_text = "A <think>first</think> B <think>second</think> C"
        expected = "A  B  C"
        result = strip_thinking_tokens(input_text)
        self.assertEqual(result, expected)
    
    def test_mixed_tags(self):
        """Test mixed think and seed:think tags"""
        input_text = "Start <think>normal</think> middle <seed:think>seed</seed:think> end"
        expected = "Start  middle  end"
        result = strip_thinking_tokens(input_text)
        self.assertEqual(result, expected)
    
    def test_no_think_tags(self):
        """Test text without any think tags"""
        input_text = "Just normal text"
        expected = "Just normal text"
        result = strip_thinking_tokens(input_text)
        self.assertEqual(result, expected)
    
    def test_empty_think_blocks(self):
        """Test empty thinking blocks"""
        input_text = "Hello <think></think> World"
        expected = "Hello  World"
        result = strip_thinking_tokens(input_text)
        self.assertEqual(result, expected)
    
    def test_nested_tags_behavior(self):
        """Test behavior with nested tags (regex matches first closing tag)"""
        input_text = "Start <think>outer <think>inner</think> content</think> End"
        # Non-greedy regex matches first </think>, leaving remainder
        expected = "Start  content</think> End"  
        result = strip_thinking_tokens(input_text)
        self.assertEqual(result, expected)
    
    def test_unclosed_seed_think(self):
        """Test the problematic case: unclosed <seed:think> tag"""
        input_text = "ECHO: <seed:think>```"
        expected = "ECHO: <seed:think>```"  # Unclosed tags are not matched by regex
        result = strip_thinking_tokens(input_text)
        self.assertEqual(result, expected)
    
    def test_eval_llm_scenarios(self):
        """Test typical eval LLM output scenarios"""
        test_cases = [
            ("Result <think>analyzing</think> code", "Result  code"),
            ("Output: <seed:think>reasoning here</seed:think> final", "Output:  final"),
            ("Mixed <think>first</think> and <seed:think>second</seed:think> content", "Mixed  and  content")
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = strip_thinking_tokens(input_text)
                self.assertEqual(result, expected)
    
    def test_gpt_oss_harmony_format(self):
        """Test GPT-OSS Harmony format thinking token removal"""
        test_cases = [
            # Full harmony format with analysis channel
            (
                "<|start|>assistant<|channel|>analysis<|message|>Let me think about this problem step by step.<|end|> <|start|>assistant<|channel|>final<|message|>The answer is 42.<|return|>",
                " <|start|>assistant<|channel|>final<|message|>The answer is 42.<|return|>"
            ),
            # Commentary channel
            (
                "Some text <|start|>assistant<|channel|>commentary<|message|>This is my reasoning process<|end|> Final result",
                "Some text  Final result"
            ),
            # Alternative pattern without assistant prefix
            (
                "Start <|channel|>analysis<|message|>Internal analysis here<|end|> End",
                "Start  End"
            ),
            # Multiple analysis blocks
            (
                "<|channel|>analysis<|message|>First analysis<|end|> Middle <|channel|>analysis<|message|>Second analysis<|end|> End",
                " Middle  End"
            ),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = strip_thinking_tokens(input_text)
                self.assertEqual(result, expected)
    
    def test_mixed_formats(self):
        """Test mixed thinking token formats in same text"""
        input_text = """
        <think>Simple thinking</think>
        <|channel|>analysis<|message|>GPT-OSS analysis<|end|>
        <seed:think>Seed thinking</seed:think>
        Final output here
        """
        expected = """
        
        
        
        Final output here
        """
        result = strip_thinking_tokens(input_text)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()