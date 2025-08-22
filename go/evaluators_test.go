package main

import (
	"testing"
)

func TestRegexEvaluator(t *testing.T) {
	testCases := []struct {
		pattern    string
		text       string
		ignoreCase bool
		expected   bool
		name       string
	}{
		{"hello", "hello world", false, true, "simple_match"},
		{"HELLO", "hello world", false, false, "case_sensitive_miss"},
		{"HELLO", "hello world", true, true, "case_insensitive_match"},
		{`\d+`, "age is 25 years", false, true, "digit_pattern"},
		{`^\d+$`, "123", false, true, "exact_digit_match"},
		{`^\d+$`, "abc123", false, false, "exact_digit_miss"},
		{"world$", "hello world", false, true, "end_anchor"},
		{"^hello", "hello world", false, true, "start_anchor"},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			evaluator, err := NewRegexEvaluator(tc.pattern, tc.ignoreCase)
			if err != nil {
				t.Fatalf("Failed to create regex evaluator: %v", err)
			}
			
			result := evaluator.Evaluate(tc.text)
			if result != tc.expected {
				t.Errorf("RegexEvaluator(%q, %q, ignoreCase=%v) = %v, want %v", 
					tc.pattern, tc.text, tc.ignoreCase, result, tc.expected)
			}
		})
	}
}

func TestSubstringEvaluator(t *testing.T) {
	testCases := []struct {
		substring       string
		text            string
		caseInsensitive bool
		expected        bool
		name            string
	}{
		{"hello", "hello world", false, true, "simple_match"},
		{"HELLO", "hello world", false, false, "case_sensitive_miss"},
		{"HELLO", "hello world", true, true, "case_insensitive_match"},
		{"world", "hello beautiful world", false, true, "substring_in_middle"},
		{"missing", "hello world", false, false, "not_found"},
		{"", "hello world", false, true, "empty_substring"},
		{"hello", "", false, false, "empty_text"},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			evaluator := NewSubstringEvaluator(tc.substring, tc.caseInsensitive)
			result := evaluator.Evaluate(tc.text)
			if result != tc.expected {
				t.Errorf("SubstringEvaluator(%q, %q, caseInsensitive=%v) = %v, want %v", 
					tc.substring, tc.text, tc.caseInsensitive, result, tc.expected)
			}
		})
	}
}

func TestIntegerEvaluator(t *testing.T) {
	testCases := []struct {
		number   int
		text     string
		expected bool
		name     string
	}{
		{42, "The answer is 42", true, "simple_integer"},
		{123, "Values: 123, 456, 789", true, "integer_in_list"},
		{999, "No such number here", false, "not_found"},
		{-5, "Temperature is -5 degrees", true, "negative_integer"},
		{1000, "Price: 1,000 dollars", true, "comma_separated"},
		{123, "Version 1.23 released", false, "decimal_not_integer"},
		{0, "Count is 0 items", true, "zero_value"},
		{42, "Code 4242 received", false, "partial_match_should_fail"},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			evaluator := NewIntegerEvaluator(tc.number)
			result := evaluator.Evaluate(tc.text)
			if result != tc.expected {
				t.Errorf("IntegerEvaluator(%d, %q) = %v, want %v", 
					tc.number, tc.text, result, tc.expected)
			}
		})
	}
}

func TestRegexEvaluatorCaching(t *testing.T) {
	// Clear cache first
	clear_evaluator_caches()
	
	pattern := "test.*pattern"
	
	// First call should create and cache
	eval1 := getRegexEvaluator(pattern, false)
	if eval1 == nil {
		t.Fatal("Failed to create regex evaluator")
	}
	
	// Second call should return cached version
	eval2 := getRegexEvaluator(pattern, false)
	if eval2 == nil {
		t.Fatal("Failed to get cached regex evaluator")
	}
	
	// Should be the same instance (pointer equality)
	if eval1 != eval2 {
		t.Error("Regex evaluator caching not working - got different instances")
	}
}

func TestInvalidRegexPattern(t *testing.T) {
	invalidPattern := "[invalid"
	evaluator, err := NewRegexEvaluator(invalidPattern, false)
	
	if err == nil {
		t.Error("Expected error for invalid regex pattern, got none")
	}
	if evaluator != nil {
		t.Error("Expected nil evaluator for invalid pattern, got instance")
	}
}

func TestEdgeCases(t *testing.T) {
	t.Run("empty_strings", func(t *testing.T) {
		// Regex with empty string
		evaluator, err := NewRegexEvaluator("", false)
		if err != nil {
			t.Fatalf("Empty regex pattern failed: %v", err)
		}
		if !evaluator.Evaluate("any text") {
			t.Error("Empty regex should match any text")
		}
		
		// Substring with empty strings
		subEval := NewSubstringEvaluator("", false)
		if !subEval.Evaluate("any text") {
			t.Error("Empty substring should match any text")
		}
	})
	
	t.Run("unicode_handling", func(t *testing.T) {
		evaluator := NewSubstringEvaluator("café", true)
		if !evaluator.Evaluate("Welcome to the café") {
			t.Error("Unicode substring matching failed")
		}
	})
}

// Benchmark tests to verify performance improvements
func BenchmarkRegexEvaluator(b *testing.B) {
	evaluator, _ := NewRegexEvaluator(`\d{4}-\d{2}-\d{2}`, false)
	text := "Today's date is 2024-01-15 and tomorrow is 2024-01-16"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evaluator.Evaluate(text)
	}
}

func BenchmarkSubstringEvaluator(b *testing.B) {
	evaluator := NewSubstringEvaluator("target", false)
	text := "This is a long text with many words including the target word we're looking for"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evaluator.Evaluate(text)
	}
}

func BenchmarkIntegerEvaluator(b *testing.B) {
	evaluator := NewIntegerEvaluator(12345)
	text := "Numbers in text: 100, 200, 12345, 300, 400"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evaluator.Evaluate(text)
	}
}