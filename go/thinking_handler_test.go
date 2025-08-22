package main

import (
	"strings"
	"testing"
)

func TestNewThinkingHandler(t *testing.T) {
	handler := NewThinkingHandler()
	if handler == nil {
		t.Fatal("NewThinkingHandler returned nil")
	}
	if len(handler.patterns) == 0 {
		t.Fatal("Handler has no patterns")
	}
}

func TestHasThinkingTokens(t *testing.T) {
	handler := NewThinkingHandler()
	
	testCases := []struct {
		text     string
		expected bool
	}{
		{"Hello <think>reasoning</think> World", true},
		{"<seed:think>analysis</seed:think> Result", true},
		{"<|start|>assistant<|channel|>analysis<|message|>thinking<|end|>", true},
		{"Just normal text", false},
		{"", false},
	}
	
	for _, tc := range testCases {
		t.Run(tc.text, func(t *testing.T) {
			result := handler.HasThinkingTokens(tc.text)
			if result != tc.expected {
				t.Errorf("HasThinkingTokens(%q) = %v, want %v", tc.text, result, tc.expected)
			}
		})
	}
}

func TestStripThinkingTokens(t *testing.T) {
	handler := NewThinkingHandler()
	
	testCases := []struct {
		input    string
		expected string
	}{
		{"Hello <think>reasoning</think> World", "Hello  World"},
		{"<seed:think>analysis</seed:think> Result", " Result"},
		{"Normal text", "Normal text"},
		{"Multiple <think>first</think> and <seed:think>second</seed:think>", "Multiple  and "},
	}
	
	for _, tc := range testCases {
		t.Run(tc.input, func(t *testing.T) {
			result := handler.StripThinkingTokens(tc.input)
			if result != tc.expected {
				t.Errorf("StripThinkingTokens(%q) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}

func TestConvertToCollapsibleHTML(t *testing.T) {
	handler := NewThinkingHandler()
	
	input := "Hello <think>Some reasoning here</think> World"
	result := handler.ConvertToCollapsibleHTML(input)
	
	// Check that result contains expected HTML elements
	if !strings.Contains(result, `class="thinking-container"`) {
		t.Error("Should contain thinking container")
	}
	if !strings.Contains(result, `class="show-thinking-btn"`) {
		t.Error("Should contain Show Thinking button")
	}
	if !strings.Contains(result, "Show Thinking") {
		t.Error("Should contain Show Thinking text")
	}
	if !strings.Contains(result, `class="thinking-content"`) {
		t.Error("Should contain thinking content")
	}
	if !strings.Contains(result, "Some reasoning here") {
		t.Error("Should preserve thinking content")
	}
	if !strings.Contains(result, "Hello") || !strings.Contains(result, "World") {
		t.Error("Should preserve non-thinking content")
	}
}

func TestConvertToCollapsibleHTMLNoThinking(t *testing.T) {
	handler := NewThinkingHandler()
	
	input := "Just normal text"
	result := handler.ConvertToCollapsibleHTML(input)
	
	// Should return unchanged text
	if result != input {
		t.Errorf("ConvertToCollapsibleHTML(%q) = %q, want %q", input, result, input)
	}
	if strings.Contains(result, "thinking-container") {
		t.Error("Should not contain thinking container")
	}
}

func TestMultipleThinkingBlocks(t *testing.T) {
	handler := NewThinkingHandler()
	
	input := "Start <think>first</think> middle <seed:think>second</seed:think> end"
	result := handler.ConvertToCollapsibleHTML(input)
	
	// Should contain both buttons with different labels
	if !strings.Contains(result, "Show Thinking") {
		t.Error("Should contain Show Thinking")
	}
	if !strings.Contains(result, "Show Seed Thinking") {
		t.Error("Should contain Show Seed Thinking")
	}
	if !strings.Contains(result, "first") {
		t.Error("Should contain first thinking content")
	}
	if !strings.Contains(result, "second") {
		t.Error("Should contain second thinking content")
	}
	
	// Check for multiple unique IDs
	if !strings.Contains(result, "thinking-1") {
		t.Error("Should contain thinking-1")
	}
	if !strings.Contains(result, "thinking-2") {
		t.Error("Should contain thinking-2")
	}
}

func TestHTMLEscaping(t *testing.T) {
	handler := NewThinkingHandler()
	
	input := "Test <think><script>alert('xss')</script></think> end"
	result := handler.ConvertToCollapsibleHTML(input)
	
	// HTML should be escaped
	if !strings.Contains(result, "&lt;script&gt;") {
		t.Error("Should escape <script>")
	}
	if !strings.Contains(result, "&lt;/script&gt;") {
		t.Error("Should escape </script>")
	}
	if strings.Contains(result, "<script>") {
		t.Error("Should not contain unescaped <script>")
	}
}

func TestHarmonyFormat(t *testing.T) {
	handler := NewThinkingHandler()
	
	input := "Start <|channel|>analysis<|message|>Deep thinking here<|end|> End"
	
	// Should detect thinking tokens
	if !handler.HasThinkingTokens(input) {
		t.Error("Should detect Harmony format thinking tokens")
	}
	
	// Should strip correctly
	stripped := handler.StripThinkingTokens(input)
	if stripped != "Start  End" {
		t.Errorf("StripThinkingTokens = %q, want %q", stripped, "Start  End")
	}
	
	// Should convert to HTML
	htmlResult := handler.ConvertToCollapsibleHTML(input)
	if !strings.Contains(htmlResult, "Show Analysis") {
		t.Error("Should contain Show Analysis button")
	}
	if !strings.Contains(htmlResult, "Deep thinking here") {
		t.Error("Should contain analysis content")
	}
}

func TestGetCSSStyles(t *testing.T) {
	handler := NewThinkingHandler()
	css := handler.GetCSSStyles()
	
	if !strings.Contains(css, ".thinking-container") {
		t.Error("CSS should contain .thinking-container")
	}
	if !strings.Contains(css, ".show-thinking-btn") {
		t.Error("CSS should contain .show-thinking-btn")
	}
	if !strings.Contains(css, ".thinking-content") {
		t.Error("CSS should contain .thinking-content")
	}
	if !strings.Contains(css, ".thinking-text") {
		t.Error("CSS should contain .thinking-text")
	}
}

func TestGetJavaScript(t *testing.T) {
	handler := NewThinkingHandler()
	js := handler.GetJavaScript()
	
	if !strings.Contains(js, "function toggleThinking") {
		t.Error("JavaScript should contain toggleThinking function")
	}
	if !strings.Contains(js, "getElementById") {
		t.Error("JavaScript should contain getElementById")
	}
	if !strings.Contains(js, "display") {
		t.Error("JavaScript should contain display manipulation")
	}
}