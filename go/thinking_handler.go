package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"html"
	"regexp"
	"strings"
	"sync"
	"unsafe"
)

// ThinkingPattern represents a thinking token pattern with its replacement
type ThinkingPattern struct {
	Pattern *regexp.Regexp
	Title   string
}

// ThinkingHandler handles thinking tokens in LLM outputs
type ThinkingHandler struct {
	patterns []ThinkingPattern
	counter  int
	mutex    sync.Mutex
}

// NewThinkingHandler creates a new thinking handler with compiled patterns
func NewThinkingHandler() *ThinkingHandler {
	patterns := []ThinkingPattern{
		{regexp.MustCompile(`<think>(.*?)</think>`), "Show Thinking"},
		{regexp.MustCompile(`<seed:think>(.*?)</seed:think>`), "Show Seed Thinking"},
		{regexp.MustCompile(`<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>`), "Show Analysis"},
		{regexp.MustCompile(`<\|start\|>assistant<\|channel\|>commentary<\|message\|>(.*?)<\|end\|>`), "Show Commentary"},
		{regexp.MustCompile(`<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>`), "Show Analysis"},
		{regexp.MustCompile(`<\|channel\|>commentary<\|message\|>(.*?)<\|end\|>`), "Show Commentary"},
	}
	
	return &ThinkingHandler{
		patterns: patterns,
		counter:  0,
	}
}

// HasThinkingTokens checks if text contains any thinking tokens
func (th *ThinkingHandler) HasThinkingTokens(text string) bool {
	for _, pattern := range th.patterns {
		if pattern.Pattern.MatchString(text) {
			return true
		}
	}
	return false
}

// StripThinkingTokens removes all thinking tokens from text
func (th *ThinkingHandler) StripThinkingTokens(text string) string {
	result := text
	for _, pattern := range th.patterns {
		// Convert capture group pattern to non-capturing for stripping
		stripPattern := strings.Replace(pattern.Pattern.String(), "(.*?)", ".*?", -1)
		stripRegex := regexp.MustCompile(stripPattern)
		result = stripRegex.ReplaceAllString(result, "")
	}
	return result
}

// ConvertToCollapsibleHTML converts thinking tokens to collapsible HTML elements
func (th *ThinkingHandler) ConvertToCollapsibleHTML(text string) string {
	th.mutex.Lock()
	defer th.mutex.Unlock()
	
	result := text
	th.counter = 0
	
	for _, pattern := range th.patterns {
		result = pattern.Pattern.ReplaceAllStringFunc(result, func(match string) string {
			return th.makeCollapsible(match, pattern.Pattern, pattern.Title)
		})
	}
	
	return result
}

// makeCollapsible creates a collapsible HTML div for thinking content
func (th *ThinkingHandler) makeCollapsible(match string, pattern *regexp.Regexp, title string) string {
	th.counter++
	
	// Extract thinking content (first capture group)
	submatches := pattern.FindStringSubmatch(match)
	var thinkingContent string
	if len(submatches) > 1 {
		thinkingContent = strings.TrimSpace(submatches[1])
	} else {
		thinkingContent = match
	}
	
	// Escape HTML in thinking content
	escapedContent := html.EscapeString(thinkingContent)
	
	return fmt.Sprintf(`<div class="thinking-container">
    <button class="show-thinking-btn" onclick="toggleThinking('thinking-%d')" type="button">
        %s
    </button>
    <div id="thinking-%d" class="thinking-content" style="display: none;">
        <pre class="thinking-text">%s</pre>
    </div>
</div>`, th.counter, title, th.counter, escapedContent)
}

// GetCSSStyles returns CSS styles for thinking containers
func (th *ThinkingHandler) GetCSSStyles() string {
	return `
.thinking-container {
    margin: 8px 0;
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: #f9f9f9;
}

.show-thinking-btn {
    background-color: #e7f3ff;
    border: none;
    padding: 8px 12px;
    cursor: pointer;
    font-size: 12px;
    color: #0366d6;
    border-radius: 4px 4px 0 0;
    width: 100%;
    text-align: left;
    transition: background-color 0.2s;
}

.show-thinking-btn:hover {
    background-color: #d1ecf1;
}

.thinking-content {
    border-top: 1px solid #ddd;
    background-color: #fff;
}

.thinking-text {
    margin: 0;
    padding: 12px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 11px;
    line-height: 1.4;
    color: #586069;
    background: none;
    border: none;
    white-space: pre-wrap;
    word-wrap: break-word;
}
`
}

// GetJavaScript returns JavaScript for toggle functionality
func (th *ThinkingHandler) GetJavaScript() string {
	return `
function toggleThinking(elementId) {
    const content = document.getElementById(elementId);
    const button = content.previousElementSibling;
    
    if (content.style.display === 'none' || content.style.display === '') {
        content.style.display = 'block';
        button.textContent = button.textContent.replace('Show', 'Hide');
    } else {
        content.style.display = 'none';
        button.textContent = button.textContent.replace('Hide', 'Show');
    }
}
`
}

// Global handler instance
var globalHandler = NewThinkingHandler()

// C-compatible exported functions

//export has_thinking_tokens
func has_thinking_tokens(text *C.char) C.int {
	goText := C.GoString(text)
	if globalHandler.HasThinkingTokens(goText) {
		return 1
	}
	return 0
}

//export strip_thinking_tokens
func strip_thinking_tokens(text *C.char) *C.char {
	goText := C.GoString(text)
	result := globalHandler.StripThinkingTokens(goText)
	return C.CString(result)
}

//export convert_thinking_to_collapsible_html
func convert_thinking_to_collapsible_html(text *C.char) *C.char {
	goText := C.GoString(text)
	result := globalHandler.ConvertToCollapsibleHTML(goText)
	return C.CString(result)
}

//export get_css_styles
func get_css_styles() *C.char {
	result := globalHandler.GetCSSStyles()
	return C.CString(result)
}

//export get_javascript
func get_javascript() *C.char {
	result := globalHandler.GetJavaScript()
	return C.CString(result)
}

//export free_string
func free_string(str *C.char) {
	C.free(unsafe.Pointer(str))
}

func main() {} // Required for C shared library