package main

import (
	"C"
	"regexp"
	"strconv"
	"strings"
)

/*
#include <stdlib.h>
*/
import "C"

// EvaluatorResult holds the result of an evaluation operation
type EvaluatorResult struct {
	Success bool
	Message string
}

// RegexEvaluator handles regex pattern matching
type RegexEvaluator struct {
	Pattern     string
	IgnoreCase  bool
	CompiledRx  *regexp.Regexp
}

// NewRegexEvaluator creates a new regex evaluator with compiled pattern
func NewRegexEvaluator(pattern string, ignoreCase bool) (*RegexEvaluator, error) {
	flags := ""
	if ignoreCase {
		flags = "(?i)"
	}
	
	compiledPattern := flags + pattern
	rx, err := regexp.Compile(compiledPattern)
	if err != nil {
		return nil, err
	}
	
	return &RegexEvaluator{
		Pattern:    pattern,
		IgnoreCase: ignoreCase,
		CompiledRx: rx,
	}, nil
}

// Evaluate tests if the text matches the regex pattern
func (re *RegexEvaluator) Evaluate(text string) bool {
	return re.CompiledRx.MatchString(text)
}

// SubstringEvaluator handles substring matching
type SubstringEvaluator struct {
	Substring     string
	CaseInsensitive bool
}

// NewSubstringEvaluator creates a new substring evaluator
func NewSubstringEvaluator(substring string, caseInsensitive bool) *SubstringEvaluator {
	return &SubstringEvaluator{
		Substring:       substring,
		CaseInsensitive: caseInsensitive,
	}
}

// Evaluate tests if the text contains the substring
func (se *SubstringEvaluator) Evaluate(text string) bool {
	if se.CaseInsensitive {
		return strings.Contains(strings.ToLower(text), strings.ToLower(se.Substring))
	}
	return strings.Contains(text, se.Substring)
}

// IntegerEvaluator handles integer detection in text
type IntegerEvaluator struct {
	Number int
}

// NewIntegerEvaluator creates a new integer evaluator
func NewIntegerEvaluator(number int) *IntegerEvaluator {
	return &IntegerEvaluator{Number: number}
}

// Evaluate tests if the text contains the specified integer
func (ie *IntegerEvaluator) Evaluate(text string) bool {
	// Pattern matches Python's: r'-?[\d,]*\d+\.?\d*'
	pattern := regexp.MustCompile(`-?[\d,]*\d+\.?\d*`)
	matches := pattern.FindAllString(text, -1)
	
	targetStr := strconv.Itoa(ie.Number)
	
	for _, match := range matches {
		// Remove commas like Python version
		cleaned := strings.ReplaceAll(match, ",", "")
		if cleaned == targetStr {
			return true
		}
	}
	
	return false
}

// Global instances for stateless operations
var (
	regexCache    = make(map[string]*RegexEvaluator)
	substringCache = make(map[string]*SubstringEvaluator)
	integerCache   = make(map[int]*IntegerEvaluator)
)

// Helper function to get or create cached regex evaluator
func getRegexEvaluator(pattern string, ignoreCase bool) *RegexEvaluator {
	key := pattern
	if ignoreCase {
		key = "(?i)" + pattern
	}
	
	if cached, exists := regexCache[key]; exists {
		return cached
	}
	
	evaluator, err := NewRegexEvaluator(pattern, ignoreCase)
	if err != nil {
		return nil
	}
	
	regexCache[key] = evaluator
	return evaluator
}

// C-compatible exported functions

//export evaluate_regex
func evaluate_regex(pattern *C.char, text *C.char, ignore_case C.int) C.int {
	goPattern := C.GoString(pattern)
	goText := C.GoString(text)
	ignoreCase := ignore_case != 0
	
	evaluator := getRegexEvaluator(goPattern, ignoreCase)
	if evaluator == nil {
		return 0 // Pattern compilation failed
	}
	
	if evaluator.Evaluate(goText) {
		return 1
	}
	return 0
}

//export evaluate_substring
func evaluate_substring(substr *C.char, text *C.char, case_insensitive C.int) C.int {
	goSubstr := C.GoString(substr)
	goText := C.GoString(text)
	caseInsensitive := case_insensitive != 0
	
	key := goSubstr
	if caseInsensitive {
		key = "ci:" + goSubstr
	}
	
	var evaluator *SubstringEvaluator
	if cached, exists := substringCache[key]; exists {
		evaluator = cached
	} else {
		evaluator = NewSubstringEvaluator(goSubstr, caseInsensitive)
		substringCache[key] = evaluator
	}
	
	if evaluator.Evaluate(goText) {
		return 1
	}
	return 0
}

//export contains_integer
func contains_integer(number C.int, text *C.char) C.int {
	goNumber := int(number)
	goText := C.GoString(text)
	
	var evaluator *IntegerEvaluator
	if cached, exists := integerCache[goNumber]; exists {
		evaluator = cached
	} else {
		evaluator = NewIntegerEvaluator(goNumber)
		integerCache[goNumber] = evaluator
	}
	
	if evaluator.Evaluate(goText) {
		return 1
	}
	return 0
}

//export clear_evaluator_caches
func clear_evaluator_caches() {
	regexCache = make(map[string]*RegexEvaluator)
	substringCache = make(map[string]*SubstringEvaluator)
	integerCache = make(map[int]*IntegerEvaluator)
}

// main function provided by thinking_handler.go