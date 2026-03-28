# Yet Another Applied LLM Benchmark - Test Investigation TODO

`★ Investigation Progress ─────────────────────────────────────`
**This document tracks our systematic investigation of all tests in the benchmark to understand failure patterns, test design issues, and LLM limitations.**
`─────────────────────────────────────────────────────────────`

## Investigation Status Legend
- ✅ **COMPLETED** - Analysis complete, findings documented
- 🔍 **IN PROGRESS** - Currently under investigation  
- ❌ **NOT STARTED** - Not yet investigated
- 🛠️ **NEEDS FIXING** - Test design issues identified, requires fixes

---

## Completed Investigations

### ✅ PrintHelloPoly & PrintHelloPoly2
**Status**: Analysis complete - Test reveals fundamental polyglot programming limitations
- **File**: `tests/print_hello_poly.py`
- **Challenge**: Create code that executes as both C/Python and Rust/Python
- **Result**: Universal failure (0/9 models) due to syntax incompatibility
- **Root Cause**: Fundamental language parser conflicts, over-complicated approaches
- **Findings**: Tests genuine cross-language synthesis capabilities

### ✅ FindBugPaperEasy  
**Status**: Analysis complete - Test design issues identified
- **File**: `tests/find_bug_in_paper.py` 
- **Challenge**: Find mathematical errors in adversarial ML objective functions
- **Result**: Universal failure (0/9 models) but likely due to test issues
- **Root Cause**: Mathematical inconsistencies in expected answer, ambiguous tie-breaking
- **Findings**: Need for improved test (created `find_bug_in_paper_improved.py`)

### 🛠️ SqlExplore  
**Status**: Root cause fixed - Test infrastructure issues identified and resolved
- **File**: `tests/explore_sql_db.py`
- **Challenge**: Interactive SQLite database exploration and data insertion
- **Result**: Universal failure (0/10 models) due to test infrastructure problems
- **Root Cause**: Docker output corruption, column schema mismatch, brittle command parsing
- **Fixes Applied**: 
  - Fixed column names (`a_name`→`name`, `person_age`→`age`, `the_profession`→`profession`)
  - Improved command extraction with robust regex patterns
  - Added schema discovery (`.schema` output) for models
  - Fixed validation query to match actual table structure
- **Findings**: LLMs showed sophisticated SQL reasoning but were sabotaged by infrastructure failures

### ✅ TrainSchedulePython
**Status**: Analysis complete - Genuine LLM limitation identified  
- **File**: `tests/data_train_timetable.py`
- **Challenge**: Parse complex train timetable and extract travel times for specific trains
- **Result**: Universal failure (0/10 models) due to systematic parsing errors
- **Root Cause**: LLM inability to accurately map column positions in complex tabular data
- **Key Issues**:
  - Models failed to correctly identify column positions for trains 703, 303, 509
  - Off-by-one errors and systematic counting mistakes in 52-column table
  - Strong algorithmic reasoning (time parsing, AM/PM conversion) but poor attention to detail
- **Minor Enhancement**: Added clearer instructions about column-position mapping
- **Findings**: Exposes fundamental LLM limitation in structured data parsing requiring precise positional accuracy

### 🛠️ MakePNGToELF (ChangeFiletype)
**Status**: Critical bug fixed - Test design issue and fundamental impossibility
- **File**: `tests/change_filetype.py`
- **Challenge**: Create a file that loads as PNG in PIL but is detected as ELF by `file` command
- **Result**: Universal failure (0/13 models) due to multiple issues
- **Root Causes**:
  1. **Critical Bug**: `Image.open().numpy()` is invalid (PIL Images don't have numpy() method)
  2. **Fundamental Impossibility**: PNG requires `\x89PNG` header, ELF requires `\x7fELF` header at start
  3. **Correct Understanding**: Most models correctly identified the theoretical impossibility
- **Bug Fix Applied**: Replaced `Image.open("image.png").numpy()` with `np.array(Image.open("image.png"))`
- **LLM Response Patterns**:
  - **Sophisticated Understanding**: Models correctly explained magic bytes, format conflicts
  - **Policy Refusals**: Some models (like GPT-OSS-120B) refused due to security concerns about polyglot files
  - **Creative Attempts**: Some tried impossible solutions (prepending ELF header) knowing they'd break PNG
- **Findings**: Test reveals both genuine impossibility and LLM security awareness, not capability limitations

### 🛠️ ImplementAssembly & ImplementAssemblyByExample
**Status**: Critical specification ambiguity fixed - Test design issue causing universal failures
- **Files**: `tests/implement_assembly_interpreter.py`, `tests/implement_assembly_interpreter_by_example.py`
- **Challenge**: Implement interpreter for custom assembly language with registers, memory, and control flow
- **Result**: Universal failure (0/10+ models) due to specification ambiguity
- **Root Cause**: **Critical Specification Gap** - Unclear argument dereferencing for register names
  - Test says arguments can be "register (e.g., R1) or constant (e.g., 10)" but doesn't specify dereferencing
  - Models implemented `STORE R3 R1` as `memory[int("R1")]` → ValueError: invalid literal for int()
  - Should be `memory[registers["R1"]]` (use R1's value as memory address)
- **Fixes Applied**:
  1. **Added Critical Clarification**: "When an argument is a register name like R1, use the VALUE stored in that register"
  2. **Fixed Typos**: "assmebly" → "assembly", " ite me" → "Write me"
  3. **Added Example**: "if R1 contains 5, then 'STORE R2 R1' stores R2's value into memory address 5"
- **Technical Analysis**:
  - **Test Programs**: `STORE R3 R1` should store R3's value at memory[R1's value (0,1,2...)]
  - **Expected Output**: `[1,4,9,16,25,36,49,64,81,100]` (squares in memory[0] through memory[19])
  - **Reference Implementation**: `tests/program_in_new_assembly.py` shows correct `lookup()` function
- **Findings**: Models showed sophisticated interpreter implementation skills but were sabotaged by ambiguous specification

### ✅ ProgramTB (PythonTraceback)
**Status**: Analysis complete - Legitimate test of advanced Python traceback knowledge
- **File**: `tests/python_traceback.py` 
- **Challenge**: Fix buggy Python code that attempts to print local variables from exception traceback frames
- **Result**: Universal failure (0/13+ models) due to advanced API knowledge requirements
- **Original Bug Issues**: Multiple critical Python API errors in the provided buggy code:
  1. **Wrong API Usage**: `traceback.extract_stack()` gets current stack, not exception traceback
  2. **Missing Attribute**: `frame.locals` is None by default (needs `capture_locals=True`)
  3. **Iteration Error**: Even with locals captured, need `frame.locals.items()` not `frame.locals`
  4. **Conceptual Error**: Stack extraction gives current location, not where exception occurred
- **Correct Solution Requires**:
  - Use `sys.exc_info()[2]` to get exception traceback
  - Walk traceback frames with `tb.tb_frame.f_locals`
  - Handle string conversion: `str(v)` for values
  - Filter out internal variables (names starting with `__`)
- **Model Response Patterns**:
  - **Sophisticated Understanding**: Models like QWQ-32B showed deep reasoning about traceback APIs
  - **Multiple Approaches Tried**: `capture_locals=True`, `extract_tb()`, frame walking
  - **Consistent Failure**: Even advanced attempts failed to produce required `x: 5` and `y: 6` output
- **Technical Distinction Tested**: Critical difference between:
  - **Current call stack** (`extract_stack()`) - where you are executing now
  - **Exception traceback** (`exc_info()[2]`) - where exception was raised
- **Findings**: Legitimate test of expert-level Python debugging knowledge; universal failure represents genuine LLM limitation in specialized API usage

### ✅ DateNewsHeadlines
**Status**: Analysis complete - Legitimate test of historical knowledge and temporal inference  
- **File**: `tests/date_news_headlines.py`
- **Challenge**: Identify the exact date when specific Hacker News headlines appeared on the front page
- **Result**: Universal failure (0/15+ models) - no model guessed the expected date `2020-04-05`
- **Expected Answer**: `2020-04-05` (April 5, 2020)
- **Key Headlines for Context**:
  - "We Made One Gram Of Remdesivir" - COVID-19 treatment research
  - "New Jersey needs COBOL programmers for unemployment" - COVID-19 unemployment system overwhelm  
  - Various tech articles (Svelte, Crafting Interpreters, Bose firmware, etc.)
- **Model Response Patterns**:
  - **Near Misses**: QWQ-32B guessed `2020-04-06` (off by 1 day)
  - **Contextual Reasoning**: Qwen3 provided detailed COVID-19/COBOL timeline analysis but guessed `2020-03-20`
  - **Reasonable Timeframe**: Most guesses in March-April 2020 (correct general period)
  - **Some Refusals**: Several models declined to guess without explicit date information
- **Sophisticated Inference Demonstrated**:
  - Models correctly linked COBOL story to COVID-19 unemployment surge
  - Recognized early pandemic timeframe (March-April 2020)
  - Showed understanding of historical context and causality
- **Test Classification**: **Historical Knowledge/Temporal Precision**
  - Not a coding problem with fixable infrastructure
  - Requires either specific memorized knowledge or extremely precise inference
  - Tests beyond typical programming/technical skills
- **Findings**: Legitimate but extremely challenging test of specific historical knowledge retention and precise temporal reasoning; universal failure represents genuine limitation in exact date recall/inference

### ✅ UnholyMatrix
**Status**: Analysis complete - Legitimate extremely challenging algorithmic test with specification issues fixed
- **File**: `tests/unholy_matrix.py`
- **Challenge**: Count 8×8 binary matrices with no isolated 1s using advanced dynamic programming
- **Result**: Near-universal failure (1/20+ models succeeded) - `openai_gpt-oss-120b-mxfp4_low` succeeded on step variant
- **Expected Answer**: `1121231537486377866` (19-digit number, ~6% of all possible matrices)
- **Root Cause Analysis**: 
  - **Specification Issues**: Ambiguous boundary conditions, unclear edge case handling, spelling errors
  - **Extreme Algorithmic Complexity**: Requires intersection of DP, bit manipulation, combinatorics
  - **Implementation Precision**: Zero tolerance for errors in complex state transitions
- **Fixes Applied**:
  1. **Fixed Spelling**: "matricies" → "matrices"
  2. **Data Type Precision**: "unsigned long" → "uint64_t" for platform consistency
  3. **Boundary Clarification**: "Positions outside matrix boundaries are considered to contain 0s"
  4. **Simplified Guidance**: Removed misleading algorithmic hints about "2^16 enumeration"
- **Technical Complexity**:
  - **Advanced DP Design**: 3D state space with row-by-row processing
  - **Bit Manipulation**: 64-bit matrix representation with neighbor detection
  - **Combinatorial Scale**: 2^64 total matrices, sophisticated counting required
- **Successful Model Analysis**: 
  - Used efficient bit manipulation and correct isolation detection
  - Implemented proper DP transitions with memoization
  - Demonstrated superior algorithmic integration capabilities
- **Findings**: Legitimate discriminator of expert-level algorithmic reasoning; universal failures represent genuine limitations in complex mathematical programming

### ✅ PythonChessGamePrefix  
**Status**: Test design issues fixed - Specification ambiguity and flawed checker resolved
- **File**: `tests/python_chess_game_prefix.py`
- **Challenge**: Generate all prefixes of a chess game in PGN notation using python-chess library
- **Result**: Universal failure (0/15+ models) due to test design flaws
- **Root Cause Analysis**:
  1. **Specification Ambiguity**: Unclear what "prefixes" means or expected output format
  2. **Flawed Checker Logic**: Arbitrary threshold (>10 occurrences) with brittle substring matching
  3. **Missing Requirements**: No explanation of prefix concept or output structure
  4. **Confusing Instructions**: "Do not give example code" misled models
- **Fixes Applied**:
  1. **Clarified Requirements**: "prints PGN notation for each prefix... from 1 to total moves, each on separate line"
  2. **Intelligent Checker**: Validates prefix property (increasing lengths), move progression, expected count
  3. **Removed Confusing Text**: Eliminated unclear instructions
  4. **Better Error Messages**: Specific feedback for different failure modes
- **Model Failure Patterns**:
  - **API Misuse**: Incorrect python-chess method calls, syntax errors
  - **Code Extraction Issues**: Generated conversational text instead of pure code
  - **Format Confusion**: Uncertainty about expected output structure
- **Enhanced Checker Logic**: 
  - Validates actual prefix relationships between consecutive lines
  - Checks reasonable output quantity and formatting
  - Provides diagnostic feedback for debugging
- **Findings**: Test design issues masked genuine API knowledge assessment; fixes enable proper evaluation of python-chess library skills

### 🛠️ FixWithPatch
**Status**: Hybrid solution implemented - Expert test preserved, guided variant added  
- **Files**: `tests/fix_with_patch.py`, `tests/fix_with_patch_guided.py`
- **Challenge**: Generate .patch file to fix regex capturing group bug in tokenizer
- **Result**: Near-universal failure (1/20+ models) - only `kimi-k2_q2kxl` succeeded on expert version
- **Success Analysis - kimi-k2_q2kxl**:
  - **Correct Bug Diagnosis**: Identified regex capturing group creates empty strings with `re.findall()`
  - **Working Solution**: Removed capturing groups AND added defensive filtering
  - **Perfect Patch Format**: Generated exact unified diff format with proper headers/context
  - **Technical Integration**: Combined regex knowledge + patch tool expertise + defensive programming
- **Root Cause of Failures**:
  1. **Hidden Requirements**: Bug not explained, must reverse-engineer problem
  2. **Multi-Domain Challenge**: Requires regex + patch format + Unix tools knowledge simultaneously  
  3. **Specification Ambiguity**: No examples of expected behavior or patch format
- **Hybrid Solution Implemented**:
  1. **Expert Test** (`fix_with_patch.py`): Original preserved as "Expert-level debugging challenge"
  2. **Guided Test** (`fix_with_patch_guided.py`): Clear problem statement with format guidance
- **Guided Version Improvements**:
  - **Clear Bug Description**: Shows current vs. expected tokenizer output
  - **Technical Education**: Explains regex capturing group behavior with `re.findall()`
  - **Format Template**: Provides unified diff format example
  - **Better Error Reporting**: Distinguishes patch format failures from logic errors
- **Findings**: Expert test legitimately identifies exceptional debugging capabilities; guided test enables systematic skill assessment for development

---

## Tests by Category

### 🧠 Explain & Code Understanding
- ❌ **ExplainPrime** (`explain_code_prime.py`) - Interpret minified JavaScript
- ❌ **ExplainPrime2** (`explain_code_prime2.py`) - Interpret obfuscated JavaScript  
- ❌ **ExplainBroadcast** (`explain_vbroadcast.py`) - Explain VPBROADCASTB instruction
- ❌ **CodeUnderstanding** (`basic_code_understanding.py`) - CTF-like C problem
- ❌ **WhatIsAutoModel** (`what_is_automodel.py`) - Interpret vague questions
- ❌ **WhatIsFloatFormat** (`what_is_formatfloat.py`) - Format f-strings with floats
- ❌ **WhatIsInv** (`what_is_inv.py`) - Identify Python tilde operator
- ❌ **WhatIsLPR** (`what_is_oraw.py`) - Knowledge of lpr commands
- ❌ **WhatIsStarStar** (`gitignore_anywhere.py`) - Understanding gitignore patterns
- ❌ **WhatIsBlockByOrb** (`what_is_blockbyorb.py`) - Domain-specific knowledge
- ❌ **WhatIsSliceStop** (`what_is_slice_stop.py`) - Python slicing behavior

### 🔧 Debugging & Fixing
- ❌ **FixDockerCuda** (`docker_cuda.py`) - Debug Docker CUDA errors
- ❌ **SimpleFix** (`fix_tokenizer.py`) - Fix tokenizer regex issues
- ❌ **FixNode** (`fix_node_error.py`) - Identify Node.js error messages
- ❌ **QuestionThreadedFix** (`fix_threading_issue.py`) - Fix threading issues
- ❌ **WhyBuggyPythonCountPar** (`debug_broken_code_parcount.py`) - Debug parallel wordcount
- ❌ **FixJSON** (`fix_json.py`) - Fix JSON formatting issues
- ❌ **FixTorchBackward** (`fix_torch_backward.py`) - Fix PyTorch gradient issues
- 🛠️ **FixWithPatch** (`fix_with_patch.py`) - Expert-level patch generation (hybrid solution)
- ❌ **FixAppendVsExtend** (`fix_append_vs_extend.py`) - Fix list operation bug
- ❌ **DebugInnerHTML** (`debug_innerhtml_eventlistener.py`) - Debug DOM issues

### 💻 Code Generation & Translation
- ❌ **PrintHello** (`print_hello.py`) - Basic Python hello world
- ❌ **PrintHelloSwift** (`print_hello_swift.py`) - Basic Swift hello world
- ❌ **ProgramRewriteCSimple** (`convert_to_c_simple.py`) - Simple Python to C
- ❌ **ProgramRewriteC** (`convert_to_c.py`) - Complex Python to C
- ❌ **PythonToCLoopUpdate** (`python_to_c_loop_update.py`) - Python to C with loops
- ❌ **TorchJnp** (`torch_to_jnp.py`) - PyTorch to JAX conversion
- ❌ **CallCFromPy** (`call_rust_from_python.py`) - Rust-Python interop
- ❌ **ProgramSqrt** (`program_sqrt.py`) - Implement sqrt function
- ❌ **CRC32** (`implement_crc32.py`) - Implement CRC-32 algorithm
- ❌ **ProgramPipesPython** (`program_pipes_python.py`) - Python pipes program
- ❌ **ProgramPipesCpp** (`program_pipes_cpp.py`) - C++ pipes program

### 🔄 Code Transformation & Optimization  
- ❌ **ShortenPyGet** (`shorten_python_if_missing.py`) - Shorten Python code
- ❌ **ShortenCFunction** (`shorten_c_function.py`) - Shorten C function
- ❌ **ShortenCFunctionHard** (`shorten_c_function_hard.py`) - Complex C shortening
- ❌ **VectorizeSmall** (`vectorize_small_update.py`) - Replace loops with vectorization
- ❌ **FastL2** (`faster_l2_diff.py`) - Optimize for speed/memory
- ❌ **DedentCodeFn** (`dedent_code_fn.py`) - Code formatting/indentation

### 🧮 Mathematical & Scientific
- ❌ **UnitConversion** (`unit_conversion_math.py`) - EE unit conversions
- ✅ **UnholyMatrix** (`unholy_matrix.py`) - Expert-level algorithmic challenge (fixed specification issues)
- ❌ **SimulateTorchGrad** (`simulate_torch_grad.py`) - Simulate gradient computation
- ❌ **JaxOneHot** (`jax_onehot.py`) - JAX one-hot encoding
- ❌ **NumbaLevenshtein** (`numba_levenshtein.py`) - Numba string distance
- ❌ **NumbaRref** (`numba_rref.py`) - Numba matrix row reduction
- ❌ **CRref** (`c_rref.py`) - C matrix row reduction  
- ❌ **StridedTrick** (`strided_trick.py`) - NumPy stride tricks
- ❌ **JnpNNBugfix** (`jnp_nn_bugfix.py`) - JAX neural network debugging

### 📊 Data Processing & Analysis
- ❌ **ExtractEmail** (`extract_emails.py`) - Extract/validate emails
- ❌ **ExtractRef** (`extract_references.py`) - Extract paper references
- ❌ **SumSomeData** (`vague_sum_data.py`) - Infer data to sum
- ❌ **DataExtraction** (`data_extraction_byyear.py`) - Extract data by year
- ❌ **DataTableProcessing** (`data_table_processing.py`) - Process data tables
- ❌ **DataTrainTimetable** (`data_train_timetable.py`) - Train schedule data
- ✅ **DateNewsHeadlines** (`date_news_headlines.py`) - News headline dating
- ❌ **VagueLoopFormat** (`vague_loop_format.py`) - Infer loop formatting

### 🗃️ Database & SQL
- ❌ **SqlMakeTable** (`make_sqlite_table.py`) - Create SQL tables  
- ❌ **SqlSubquery** (`fancy_sql_process.py`) - Complex SQL queries
- ❌ **SqlExplore** (`explore_sql_db.py`) - Database exploration

### 🌐 Web & Frontend
- ❌ **Flexbox** (`flexbox_webpage.py`) - HTML flexbox layouts
- ❌ **RecoverExpiredPage** (`save_expired_html.py`) - Recover HTML pages
- ❌ **WebGLTriangle** (`webgl_triangle.py`) - WebGL 3D programming

### 🛠️ System Administration & DevOps
- ❌ **GitMerge** (`git_merge.py`) - Git merge operations
- ❌ **BasicGitSetup** (`basic_git_setup.py`) - Basic git configuration
- ❌ **BashFindDontContain** (`bash_find_dont_contain.py`) - Bash file search
- ❌ **BashListSize** (`bash_list_files_by_size_mod_ten.py`) - Bash file sorting
- ❌ **BashRenamer** (`bash_renamer.py`) - Bash file renaming
- 🛠️ **ChangeFiletype** (`change_filetype.py`) - File type conversion (MakePNGToELF)

### 🔍 Reverse Engineering & Decompilation
- ❌ **Disas1** (`decompile_py_simple.py`) - Simple Python bytecode
- ❌ **DisasPrimes** (`decompile_py_mid.py`) - Medium Python bytecode  
- ❌ **DisasRref** (`decompile_py_rref.py`) - Complex Python bytecode

### 🧠 Knowledge & Domain Expertise
- ❌ **LlamaKnowledge** (`knowledge_llama.py`) - LLAMA-2 architecture knowledge
- ❌ **DB9** (`db9_pinout.py`) - Hardware connector knowledge
- ❌ **GetVocab** (`tokenizer_vocab.py`) - Tokenizer vocabulary access
- ❌ **FreeCADCircle** (`freecad_construction.py`) - CAD software usage

### 🎮 Games & Interactive
- ❌ **TwentyQuestionsBook** (`play_20_questions.py`) - Interactive game playing
- ✅ **ChessGamePrefix** (`python_chess_game_prefix.py`) - Python-chess API usage (fixed specification)
- ❌ **EmojiMovies** (`emoji_movies.py`) - Emoji-based movie guessing

### 📝 Text Processing & Parsing
- ❌ **Regex** (`regex_remove_5_words.py`) - Regex text processing
- ❌ **SimpleBNF** (`easy_parser_generator.py`) - BNF grammar parsing
- ❌ **MakeTreeEasy** (`make_tree_from_text.py`) - Tree structure creation
- ❌ **MakeJSON** (`make_json.py`) - JSON generation
- ❌ **DoUUDecode** (`do_uudecode.py`) - UUEncoding/decoding
- ✅ **ProgramTB (PythonTraceback)** (`python_traceback.py`) - Error traceback analysis

### 🎨 Graphics & Media
- ❌ **ImgResize** (`py_image_resize.py`) - Image resizing operations
- ❌ **DrawFlagBMP** (`draw_flag_bmp.py`) - Bitmap flag drawing
- ❌ **PythonJPEG** (`python_jpeg.py`) - JPEG image processing

### 📚 Documentation & LaTeX
- ❌ **LatexNewline** (`latex_protect.py`) - LaTeX formatting fixes
- ❌ **LatexRedef** (`latex_redef.py`) - LaTeX redefinition issues
- ❌ **HallucinateReference** (`hallucinate_reference.py`) - Generate references

### 🔬 Specialized Libraries & Frameworks
- ❌ **UPythonMQTT** (`upython_mqtt.py`) - MicroPython MQTT
- ❌ **WhisperMerge** (`whisper_merge.py`) - Whisper model operations
- ❌ **EmacsLispSilence** (`emacs_lisp_silence_cmd.py`) - Emacs Lisp scripting
- ❌ **RustWordCount** (`rust_word_count.py`) - Rust programming
- ❌ **RustParallelWordcount** (`rust_parallel_wordcount.py`) - Rust concurrency
- ❌ **PythonParallelWordcount** (`python_parallel_wordcount.py`) - Python multiprocessing
- ❌ **RewriteMacCrypto** (`rewrite_mac_crypto.py`) - Cryptography operations

### 🧩 Advanced Programming Concepts
- ❌ **NumpyIx** (`numpy_ix.py`) - NumPy advanced indexing
- ❌ **NumpyAdvancedIndex** (`numpy_advanced_index.py`) - Complex NumPy indexing
- ❌ **GOLRLEDecode** (`gol_rle_decode.py`) - Game of Life RLE decoding
- ❌ **CWeirdExpression** (`c_weird_expression.py`) - Complex C expressions
- ❌ **WhichPackageSbox** (`which_package_sbox.py`) - Package identification
- 🛠️ **ImplementAssembly** (`implement_assembly_interpreter.py`) - Assembly language interpreter
- 🛠️ **ImplementAssemblyByExample** (`implement_assembly_interpreter_by_example.py`) - Assembly interpreter from examples

### 🏗️ File & Project Management
- ❌ **Make16FilesEasy** (`merge_into_16.py`) - File merging operations
- ❌ **NewAssemblyPrimeNumbers** (`program_in_new_assembly.py`) - Assembly programming
- ❌ **ProgramStringSlice** (`generate_string_moves.py`) - String manipulation

### 🍰 Non-Technical (Control Tests)
- ❌ **MissingStep** (`baking_help.py`) - Recipe ingredient identification

---

## Investigation Methodology

For each test investigation, document:

1. **Test Objective**: What capability is being tested?
2. **Success Rate**: How many models succeeded? (X/9 format)
3. **Failure Patterns**: Common types of failures across models
4. **Root Cause Analysis**: Test design issues vs. genuine LLM limitations
5. **Sample Analysis**: Examine specific model outputs for insights
6. **Recommendations**: Test improvements or findings about LLM capabilities

## Priority Investigation Queue

### High Priority (Fundamental Capabilities)
1. **PrintHello** - Basic code generation baseline
2. **ExplainPrime** - Code comprehension baseline  
3. **SimpleFix** - Basic debugging capabilities
4. **UnitConversion** - Mathematical reasoning
5. **Regex** - Text processing fundamentals

### Medium Priority (Domain-Specific Skills)  
1. **TorchJnp** - Framework translation
2. **SqlMakeTable** - Database operations
3. **Flexbox** - Web development
4. **GitMerge** - Version control
5. **CRC32** - Algorithm implementation

### Lower Priority (Specialized/Complex)
1. Reverse engineering tests (Disas*)
2. Advanced mathematical tests (NumbaRref, etc.)
3. Hardware knowledge tests (DB9)
4. Game implementations
5. Specialized library tests

---

## Global Patterns to Investigate

1. **Output Format Failures**: How many tests fail due to parsing issues?
2. **Domain Knowledge Gaps**: Which knowledge areas show systematic failures?
3. **Code Generation vs. Understanding**: Different failure rates between tasks?
4. **Language-Specific Patterns**: Do certain programming languages show more failures?
5. **Complexity Correlation**: Do more complex tests show predictably worse performance?

---

**Last Updated**: 2025-01-23  
**Tests Investigated**: 13/150+ (including 2 ImplementAssembly variants + hybrid FixWithPatch solution)  
**Completion Rate**: ~8.7%

## Session Summary (Latest Investigation)
**Completed Today**:
- ✅ **UnholyMatrix**: Fixed specification ambiguities while preserving extreme algorithmic challenge
- ✅ **PythonChessGamePrefix**: Fixed test design flaws enabling proper API assessment  
- 🛠️ **FixWithPatch**: Implemented hybrid solution (expert + guided variants) for comprehensive evaluation
- 📝 Updated test categorizations and comprehensive documentation of findings

**Key Insights Gained**:
1. **Specification Quality Critical**: Minor ambiguities can cause systematic universal failures
2. **Expert vs. Guided Assessment**: Different test designs serve different evaluation purposes
3. **Technical Integration Skills**: Success often requires combining multiple technical domains
4. **Defensive Programming Value**: Over-engineered solutions sometimes work better than minimal fixes