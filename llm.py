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

import os
import time
import logging
import importlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from llms.openai_model import OpenAIModel
from llm_cache import LLMCache

BACKOFF_TIMES = [10, 20, 30, 60, 90, 120, 300]
DEFAULT_TEST_MODEL = os.getenv("BENCHMARK_DEFAULT_MODEL", "o1-mini")
DEFAULT_EVAL_MODEL = os.getenv("BENCHMARK_DEFAULT_EVAL_MODEL", "openai_eval_qwen/qwen3-30b-a3b")
DEFAULT_VISION_EVAL_MODEL = os.getenv("BENCHMARK_DEFAULT_VISION_MODEL")
OPENAI_CLIENT_TIMEOUT_SECONDS = 3600
STREAM_CHUNK_TIMEOUT_SECONDS = 3600
REQUEST_OVERALL_TIMEOUT_SECONDS = 3600


def _load_provider_class(module_name, class_name, dependency_name=None):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        package_hint = dependency_name or module_name.rsplit(".", 1)[-1]
        raise RuntimeError(
            f"Provider '{module_name}' requires optional dependency '{package_hint}'. "
            f"Install the matching extra or dependency group before using this model."
        ) from exc
    return getattr(module, class_name)


def should_retry_request_error(error):
    error_type = type(error).__name__
    error_str = str(error)
    if "APIConnectionError" in error_type or "ConnectionError" in error_type:
        return True
    if "RemoteProtocolError" in error_str and "peer closed connection" in error_str:
        return True
    if "ReadTimeout" in error_str or "timed out" in error_str:
        return True
    return False


def _build_model(name):
    if name.startswith("openai_eval_"):
        return OpenAIModel(name.removeprefix("openai_eval_"), config_key="openai_eval")
    if name.startswith("openai_local_"):
        return OpenAIModel(name.removeprefix("openai_local_"), config_key="openai_local")
    if name.startswith("openai_"):
        return OpenAIModel(name.removeprefix("openai_"))
    if "gpt" in name or name.startswith("o1") or name.startswith("o3") or name.startswith("o4"):
        return OpenAIModel(name)
    if name.startswith("ollama_"):
        OllamaModel = _load_provider_class("llms.ollama_model", "OllamaModel")
        return OllamaModel(name.removeprefix("ollama_"))
    if "mistral" in name:
        MistralModel = _load_provider_class("llms.mistral_model", "MistralModel")
        return MistralModel(name)
    if "bison" in name or "gemini" in name:
        VertexAIModel = _load_provider_class("llms.vertexai_model", "VertexAIModel", "google-cloud-aiplatform")
        return VertexAIModel(name)
    if "claude" in name:
        AnthropicModel = _load_provider_class("llms.anthropic_model", "AnthropicModel", "anthropic")
        return AnthropicModel(name)
    if "moonshot" in name:
        MoonshotAIModel = _load_provider_class("llms.moonshot_model", "MoonshotAIModel")
        return MoonshotAIModel(name)
    if "command" in name:
        CohereModel = _load_provider_class("llms.cohere_model", "CohereModel", "cohere")
        return CohereModel(name)
    if "llama3" in name or "mixtral" in name or "gemma" in name:
        GroqModel = _load_provider_class("llms.groq_model", "GroqModel", "groq")
        return GroqModel(name)
    raise RuntimeError(f"Unknown model type for name: {name}")


def _uses_openai_compatible_api(name):
    return (
        name.startswith("openai_eval_")
        or name.startswith("openai_local_")
        or name.startswith("openai_")
        or "gpt" in name
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    )

class LLM:
    def __init__(self, name="gpt-3.5-turbo", use_cache=True, override_hparams=None):
        self.original_name = name  # Store original name for cache consistency
        self.name = name
        logging.debug(f"Initializing LLM with name: {name}")
        self.model = _build_model(name)
        self.model.hparams.update(override_hparams or {})

        # Clean up unsupported hparams *after* overrides are applied
        if _uses_openai_compatible_api(name):
            if 'repeat_penalty' in self.model.hparams:
                del self.model.hparams['repeat_penalty']
                logging.debug("Removed 'repeat_penalty' from hparams, reason: unsupported by OpenAIModel")
            if 'top_k' in self.model.hparams:
                del self.model.hparams['top_k']
                logging.debug("Removed 'top_k' from hparams, reason: unsupported by OpenAIModel")

        # Update name based on actual model name if prefix was used
        self.name = self.model.name

        # Initialize cache with clean interface
        self.use_cache = use_cache
        if use_cache:
            self.cache = LLMCache(self.original_name)
        else:
            self.cache = None


    def _log_stream_progress(self, chunk_count, start_time, log_accumulator, final_log=False):
        """Helper method to log stream progress efficiently."""
        # SPARC: Simple logging - only log summary metrics, not full content
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            current_time = time.monotonic()
            elapsed_time = int(current_time - start_time)
            content_length = len(log_accumulator) if log_accumulator else 0
            if final_log:
                logging.debug(f"Stream completed: {chunk_count} chunks, {elapsed_time}s, {content_length} chars")
            else:
                logging.debug(f"Stream progress: ~{chunk_count} chunks, {elapsed_time}s, {content_length} chars")

    def __call__(self, conversation, add_image=None, max_tokens=None, skip_cache=False, json=False):
        if type(conversation) == str:
            conversation = [conversation]

        # Check cache if enabled
        if self.use_cache and not skip_cache and self.cache:
            import time
            import os
            cache_start = time.time()
            pid = os.getpid()
            logging.debug(f"LLM_TRACE[{pid}]: Starting cache operations for {self.name}")
            
            # Generate cache key with all parameters that affect response
            hparams = self.model.hparams if hasattr(self.model, 'hparams') else None
            key_start = time.time()
            cache_key = self.cache.get_cache_key(
                conversation, 
                max_tokens=max_tokens, 
                json=json, 
                add_image=add_image,
                hparams=hparams
            )
            logging.debug(f"LLM_TRACE[{pid}]: Cache key generated in {time.time() - key_start:.3f}s")
            
            # Try cache lookup
            lookup_start = time.time()
            logging.debug(f"LLM_TRACE[{pid}]: About to call cache.get()")
            cached_response = self.cache.get(cache_key)
            logging.debug(f"LLM_TRACE[{pid}]: Cache lookup completed in {time.time() - lookup_start:.3f}s, found: {cached_response is not None}")
            
            if cached_response:
                logging.debug(f"LLM_TRACE[{pid}]: Returning cached response, total cache time: {time.time() - cache_start:.3f}s")
                return cached_response

        import os
        pid = os.getpid()
        log_prompt_snippet = repr(conversation[0])[:200] + ("..." if len(conversation[0]) > 200 else "")
        logging.info(f"LLM_TRACE[{pid}]: {self.name} CACHE MISS. Prompt starts: {log_prompt_snippet}")
        logging.debug(f"LLM_TRACE[{pid}]: Proceeding to model API call")

        response = "Model API request failed"

        def process_request_and_stream(json_arg, overall_timeout=REQUEST_OVERALL_TIMEOUT_SECONDS):
            stream = None
            full_response_content = ""
            log_accumulator = "" # Accumulator for logging
            log_chunk_interval = 500 # Log every N chunks
            chunk_count = 0
            start_time = time.monotonic()
            last_chunk_time = start_time
            chunk_timeout = STREAM_CHUNK_TIMEOUT_SECONDS
            
            stream = None  # Initialize to prevent UnboundLocalError in finally block
            try:
                stream = self.model.make_request(
                    conversation,
                    add_image=add_image,
                    max_tokens=max_tokens,
                    json=json_arg,
                    stream=True
                )
                logging.debug(f"Stream opened for {self.name}. Timeout: {overall_timeout}s, chunk timeout: {chunk_timeout}s.")
                
                # Log connection state for debugging
                import psutil
                import os
                try:
                    current_process = psutil.Process(os.getpid())
                    connections = [conn for conn in current_process.connections() if conn.status == psutil.CONN_ESTABLISHED]
                    logging.debug(f"CONN_MONITOR: {len(connections)} active connections after stream open")
                except Exception as conn_e:
                    logging.debug(f"CONN_MONITOR: Could not check connections: {conn_e}")

                received_chunks = 0
                content_chunks = 0
                
                for chunk in stream:
                    received_chunks += 1
                    current_time = time.monotonic()
                    
                    # Check overall timeout
                    if current_time - start_time > overall_timeout:
                        logging.warning(f"Stream processing exceeded overall timeout ({overall_timeout}s) for {self.name}.")
                        raise TimeoutError("Stream processing exceeded overall timeout.")
                    
                    # Check chunk timeout (time since last chunk)
                    if current_time - last_chunk_time > chunk_timeout:
                        logging.warning(f"Stream chunk timeout ({chunk_timeout}s) exceeded for {self.name}.")
                        raise TimeoutError("Stream chunk timeout exceeded - stream appears dead.")

                    content_part = ""
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        # Standard content field
                        if delta.content:
                            content_part = delta.content
                        # Thinking/reasoning models may use reasoning_content
                        elif getattr(delta, 'reasoning_content', None):
                            # Reasoning tokens — don't include in response but count as activity
                            last_chunk_time = current_time
                            received_chunks += 0  # already counted above
                            continue

                    if content_part:
                        full_response_content += content_part
                        log_accumulator += content_part
                        chunk_count += 1
                        content_chunks += 1
                        last_chunk_time = current_time

                        if chunk_count % (log_chunk_interval * 2) == 0:
                            self._log_stream_progress(chunk_count, start_time, log_accumulator)
                            log_accumulator = ""
                    
                    # Check for completion markers that indicate stream should end
                    if chunk.choices and chunk.choices[0].finish_reason:
                        logging.debug(f"Stream completion detected: {chunk.choices[0].finish_reason}")
                        break

                # Log any remaining accumulated content after the loop
                self._log_stream_progress(chunk_count, start_time, log_accumulator, final_log=True)

                logging.debug(f"Stream finished for {self.name} after {int(time.monotonic() - start_time)}s. Chunks: {chunk_count}, Length: {len(full_response_content)}")
                
                # Log connection state after completion
                try:
                    current_process = psutil.Process(os.getpid())
                    connections = [conn for conn in current_process.connections() if conn.status == psutil.CONN_ESTABLISHED]
                    logging.debug(f"CONN_MONITOR: {len(connections)} active connections after stream completion")
                except Exception as conn_e:
                    logging.debug(f"CONN_MONITOR: Could not check connections after completion: {conn_e}")
                
                if not full_response_content.strip():
                    logging.warning(f"Stream completed for {self.name} but produced empty response. Received {received_chunks} chunks, {content_chunks} with content")
                
                return full_response_content

            except TimeoutError:
                logging.error(f"Stream timeout for {self.name} after {int(time.monotonic() - start_time)}s. Chunks received: {chunk_count}")
                
                # Log connection state on timeout
                try:
                    current_process = psutil.Process(os.getpid())
                    connections = [conn for conn in current_process.connections() if conn.status == psutil.CONN_ESTABLISHED]
                    logging.error(f"CONN_MONITOR: {len(connections)} active connections during TIMEOUT")
                except Exception:
                    pass
                    
                return ""  # Return empty string instead of raising, to trigger empty response logging
            except Exception as e:
                error_str = str(e)
                # Re-raise connection/transient errors so the backoff retry loop can handle them
                if should_retry_request_error(e):
                    logging.warning(f"Retryable request error for {self.name}: {e}")
                    raise

                logging.error(f"Error processing stream for {self.name}: {e}", exc_info=True)

                # Log connection state on error
                try:
                    current_process = psutil.Process(os.getpid())
                    connections = [conn for conn in current_process.connections() if conn.status == psutil.CONN_ESTABLISHED]
                    logging.error(f"CONN_MONITOR: {len(connections)} active connections during ERROR")
                except Exception:
                    pass

                return ""  # Return empty string for non-retryable errors
            finally:
                # Ensure stream is properly closed to prevent connection leaks
                if stream and hasattr(stream, 'close'):
                    try:
                        stream.close()
                        logging.debug(f"Stream closed for {self.name}")
                    except Exception as close_e:
                        logging.warning(f"Failed to close stream for {self.name}: {close_e}")
                        
                # Final connection state check
                try:
                    current_process = psutil.Process(os.getpid())
                    connections = [conn for conn in current_process.connections() if conn.status == psutil.CONN_ESTABLISHED]
                    logging.debug(f"CONN_MONITOR: {len(connections)} active connections after stream cleanup")
                except Exception:
                    pass

        for i in range(len(BACKOFF_TIMES)):
            logging.debug(f"Attempt {i+1}/{len(BACKOFF_TIMES)} for model {self.name}")
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        process_request_and_stream,
                        json_arg=json,
                        overall_timeout=REQUEST_OVERALL_TIMEOUT_SECONDS,
                    )
                    response = future.result()
                    if response and response.strip():
                        break  # Got a valid response
                    logging.warning(f"Empty response on attempt {i+1} for model {self.name}, will retry")
                    response = "Model API request failed"

            except TimeoutError:
                logging.warning(f"Caught timeout on attempt {i+1} for model {self.name}")
                response = "Model API request failed due to timeout"
            except Exception as e:
                logging.error(f"RUN FAILED on attempt {i+1} for model {self.name}: {e}", exc_info=False)
                response = f"Model API request failed: {type(e).__name__}"

            if i < len(BACKOFF_TIMES) - 1:  # Don't sleep after the last attempt
                wait_time = BACKOFF_TIMES[i]
                logging.info(f"Request failed on attempt {i+1}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        # Save successful response to cache
        if self.use_cache and self.cache and "Model API request failed" not in response:
            # Regenerate cache key (same logic as above)
            hparams = self.model.hparams if hasattr(self.model, 'hparams') else None
            cache_key = self.cache.get_cache_key(
                conversation, 
                max_tokens=max_tokens, 
                json=json, 
                add_image=add_image,
                hparams=hparams
            )
            self.cache.put(cache_key, response)

        return response

llm = None
eval_llm = None
vision_eval_llm = None


def ensure_default_models(test_model_name=None):
    global llm, eval_llm, vision_eval_llm

    if test_model_name and (llm is None or llm.original_name != test_model_name):
        llm = LLM(test_model_name)
    elif llm is None and DEFAULT_TEST_MODEL:
        llm = LLM(DEFAULT_TEST_MODEL)

    if eval_llm is None and DEFAULT_EVAL_MODEL:
        eval_llm = LLM(DEFAULT_EVAL_MODEL)

    if vision_eval_llm is None and DEFAULT_VISION_EVAL_MODEL:
        vision_eval_llm = LLM(DEFAULT_VISION_EVAL_MODEL)

    return llm, eval_llm, vision_eval_llm
