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

from io import BytesIO
import os
import base64
import requests
import json
import pickle
import time
import logging
import fcntl
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from llms.openai_model import OpenAIModel
from llms.anthropic_model import AnthropicModel
from llms.mistral_model import MistralModel
from llms.vertexai_model import VertexAIModel
from llms.cohere_model import CohereModel
from llms.moonshot_model import MoonshotAIModel
from llms.groq_model import GroqModel
from llms.ollama_model import OllamaModel

class LLM:
    def __init__(self, name="gpt-3.5-turbo", use_cache=True, override_hparams={}):
        self.original_name = name  # Store original name for cache consistency
        self.name = name
        logging.debug(f"Initializing LLM with name: {name}")
        if 'openai_eval_' in name:
            model = name.replace('openai_eval_', '')
            self.model = OpenAIModel(model, config_key='openai_eval')
        elif 'openai_' in name:
            model = name.replace('openai_', '')
            self.model = OpenAIModel(model)
        elif 'gpt' in name or name.startswith('o1'):
            self.model = OpenAIModel(name)
        # elif 'llama' in name:
        #     self.model = LLAMAModel(name)
        elif name.startswith('ollama_'):
            model = name.replace('ollama_', '')
            self.model = OllamaModel(model)
        elif 'mistral' in name:
            self.model = MistralModel(name)
        elif 'bison' in name or 'gemini' in name:
            self.model = VertexAIModel(name)
        #elif 'gemini' in name:
        #    self.model = GeminiModel(name)
        elif 'claude' in name:
            self.model = AnthropicModel(name)
        elif 'moonshot' in name:
            self.model = MoonshotAIModel(name)            
        elif 'command' in name:
            self.model = CohereModel(name)
        elif 'llama3' in name or 'mixtral' in name or 'gemma' in name:
            self.model = GroqModel(name)
        else:
            raise RuntimeError(f"Unknown model type for name: {name}")
        self.model.hparams.update(override_hparams)

        # Clean up unsupported hparams *after* overrides are applied
        if isinstance(self.model, OpenAIModel):
            if 'repeat_penalty' in self.model.hparams:
                del self.model.hparams['repeat_penalty']
                logging.debug("Removed 'repeat_penalty' from hparams, reason: unsupported by OpenAIModel")
            if 'top_k' in self.model.hparams:
                del self.model.hparams['top_k']
                logging.debug("Removed 'top_k' from hparams, reason: unsupported by OpenAIModel")

        # Update name based on actual model name if prefix was used
        self.name = self.model.name

        # Use original name for cache to ensure consistency and avoid collisions
        self.cache_filename = f"tmp/cache-{self.original_name.replace('/', '_').replace(':', '_')}.p"
        
        self.use_cache = use_cache
        self.last_cache_mtime = 0  # Track last modification time for intelligent reloading
        if use_cache:
            try:
                if not os.path.exists("tmp"):
                    os.makedirs("tmp", exist_ok=True)
                self.cache = self._load_cache_safely()
                self._update_cache_mtime()
            except Exception as e:
                logging.debug(f"Cache initialization failed for {self.original_name}: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def _update_cache_mtime(self):
        """Update cached modification time. Simple helper following SPARC principles."""
        try:
            if os.path.exists(self.cache_filename):
                self.last_cache_mtime = os.path.getmtime(self.cache_filename)
        except OSError:
            self.last_cache_mtime = 0
    
    def _should_reload_cache(self):
        """Check if cache should be reloaded based on file modification time."""
        try:
            if not os.path.exists(self.cache_filename):
                return False
            current_mtime = os.path.getmtime(self.cache_filename)
            return current_mtime > self.last_cache_mtime
        except OSError:
            return False

    def _load_cache_safely(self):
        """Load cache with file locking for multiprocessing safety."""
        try:
            with open(self.cache_filename, "rb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                cache = pickle.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
                logging.debug(f"Loaded {len(cache)} cache entries for {self.original_name}")
                return cache
        except (FileNotFoundError, pickle.PickleError, OSError):
            logging.debug(f"Cache file not found or corrupted for {self.original_name}, starting fresh")
            return {}

    def _save_cache_safely(self, cache):
        """Save cache with file locking and atomic write for multiprocessing safety."""
        try:
            # Use atomic write pattern: write to temp file, then rename
            temp_file = self.cache_filename + ".tmp"
            with open(temp_file, "wb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                pickle.dump(cache, f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
            
            # Atomic rename ensures other processes don't see partial writes
            os.rename(temp_file, self.cache_filename)
            self._update_cache_mtime()  # Update our tracking after successful save
            logging.debug(f"Saved {len(cache)} cache entries for {self.original_name}")
            
        except Exception as e:
            logging.error(f"Failed to save cache for {self.original_name}: {e}")
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

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

        # Create comprehensive cache key including all parameters that affect response
        cache_key_components = [tuple(conversation)]
        
        if add_image is not None:
            cache_key_components.append(add_image.tobytes())
            
        if max_tokens is not None:
            cache_key_components.append(('max_tokens', max_tokens))
            
        if json:
            cache_key_components.append(('json_mode', True))
            
        # Include model hyperparameters that affect response
        if hasattr(self.model, 'hparams') and self.model.hparams:
            hparams_tuple = tuple(sorted(self.model.hparams.items()))
            cache_key_components.append(('hparams', hparams_tuple))
            
        cache_key = tuple(cache_key_components)

        # Intelligently reload cache only when file has been modified by other processes
        if self.use_cache and not skip_cache and self._should_reload_cache():
            self.cache = self._load_cache_safely()
            self._update_cache_mtime()
            
        if cache_key in self.cache and not skip_cache and self.use_cache:
            logging.info(f"{self.name} GETCACHE")
            if len(self.cache[cache_key]) > 0:
                return self.cache[cache_key]
            else:
                logging.info("Empty cache hit")

        log_prompt_snippet = repr(conversation[0])[:200] + ("..." if len(conversation[0]) > 200 else "")
        logging.info(f"{self.name} CACHE MISS. Prompt starts: {log_prompt_snippet}")

        response = "Model API request failed"

        def process_request_and_stream(json_arg, overall_timeout=300):
            stream = None
            full_response_content = ""
            log_accumulator = "" # Accumulator for logging
            log_chunk_interval = 500 # Log every N chunks
            chunk_count = 0
            start_time = time.monotonic()
            last_chunk_time = start_time
            chunk_timeout = 30  # Max time between chunks before considering stream dead
            
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

                for chunk in stream:
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
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content_part = chunk.choices[0].delta.content
                        full_response_content += content_part
                        log_accumulator += content_part # Add to accumulator
                        chunk_count += 1
                        last_chunk_time = current_time  # Update last chunk time

                        # Log progress periodically (reduced frequency for performance)
                        if chunk_count % (log_chunk_interval * 2) == 0:  # Log every 1000 chunks instead of 500
                            self._log_stream_progress(chunk_count, start_time, log_accumulator)
                            log_accumulator = "" # Reset accumulator
                    
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
                    
                raise
            except Exception as e:
                logging.error(f"Error processing stream for {self.name}: {e}", exc_info=True)
                
                # Log connection state on error
                try:
                    current_process = psutil.Process(os.getpid())
                    connections = [conn for conn in current_process.connections() if conn.status == psutil.CONN_ESTABLISHED]
                    logging.error(f"CONN_MONITOR: {len(connections)} active connections during ERROR")
                except Exception:
                    pass
                    
                raise
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

        backoff_times = [10, 20, 30, 60, 90, 120, 300]  # New backoff times
        for i in range(len(backoff_times)):
            logging.debug(f"Attempt {i+1}/{len(backoff_times)} for model {self.name}")
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(process_request_and_stream, json_arg=json, overall_timeout=3600)
                    response = future.result()
                    break

            except TimeoutError: 
                logging.warning(f"Caught timeout on attempt {i+1} for model {self.name}")
                response = "Model API request failed due to timeout"
            except Exception as e:
                logging.error(f"RUN FAILED on attempt {i+1} for model {self.name}: {e}", exc_info=False) 
                response = f"Model API request failed: {type(e).__name__}" 
            
            if i < len(backoff_times) - 1:  # Don't sleep after the last attempt
                wait_time = backoff_times[i]
                logging.info(f"Request failed on attempt {i+1}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        if self.use_cache and "Model API request failed" not in response:
            # Update local cache and save to disk atomically
            self.cache[cache_key] = response
            self._save_cache_safely(self.cache)

        return response

#llm = LLM("command")
#llm = """LLM("gpt-3.5-turbo")"""

# llm = LLM("ollama_ollama_deepseek-coder-v2")
#llm = LLM("gpt-4-turbo-2024-04-09")
#llm = LLM("gemini-1.5-pro-preview-0409")
llm = LLM("o1-mini")

eval_llm = LLM("openai_eval_qwen2.5-coder:latest")
#eval_llm = LLM("gpt-3.5-turbo", override_hparams={'temperature': 0.1})

# Set to None to skip vision tests, or configure a vision-capable model
vision_eval_llm = None
#vision_eval_llm = LLM("openai_gpt-4-vision", override_hparams={'temperature': 0.1})
