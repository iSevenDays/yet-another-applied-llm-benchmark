#!/usr/bin/env python3
"""
LLM Response Cache - SPARC-compliant implementation.

Simple, focused cache for LLM responses with:
- Deterministic cache key generation
- Thread-safe file operations
- Clean error handling
"""
import os
import pickle
import fcntl
import logging
from typing import Dict, Tuple, Optional


class LLMCache:
    """Simple, reliable cache for LLM responses following SPARC principles."""
    
    def __init__(self, model_name: str, cache_dir: str = "tmp"):
        """Initialize cache with model-specific storage."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Create safe filename from model name
        safe_name = model_name.replace('/', '_').replace(':', '_')
        self.cache_file = os.path.join(cache_dir, f"cache-{safe_name}.p")
        
        # Internal state
        self._cache: Dict[Tuple, str] = {}
        self._file_mtime = 0
        
        # Initialize
        self._ensure_cache_dir()
        self._load_cache()
    
    def get_cache_key(self, conversation: list, **kwargs) -> Tuple:
        """Generate deterministic cache key from all parameters that affect LLM response."""
        # Input validation
        if not isinstance(conversation, (list, tuple)):
            raise TypeError("conversation must be a list or tuple")
        
        # Normalize conversation by stripping leading/trailing whitespace
        # This ensures cache hits regardless of prompt formatting variations
        normalized_conversation = [str(msg).strip() if msg is not None else "" for msg in conversation]
        key_parts = [tuple(normalized_conversation)]
        
        # Add image data if present
        if kwargs.get('add_image') is not None:
            key_parts.append(('image', kwargs['add_image'].tobytes()))
        
        # Add JSON mode flag
        if kwargs.get('json'):
            key_parts.append(('json_mode', True))
        
        # Normalize parameters into hparams for consistent cache keys
        # This ensures backward compatibility with legacy cache entries
        normalized_hparams = {}
        
        # Start with existing hparams if provided
        if kwargs.get('hparams'):
            normalized_hparams.update(kwargs['hparams'])
        
        # Merge explicit max_tokens into hparams (explicit takes precedence)
        if kwargs.get('max_tokens') is not None:
            normalized_hparams['max_tokens'] = kwargs['max_tokens']
        
        # Only add hparams component if we have parameters
        if normalized_hparams:
            hparams_tuple = tuple(sorted(normalized_hparams.items()))
            key_parts.append(('hparams', hparams_tuple))
        
        cache_key = tuple(key_parts)
        
        # Debug logging for troubleshooting
        conv_len = len(conversation[0]) if conversation and conversation[0] else 0
        logging.debug(f"Cache key for {self.model_name}: {len(cache_key)} components, conv_len={conv_len}")
        
        return cache_key
    
    def get(self, cache_key: Tuple) -> Optional[str]:
        """Retrieve cached response if available with backward compatibility."""
        self._reload_if_changed()
        
        # Try direct lookup first (normalized key)
        response = self._cache.get(cache_key)
        if response:
            logging.info(f"Cache HIT for {self.model_name}")
            return response
        
        # For backward compatibility: try to find matching entries with different whitespace
        # This handles legacy cache entries that may have been stored with different formatting
        if len(cache_key) > 0 and isinstance(cache_key[0], tuple):
            target_conversation = cache_key[0]
            
            # Extract target hparams correctly from cache key structure
            target_hparams = None
            for part in cache_key[1:]:
                if isinstance(part, tuple) and len(part) == 2 and part[0] == 'hparams':
                    target_hparams = part
                    break
            
            # Limit backward compatibility search to prevent O(N) performance issues
            search_limit = min(100, len(self._cache))  # Only search first 100 entries
            searched = 0
            
            for existing_key, cached_response in self._cache.items():
                if searched >= search_limit:
                    break
                searched += 1
                
                if len(existing_key) > 0 and isinstance(existing_key[0], tuple):
                    # Compare normalized conversations
                    existing_conversation = existing_key[0]
                    
                    # Extract existing hparams correctly
                    existing_hparams = None
                    for part in existing_key[1:]:
                        if isinstance(part, tuple) and len(part) == 2 and part[0] == 'hparams':
                            existing_hparams = part
                            break
                    
                    # Check if conversations match when normalized
                    if (len(target_conversation) == len(existing_conversation) and
                        all(target.strip() == existing.strip() for target, existing in zip(target_conversation, existing_conversation)) and
                        target_hparams == existing_hparams):
                        
                        logging.info(f"Cache HIT (legacy format) for {self.model_name}")
                        return cached_response
        
        logging.info(f"Cache MISS for {self.model_name}")
        return None
    
    def put(self, cache_key: Tuple, response: str):
        """Store response in cache with immediate persistence."""
        if not response or response.strip() == "":
            logging.warning(f"Not caching empty response for {self.model_name}")
            return
            
        self._cache[cache_key] = response
        self._save_cache()
        logging.debug(f"Cached response for {self.model_name} ({len(response)} chars)")
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_cache(self):
        """Load cache from disk with error handling."""
        try:
            if not os.path.exists(self.cache_file):
                logging.debug(f"No cache file for {self.model_name}, starting fresh")
                return
            
            with open(self.cache_file, "rb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                self._cache = pickle.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            self._update_file_mtime()
            logging.info(f"Loaded {len(self._cache)} cache entries for {self.model_name}")
            
        except Exception as e:
            logging.error(f"Cache load failed for {self.model_name}: {e}")
            self._cache = {}
    
    def _save_cache(self):
        """Save cache to disk atomically with file locking."""
        try:
            # Ensure cache directory exists (handles multiprocessing race conditions)
            self._ensure_cache_dir()
            
            # Atomic write: write to temp file then rename
            temp_file = self.cache_file + ".tmp"
            
            with open(temp_file, "wb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                pickle.dump(self._cache, f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            os.rename(temp_file, self.cache_file)
            self._update_file_mtime()
            
        except Exception as e:
            logging.error(f"Cache save failed for {self.model_name}: {e}")
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
    
    def _reload_if_changed(self):
        """Reload cache if file was modified by another process."""
        try:
            if not os.path.exists(self.cache_file):
                return
            
            # Reload if cache is empty but file exists (failed previous load)
            if not self._cache:
                logging.debug(f"Cache empty but file exists for {self.model_name}, reloading")
                self._load_cache()
                return
            
            # Reload if file was modified
            current_mtime = os.path.getmtime(self.cache_file)
            if current_mtime > self._file_mtime:
                logging.debug(f"Cache file updated for {self.model_name}, reloading")
                self._load_cache()
        
        except OSError as e:
            logging.debug(f"Cache file check failed for {self.model_name}: {e}")
    
    def _update_file_mtime(self):
        """Update tracked modification time."""
        try:
            self._file_mtime = os.path.getmtime(self.cache_file)
        except OSError:
            self._file_mtime = 0