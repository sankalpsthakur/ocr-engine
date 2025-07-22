"""Model caching utilities for efficient loading"""

import os
import time
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)


class ModelCacheManager:
    """Manage model caching for efficient reuse"""
    
    # Class-level cache shared across instances
    _model_cache: Dict[str, Tuple[Any, Any]] = {}
    _cache_timestamps: Dict[str, float] = {}
    _cache_ttl = 3600  # 1 hour TTL
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for persistent cache (not implemented yet)
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "qwen_vl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model(self, model_name: str) -> Optional[Tuple[Any, Any]]:
        """
        Get model from cache if available and not expired.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Tuple of (model, processor) if cached, None otherwise
        """
        if model_name in self._model_cache:
            # Check if cache is still valid
            timestamp = self._cache_timestamps.get(model_name, 0)
            if time.time() - timestamp < self._cache_ttl:
                logger.info(f"Found {model_name} in cache")
                return self._model_cache[model_name]
            else:
                # Cache expired, remove it
                logger.info(f"Cache expired for {model_name}")
                self._remove_from_cache(model_name)
        
        return None
    
    def cache_model(self, model_name: str, model_tuple: Tuple[Any, Any]):
        """
        Cache a model and processor.
        
        Args:
            model_name: Model identifier
            model_tuple: Tuple of (model, processor)
        """
        # Check memory before caching
        if self._should_cache():
            self._model_cache[model_name] = model_tuple
            self._cache_timestamps[model_name] = time.time()
            logger.info(f"Cached {model_name}")
        else:
            logger.warning("Insufficient memory for caching")
    
    def clear_cache(self):
        """Clear all cached models"""
        model_names = list(self._model_cache.keys())
        for name in model_names:
            self._remove_from_cache(name)
        logger.info("Cleared model cache")
    
    def _remove_from_cache(self, model_name: str):
        """Remove a specific model from cache"""
        if model_name in self._model_cache:
            # Delete model to free memory
            model, processor = self._model_cache[model_name]
            del model
            del processor
            
            # Remove from cache
            del self._model_cache[model_name]
            del self._cache_timestamps[model_name]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _should_cache(self) -> bool:
        """Check if there's enough memory to cache a model"""
        try:
            if torch.cuda.is_available():
                # Check GPU memory
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                gpu_used = torch.cuda.memory_allocated(0)
                gpu_free = gpu_mem - gpu_used
                
                # Need at least 4GB free for caching
                return gpu_free > 4 * 1024 * 1024 * 1024
            else:
                # Check system memory
                import psutil
                mem = psutil.virtual_memory()
                # Need at least 8GB free for CPU caching
                return mem.available > 8 * 1024 * 1024 * 1024
        except:
            # If we can't check, assume it's ok
            return True
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models"""
        info = {
            "cached_models": list(self._model_cache.keys()),
            "cache_size": len(self._model_cache),
            "cache_ttl": self._cache_ttl
        }
        
        # Add age information
        for model_name, timestamp in self._cache_timestamps.items():
            age = time.time() - timestamp
            info[f"{model_name}_age_seconds"] = int(age)
        
        return info