"""
Redis Caching Layer for API Responses and Weather Data

Provides high-performance caching with TTL management and cache invalidation.
"""
import logging
import json
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from functools import wraps
import pickle

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not installed. Install with: pip install redis")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Unified cache manager with Redis backend and fallback to in-memory cache"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.redis_client = None
        self.in_memory_cache = {}  # Fallback cache
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0
        }
        
        # Default TTL values (in seconds)
        self.default_ttls = {
            'weather_data': 300,      # 5 minutes
            'predictions': 600,        # 10 minutes
            'satellite_images': 1800,  # 30 minutes
            'model_results': 300,      # 5 minutes
            'historical_data': 3600,   # 1 hour
            'api_response': 60         # 1 minute
        }
        
        # Connect to Redis
        if REDIS_AVAILABLE:
            self._connect_redis()
        else:
            logger.warning("Using in-memory cache (Redis not available)")
    
    def _connect_redis(self):
        """Establish Redis connection"""
        try:
            redis_config = {
                'host': self.config.get('redis_host', 'localhost'),
                'port': self.config.get('redis_port', 6379),
                'db': self.config.get('redis_db', 0),
                'decode_responses': False,  # Handle binary data
                'socket_timeout': 5,
                'socket_connect_timeout': 5
            }
            
            # Add password if provided
            if self.config.get('redis_password'):
                redis_config['password'] = self.config['redis_password']
            
            self.redis_client = redis.Redis(**redis_config)
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {redis_config['host']}:{redis_config['port']}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.info("Falling back to in-memory cache")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """
        Generate cache key from prefix and parameters
        
        Args:
            prefix: Cache key prefix (e.g., 'weather', 'prediction')
            **kwargs: Key-value pairs to include in key
            
        Returns:
            Cache key string
        """
        # Sort kwargs for consistent key generation
        key_parts = [prefix]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        
        key_string = ":".join(str(p) for p in key_parts)
        
        # Hash long keys
        if len(key_string) > 200:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"{prefix}:{key_hash}"
        
        return key_string
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            if self.redis_client:
                # Try Redis first
                value = self.redis_client.get(key)
                if value is not None:
                    self.cache_stats['hits'] += 1
                    return pickle.loads(value)
            else:
                # Use in-memory cache
                if key in self.in_memory_cache:
                    value, expiry = self.in_memory_cache[key]
                    if expiry is None or datetime.now() < expiry:
                        self.cache_stats['hits'] += 1
                        return value
                    else:
                        # Expired
                        del self.in_memory_cache[key]
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.cache_stats['errors'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, category: str = 'api_response'):
        """
        Set value in cache with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for no expiry)
            category: Cache category for default TTL lookup
        """
        try:
            # Use category-specific TTL if not provided
            if ttl is None:
                ttl = self.default_ttls.get(category, 300)
            
            if self.redis_client:
                # Store in Redis
                serialized = pickle.dumps(value)
                if ttl:
                    self.redis_client.setex(key, ttl, serialized)
                else:
                    self.redis_client.set(key, serialized)
            else:
                # Store in memory
                expiry = datetime.now() + timedelta(seconds=ttl) if ttl else None
                self.in_memory_cache[key] = (value, expiry)
            
            self.cache_stats['sets'] += 1
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.cache_stats['errors'] += 1
    
    def delete(self, key: str):
        """Delete key from cache"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                self.in_memory_cache.pop(key, None)
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
    
    def delete_pattern(self, pattern: str):
        """
        Delete all keys matching pattern
        
        Args:
            pattern: Pattern with wildcards (e.g., 'weather:*')
        """
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Deleted {len(keys)} keys matching pattern: {pattern}")
            else:
                # In-memory pattern matching
                to_delete = [k for k in self.in_memory_cache.keys() 
                           if self._match_pattern(k, pattern)]
                for k in to_delete:
                    del self.in_memory_cache[k]
                logger.info(f"Deleted {len(to_delete)} keys matching pattern: {pattern}")
        except Exception as e:
            logger.error(f"Cache delete pattern error: {e}")
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for in-memory cache"""
        import re
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(f"^{regex_pattern}$", key))
    
    def clear_all(self):
        """Clear all cache entries"""
        try:
            if self.redis_client:
                self.redis_client.flushdb()
            else:
                self.in_memory_cache.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        stats = self.cache_stats.copy()
        
        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        stats['hit_rate'] = stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        # Get cache size
        if self.redis_client:
            try:
                info = self.redis_client.info('memory')
                stats['used_memory'] = info.get('used_memory_human', 'N/A')
                stats['keys_count'] = self.redis_client.dbsize()
            except:
                stats['used_memory'] = 'N/A'
                stats['keys_count'] = 'N/A'
        else:
            stats['used_memory'] = 'N/A'
            stats['keys_count'] = len(self.in_memory_cache)
        
        stats['backend'] = 'redis' if self.redis_client else 'in-memory'
        
        return stats


# Global cache instance
_cache_manager = None


def get_cache_manager(config: Dict = None) -> CacheManager:
    """Get or create global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(config)
    return _cache_manager


# Decorator for caching function results
def cached(category: str = 'api_response', ttl: Optional[int] = None, key_prefix: str = None):
    """
    Decorator to cache function results
    
    Args:
        category: Cache category for TTL lookup
        ttl: Custom TTL in seconds
        key_prefix: Custom key prefix (default: function name)
        
    Example:
        @cached(category='weather_data', ttl=300)
        def get_weather(lat, lon):
            # Expensive API call
            return fetch_weather_api(lat, lon)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key
            prefix = key_prefix or func.__name__
            key_params = {
                f'arg{i}': str(arg) for i, arg in enumerate(args)
            }
            key_params.update({k: str(v) for k, v in kwargs.items()})
            
            cache_key = cache._generate_key(prefix, **key_params)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value
            
            # Execute function
            logger.debug(f"Cache miss: {cache_key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl, category=category)
            
            return result
        
        return wrapper
    return decorator


# Specific cache functions for common use cases
class WeatherCache:
    """Specialized cache for weather data"""
    
    def __init__(self, cache_manager: CacheManager = None):
        self.cache = cache_manager or get_cache_manager()
    
    def get_weather(self, latitude: float, longitude: float) -> Optional[Dict]:
        """Get cached weather data for location"""
        key = self.cache._generate_key('weather', lat=latitude, lon=longitude)
        return self.cache.get(key)
    
    def set_weather(self, latitude: float, longitude: float, data: Dict):
        """Cache weather data for location"""
        key = self.cache._generate_key('weather', lat=latitude, lon=longitude)
        self.cache.set(key, data, category='weather_data')
    
    def invalidate_weather(self, latitude: float = None, longitude: float = None):
        """Invalidate weather cache"""
        if latitude is not None and longitude is not None:
            key = self.cache._generate_key('weather', lat=latitude, lon=longitude)
            self.cache.delete(key)
        else:
            self.cache.delete_pattern('weather:*')


class PredictionCache:
    """Specialized cache for predictions"""
    
    def __init__(self, cache_manager: CacheManager = None):
        self.cache = cache_manager or get_cache_manager()
    
    def get_prediction(self, latitude: float, longitude: float, model: str = 'default') -> Optional[Dict]:
        """Get cached prediction"""
        key = self.cache._generate_key('prediction', lat=latitude, lon=longitude, model=model)
        return self.cache.get(key)
    
    def set_prediction(self, latitude: float, longitude: float, prediction: Dict, model: str = 'default'):
        """Cache prediction result"""
        key = self.cache._generate_key('prediction', lat=latitude, lon=longitude, model=model)
        self.cache.set(key, prediction, category='predictions')
    
    def invalidate_predictions(self):
        """Clear all prediction cache"""
        self.cache.delete_pattern('prediction:*')


# Example FastAPI integration
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class CacheMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for response caching"""
    
    def __init__(self, app, cache_manager: CacheManager = None):
        super().__init__(app)
        self.cache = cache_manager or get_cache_manager()
    
    async def dispatch(self, request: Request, call_next):
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Generate cache key from URL and query params
        cache_key = self.cache._generate_key(
            'http',
            url=str(request.url)
        )
        
        # Check cache
        cached_response = self.cache.get(cache_key)
        if cached_response:
            logger.debug(f"Returning cached response for {request.url}")
            from starlette.responses import Response
            return Response(
                content=cached_response['body'],
                status_code=cached_response['status_code'],
                headers=cached_response['headers'],
                media_type=cached_response.get('media_type')
            )
        
        # Get fresh response
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Cache response
            cached_data = {
                'body': body,
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'media_type': response.media_type
            }
            self.cache.set(cache_key, cached_data, category='api_response')
            
            # Return response
            from starlette.responses import Response
            return Response(
                content=body,
                status_code=response.status_code,
                headers=response.headers,
                media_type=response.media_type
            )
        
        return response


if __name__ == "__main__":
    # Example usage
    cache = get_cache_manager()
    
    # Test caching
    cache.set('test_key', {'data': 'test_value'}, ttl=60)
    value = cache.get('test_key')
    print(f"Cached value: {value}")
    
    # Test decorator
    @cached(category='weather_data', ttl=300)
    def expensive_function(x, y):
        print(f"Computing expensive_function({x}, {y})")
        return x + y
    
    result1 = expensive_function(1, 2)  # Cache miss
    result2 = expensive_function(1, 2)  # Cache hit
    
    # Show stats
    print("\nCache Statistics:")
    print(json.dumps(cache.get_stats(), indent=2))
