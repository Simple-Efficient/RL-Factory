from .cache.cache_base import CacheBase
from .persist.persist_base import PersistBase

__all__ = ['CacheBase', 'PersistBase']

CACHE_REGISTRY = {
    'cache': CacheBase,
    'persist': PersistBase
}

