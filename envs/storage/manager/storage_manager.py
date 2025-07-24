import argparse
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from ..cache.cache_base import CacheMode, EvictionPolicy
from ..cache.cachebox_cache import CacheBoxCache
from ..persist.disk_persist import DiskPersist


class StorageManager:
    def __init__(self, cache_mode: CacheMode = CacheMode.SINGLE,
                 enable_persist: bool = False,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 max_cache_size: int = 1000,
                 persist_dir: str = "cache_data",
                 persist_interval: int = 300,  # 持久化间隔，默认5分钟
                 sync_interval: int = 600):  # 同步间隔，默认10分钟
        """初始化存储管理器
        
        Args:
            cache_mode: 缓存模式
            enable_persist: 是否启用持久化
            eviction_policy: 缓存淘汰策略
            max_cache_size: 最大缓存条目数
            persist_dir: 持久化存储目录
            persist_interval: 持久化间隔（秒）
            sync_interval: 同步间隔（秒）
        """
        self.cache_mode = cache_mode
        self.enable_persist = enable_persist
        self.persist_interval = persist_interval
        self.sync_interval = sync_interval

        # 初始化缓存
        self.cache = CacheBoxCache(
            max_size=max_cache_size,
            mode=cache_mode,
            eviction_policy=eviction_policy
        )

        # 如果启用持久化，初始化持久化存储
        self.persist = DiskPersist(base_dir=persist_dir) if enable_persist else None

        # 用于批量写入的缓冲区
        self.persist_buffer: Dict[str, Dict[str, Any]] = {}
        self.last_persist_time = datetime.now()

        # 如果启用持久化，启动定时任务
        if self.enable_persist and self.persist:
            asyncio.create_task(self._start_persist_task())
            asyncio.create_task(self._start_sync_task())

    async def _start_persist_task(self):
        """启动定时持久化任务"""
        while True:
            await asyncio.sleep(self.persist_interval)
            await self._flush_persist_buffer()

    async def _start_sync_task(self):
        """启动定时同步任务"""
        while True:
            await asyncio.sleep(self.sync_interval)
            await self._sync_from_persist()

    async def _flush_persist_buffer(self):
        """将缓冲区数据写入持久化存储"""
        if not self.persist_buffer:
            return

        try:
            for method_name, data in self.persist_buffer.items():
                await asyncio.to_thread(self.persist.save, data, method_name)
            self.persist_buffer.clear()
            self.last_persist_time = datetime.now()
        except Exception as e:
            print(f"Failed to flush persist buffer: {e}")

    async def _sync_from_persist(self):
        """从持久化存储同步数据到缓存"""
        if not self.persist:
            return

        try:
            # 获取所有缓存文件
            cache_files = self.persist.list_cache_files()
            for file_path in cache_files:
                data = await asyncio.to_thread(self.persist.load, file_path)
                for cache_key, value in data.items():
                    # 使用缓存的hash_key方法确保一致性
                    self.cache.set(cache_key, value)
        except Exception as e:
            print(f"Failed to sync from persist: {e}")

    async def get(self, method_name: str, params: dict) -> Optional[Any]:
        """获取缓存数据
        
        Args:
            method_name: 方法名
            params: 参数字典
            
        Returns:
            Optional[Any]: 缓存数据
        """
        cache_key = self.cache._hash_key(method_name, params)

        # 尝试从缓存获取
        if self.cache.has(cache_key):
            return self.cache.get(cache_key)

        # 如果启用持久化，尝试从持久化存储加载
        if self.enable_persist and self.persist:
            try:
                # 查找相关的缓存文件
                cache_files = self.persist.list_cache_files(method_name)
                for file_path in cache_files:
                    data = await asyncio.to_thread(self.persist.load, file_path)
                    if cache_key in data:
                        value = data[cache_key]
                        # 写入缓存
                        self.cache.set(cache_key, value)
                        return value
            except Exception as e:
                print(f"Failed to load from persist: {e}")

        return None

    async def set(self, method_name: str, params: dict, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存数据
        
        Args:
            method_name: 方法名
            params: 参数字典
            value: 要缓存的值
            ttl: 过期时间（秒）
        """
        cache_key = self.cache._hash_key(method_name, params)

        # 写入缓存
        self.cache.set(cache_key, value, ttl)

        # 如果启用持久化，写入缓冲区
        if self.enable_persist and self.persist:
            if method_name not in self.persist_buffer:
                self.persist_buffer[method_name] = {}
            self.persist_buffer[method_name][cache_key] = value

            # 如果缓冲区过大或距离上次持久化时间过长，触发持久化
            current_time = datetime.now()
            if (len(self.persist_buffer) > 1000 or
                    (current_time - self.last_persist_time).total_seconds() > self.persist_interval):
                await self._flush_persist_buffer()

    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            "cache_mode": self.cache_mode.value,
            "enable_persist": self.enable_persist,
            "eviction_policy": self.cache.get_eviction_policy().value,
            "cache_stats": self.cache.get_stats(),
            "persist_buffer_size": len(self.persist_buffer) if self.enable_persist else 0,
            "last_persist_time": self.last_persist_time.isoformat() if self.enable_persist else None
        }
        return stats


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Storage Manager")
    parser.add_argument("-cache", choices=["single", "multi"], default="single",
                        help="Cache mode: single (default) or multi")
    parser.add_argument("-persist", action="store_true",
                        help="Enable persistence")
    parser.add_argument("-eviction", choices=["lru", "lfu", "fifo", "ttl"], default="lru",
                        help="Cache eviction policy")
    parser.add_argument("-max-size", type=int, default=1000,
                        help="Maximum cache size")
    parser.add_argument("-persist-dir", default="cache_data",
                        help="Persistence directory")
    return parser.parse_args()


def create_storage_manager(args) -> StorageManager:
    """创建存储管理器
    
    Args:
        args: 命令行参数
        
    Returns:
        StorageManager: 存储管理器实例
    """
    cache_mode = CacheMode.SINGLE if args.cache == "single" else CacheMode.MULTI
    eviction_policy = EvictionPolicy(args.eviction)

    return StorageManager(
        cache_mode=cache_mode,
        enable_persist=args.persist,
        eviction_policy=eviction_policy,
        max_cache_size=args.max_size,
        persist_dir=args.persist_dir
    )


def create_config_storage_manager(verl_config) -> StorageManager:
    """创建存储管理器

    Args:
        verl_config: 配置文件

    Returns:
        StorageManager: 存储管理器实例
    """
    cache_mode = CacheMode.SINGLE if verl_config.cache == "single" else CacheMode.MULTI
    eviction_policy = EvictionPolicy(verl_config.eviction)

    return StorageManager(
        cache_mode=cache_mode,
        enable_persist=verl_config.persist,
        eviction_policy=eviction_policy,
        max_cache_size=verl_config.max_size,
        persist_dir=verl_config.persist_dir
    )
