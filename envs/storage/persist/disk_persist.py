import os
import json
import gzip
from datetime import datetime
from typing import Any, Dict, Optional
from .persist_base import PersistBase

class DiskPersist(PersistBase):
    def __init__(self, base_dir: str = "cache_data"):
        """初始化持久化存储
        
        Args:
            base_dir: 缓存文件存储的基础目录
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def _get_file_path(self, method_name: str, partition: Optional[int] = None) -> str:
        """生成文件路径
        
        Args:
            method_name: 方法名称
            partition: 分区号，用于分布式场景
            
        Returns:
            str: 文件路径
        """
        date_str = datetime.now().strftime("%Y%m%d")
        method_hash = hashlib.md5(method_name.encode()).hexdigest()
        partition_str = f"_p{partition}" if partition is not None else ""
        file_name = f"{method_hash}_{date_str}{partition_str}.cache"
        return os.path.join(self.base_dir, file_name)
    
    def save(self, data: Dict[str, Any], method_name: str, partition: Optional[int] = None) -> str:
        """保存数据到磁盘
        
        Args:
            data: 要保存的数据字典
            method_name: 方法名称
            partition: 分区号，用于分布式场景
            
        Returns:
            str: 保存的文件路径
        """
        file_path = self._get_file_path(method_name, partition)
        
        # 使用gzip压缩
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
        
        return file_path
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """从磁盘加载数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 加载的数据字典
        """
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load cache file {file_path}: {e}")
            return {}
    
    def merge(self, file_paths: list[str], output_path: str) -> str:
        """合并多个缓存文件
        
        Args:
            file_paths: 要合并的文件路径列表
            output_path: 输出文件路径
            
        Returns:
            str: 合并后的文件路径
        """
        merged_data = {}
        for file_path in file_paths:
            data = self.load(file_path)
            merged_data.update(data)
        
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(merged_data, f)
        
        return output_path
    
    def list_cache_files(self, method_name: Optional[str] = None) -> list[str]:
        """列出缓存文件
        
        Args:
            method_name: 方法名称，如果指定则只列出该方法的缓存文件
            
        Returns:
            list[str]: 缓存文件路径列表
        """
        files = []
        for file_name in os.listdir(self.base_dir):
            if not file_name.endswith('.cache'):
                continue
                
            if method_name is not None:
                method_hash = hashlib.md5(method_name.encode()).hexdigest()
                if not file_name.startswith(method_hash):
                    continue
            
            files.append(os.path.join(self.base_dir, file_name))
        
        return files 