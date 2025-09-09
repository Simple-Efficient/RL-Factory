#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis缓存管理器
用于MCP工具调用和其他组件的缓存功能
"""

import json
import hashlib
from typing import Any, Optional, Dict, Union
from redis_server.client import RedisClient


class RedisCacheManager:
    """Redis缓存管理器 - 基于RedisClient的通用缓存服务"""
    
    def __init__(self, 
                 redis_host='localhost', 
                 redis_port=6379, 
                 default_ttl=3600,
                 prefix="cache",
                 enable_logging=True):
        """初始化Redis缓存管理器"""
        self.redis_client = RedisClient(host=redis_host, port=redis_port)
        self.default_ttl = default_ttl
        self.prefix = prefix
        self.enable_logging = enable_logging
        
        if self.enable_logging:
            print(f"🚀 RedisCacheManager初始化完成: {prefix} (TTL: {default_ttl}s)")
    
    def get_cache_key(self, 
                     key_type: str, 
                     identifier: Union[str, Dict, Any], 
                     sub_key: Optional[str] = None) -> str:
        """生成缓存键"""
        # 处理不同类型的identifier
        if isinstance(identifier, dict):
            # 字典类型：排序后序列化
            identifier_str = json.dumps(identifier, sort_keys=True, ensure_ascii=False)
        elif isinstance(identifier, (list, tuple)):
            # 列表/元组类型：序列化
            identifier_str = json.dumps(identifier, ensure_ascii=False)
        else:
            # 其他类型：转为字符串
            identifier_str = str(identifier)
        
        # 生成哈希值确保键的唯一性
        hash_value = hashlib.md5(identifier_str.encode('utf-8')).hexdigest()
        
        # 构建完整键
        if sub_key:
            full_key = f"{key_type}:{sub_key}:{hash_value}"
        else:
            full_key = f"{key_type}:{hash_value}"
        
        return self.redis_client.generate_key(full_key, prefix=self.prefix)
    
    def get(self, cache_key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            result = self.redis_client.get_cache(cache_key)
            if result is not None and self.enable_logging:
                print(f"🎯 缓存命中: {cache_key}")
            return result
        except Exception as e:
            if self.enable_logging:
                print(f"❌ 缓存获取失败: {cache_key} - {e}")
            return None
    
    def set(self, 
            cache_key: str, 
            value: Any, 
            ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            success = self.redis_client.set_cache(cache_key, value, ttl)
            if success and self.enable_logging:
                print(f"💾 缓存设置: {cache_key} (TTL: {ttl}s)")
            return success
        except Exception as e:
            if self.enable_logging:
                print(f"❌ 缓存设置失败: {cache_key} - {e}")
            return False
    
    def delete(self, cache_key: str) -> bool:
        """删除缓存"""
        try:
            success = self.redis_client.delete_cache(cache_key)
            if success and self.enable_logging:
                print(f"🗑️ 缓存删除: {cache_key}")
            return success
        except Exception as e:
            if self.enable_logging:
                print(f"❌ 缓存删除失败: {cache_key} - {e}")
            return False
    
    def should_cache(self, 
                    key_type: str, 
                    value: Any, 
                    custom_rules: Optional[Dict] = None) -> bool:
        """判断是否应该缓存"""
        # 基本检查
        if value is None:
            return False
        
        # 字符串类型检查
        if isinstance(value, str):
            if not value.strip():
                return False
            # 检查错误指示符
            error_indicators = ["error", "failed", "timeout", "exception", "❌"]
            if any(indicator in value.lower() for indicator in error_indicators):
                return False
        
        # 字典类型检查
        if isinstance(value, dict):
            # 检查是否包含错误信息
            if "error" in value or "failed" in value:
                return False
        
        # 自定义规则检查
        if custom_rules:
            for rule_name, rule_func in custom_rules.items():
                if not rule_func(value):
                    if self.enable_logging:
                        print(f"🚫 缓存被拒绝 (规则: {rule_name}): {key_type}")
                    return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            stats = self.redis_client.get_stats()
            stats['prefix'] = self.prefix
            stats['default_ttl'] = self.default_ttl
            return stats
        except Exception as e:
            if self.enable_logging:
                print(f"❌ 获取统计信息失败: {e}")
            return {}
    
    def clear_by_prefix(self, key_type: str) -> int:
        """根据键类型清理缓存"""
        try:
            # 这里需要实现批量删除逻辑
            # 由于RedisClient没有提供scan功能，这里先返回0
            if self.enable_logging:
                print(f"🧹 清理缓存: {key_type} (功能待实现)")
            return 0
        except Exception as e:
            if self.enable_logging:
                print(f"❌ 清理缓存失败: {e}")
            return 0
    
    # 便捷方法
    def cache_mcp_tool(self, tool_name: str, tool_args: Dict, result: Any, ttl: Optional[int] = None) -> bool:
        """缓存MCP工具调用结果"""
        cache_key = self.get_cache_key("mcp_tool", f"{tool_name}:{tool_args}")
        if self.should_cache("mcp_tool", result):
            return self.set(cache_key, result, ttl)
        return False
    
    def get_mcp_tool(self, tool_name: str, tool_args: Dict) -> Optional[Any]:
        """获取MCP工具调用缓存"""
        cache_key = self.get_cache_key("mcp_tool", f"{tool_name}:{tool_args}")
        return self.get(cache_key)


# 单例模式
_cache_managers = {}

def get_cache_manager(**kwargs) -> RedisCacheManager:
    """获取MCP工具缓存管理器单例"""
    cache_key = f"mcp_tool:{hash(str(kwargs))}"
    
    if cache_key not in _cache_managers:
        default_config = {
            "prefix": "mcp_tool",
            "default_ttl": 0,  # 永不过期
        }
        default_config.update(kwargs)
        _cache_managers[cache_key] = RedisCacheManager(**default_config)
    
    return _cache_managers[cache_key]


