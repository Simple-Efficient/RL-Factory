# MCP工具缓存功能使用说明

## 功能概述

本功能为MCP工具调用添加了Redis缓存支持，可以显著提升重复工具调用的性能，减少对远程MCP服务的请求。

## 核心特性

- **透明缓存**: 对现有代码完全透明，无需修改工具调用逻辑
- **智能过滤**: 自动过滤错误结果，只缓存成功的工具调用
- **配置灵活**: 支持多种缓存配置选项
- **性能优化**: 避免重复的远程MCP调用
- **降级处理**: Redis连接失败时自动降级到无缓存模式

## 使用方法

### 1. 启用缓存

在训练脚本中添加以下配置参数：

```bash
python3 -m verl.trainer.main_ppo --config-name=rl_factory_ppo_trainer \
    # ... 其他参数 ... \
    actor_rollout_ref.env.enable_redis_cache=True \
    actor_rollout_ref.env.redis_host=localhost \
    actor_rollout_ref.env.redis_port=6379 \
    actor_rollout_ref.env.cache_ttl=0 \
    actor_rollout_ref.env.cache_prefix=mcp_tool \
    actor_rollout_ref.env.cache_logging=True
```

### 2. 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_redis_cache` | `False` | 是否启用Redis缓存 |
| `redis_host` | `localhost` | Redis服务器地址 |
| `redis_port` | `6379` | Redis服务器端口 |
| `cache_ttl` | `0` | 缓存过期时间（秒），0表示永不过期 |
| `cache_prefix` | `mcp_tool` | 缓存键前缀 |
| `cache_logging` | `True` | 是否启用缓存日志 |

### 3. 缓存策略

- **缓存键格式**: `{prefix}:mcp_tool:{tool_name}:{hash(tool_args)}`
- **缓存内容**: 工具调用的完整返回结果
- **过滤规则**: 自动过滤包含错误信息的返回结果
- **过期策略**: 默认永不过期，依赖Redis内存管理

## 文件结构

```
envs/
├── utils/
│   └── redis_cache_manager.py    # Redis缓存管理器
└── tool_manager/
    └── qwen3_manager.py          # 集成了缓存功能的工具管理器

redis_server/
├── start_redis.sh               # Redis服务管理脚本
├── redis.conf                   # Redis配置文件
├── data/                        # Redis数据目录
└── logs/                        # Redis日志目录

tests/
└── test_mcp_cache_real.py       # 真实MCP缓存功能测试

docs/MCP_CACHE_README.md         # 使用说明
```

## 测试方法

### 运行MCP缓存功能测试

**使用真实测试脚本**，该脚本会完整测试MCP工具的缓存功能：

```bash
python3 tests/test_mcp_cache_real.py
```

**测试特性：**
- 自动启动和停止Redis服务
- 测试前清除相关缓存数据
- 完整的缓存命中率测试
- 详细的性能统计和分析

**测试流程：**
1. 启动Redis服务
2. 初始化工具管理器和缓存管理器
3. 清除测试相关的缓存数据
4. 执行首次查询（应该不命中缓存）
5. 执行重复查询（应该命中缓存）
6. 分析缓存命中率和性能指标
7. 停止Redis服务

### 运行训练脚本

```bash
# 确保Redis服务已启动
cd redis_server
./start_redis.sh start

# 运行训练脚本（已启用缓存）
./main_web_search.sh
```

## 缓存效果

### 性能提升

- **首次调用**: 执行实际MCP工具调用，结果缓存到Redis
- **重复调用**: 直接从Redis获取缓存结果，响应时间显著降低
- **内存优化**: 避免重复的远程网络请求

### 测试结果示例

运行真实测试脚本的典型输出：

```
🧪 MCP工具缓存功能真实测试脚本
============================================================
🚀 开始MCP缓存功能完整真实测试
============================================================
🚀 启动Redis服务...
✅ Redis服务启动成功
✅ Redis连接成功: localhost:6379
✅ Redis连接初始化成功
🚀 MCP工具缓存已启用: mcp_tool
✅ 工具管理器初始化成功

🧹 清除测试相关的缓存数据...
✅ 已清除 7 个测试缓存项

📊 测试前Redis状态: {'connected_clients': 2, 'used_memory_human': '816.46K', 'total_commands_processed': 16, 'keyspace_hits': 0, 'keyspace_misses': 0}

============================================================
🧪 开始MCP缓存功能真实测试
============================================================

📝 测试1: 首次查询（应该不命中缓存）
----------------------------------------
查询 1: Python编程教程
🔧 调用MCP工具: meituan_search-web_search (服务器: meituan_search, 工具: web_search)
🔍 缓存未命中: mcp_tool:4420c19b3e4a91637b478e877bcf59aa
📝 永久缓存设置: mcp_tool:4420c19b3e4a91637b478e877bcf59aa -> 8551 bytes
💾 缓存设置: mcp_tool:4420c19b3e4a91637b478e877bcf59aa (TTL: 0s)
✅ 查询成功

📝 测试2: 重复查询（应该命中缓存）
----------------------------------------
重复查询 1: Python编程教程
🔧 调用MCP工具: meituan_search-web_search (服务器: meituan_search, 工具: web_search)
📖 缓存命中: mcp_tool:4420c19b3e4a91637b478e877bcf59aa
🎯 缓存命中: mcp_tool:4420c19b3e4a91637b478e877bcf59aa
🎯 缓存命中: meituan_search-web_search
✅ 查询成功

============================================================
📊 测试结果分析
============================================================

📈 查询统计:
  总查询数: 9
  成功查询数: 9
  成功率: 100.00%

🎯 缓存命中率: 22.22%
✅ 缓存功能工作正常

💾 Redis内存使用情况:
  测试前: 816.46K
  测试后: 971.77K

📈 Redis命令处理统计:
  总命令数: 33
  缓存命中: 2
  缓存未命中: 7

✅ 测试完成!
```