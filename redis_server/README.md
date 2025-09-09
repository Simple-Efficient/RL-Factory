# Redis缓存服务

本项目使用Redis作为web search API调用的缓存服务，以减少重复API调用，降低训练成本。

## 📁 文件结构

```
redis_server/
├── README.md          # 本文档
├── client.py          # Redis客户端使用示例
├── start_redis.sh     # Redis服务启动脚本
├── redis.conf         # Redis配置文件（自动生成）
├── redis.pid          # Redis进程ID文件（自动生成）
├── data/              # Redis数据目录（自动创建）
│   ├── dump.rdb       # RDB快照文件
│   └── appendonly.aof # AOF日志文件
└── logs/              # Redis日志目录（自动创建）
    └── redis.log      # Redis日志文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装Redis服务器（Ubuntu/Debian）
sudo apt-get update
sudo apt-get install redis-server

# 安装Python Redis客户端
pip install redis
```

### 2. 启动Redis服务

```bash
# 启动Redis服务器（推荐）
./start_redis.sh start

# 其他可用命令
./start_redis.sh stop      # 停止服务
./start_redis.sh restart   # 重启服务
./start_redis.sh status    # 查看状态
./start_redis.sh help      # 显示帮助

# 或者手动启动
redis-server --port 6379 --daemonize yes
```

### 3. 测试连接

```bash
# 运行客户端测试
python client.py
```

## 🔧 配置说明

### Redis服务器配置

- **默认端口**: 6379
- **数据库**: 0
- **持久化**: RDB + AOF
- **内存限制**: 16GB
- **内存策略**: allkeys-lru（最近最少使用）

### 环境变量配置

```bash
# 可通过环境变量自定义配置
export REDIS_PORT=6379                    # Redis端口
export REDIS_HOST=localhost               # Redis主机
export REDIS_DATA_DIR=./data              # 数据目录
export REDIS_LOG_DIR=./logs               # 日志目录

# 使用自定义配置启动
REDIS_PORT=6380 ./start_redis.sh start
```

### 缓存策略

- **键格式**: `search:{query_md5_hash}`
- **默认TTL**: 30天（2592000秒）
- **永久缓存**: 支持永不过期缓存
- **序列化**: JSON格式
- **编码**: UTF-8

### 缓存模式

#### 1. 默认模式（30天过期）
```python
# 默认30天过期
redis_client = RedisClient()
redis_client.set_cache(key, data)
```

#### 2. 永久缓存模式
```python
# 启用永久缓存模式
redis_client = RedisClient(default_permanent=True)
redis_client.set_cache(key, data)  # 自动永久缓存
```

#### 3. 混合模式
```python
# 灵活使用不同缓存策略
redis_client = RedisClient()

# 永久缓存
redis_client.set_permanent_cache(key, data)

# 临时缓存（指定时间）
redis_client.set_temporary_cache(key, data, ttl=3600)  # 1小时

# 使用默认策略
redis_client.set_cache(key, data)  # 30天
```

## 📖 使用方法

### Python客户端

#### 基本使用
```python
from redis_server.client import RedisClient

# 初始化客户端（默认30天过期）
redis_client = RedisClient()

# 设置缓存
data = {"results": ["result1", "result2"]}
key = redis_client.generate_key("搜索查询")
redis_client.set_cache(key, data)  # 30天过期

# 获取缓存
cached_data = redis_client.get_cache(key)

# 删除缓存
redis_client.delete_cache(key)
```

#### 强化学习训练场景（永久缓存）
```python
# 启用永久缓存模式，适合长期训练
redis_client = RedisClient(default_permanent=True)

# 检查缓存
cached_result = redis_client.get_search_result(query)
if cached_result:
    # 使用缓存结果，避免API调用
    results = cached_result['results']
else:
    # 调用API搜索
    results = call_search_api(query)
    # 永久缓存结果
    redis_client.set_cache(key, results)
```

#### 灵活缓存策略
```python
redis_client = RedisClient()

# 重要数据永久缓存
redis_client.set_permanent_cache("important_data", data)

# 临时数据短期缓存
redis_client.set_temporary_cache("temp_data", data, ttl=3600)  # 1小时

# 使用默认策略
redis_client.set_cache("normal_data", data)  # 30天
```

### 命令行操作

```bash
# 连接Redis
redis-cli

# 查看所有键
KEYS *

# 查看特定键
GET search:abc123

# 设置键值
SET search:abc123 '{"results": ["data"]}'

# 设置过期时间
EXPIRE search:abc123 3600

# 查看键的TTL
TTL search:abc123
```

## 🔍 监控和维护

### 查看Redis状态

```bash
# 查看Redis信息
redis-cli INFO

# 查看内存使用
redis-cli INFO memory

# 查看连接数
redis-cli INFO clients
```

### 清理缓存

```bash
# 清空当前数据库
redis-cli FLUSHDB

# 清空所有数据库
redis-cli FLUSHALL

# 删除特定模式的键
redis-cli --scan --pattern "search:*" | xargs redis-cli DEL
```

## ⚠️ 注意事项

1. **内存管理**: 定期监控Redis内存使用，避免内存溢出
2. **数据持久化**: 确保RDB和AOF配置正确，防止数据丢失
3. **网络安全**: 生产环境建议设置密码和网络访问控制
4. **性能优化**: 根据实际使用情况调整缓存TTL和内存限制

## 🐛 故障排除

### 常见问题

1. **连接失败**
   ```bash
   # 检查Redis是否运行
   ps aux | grep redis
   
   # 检查端口是否开放
   netstat -tlnp | grep 6379
   ```

2. **内存不足**
   ```bash
   # 查看内存使用
   redis-cli INFO memory
   
   # 清理过期键
   redis-cli --scan --pattern "*" | xargs redis-cli EXPIRE
   ```

3. **权限问题**
   ```bash
   # 检查Redis用户权限
   sudo chown redis:redis /var/lib/redis/
   ```

## 📊 性能指标

- **缓存命中率**: 目标 > 80%
- **响应时间**: < 10ms
- **内存使用**: 根据缓存数据量动态调整
- **并发连接**: 支持多进程训练

## 🔄 集成到训练流程

1. 启动Redis服务
2. 修改训练脚本使用缓存工具
3. 监控缓存命中率和性能
4. 根据实际情况调整缓存策略