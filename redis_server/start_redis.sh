#!/bin/bash

# Redis服务启动脚本
# 用于启动和管理Redis缓存服务

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置变量
REDIS_PORT=${REDIS_PORT:-6379}
REDIS_HOST=${REDIS_HOST:-localhost}
REDIS_DATA_DIR=${REDIS_DATA_DIR:-$SCRIPT_DIR/data}
REDIS_LOG_DIR=${REDIS_LOG_DIR:-$SCRIPT_DIR/logs}
REDIS_CONF_FILE=${REDIS_CONF_FILE:-$SCRIPT_DIR/redis.conf}
REDIS_PID_FILE=${REDIS_PID_FILE:-$SCRIPT_DIR/redis.pid}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Redis安装检查已删除 - 假设Redis已预安装

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    mkdir -p "$REDIS_DATA_DIR"
    mkdir -p "$REDIS_LOG_DIR"
    
    log_success "目录创建完成"
}

# 生成Redis配置文件
generate_redis_config() {
    log_info "生成Redis配置文件..."
    
    cat > "$REDIS_CONF_FILE" << EOF
# Redis配置文件
# 端口配置
port $REDIS_PORT
bind $REDIS_HOST

# 数据目录
dir $REDIS_DATA_DIR

# 日志配置
logfile $REDIS_LOG_DIR/redis.log
loglevel notice

# 持久化配置
save 900 1
save 300 10
save 60 10000

# AOF持久化
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec

# 内存配置
maxmemory 16gb
maxmemory-policy allkeys-lru

# 安全配置
# requirepass your_password_here

# 客户端配置
timeout 300
tcp-keepalive 300

# 其他配置
daemonize yes
pidfile $REDIS_PID_FILE
EOF

    log_success "Redis配置文件生成完成: $REDIS_CONF_FILE"
}

# 检查Redis是否正在运行
check_redis_running() {
    if [ -f "$REDIS_PID_FILE" ]; then
        PID=$(cat "$REDIS_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_warning "Redis已在运行 (PID: $PID)"
            return 0
        else
            log_warning "PID文件存在但进程不存在，清理PID文件"
            rm -f "$REDIS_PID_FILE"
        fi
    fi
    
    # 检查端口是否被占用
    if netstat -tlnp 2>/dev/null | grep ":$REDIS_PORT " >/dev/null; then
        log_warning "端口 $REDIS_PORT 已被占用"
        return 0
    fi
    
    return 1
}

# 启动Redis服务
start_redis() {
    log_info "启动Redis服务..."
    
    if check_redis_running; then
        log_warning "Redis已在运行，跳过启动"
        return 0
    fi
    
    # 启动Redis
    redis-server "$REDIS_CONF_FILE"
    
    # 等待服务启动
    sleep 2
    
    # 检查服务是否启动成功
    if check_redis_running; then
        log_success "Redis服务启动成功"
        log_info "服务信息:"
        log_info "  端口: $REDIS_PORT"
        log_info "  数据目录: $REDIS_DATA_DIR"
        log_info "  日志文件: $REDIS_LOG_DIR/redis.log"
        log_info "  PID文件: $REDIS_PID_FILE"
        return 0
    else
        log_error "Redis服务启动失败"
        return 1
    fi
}

# 停止Redis服务
stop_redis() {
    log_info "停止Redis服务..."
    
    if [ -f "$REDIS_PID_FILE" ]; then
        PID=$(cat "$REDIS_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            kill "$PID"
            sleep 2
            
            if ps -p "$PID" > /dev/null 2>&1; then
                log_warning "正常停止失败，强制停止"
                kill -9 "$PID"
            fi
            
            rm -f "$REDIS_PID_FILE"
            log_success "Redis服务已停止"
        else
            log_warning "Redis服务未运行"
        fi
    else
        log_warning "PID文件不存在，Redis可能未运行"
    fi
}

# 重启Redis服务
restart_redis() {
    log_info "重启Redis服务..."
    stop_redis
    sleep 1
    create_directories
    generate_redis_config
    start_redis
}

# 检查Redis服务状态
status_redis() {
    log_info "检查Redis服务状态..."
    
    if check_redis_running; then
        PID=$(cat "$REDIS_PID_FILE")
        log_success "Redis服务正在运行 (PID: $PID)"
        
        # 测试连接
        if redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1; then
            log_success "Redis连接测试成功"
        else
            log_warning "Redis连接测试失败"
        fi
        
        # 显示基本信息
        log_info "服务信息:"
        log_info "  端口: $REDIS_PORT"
        log_info "  数据目录: $REDIS_DATA_DIR"
        log_info "  日志文件: $REDIS_LOG_DIR/redis.log"
        
    else
        log_error "Redis服务未运行"
    fi
}

# Redis连接测试已删除 - 简化脚本功能

# 显示帮助信息
show_help() {
    echo "Redis服务管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start     启动Redis服务"
    echo "  stop      停止Redis服务"
    echo "  restart   重启Redis服务"
    echo "  status    检查服务状态"
    echo "  help      显示帮助信息"
    echo ""
    echo "环境变量:"
    echo "  REDIS_PORT      Redis端口 (默认: 6379)"
    echo "  REDIS_HOST      Redis主机 (默认: localhost)"
    echo "  REDIS_DATA_DIR  数据目录 (默认: ./data)"
    echo "  REDIS_LOG_DIR   日志目录 (默认: ./logs)"
    echo ""
    echo "示例:"
    echo "  $0 start"
    echo "  REDIS_PORT=6380 $0 start"
}

# 主函数
main() {
    case "${1:-start}" in
        "start")
            create_directories
            generate_redis_config
            start_redis
            ;;
        "stop")
            stop_redis
            ;;
        "restart")
            restart_redis
            ;;
        "status")
            status_redis
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
