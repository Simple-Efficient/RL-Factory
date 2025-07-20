import sys
import builtins
from typing import Any, TextIO


class StderrPrint:
    """将所有print调用重定向到stderr的类"""
    
    def __init__(self):
        self.original_print = builtins.print
    
    def __call__(self, *args, **kwargs):
        # 强制设置file=sys.stderr和flush=True
        kwargs['file'] = sys.stderr
        kwargs['flush'] = True
        return self.original_print(*args, **kwargs)


def enable_stderr_print():
    """启用全局print重定向到stderr"""
    builtins.print = StderrPrint()


def disable_stderr_print():
    """禁用全局print重定向，恢复原始print"""
    if hasattr(builtins.print, 'original_print'):
        builtins.print = builtins.print.original_print


def format_error_print(error: Exception, context: str = "解析错误") -> None:
    """标准化的错误打印格式"""
    print(f"{context}: {error}")


# 上下文管理器版本
class StderrPrintContext:
    """临时重定向print到stderr的上下文管理器"""
    
    def __enter__(self):
        self.original_print = builtins.print
        builtins.print = StderrPrint()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.print = self.original_print


# 装饰器版本
def stderr_print_decorator(func):
    """装饰器：在函数执行期间将print重定向到stderr"""
    def wrapper(*args, **kwargs):
        with StderrPrintContext():
            return func(*args, **kwargs)
    return wrapper 