# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from typing import Any


import torch


def set_basic_config(level):
    """
    This function sets the global logging format and level. It will be called when import verl
    """
    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level=level)


def log_to_file(string):
    print(string)
    if os.path.isdir("logs"):
        with open(f"logs/log_{torch.distributed.get_rank()}", "a+") as f:
            f.write(string + "\n")


def error_print(message: Any, *args, **kwargs) -> None:
    """
    统一的错误输出函数，所有消息都输出到stderr并立即刷新
    
    Args:
        message: 要打印的消息
        *args: 额外的位置参数
        **kwargs: 额外的关键字参数（会被忽略，因为我们强制使用stderr和flush=True）
    """
    # 忽略传入的file和flush参数，强制使用我们的标准
    print(message, *args, file=sys.stderr, flush=True)


def format_error(error: Exception, context: str = "") -> None:
    """
    格式化错误输出
    
    Args:
        error: 异常对象
        context: 错误上下文描述
    """
    if context:
        error_print(f"{context}: {error}")
    else:
        error_print(f"解析错误: {error}")


# 为了向后兼容，也可以直接替换print
def unified_print(message: Any, *args, **kwargs) -> None:
    """
    统一的打印函数，强制输出到stderr
    """
    print(message, *args, file=sys.stderr, flush=True)
