import torch
import time
import psutil
import os
import argparse
import numpy as np
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetCount

def get_gpu_memory_info(device_id=0):
    """获取GPU显存信息"""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    total_memory = info.total / 1024**2  # MB
    used_memory = info.used / 1024**2    # MB
    free_memory = info.free / 1024**2    # MB
    return total_memory, used_memory, free_memory

def get_gpu_utilization(device_id=0):
    """获取GPU利用率"""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_id)
    util = nvmlDeviceGetUtilizationRates(handle)
    return util.gpu  # GPU利用率百分比

def get_gpu_count():
    """获取可用的GPU数量"""
    nvmlInit()
    return nvmlDeviceGetCount()

def occupy_gpu_memory(target_percentage=20, target_util=20, gpu_ids=None):
    """
    占用指定百分比的GPU显存并保持指定的利用率
    
    参数:
        target_percentage: 目标显存占用百分比
        target_util: 目标GPU利用率百分比
        gpu_ids: 要使用的GPU ID列表，如果为None则使用所有可用GPU
    """
    print(f"正在尝试占用 {target_percentage}% 的GPU显存并保持 {target_util}% 的利用率...")
    
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        print("没有可用的GPU!")
        return
    
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"系统中共有 {num_gpus} 个可用的GPU")
    
    # 如果没有指定GPU ID，则使用所有可用的GPU
    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))
    else:
        # 确保所有指定的GPU ID都是有效的
        gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < num_gpus]
        if not gpu_ids:
            print("指定的GPU ID无效!")
            return
    
    print(f"将使用以下GPU: {gpu_ids}")
    
    # 为每个GPU创建一个字典来存储分配的张量和计算张量
    gpu_data = {}
    
    # 初始化每个GPU的数据
    for gpu_id in gpu_ids:
        # 获取GPU总显存
        total_memory, used_memory, free_memory = get_gpu_memory_info(gpu_id)
        print(f"GPU {gpu_id} 总显存: {total_memory:.2f} MB")
        print(f"GPU {gpu_id} 当前已使用: {used_memory:.2f} MB ({used_memory/total_memory*100:.2f}%)")
        print(f"GPU {gpu_id} 当前可用: {free_memory:.2f} MB")
        
        # 计算需要分配的显存大小
        target_memory = (total_memory * target_percentage / 100) - used_memory
        if target_memory <= 0:
            print(f"GPU {gpu_id} 已经使用了超过 {target_percentage}% 的显存，无需额外分配")
            target_memory = 0
        else:
            print(f"GPU {gpu_id} 将分配约 {target_memory:.2f} MB 显存...")
        
        gpu_data[gpu_id] = {
            'total_memory': total_memory,
            'target_memory': target_memory,
            'allocated_memory': 0,
            'allocated_tensors': [],
            'compute_tensors': None
        }
    
    # 每次分配的块大小 (MB)
    block_size = 100  # MB
    
    # 为每个GPU分配显存
    try:
        for gpu_id in gpu_ids:
            # 设置当前设备
            torch.cuda.set_device(gpu_id)
            
            target_memory = gpu_data[gpu_id]['target_memory']
            if target_memory <= 0:
                continue
                
            allocated_memory = 0
            allocated_tensors = []
            
            while allocated_memory < target_memory:
                # 计算当前块的大小
                current_block = min(block_size, target_memory - allocated_memory)
                if current_block <= 0:
                    break
                    
                # 分配显存 (每个float32值占4字节)
                tensor_size = int(current_block * 1024 * 1024 / 4)
                tensor = torch.rand(tensor_size, device=f'cuda:{gpu_id}')
                allocated_tensors.append(tensor)
                allocated_memory += current_block
                
                # 打印进度
                progress = allocated_memory / target_memory * 100
                print(f"GPU {gpu_id} 已分配: {allocated_memory:.2f} MB / {target_memory:.2f} MB ({progress:.2f}%)", end='\r')
            
            print(f"\nGPU {gpu_id} 成功分配了约 {allocated_memory:.2f} MB 显存")
            
            # 更新GPU数据
            gpu_data[gpu_id]['allocated_memory'] = allocated_memory
            gpu_data[gpu_id]['allocated_tensors'] = allocated_tensors
            
            # 获取当前显存使用情况
            _, current_used, _ = get_gpu_memory_info(gpu_id)
            print(f"GPU {gpu_id} 当前显存使用: {current_used:.2f} MB ({current_used/gpu_data[gpu_id]['total_memory']*100:.2f}%)")
            
            # 创建用于计算的张量
            compute_size = 2000  # 调整大小以影响利用率
            a = torch.rand((compute_size, compute_size), device=f'cuda:{gpu_id}')
            b = torch.rand((compute_size, compute_size), device=f'cuda:{gpu_id}')
            gpu_data[gpu_id]['compute_tensors'] = (a, b)
            
    except RuntimeError as e:
        print(f"\n无法分配更多显存: {e}")
    
    # 保持GPU利用率
    print(f"正在执行计算以保持所有GPU利用率在 {target_util}% 左右...")
    
    try:
        while True:
            for gpu_id in gpu_ids:
                # 获取当前GPU利用率
                current_util = get_gpu_utilization(gpu_id)
                
                # 设置当前设备
                torch.cuda.set_device(gpu_id)
                
                # 获取计算张量
                if gpu_data[gpu_id]['compute_tensors'] is None:
                    continue
                    
                a, b = gpu_data[gpu_id]['compute_tensors']
                
                # 根据当前利用率调整计算强度
                if current_util < target_util - 5:
                    # 增加计算强度
                    for _ in range(10):
                        c = torch.matmul(a, b)
                        d = torch.nn.functional.relu(c)
                        e = torch.matmul(d, b)
                elif current_util > target_util + 5:
                    # 降低计算强度，休息一下
                    time.sleep(0.1)
                else:
                    # 保持适中的计算强度
                    c = torch.matmul(a, b)
                    d = torch.nn.functional.relu(c)
                
                # 获取当前显存使用情况
                _, current_used, _ = get_gpu_memory_info(gpu_id)
                memory_percentage = current_used / gpu_data[gpu_id]['total_memory'] * 100
                print(f"GPU {gpu_id} - 显存: {current_used:.2f} MB ({memory_percentage:.2f}%), 利用率: {current_util}%", end='  ')
            
            print("", end='\r')
            # 短暂休息，避免打印过快
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 清理分配的显存
        for gpu_id in gpu_ids:
            gpu_data[gpu_id]['allocated_tensors'].clear()
            gpu_data[gpu_id]['compute_tensors'] = None
        torch.cuda.empty_cache()
        print("\n已释放分配的显存")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='占用指定百分比的GPU显存并保持指定的利用率')
    parser.add_argument('--memory', type=float, default=40.0, help='目标显存占用百分比 (默认: 80%%)')
    parser.add_argument('--util', type=float, default=40.0, help='目标GPU利用率百分比 (默认: 80%%)')
    parser.add_argument('--gpus', type=str, default="1,2", help='要使用的GPU ID，用逗号分隔，例如"0,1,2,3" (默认: 使用所有可用GPU)')
    args = parser.parse_args()
    
    # 解析GPU ID
    gpu_ids = None
    if args.gpus is not None:
        try:
            gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',') if gpu_id.strip()]
        except ValueError:
            print("无效的GPU ID格式，应为逗号分隔的整数")
            exit(1)
    
    occupy_gpu_memory(args.memory, args.util, gpu_ids)
