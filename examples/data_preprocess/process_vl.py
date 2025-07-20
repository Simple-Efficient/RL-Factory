# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import argparse
import logging
import os
import tempfile
import random
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

instruction_following = (
    r"Given a question and an image, answer the question based strictly on visual content. "
    r"Any time you receive new information, you should reason step by step inside the <think> and </think> XML tag. "
    r"Afterwards, you can either choose to call tool functions or directly provide the answer. "
    r"If the input is an inverted/rotated image, automatically detect its orientation and use the tool to correct it to a standard upright position. "
    "Only after correction can you provide the final answer wrapped with <answer></answer> XML tag. All response must be in English and final answer should be brief and concise wrapped with <answer></answer> XML tag. \n"
)
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
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os

import datasets
import logging
import os

# 配置日志，包含进程ID
logging.basicConfig(level=logging.INFO, format='%(asctime)s - PROCESS %(process)d - %(message)s')
from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data/user/qxiao183/qxiao183test2/jjw/datasets/textvqav4")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # data_source = "/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/huggingface.co/datasets/hiyouga/geometry3k"
    data_source = "/data/user/qxiao183/qxiao183test2/jjw/datasets/textvqa/train-00001-of-00020.parquet"
    data_source1 = "/data/user/qxiao183/qxiao183test2/jjw/datasets/textvqa/train-00002-of-00020.parquet"
    # dataset = datasets.load_dataset(data_source)
    # breakpoint()
    train_dataset = datasets.load_dataset("parquet",data_files=data_source)["train"].train_test_split(test_size=0.45,)['train'] 
    # breakpoint()
    test_dataset = datasets.load_dataset("parquet",data_files=data_source1)["train"].train_test_split(test_size=0.45,)['train'] 
    # breakpoint()
    
    from PIL import Image, ImageFile

    # Allow loading of truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # add a row to each data item that represents a unique id
    from typing import List
    def make_map_fn(split):
        def process_fn(example, idx):
            # logging.info(f"开始处理索引 {idx}...")
            try:
                problem = example.pop("question")
                # prompt = problem + " " + instruction_following
                prompt = "Question:" + "<image>" +  problem
                answer = example.pop("answers")
                images = example.pop("image")
                # print(f"image size: {images.size}")
                images = images.resize((1024,1024), Image.Resampling.BILINEAR)
                angle = random.choice([90, 180, 270])
                imgs_pil: List[Image.Image] = []
                if isinstance(images, list):
                    for img in images:
                        if isinstance(img, Image.Image):
                            imgs_pil.append(img)
                        elif isinstance(img, bytes):
                            imgs_pil.append(Image.open(io.BytesIO(img)))
                        elif isinstance(img, str):
                            imgs_pil.append(Image.open(img))
                        # 其它类型可按需求添加转换
                else:
                    if isinstance(images, Image.Image):
                        imgs_pil.append(images)
                    elif isinstance(images, bytes):
                        imgs_pil.append(Image.open(io.BytesIO(images)))
                    elif isinstance(images, str):
                        imgs_pil.append(Image.open(images))
                
                    
                data = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant. "
                                "\n# Tools\nEvery turn you can call one function at most among the following functions to assist with the user query."
                                "\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
                                '{"type": "function", "function": {"name": "rotate", "description": "Rotate a Pillow image by specified degrees", "parameters": {"type": "object", "properties": {"degree": {"type": "integer", "description": "Rotation angle in degrees"}}, "required": ["degree"]}}}\n</tools>\n'
                                'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
                                '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\nThe tool response will wrapped with <tool_response></tool_response> XML tags. '
                                
                                
    )  + instruction_following
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "images": imgs_pil,
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "answer": answer,
                        "question": problem,
                    },
                }
            except Exception as e:
                print(f"Error on example {example.get('id', 'unknown')}: {e}")
                return None  # 跳过错误样本
            
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=10)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True,num_proc=10)

    local_dir = args.local_dir


    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

