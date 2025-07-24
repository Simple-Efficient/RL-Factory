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
import io
import random
import argparse
import os
import datasets
import logging
import os
from PIL import Image, ImageFile
from verl.utils.hdfs_io import copy, makedirs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - PROCESS %(process)d - %(message)s')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

instruction_following = (
    r"Given a question and an image, answer the question based strictly on visual content. "
    r"Any time you receive new information, you should reason step by step inside the <think> and </think> XML tag. "
    r"Afterwards, you can either choose to call tool functions or directly provide the answer. "
    r"If the input is an inverted/rotated image, automatically detect its orientation and use the tool to correct it to a standard upright position. "
    "Only after correction can you provide the final answer wrapped with <answer></answer> XML tag. All response must be in English and final answer should be brief and concise wrapped with <answer></answer> XML tag. \n"
)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data/user/qxiao183/qxiao183test2/jjw/datasets/textvqav4")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # data_source = "/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/huggingface.co/datasets/hiyouga/geometry3k"
    data_source = "/data/user/qxiao183/qxiao183test2/jjw/datasets/textvqa/train-00001-of-00020.parquet"
    data_source1 = "/data/user/qxiao183/qxiao183test2/jjw/datasets/textvqa/train-00002-of-00020.parquet"
    train_dataset = datasets.load_dataset("parquet",data_files = data_source)["train"]
    test_dataset = datasets.load_dataset("parquet",data_files = data_source1)["train"]


    # Allow loading of truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    def make_map_fn(split):
        def process_fn(example, idx):
            try:
                problem = example.pop("question")
                prompt = "Question:" + "<image>" +  problem
                answer = example.pop("answers")
                images = example.pop("image")
                images = images.resize((1024,1024), Image.Resampling.BILINEAR)
                angle = random.choice([90, 180, 270])
                images = images.rotate(angle)
                    
                    
                data = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "system",
                            "content": ("You are a helpful assistant. ")  + instruction_following
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "images": images,
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
                return None  
            
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=10)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True,num_proc=10)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

