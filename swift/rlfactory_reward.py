from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
import asyncio
import re
import textwrap
from copy import deepcopy
from typing import Dict, List, Optional
import math

import json
import torch
from json import JSONDecodeError
from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

from collections import Counter

logger = get_logger()

# infos: [{'rollout_reward': [0.1], 'num_turns': 2}, {'rollout_reward': [0.1], 'num_turns': 2}]
class RLFactoryReward(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        n = len(completions) # batch number

        infos = kwargs.get('rollout_infos', [{}]*n)
        #print("infos:" + str(infos) + "; completions: \n" + str([str(c)+'\n' for c in completions]))
        for i, d in zip(range(n), infos):
            completion = completions[i]
            rw = d.get("rollout_reward", 0)
            if isinstance(rw, list):
                rw = rw[-1]
            if math.isnan(rw):
                rw = 0
            rewards.append(rw)
        return rewards

orms['rlfactory'] = RLFactoryReward