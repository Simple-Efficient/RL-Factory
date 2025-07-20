from .llama3_manager import Llama3Manager
from .config_manager import ConfigManager
from .qwen3_manager import QwenManager
from .qwen2_5_manager import Qwen25Manager
from .qwen2_5_vl_manager import Qwen25VLManager
from .centralized.centralized_qwen3_manager import CentralizedQwenManager



__all__ = ['ConfigManager', 'QwenManager', 'Qwen25Manager','Qwen25VLManager', 'Llama3Manager', 'CentralizedQwenManager']

TOOL_MANAGER_REGISTRY = {
    'config': ConfigManager,
    'qwen3': QwenManager,
    'qwen2_5': Qwen25Manager,
    'qwen2_5_vl': Qwen25VLManager,
    'llama3' : Llama3Manager,
    'centralized_qwen3': CentralizedQwenManager,
}
