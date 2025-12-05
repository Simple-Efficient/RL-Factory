import asyncio
import json
import traceback
from typing import List, Union
from abc import ABC, abstractmethod

from envs.storage.manager.storage_manager import create_config_storage_manager
from envs.utils.util import ToolServiceError, DocParserError


class ToolManager(ABC):
    """
    Abstract base class for managing tools in reinforcement learning environments.
    
    This class provides the foundation for tool management systems that handle
    execution of various tools by agents in RL environments. It supports tool
    registration, storage management, and tool calling functionality.
    
    Attributes:
        verl_config (dict): Configuration dictionary for the VERL framework
        tool_map (dict): Mapping of tool names to tool instances
        storage_manager: Optional storage manager for caching tool results
        
    The class handles:
    - Tool registration and retrieval
    - Tool name resolution (full names vs short names)
    - Tool execution with error handling
    - Optional caching of tool results
    - Asynchronous storage operations
    """
    def __init__(self, verl_config) -> None:
        self.verl_config = verl_config
        self.tool_map = {}
        self._build_tools()
        if self.verl_config.get("use_storage_manager", False):
            self.build_storage_manager(self.verl_config)
        else:
            self.storage_manager = None

    def build_storage_manager(self, verl_config):
        self.storage_manager = create_config_storage_manager(verl_config)

    def _full_name_str(self, name):
        if name in self.tool_map.keys():
            return name
        else:
            for key in self.tool_map.keys():
                # matching the 2nd half of the name
                if "-" + name in key:
                    return key
            return name

    def _full_name(self, name):
        if isinstance(name, str):
            return self._full_name_str(name)
        if isinstance(name, dict):
            # modify the "name"
            name["name"] = self._full_name_str(name["name"])
            return name
    
    def full_name(self, name):
        if isinstance(name, list):
            return [self._full_name(n) for n in name]
        return self._full_name(name)

    def get_tool(self, name_or_short_name: str):
        """Get tool by name or short name
        Args:
            name_or_short_name: Tool name or short name
            
        Returns:
            The found tool, or None if not found
        """
        name_or_short_name = str(name_or_short_name)
        return self.tool_map.get(name_or_short_name, None)

    @property
    @abstractmethod
    def all_tools(self):
        raise NotImplementedError

    @abstractmethod
    def _build_tools(self):
        raise NotImplementedError

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs):
        """The interface of calling tools for the agent.

          Args:
              tool_name: The name of one tool.
              tool_args: Model generated or user given tool parameters.

          Returns:
              The output of tools.
          """
        if tool_name not in self.tool_map:
            return f'Tool {tool_name} does not exists.'
        tool = self.get_tool(tool_name)

        tool_result = None
        try:
            if self.storage_manager is not None:
                tool_result = asyncio.get_event_loop().run_until_complete(self.storage_manager.get(tool_name, tool_args))
            if tool_result is not None:
                return tool_result
            else:
                tool_result = tool.call(tool_args, **kwargs)

            if self.storage_manager is not None:
                asyncio.get_event_loop().run_until_complete(self.storage_manager.set(tool_name, tool_args, tool_result, ttl=30))

        except (ToolServiceError, DocParserError) as ex:
            raise ex
        except Exception as ex:
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            print(f"error_message: {error_message}")
            return error_message

        if isinstance(tool_result, str):
            return tool_result
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    @abstractmethod
    def execute_actions(self, responses: List[str]):
        raise NotImplementedError
