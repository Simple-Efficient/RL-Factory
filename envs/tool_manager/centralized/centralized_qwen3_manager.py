import ray
import json
import torch
import asyncio
import datetime
from typing import List, Optional, Any, Dict
from envs.tool_manager.qwen3_manager import QwenManager


class CentralizedQwenManager(QwenManager):
    """集中式QwenManager - 工具调用转发到Ray Actor"""
    
    def __init__(self, verl_config, centralized_actor_handle=None, mock=False):
        if not mock and centralized_actor_handle is None:
            raise ValueError("集中式QwenManager需要centralized_actor_handle参数")
        
        # 先调用父类初始化，获得完整的基础功能
        super().__init__(verl_config)
        
        # 保存集中式Actor句柄
        self.centralized_actor_handle = centralized_actor_handle
        print("集中式QwenManager初始化完成，工具调用将转发到集中式Actor")
        
    def execute_actions(self, responses: List[str]):
        assert self.centralized_actor_handle is not None, "集中式QwenManager需要centralized_actor_handle参数"
        """执行动作 - 转发到集中式Actor"""
        actions, tools = [], []
        for response in responses:
            temp_action, temp_tool_list = self.parse_response(response_content=response)
            # temp_action: answer or tools
            # if temp_action is 'answer', temp_tool_list is the answer
            # else, temp_tool_list is the list of the 'Tool' instances
            actions.append(temp_action)
            tools.append(temp_tool_list)

        try:
            tool_results = asyncio.run(self.execute_all_tools(actions, tools))
        except RuntimeError:
            # 如果事件循环已经在运行，则获取当前循环
            loop = asyncio.get_event_loop()
            tool_results = loop.run_until_complete(self.execute_all_tools(actions, tools))
        
        return actions, tool_results
    
    async def execute_all_tools(self, actions, tool_list):
        """异步并行执行所有工具列表
        
        Args:
            tool_list: 工具列表的列表
            
        Returns:
            所有工具执行结果的列表
        """
        
        # 并行执行每个工具列表
        tasks = []
        for temp_action, temp_tool_list in zip(actions, tool_list):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list))
        
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def _execute_tool_batch(self, action, tools):
        if action == 'answer':
            # 'tools' is a str (the answer)
            results = {'role': 'assitant', 'content': tools}
        elif action == 'error':
            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            # ('Cannot extract the actions tag' or 'There is no action after extraction')
            results = {'role': 'assitant', 'content': """# Extract the tools failed due to: {}""".format(tools)}
        elif action == 'actions':
            # 'tools' is the list of the 'Tool' instances
            object_refs = [self.centralized_actor_handle.execute_single_tool.remote(temp_tool) for temp_tool in tools]
            tool_results = await asyncio.gather(*object_refs)
            results = [{'role': 'tool', 'content': temp_tool_result} for temp_tool_result in tool_results]
        else:
            raise ValueError('Unexpected action: {}'.format(action))

        return results


@ray.remote(max_concurrency=10)
class CentralizedToolActor:
    """集中式工具管理器Actor - 集中管理所有工具调用的并发"""
    
    def __init__(self, verl_config):
        self.verl_config = verl_config
        self.tool_manager = CentralizedQwenManager(verl_config, mock=True)
        # 设置默认超时时间（秒）
        self.timeout = getattr(verl_config, 'tool_timeout', 30)
        print(f"集中式工具Actor初始化完成，工具执行超时时间：{self.timeout}秒")

    async def execute_single_tool(self, tool):
        tool_instance = self.tool_manager.get_tool(tool["name"])
        args = tool["args"]
        if tool_instance is not None:
            try:
                args = json.loads(args)
            except Exception as e:
                pass

            if type(args) is dict:
                try:
                    try:
                        # 使用asyncio.to_thread包装self._call_tool以保持异步特性，使用asyncio.wait_for添加超时控制
                        tool_result = await asyncio.wait_for(
                            asyncio.to_thread(
                                self.tool_manager._call_tool, 
                                tool["name"], 
                                json.dumps(args, ensure_ascii=False, indent=4)
                            ),
                            timeout=self.timeout
                        )
                        if len(tool_result) == 0:
                            tool_result = "The tool call format is correct, but the return result is empty. Please try other inputs or tools. "
                        result = f"# Execute the tool {tool['name']} successed\n" \
                                f"- The args are: {args}\n" \
                                f"- The result is:\n{tool_result}"
                    except asyncio.TimeoutError:
                        result = f"# Execute the tool {tool['name']} timeout\n" \
                                f"- The tool execution exceeded the timeout limit of {self.timeout} seconds\n" \
                                f"- The original args are: {args}"
                except Exception as e:
                    result = f"# Execute the tool {tool['name']} failed\n" \
                            f"- The original args are: {args}\n" \
                            f"- Error message:\n{str(e)}"
            elif type(args) is str:
                # Json decode error: xxx
                result = args
            else:
                result = 'Unexpected type of args: {} (args: {})'.format(type(args), args)
        else:
            if tool['name'] == '<empty>':
                result = args
            else:
                result = "# Failed to find the tool {} in the tool map".format(tool['name'])
        
        return result