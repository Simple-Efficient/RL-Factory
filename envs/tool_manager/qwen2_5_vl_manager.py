import re
import copy
import json
import json5, sys
import asyncio
import traceback
import base64
from PIL import Image
from ast import literal_eval
from omegaconf import OmegaConf
from envs.tool_manager.base_manager import ToolManager
from envs.utils.mcp_manager import MCPManager
from typing import Union, List, Tuple, Optional, Any, Dict
from envs.utils.util import ToolServiceError, DocParserError
# from qwen_agent.tools import TOOL_REGISTRY, MCPManager, BaseTool
from qwen_agent.tools import TOOL_REGISTRY, BaseTool
from qwen_agent.llm.schema import ASSISTANT, SYSTEM, USER, FUNCTION, ContentItem
import json
import io
import base64


def parse_mcp_tools_config(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        # 使用 literal_eval 安全地解析 Python 字面量
        data = literal_eval(content)
        return data
    except Exception as e:
        print(f"解析错误: {e}",file=sys.stderr, flush=True)
        return None


class Qwen25VLManager(ToolManager):    
    def __init__(self, verl_config):
        if isinstance(verl_config, dict):
            verl_config = OmegaConf.to_container(verl_config)
        super().__init__(verl_config)
        # self.functions = []
        self.generate_cfg = {
            'fncall_prompt_type': 'nous',
            'function_choice': 'auto',  # 注释掉这行
            'parallel_function_calls': False,
            'lang': 'en',
            'max_input_tokens': 10000
        }

    def get_tool(self, name_or_short_name: str):
        """通过名称或简写获取工具
        
        Args:
            name_or_short_name: 工具名称或简写
            
        Returns:
            找到的工具，如果没找到则返回None
        """
        name_or_short_name = str(name_or_short_name)
        return self.tool_map.get(name_or_short_name, None)
    
    @property
    def all_tools(self):
        """获取所有工具
        
        Returns:
            所有工具的列表
        """
        return self.tool_map

    def _build_tools(self):
        config_path = self.verl_config.config_path
        if config_path is not None:
            function_list = parse_mcp_tools_config(config_path)

            if function_list:
                for tool in function_list:
                    self._init_tool(tool)
            
            self.functions = [func.function for func in self.tool_map.values()]
        else:
            print("The config_path is None!")
            self.functions = []

    async def execute_all_tools(self, actions, tool_list, image_data: List[List[Image.Image]]):
        """异步并行执行所有工具列表
        
        Args:
            tool_list: 工具列表的列表
            
        Returns:
            所有工具执行结果的列表
        """
        
        # 并行执行每个工具列表
        tasks = []
        for temp_action, temp_tool_list, temp_image_data in zip(actions, tool_list, image_data):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list, temp_image_data))
        
        results = await asyncio.gather(*tasks)
        
        return results
        
    async def _execute_tool_batch(self, action, tools, image_data: List[Image.Image]):
        """异步并行执行一批工具
        
        Args:
            tools: 工具列表
            
        Returns:
            工具执行结果的列表
        """        
        async def execute_single_tool(tool, image_data:Image.Image):
            
            tool_instance = self.get_tool(tool["name"])
            args = tool["args"]
            if tool_instance is not None:
                try:
                    args = json.loads(args)
                except Exception as e:
                    pass

                if type(args) is dict:
                    try:
                        # 使用asyncio.to_thread包装self._call_tool以保持异步特性
                        imqge_result = await asyncio.to_thread(
                            self._call_tool, 
                            tool["name"], image_data, json.dumps(args, ensure_ascii=False, indent=4)
                        )
                        
                        assert isinstance(imqge_result,  Image.Image), f"tool_result is not a PIL.Image.Image instance, got {type(tool_result)}"
                        text_result = [
                            {"type": "text", "text": f"Execute the tool {tool['name']} successed. The args are: {args}. The image result is: "},
                            {"type": "image"}
                        ]
                        return (text_result, imqge_result)
                        
                        

                    except Exception as e:
                        result = f"Execute the tool {tool['name']} failed. The original args are: {args}. Error message: {str(e)}"                        
            else:
                result = f"Failed to find the tool {tool['name']} in the tool map."
                
            if isinstance(result, str):
                return [{"type": "text", "text": result}]
            else:
                return result

        
        if action == 'answer':
            # 'tools' is a str (the answer)
            results = {'role': 'assistant', 'content': tools}
        elif action == 'error':
            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            # ('Cannot extract the actions tag' or 'There is no action after extraction')
            results = {'role': 'assistant', 'content': """# Extract the tools failed due to: {}""".format(tools)}
        elif action == 'actions':
            # 'tools' is the list of the 'Tool' instances
            tasks = [execute_single_tool(temp_tool, image_data[-1]) for temp_tool in tools]
            tool_results = await asyncio.gather(*tasks)
            # breakpoint()
            results = [{'role': 'tool', 'content': temp_tool_result} for temp_tool_result in tool_results]
        else:
            raise ValueError('Unexpected action: {}'.format(action))

        return results
    
    def _call_tool(self, tool_name: str, image_data: Image.Image = None, **kwargs) -> Union[str, List[ContentItem]]:
        """The interface of calling tools for the agent.

        Args:
            tool_name: The name of one tool.
            image_data: Image data for the tool.

        Returns:
            The output of tools.
        """
        if tool_name not in self.tool_map:
            return f'Tool {tool_name} does not exists.'
        tool = self.get_tool(tool_name)
        # breakpoint()
        try:
            tool_args = json.dumps({"img_base64": self.pil_to_base64(image_data), "instruction": "top"})
            tool_result = tool.call(tool_args)
            # import pdb; pdb.set_trace()

        except (ToolServiceError, DocParserError) as ex:
            raise ex
        except Exception as ex:
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            return error_message

        # 如果工具结果是PIL图像，直接返回
        if self.is_base64(tool_result):
            # print("tool_result is base64", file=sys.stderr,flush=True)
            return self.base64_to_pil(tool_result)
        elif isinstance(tool_result, str):
            return tool_result
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    def _init_tool(self, tool: Union[str, BaseTool]):
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            self.tool_map[tool_name] = tool
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            tools = MCPManager().initConfig(tool)
            for tool in tools:
                tool_name = tool.name
                if "-" in tool_name:
                    tool_name = tool_name[0:len(tool_name)//2]
                self.tool_map[tool_name] = tool
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                tool_name = tool
                tool_cfg = None
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')

            self.tool_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)

    def execute_actions(self, responses: List[str], image_data: List[List[Image.Image]]):
        
        actions, tools = [], []
        for response in responses:
            temp_action, temp_tool_list = self.parse_response(response_content=response)
            # temp_action: answer or tools
            # if temp_action is 'answer', temp_tool_list is the answer
            # else, temp_tool_list is the list of the 'Tool' instances
            actions.append(temp_action)
            tools.append(temp_tool_list)

        # 使用asyncio.run同步运行异步函数

        try:
            tool_results = asyncio.run(self.execute_all_tools(actions, tools, image_data))
        except RuntimeError:
            # 如果事件循环已经在运行，则获取当前循环
            loop = asyncio.get_event_loop()
            tool_results = loop.run_until_complete(self.execute_all_tools(actions, tools, image_data))
        
        return actions, tool_results
    
    def parse_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """执行动作
        
        Args:
            response_content: 响应文本
            
        Returns:
            解析后的动作信息，包含name和args（如果存在）
            如果没有动作标签或格式不正确，返回None
        """
        # 提取answers
        if_answer, answer = self.parse_end_flag(response_content)
        if if_answer:
            return 'answer', answer
        
        # 提取tools
        tools = self.parse_tools(response_content)
        if type(tools) == list:
            return 'actions', tools
        else:
            assert type(tools) == str
            # if the response is not a tool call, it is an answer
            return 'answer', tools
    
    def parse_end_flag(self, response_content: str) -> bool:
        answer = None
        answer_section = re.findall(r'(<answer>.*?</answer>)', response_content, re.DOTALL)
        if len(answer_section) > 0:
            answer = answer_section[-1]
            return True, answer
        else:
            return False, None
        

    def parse_tools(self, response: str):
        parsed_tools = []
        i = response.find('<tool_call>')
        
        # If no function call, return original response
        if i < 0:
            return response

        # Split tool-call to separate assistant msg
        tool_call_list = response.split('<tool_call>')
        pre_thought = tool_call_list[0].strip()
        
        # Only process the first valid tool call
        for txt in tool_call_list[1:]:
            if not txt.strip():
                continue

            if '</tool_call>' not in txt:
                # Incomplete tool call
                parsed_tools.append({
                    "name": "<empty>",
                    "args": "Extract the tool name failed"
                })
                break  # Stop after the first tool call (even if incomplete)
            else:
                one_tool_call_txt = txt.split('</tool_call>')
                try:
                    if not one_tool_call_txt[0].strip():
                        raise ValueError("Empty tool call content")
                    
                    # Try to parse JSON
                    fn = json5.loads(one_tool_call_txt[0].strip())

                    # Only append the first valid tool
                    # 检查必须字段是否存在
                    if type(fn) is not dict or 'name' not in fn or 'arguments' not in fn:
                        raise KeyError("Missing required fields")
                    parsed_tools.append({
                        "name": fn['name'],
                        "args": json.dumps(fn['arguments'], ensure_ascii=False, indent=4),
                    })
                    break  # Stop after the first valid tool call
                
                except (IndexError, KeyError, ValueError) as e:
                    parsed_tools.append({
                        "name": "<empty>",
                        "args": "Extract the tool name failed"
                    })
                    break  # Stop after the first error

        return parsed_tools

    
    def get_prompt(self, input_data, tokenizer, processor, mode='initial', add_generation_prompt=True):
        assert mode in ['initial', 'tool_call', 'assistant_response','multimodal_tool_call'], 'Invalid mode: {}'.format(mode)
        base_chat = [
            {'role': SYSTEM, 'content': 'base'},
            {'role': USER, 'content': 'base'},
        ]
        base_prompt = tokenizer.apply_chat_template(
            conversation=base_chat,
            tools=self.functions,
            tokenize=False, add_generation_prompt=False
        )

        if mode == 'initial':
            chat = input_data
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tokenize=False, tools=self.functions, 
                add_generation_prompt=add_generation_prompt
            )
        elif mode in ['tool_call', 'assistant_response']:
            # NOTE: the assistant response might not be used
            role = 'tool' if mode == 'tool_call' else ASSISTANT
            if type(input_data) == str:
                chat = {'role': role, 'content': input_data}
            elif type(input_data) == list:
                chat = input_data
            else:
                raise ValueError('Unexpected type of input_data {} ({})'.format(type(input_data), input_data))
            
            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=base_chat + chat, tools=self.functions, 
                tokenize=False, add_generation_prompt=add_generation_prompt
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        elif mode == 'multimodal_tool_call':
            chat = input_data
            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=base_chat + chat, tools=[func.function for func in self.tool_map.values()], 
                tokenize=False, add_generation_prompt=add_generation_prompt
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        else:
            raise ValueError('Invalid mode: {}'.format(mode))
        
        return prompt_with_chat_template
    
    def pil_to_base64(self, image):
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return img_base64
    
    def base64_to_pil(self, base64_str):
        img_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(img_data))
        assert isinstance(image, Image.Image), "image is not an Image.Image"
        return image
   

    def is_base64(self, s):
        head = base64.b64decode(s[:100], validate=True)
        return head.startswith((b'\xff\xd8\xff', b'\x89PNG\r\n\x1a\n', b'GIF87a', b'GIF89a', b'BM'))
    
    def _extract_image_data(self, data):
            """
            从数据中提取第一个图像数据
            
            Args:
                data: 包含图像的数据结构
                
            Returns:
                PIL.Image: 提取出的第一个图像，如果没有找到则返回None
            """
            if isinstance(data, Image.Image):
                return data
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, Image.Image):
                        return item
                    elif isinstance(item, dict):
                        # 检查字典中是否包含图像
                        if 'image' in item and isinstance(item['image'], Image.Image):
                            return item['image']
                        # 递归提取嵌套结构中的图像
                        nested_image = self._extract_image_data(item)
                        if nested_image is not None:
                            return nested_image
                    elif isinstance(item, list):
                        # 递归提取嵌套列表中的图像
                        nested_image = self._extract_image_data(item)
                        if nested_image is not None:
                            return nested_image
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, Image.Image):
                        return value
                    elif key in ['image', 'image_url', 'img'] and isinstance(value, Image.Image):
                        return value
                    elif isinstance(value, (list, dict)):
                        # 递归提取嵌套结构中的图像
                        nested_image = self._extract_image_data(value)
                        if nested_image is not None:
                            return nested_image
            
            return None