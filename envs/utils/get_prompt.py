import copy
import random
from abc import abstractmethod
from qwen_agent.tools.base import BaseTool
from qwen_agent.llm.base import ModelServiceError
from qwen_agent.tools.mcp_manager import MCPManager
from qwen_agent.settings import DEFAULT_MAX_INPUT_TOKENS
from qwen_agent.utils.tokenization_qwen import tokenizer
from typing import List, Dict, Optional, Literal, Union, Iterator
from qwen_agent.utils.utils import extract_text_from_message, format_as_text_message, has_chinese_messages
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, SYSTEM, USER, FUNCTION, ContentItem, Message


def _truncate_input_messages_roughly(messages: List[Message], max_tokens: int) -> List[Message]:
    if len([m for m in messages if m.role == SYSTEM]) >= 2:
        raise ModelServiceError(
            code='400',
            message='The input messages must contain no more than one system message. '
            ' And the system message, if exists, must be the first message.',
        )

    turns = []
    for m in messages:
        if m.role == SYSTEM:
            continue
        elif m.role == USER:
            turns.append([m])
        else:
            if turns:
                turns[-1].append(m)
            else:
                raise ModelServiceError(
                    code='400',
                    message='The input messages (excluding the system message) must start with a user message.',
                )

    def _count_tokens(msg: Message) -> int:
        return tokenizer.count_tokens(extract_text_from_message(msg, add_upload_info=True))

    def _truncate_message(msg: Message, max_tokens: int, keep_both_sides: bool = False):
        if isinstance(msg.content, str):
            content = tokenizer.truncate(msg.content, max_token=max_tokens, keep_both_sides=keep_both_sides)
        else:
            text = []
            for item in msg.content:
                if not item.text:
                    return None
                text.append(item.text)
            text = '\n'.join(text)
            content = tokenizer.truncate(text, max_token=max_tokens, keep_both_sides=keep_both_sides)
        return Message(role=msg.role, content=content)
    
    if messages and messages[0].role == SYSTEM:
        sys_msg = messages[0]
        available_token = max_tokens - _count_tokens(sys_msg)
    else:
        sys_msg = None
        available_token = max_tokens
    
    token_cnt = 0
    new_messages = []
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == SYSTEM:
            continue
        cur_token_cnt = _count_tokens(messages[i])
        if cur_token_cnt <= available_token:
            new_messages = [messages[i]] + new_messages
            available_token -= cur_token_cnt
        else:
            if (messages[i].role == USER) and (i != len(messages) - 1):
                _msg = _truncate_message(messages[i], max_tokens=available_token)
                if _msg:
                    new_messages = [_msg] + new_messages
                break
            elif messages[i].role == FUNCTION:
                _msg = _truncate_message(messages[i], max_tokens=available_token, keep_both_sides=True)
                if _msg:
                    new_messages = [_msg] + new_messages
                else:
                    break
            else:
                token_cnt = (max_tokens - available_token) + cur_token_cnt
                break
    
    if sys_msg is not None:
        new_messages = [sys_msg] + new_messages

    if (sys_msg is not None and len(new_messages) < 2) or (sys_msg is None and len(new_messages) < 1):
        raise ModelServiceError(
            code='400',
            message=f'The input messages exceed the maximum context length ({max_tokens} tokens) after '
            f'keeping only the system message (if exists) and the latest one user message (around {token_cnt} tokens). '
            'To configure the context limit, please specifiy "max_input_tokens" in the model generate_cfg. '
            f'Example: generate_cfg = {{..., "max_input_tokens": {(token_cnt // 100 + 1) * 100}}}',
        )
    return new_messages


def _remove_fncall_messages(messages: List[Message], lang: Literal['en', 'zh']) -> List[Message]:
        # Change function calls into user messages so that the model won't try
        # to generate function calls when given functions and function_choice="none".
        new_messages = []
        for msg in messages:
            if (msg.role == FUNCTION) or msg.function_call:
                if (not new_messages) or (new_messages[-1].role != USER):
                    new_messages.append(Message(role=USER, content=[]))
                if msg.function_call:
                    tool_name = msg.function_call.name
                    tool_args = msg.function_call.arguments
                    if lang == 'zh':
                        tool_text = f'\n\n工具"{tool_name}"被调用时使用了以下参数：\n{tool_args}'
                    else:
                        tool_text = f'\n\nThe tool "{tool_name}" was called with these arguments:\n{tool_args}'
                else:
                    assert msg.role == FUNCTION
                    if msg.content:
                        assert len(msg.content) == 1
                        assert isinstance(msg.content[0], ContentItem)
                        assert isinstance(msg.content[0].text, str)
                        tool_result = msg.content[0].text
                    else:
                        tool_result = 'No result.'
                    if lang == 'zh':
                        tool_text = f'\n\n该工具返回了以下结果：\n{tool_result}'
                    else:
                        tool_text = f'\n\nThe tool has returned the following result:\n{tool_result}'
                new_messages[-1].content.append(ContentItem(text=tool_text))
            else:
                if (msg.role == USER) and new_messages and (new_messages[-1].role == USER):
                    # Separate two user messages with an assistant message to make the bot focus on the latter:
                    new_messages.append(Message(role=ASSISTANT, content=[ContentItem(text='...')]))
                new_messages.append(msg)
        return new_messages


def _preprocess_messages(
        messages: List[Message],
        lang: Literal['en', 'zh'],
        generate_cfg: dict,
        functions: Optional[List[Dict]] = None,
    ) -> List[Message]:
        fncall_prompt_type = generate_cfg.get('fncall_prompt_type', 'nous')
        if fncall_prompt_type == 'qwen':
            from qwen_agent.llm.fncall_prompts.qwen_fncall_prompt import FN_STOP_WORDS, QwenFnCallPrompt
            fncall_prompt = QwenFnCallPrompt()
            stop = generate_cfg.get('stop', [])
            generate_cfg['stop'] = stop + [x for x in FN_STOP_WORDS if x not in stop]
        elif fncall_prompt_type == 'nous':
            from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt
            fncall_prompt = NousFnCallPrompt()
        else:
            raise NotImplementedError
        if 'fncall_prompt_type' in generate_cfg:
            del generate_cfg['fncall_prompt_type']


        if (not functions) or (generate_cfg.get('function_choice', 'auto') == 'none'):
            messages = _remove_fncall_messages(messages, lang=lang)
        else:
            # validate_num_fncall_results(
            #     messages=messages,
            #     support_multimodal_input=self.support_multimodal_input,
            # )
            messages = fncall_prompt.preprocess_fncall_messages(
                messages=messages,
                functions=functions,
                lang=lang,
                parallel_function_calls=generate_cfg.get('parallel_function_calls', False),
                function_choice=generate_cfg.get('function_choice', 'auto'),
            )
        return messages


def _chat(
        messages: List[Union[Message, Dict]],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Union[List[Message], Iterator[List[Message]]]:
        if stream:
            return _chat_stream(messages, delta_stream=delta_stream, generate_cfg=generate_cfg)
        else:
            return _chat_no_stream(messages, generate_cfg=generate_cfg)


def _chat_with_functions(
        messages: List[Message],
        functions: List[Dict],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
        lang: Literal['en', 'zh'],
    ) -> Union[List[Message], Iterator[List[Message]]]:
        if delta_stream:
            raise NotImplementedError('Please use stream=True with delta_stream=False, because delta_stream=True'
                                      ' is not implemented for function calling due to some technical reasons.')
        generate_cfg = copy.deepcopy(generate_cfg)
        for k in ['parallel_function_calls', 'function_choice', 'thought_in_content']:
            if k in generate_cfg:
                del generate_cfg[k]
        return _continue_assistant_response(messages, generate_cfg=generate_cfg, stream=stream)


def _continue_assistant_response(
    messages: List[Message],
    generate_cfg: dict,
    stream: bool,
) -> Iterator[List[Message]]:
    messages = simulate_response_completion_with_chat(messages)
    return _chat(messages, stream=stream, delta_stream=False, generate_cfg=generate_cfg)


def simulate_response_completion_with_chat(messages: List[Message]) -> List[Message]:
    if messages and (messages[-1].role == ASSISTANT):
        assert (len(messages) > 1) and (messages[-2].role == USER)
        assert messages[-1].function_call is None
        usr = messages[-2].content
        bot = messages[-1].content
        sep = '\n\n'
        if isinstance(usr, str) and isinstance(bot, str):
            usr = usr + sep + bot
        elif isinstance(usr, list) and isinstance(bot, list):
            usr = usr + [ContentItem(text=sep)] + bot
        else:
            raise NotImplementedError
        text_to_complete = copy.deepcopy(messages[-2])
        text_to_complete.content = usr
        messages = messages[:-2] + [text_to_complete]
    return messages


def _chat_stream(
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
    messages = convert_messages_to_dicts(messages)
    #response = self._chat_complete_create(model=self.model, messages=messages, stream=True, **generate_cfg)
    return messages
            

def _chat_no_stream(
    messages: List[Message],
    generate_cfg: dict,
) -> List[Message]:
    messages = convert_messages_to_dicts(messages)
    #response = _chat_complete_create(model=model, messages=messages, stream=False, **generate_cfg)
    return messages


def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
    # TODO: Change when the VLLM deployed model needs to pass reasoning_complete.
    #  At this time, in order to be compatible with lower versions of vLLM,
    #  and reasoning content is currently not useful
    messages = [msg.model_dump(exclude={'reasoning_content'}) for msg in messages]

    return messages


def build_prompt_from_dict(messages: List[dict]) -> str:
    """Convert a list of message dictionaries to a prompt string.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
        
    Returns:
        A formatted prompt string.
    """
    prompt = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return prompt


def get_prompt(
        messages: List[Message],
        generate_cfg: dict,
        stream: bool = True,
        delta_stream: bool = False,
        functions: Optional[List[Dict]] = None,
    ) -> List[Message]:
    """Convert messages to prompt string.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
            Example: [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]
        generate_cfg: Generation configuration dictionary.
            Example: {'fncall_prompt_type': 'nous', 'function_choice': 'auto', 'lang': 'en'}
        stream: Whether to use streaming generation. Default is True.
        delta_stream: Whether to stream the response incrementally. Default is False.
        functions: Optional list of function definitions for function calling.
            Example: [{'name': 'function_name', 'description': '...', 'parameters': {...}}]

    Returns:
        A formatted prompt string.
    """

        # Unify the input messages to type List[Message]:
    messages = copy.deepcopy(messages)
    _return_message_type = 'dict'
    #print('check1:{}'.format(messages))
    new_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            new_messages.append(Message(**msg))
        else:
            new_messages.append(msg)
            _return_message_type = 'message'
    messages = new_messages

    if not messages:
        raise ValueError("Messages can not be empty.")
    #print('check2:{}'.format(messages))

    if 'seed' not in generate_cfg:
        generate_cfg['seed'] = random.randint(a=0, b=2**30)
    if 'lang' in generate_cfg:
        lang: Literal['en', 'zh'] = generate_cfg.pop('lang')
    else:
        lang: Literal['en', 'zh'] = 'zh' if has_chinese_messages(messages) else 'en'
    if not stream and 'incremental_output' in generate_cfg:
        generate_cfg.pop('incremental_output')

    if DEFAULT_SYSTEM_MESSAGE and messages[0].role != SYSTEM:
        messages = [Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE)] + messages
    # Not precise. It's hard to estimate tokens related with function calling and multimodal items.
    max_input_tokens = generate_cfg.pop('max_input_tokens', DEFAULT_MAX_INPUT_TOKENS)
    if max_input_tokens > 0:
        messages = _truncate_input_messages_roughly(
            messages=messages,
            max_tokens=max_input_tokens,
        )
    #print('check3:{}'.format(messages))
    
    if functions:
        fncall_mode = True
    else:
        fncall_mode = False
    if 'function_choice' in generate_cfg:
        fn_choice = generate_cfg['function_choice']
        valid_fn_choices = [f.get('name', f.get('name_for_model', None)) for f in (functions or [])]
        valid_fn_choices = ['auto', 'none'] + [f for f in valid_fn_choices if f]
        if fn_choice not in valid_fn_choices:
            raise ValueError(f'The value of function_choice must be one of the following: {valid_fn_choices}. '
                                f'But function_choice="{fn_choice}" is received.')
        if fn_choice == 'none':
            fncall_mode = False
    # Note: the preprocessor's behavior could change if it receives function_choice="none"
    
    #print('checkfinal:{}'.format(messages))
    messages = _preprocess_messages(messages, lang=lang, generate_cfg=generate_cfg, functions=functions)
    messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]
    print('check:{}'.format(messages))
    if not fncall_mode:
            for k in ['parallel_function_calls', 'function_choice', 'thought_in_content']:
                if k in generate_cfg:
                    del generate_cfg[k]
    if fncall_mode:
        messages = _chat_with_functions(
            messages=messages,
            functions=functions,
            stream=stream,
            delta_stream=delta_stream,
            generate_cfg=generate_cfg,
            lang=lang,
        )
    else:
        # TODO: Optimize code structure
        if messages[-1].role == ASSISTANT:
            assert not delta_stream, 'Continuation mode does not currently support `delta_stream`'
            messages = _continue_assistant_response(messages, generate_cfg=generate_cfg, stream=stream)
        else:
            messages = _chat(
                messages,
                stream=stream,
                delta_stream=delta_stream,
                generate_cfg=generate_cfg,
            )
    #print('checkfinal:{}'.format(messages))
    prompt = build_prompt_from_dict(messages)
    return prompt


# Generate functions for testing
def _init_tool(function_list):
    function_map = {}
    if function_list:
        for tool in function_list:
            if isinstance(tool, BaseTool):
                tool_name = tool.name
                function_map[tool_name] = tool
            elif isinstance(tool, dict) and 'mcpServers' in tool:
                tools = MCPManager().initConfig(tool)
                for tool in tools:
                    tool_name = tool.name
                    function_map[tool_name] = tool
    
    functions=[func.function for func in function_map.values()]
    return functions
