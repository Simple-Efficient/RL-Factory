from PIL import Image
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.tools.base import Tool
from mcp.server.fastmcp.utilities.func_metadata import func_metadata
from io import BytesIO
import base64
import binascii
import inspect # For inspecting function signature in decorator
from typing import Any, Callable, List # For type hints

# 初始化 MCP 服务
mcp = FastMCP("ImageRotateServer")

def mcp_tool_with_hidden_params(hidden_params: List[str] = None, **kwargs):
    """
    自定义装饰器，用于创建FastMCP工具并隐藏特定的参数不暴露在inputSchema中。
    它会包装mcp.tool()装饰器，并在其后修改生成的Tool对象的parameters。
    """
    if hidden_params is None:
        hidden_params = []

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # 1. 使用原始的mcp.tool()装饰器来注册工具
        # 这会创建初始的Tool对象，其inputSchema包含所有函数参数
        original_tool_decorator = mcp.tool(**kwargs)
        decorated_func = original_tool_decorator(func)

        # 2. 获取刚刚由mcp.tool()注册的Tool对象
        # 它已经存在于mcp._tool_manager._tools中
        registered_tool = mcp._tool_manager._tools[func.__name__]

        # 3. 收集所有需要隐藏的参数：用户指定的 + Context参数
        all_skip_names = list(hidden_params)
        if registered_tool.context_kwarg and registered_tool.context_kwarg not in all_skip_names:
            all_skip_names.append(registered_tool.context_kwarg)

        # 4. 重新生成参数模型，明确跳过所有all_skip_names中的参数
        # 这将覆盖原始装饰器生成的parameters
        temp_fn_metadata = func_metadata(
            func, # 传入原始函数
            skip_names=all_skip_names
        )
        registered_tool.parameters = temp_fn_metadata.arg_model.model_json_schema(by_alias=True)

        return decorated_func
    return decorator


@mcp_tool_with_hidden_params(hidden_params=["img_base64"])
def rotate(degree: int, img_base64: str) -> str:
    """Rotate the image by a specified angle.

    Args:
        degree (int): The angle of rotation (positive for clockwise, negative for counter-clockwise).

    Returns:
        str: The Base64 encoded string of the rotated image.
    """
    print("================= Calling image_rotate tool ==================")
    print(f"Received parameters: degree={degree}, img_base64 length={len(img_base64) if img_base64 else 0}")

    if img_base64 is None:
        return "Error: Missing image data"

    if degree is None:
        return "Error: Missing degree parameter"

    try:
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data))

        rotated_img = img.rotate(degree, expand=True)

        buffer = BytesIO()
        rotated_img.save(buffer, format='PNG')
        img_base64_output = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return img_base64_output

    except binascii.Error:
        return "Error: Invalid base64 image data"
    except Exception as e:
        return f"Image rotation failed: {str(e)}"


if __name__ == "__main__":
    print("\n启动 MCP 图像旋转服务...")
    mcp.run(transport='stdio')
