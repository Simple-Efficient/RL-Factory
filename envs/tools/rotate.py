
from PIL import Image
from mcp.server.fastmcp import FastMCP,Context
from io import BytesIO
import base64
import binascii

# Initialize MCP server
mcp = FastMCP("ImageRotateServer")


@mcp.tool()
def rotate(degree: int, ctx:Context) -> str:
    """Rotate a Pillow image by specified degrees
    
    Args:
        img_base64: Input image as base64 encoded string
        degree: Rotation angle in degrees (positive for clockwise, negative for counterclockwise)
        
    Returns:
        str: Rotated image as base64 encoded string
    """
    print("================= call image_rotate tool ==================")
    # img_base64 = context.get('img_base64')
    img_base64 = ctx
    # img_base64 = 
    # Validate required parameters
    if img_base64 is None:
        return "⚠️ Error: img_base64 parameter is required"
    
    if degree is None:
        return "⚠️ Error: degree parameter is required"
    
    try:
        # Convert base64 string to PIL Image
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data))
        
        # Rotate the image
        rotated_img = img.rotate(degree, expand=True)
        
        # Convert PIL Image back to base64 string
        buffer = BytesIO()
        rotated_img.save(buffer, format='PNG')
        img_base64_output = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64_output
        
    except (base64.binascii.Error, binascii.Error):
        return "⚠️ Error: Invalid base64 image data"
    except Exception as e:
        return f"⚠️ Image rotation failed: {str(e)}"


if __name__ == "__main__":
    print("\nStarting MCP Image Rotation Service...")
    mcp.run(transport='stdio')