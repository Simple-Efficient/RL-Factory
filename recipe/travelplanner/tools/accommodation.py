import requests
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP("LocalServer")

@mcp.tool()
def accommodation(city: str) -> Dict[str, Any]:
    """Query accommodation information
    
    Args:
        city: City name
        
    Returns:
        Dict: Dictionary containing accommodation information
    """
    try:
        request_data = {"city": city}

        headers= {
            "Content-Type": "application/json"
        }
        proxies = {
            "http": None,
            "https": None
        }
        
        response = requests.post(
            "http://127.0.0.1:8009/accommodations",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=10
        )
        
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        return {"error": f"Accommodation query failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport='stdio')