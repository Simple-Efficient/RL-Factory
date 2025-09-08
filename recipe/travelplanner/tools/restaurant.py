import requests
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP("LocalServer")

@mcp.tool()
def restaurant(city: str) -> Dict[str, Any]:
    """Query restaurant information
    
    Args:
        city: City name
        
    Returns:
        Dict: Dictionary containing restaurant information
    """
    try:
        request_data = {"city": city}

        # Set request headers and proxy
        headers = {
            "Content-Type": "application/json"
        }
        
        # Use local connection, bypass proxy
        proxies = {
            "http": None,
            "https": None
        }
        
        response = requests.post(
            "http://127.0.0.1:8003/restaurants",
            headers=headers,
            proxies=proxies,
            json=request_data,
            timeout=10
        )
        
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        return {"error": f"Restaurant query failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport='stdio')