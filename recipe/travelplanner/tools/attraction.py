import requests
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP("LocalServer")

@mcp.tool()
def attraction(city: str) -> Dict[str, Any]:
    """Query attraction information
    
    Args:
        city: City name
        
    Returns:
        Dict: Dictionary containing attraction information
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
            "http://127.0.0.1:8004/attractions",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=10
        )
        
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        return {"error": f"Attraction query failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport='stdio')