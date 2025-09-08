import requests
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP("LocalServer")

@mcp.tool()
def distance(origin: str, destination: str) -> Dict[str, Any]:
    """Query distance information between two locations
    
    Args:
        origin: Origin location
        destination: Destination location
        
    Returns:
        Dict: Dictionary containing distance information
    """
    try:
        request_data = {
            "origin": origin,
            "destination": destination,
        }

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
            "http://127.0.0.1:8006/distance",
            json=request_data,
            headers = headers,
            proxies = proxies,
            timeout=10
        )
        
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        return {"error": f"Distance query failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport='stdio')