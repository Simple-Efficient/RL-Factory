import requests
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP("LocalServer")

@mcp.tool()
def flight(origin: str, destination: str, date: str) -> Dict[str, Any]:
    """Query flight information
    
    Args:
        origin: Origin city
        destination: Destination city
        date: Departure date (YYYY-MM-DD format)
        
    Returns:
        Dict: Dictionary containing flight information
    """
    try:
        request_data = {
            "origin": origin,
            "destination": destination,
            "date": date
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
            "http://127.0.0.1:8007/flights",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=10
        )
        
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        return {"error": f"Flight query failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport='stdio')