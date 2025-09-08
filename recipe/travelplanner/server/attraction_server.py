from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from base_server import BaseServer

class AttractionRequest(BaseModel):
    city: str = Field(..., description="城市名称")

class AttractionResponse(BaseModel):
    name: str
    latitude: float
    longitude: float
    address: str
    phone: Optional[str]
    website: Optional[str]
    city: str

class AttractionServer(BaseServer):
    def __init__(self):
        super().__init__("景点", 8004)
        self.load_data()
        self.setup_routes()
        
    def load_data(self):
        self.data = pd.read_csv("envs/reward_score/database/attractions/attractions.csv")
        # print(self.data.head())
        
    def setup_routes(self):
        @self.app.post("/attractions", response_model=List[AttractionResponse])
        async def query_attractions(request: AttractionRequest):
            try:
                results = self.data[self.data["City"] == request.city]
                
                if len(results) == 0:
                    return []
                    
                return [
                    AttractionResponse(
                        name=row["Name"],
                        latitude=row["Latitude"],
                        longitude=row["Longitude"],
                        address=row["Address"],
                        phone=row["Phone"],
                        website=row["Website"],
                        city=row["City"]
                    ) for _, row in results.iterrows()
                ]
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    server = AttractionServer()
    server.run()