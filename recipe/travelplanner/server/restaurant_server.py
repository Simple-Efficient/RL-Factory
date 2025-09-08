from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from base_server import BaseServer

class RestaurantRequest(BaseModel):
    city: str = Field(..., description="城市名称")

class RestaurantResponse(BaseModel):
    name: str
    average_cost: float
    cuisines: str
    rating: float
    city: str

class RestaurantServer(BaseServer):
    def __init__(self):
        super().__init__("餐厅", 8003)
        self.load_data()
        self.setup_routes()
        
    def load_data(self):
        self.data = pd.read_csv("envs/reward_score/database/restaurants/clean_restaurant_2022.csv")
        
    def setup_routes(self):
        @self.app.post("/restaurants", response_model=List[RestaurantResponse])
        async def query_restaurants(request: RestaurantRequest):
            try:
                results = self.data[self.data["City"] == request.city]
                
                if len(results) == 0:
                    return []
                    
                return [
                    RestaurantResponse(
                        name=row["Name"],
                        average_cost=row["Average Cost"],
                        cuisines=row["Cuisines"],
                        rating=row["Aggregate Rating"],
                        city=row["City"]
                    ) for _, row in results.iterrows()
                ]
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    server = RestaurantServer()
    server.run()