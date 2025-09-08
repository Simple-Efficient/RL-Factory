import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from base_server import BaseServer

class FlightRequest(BaseModel):
    origin: str = Field(..., description="出发城市")
    destination: str = Field(..., description="目的城市") 
    date: str = Field(..., description="出发日期")

class FlightResponse(BaseModel):
    flight_number: str
    price: float
    departure_time: str
    arrival_time: str
    duration: str
    date: str
    origin: str
    destination: str
    distance: float

class FlightServer(BaseServer):
    def __init__(self):
        super().__init__("航班", 8007)
        self.load_data()
        self.setup_routes()
        
    def load_data(self):
        self.data = pd.read_csv("envs/reward_score/database/flights/clean_Flights_2022.csv")
        
    def setup_routes(self):
        @self.app.post("/flights", response_model=List[FlightResponse])
        async def query_flights(request: FlightRequest):
            try:
                results = self.data[
                    (self.data["OriginCityName"] == request.origin) &
                    (self.data["DestCityName"] == request.destination) &
                    (self.data["FlightDate"] == request.date)
                ]
                
                if len(results) == 0:
                    return []
                    
                return [
                    FlightResponse(
                        flight_number=row["Flight Number"],
                        price=row["Price"],
                        departure_time=row["DepTime"],
                        arrival_time=row["ArrTime"],
                        duration=row["ActualElapsedTime"],
                        date=row["FlightDate"],
                        origin=row["OriginCityName"],
                        destination=row["DestCityName"],
                        distance=row["Distance"]
                    ) for _, row in results.iterrows()
                ]
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    server = FlightServer()
    server.run()