from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
from abc import ABC, abstractmethod

class BaseServer(ABC):
    def __init__(self, name: str, port: int):
        self.app = FastAPI(title=f"{name}查询服务")
        self.port = port
        self.data = None
        
    @abstractmethod
    def load_data(self):
        pass
        
    def run(self):
        print(f'启动{self.app.title}... 监听地址: 127.0.0.1:{self.port}')
        uvicorn.run(self.app, host="127.0.0.1", port=self.port)