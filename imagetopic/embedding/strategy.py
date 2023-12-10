

from typing import Any 
from abc import ABC, abstractmethod

class ABCStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def process_task(self, task:str) -> Any:
        pass 

    def __call__(self, task:str) -> Any:
        return self.process_task(task=task)
    
    def __del__(self):
        pass 