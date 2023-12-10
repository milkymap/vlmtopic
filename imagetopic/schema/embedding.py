
from enum import Enum 
from pydantic import BaseModel

from typing import Any, Optional
    
class IMAGE_MODEL(str, Enum):
    CLIP_VIT_B32:str='clip-ViT-B-32'
    CLIP_VIT_B16:str='clip-ViT-B-16'
    CLIP_VIT_L14:str='clip-ViT-L-14'
    CLIP_VIT_B32_MULTILINGUAL_V1:str='clip-ViT-B-32-multilingual-v1'

class TaskResponse(BaseModel):
    status:bool 
    content:Any 
    error_message:Optional[str]