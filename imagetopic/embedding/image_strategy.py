import json 


from PIL import Image 
from PIL.Image import Image as PILImage 

from typing import Any, List, Dict 
from imagetopic.embedding.strategy import ABCStrategy
from sentence_transformers import SentenceTransformer

from imagetopic.schema.embedding import IMAGE_MODEL

class IMAGEStrategy(ABCStrategy):
    def __init__(self, model_name:IMAGE_MODEL, cache_folder:str, device:str) -> None:
        assert isinstance(model_name, IMAGE_MODEL)

        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            cache_folder=cache_folder,
            device=device
        )

    def process_task(self, task: str) -> Dict[str, List[float]]:
        map_id2image_path:Dict[str, str] = json.loads(task)
        ids, image_paths = list(map(list, zip(*map_id2image_path.items())))
        pil_images_batch:List[PILImage] = []
        for path2img in image_paths:
            with Image.open(path2img) as image:
                pil_images_batch.append(image.copy())
        
        embeddings = self.model.encode(pil_images_batch, show_progress_bar=False)
        return dict(zip(ids, embeddings.tolist()))

