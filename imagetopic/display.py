import cv2 

import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib.image import imread

from PIL import Image 
from sklearn.metrics import euclidean_distances

from typing import List, Optional  

from llama_cpp import Llama, CreateChatCompletionResponse
from llama_cpp.llama_chat_format import Llava15ChatHandler

from imagetopic.utilities import encode_image

class Display:
    def __init__(self, image_paths:List[str], reduced_embedding:np.ndarray, clusters:List[List[int]], path2llava_model:str, path2llava_clip_model:str):
        self.image_paths = image_paths
        self.reduced_embedding = reduced_embedding
        self.clusters = clusters 
        self.point:Optional[np.ndarray] = None 
        self.target:Optional[int] = None  
        chat_handler = Llava15ChatHandler(clip_model_path=path2llava_clip_model)
        self.llm = Llama(
            model_path=path2llava_model, chat_handler=chat_handler, 
            n_ctx=2048,  n_gpu_layers=32, verbose=True, logits_all=True,
            chat_format="llava-1-5"
        )

    def _event_handler(self):
        def on_click(event):
            x, y = event.xdata, event.ydata 
            self.point = np.array([x, y])
            distances = np.ravel(euclidean_distances(X=self.point[None, :], Y=self.reduced_embedding))
            self.target = np.argmin(distances)

        return on_click

    def describe_image(self, path2image:str) -> CreateChatCompletionResponse:
        b64encoded_image = encode_image(path2image)
        stream = self.llm.create_chat_completion(
            messages = [
                {"role": "system", "content": "You are an assistant who perfectly describes images"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": b64encoded_image}},
                        {"type" : "text", "text": "please describe this image with one sentence"}
                    ]
                }

            ],
            stream=True 
        )
        return stream 

    def show(self):
        rgb_colors = np.random.rand(len(self.clusters), 3)

        c = np.zeros((len(self.reduced_embedding), 3))
        for group_idx, group in enumerate(self.clusters):
            for node_idx in group:
                c[node_idx] = rgb_colors[group_idx]

        fig = plt.figure(figsize=(16, 8))
        fig.canvas.mpl_connect('button_press_event', self._event_handler())
        ax0, ax1 = fig.subplots(1, 2)
        ax0.scatter(self.reduced_embedding[:,0], self.reduced_embedding[:,1], c=c)

        res = None 
        last_target = self.target
        tokens = []
        while True:
            try:
                plt.pause(interval=0.01)
                if self.target is not None:
                    if last_target != self.target:
                        tokens.clear()

                        image = Image.open(self.image_paths[self.target])
                        image = image.resize((512, 512))
                        image_array = np.asarray(image)
                        ax1.imshow(image_array)
                        image.close()
                        last_target = self.target
                        plt.pause(0.001)

                        stream = self.describe_image(self.image_paths[self.target])
                        x,y = self.point.tolist()
                        for chunk in stream:
                            data = chunk["choices"][0]["delta"].get('content', '')
                            tokens.append(data)
                            text = ''.join(tokens)
                            if res is not None:
                                res.remove()
                            res = ax0.annotate(
                                text=text, 
                                xy=(x, y), xycoords="data", 
                                xytext=(x - 7, y - 3), textcoords='data',
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3),
                                fontsize=12, bbox=dict(facecolor='wheat', edgecolor='black', boxstyle='round,pad=1')
                            )
                            plt.pause(0.001)
            except KeyboardInterrupt:
                break 
            except Exception:
                break 
        
        plt.clf()