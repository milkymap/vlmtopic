import click 

import json 
import pickle 

import zmq 

from typing import List, Dict 

from os import path 
from glob import glob 

import numpy as np 
import umap 

from rich.progress import Progress
from dotenv import load_dotenv

from matplotlib import pyplot as plt 

from imagetopic.log import logger
from imagetopic.embedding import VLMEmbedding 
from imagetopic.schema.embedding import IMAGE_MODEL, TaskResponse

from imagetopic.utilities import enhance_embedding, find_communities

from imagetopic.display import Display

@click.group()
@click.pass_context
def handler(ctx:click.core.Context):
    ctx.ensure_object(dict)

@handler.command()
@click.option('--path2image_dir', type=click.Path(exists=True, file_okay=False), required=True)
@click.option('--image_extension', type=str, default='*.jpg')
@click.option('--model_name', type=IMAGE_MODEL, default=IMAGE_MODEL.CLIP_VIT_B32)
@click.option('--cache_folder', envvar='TRANSFORMERS_CACHE', required=True)
@click.option('--device', type=str, default='cpu')
@click.option('--batch_size', type=int, default=8)
@click.option('--nb_workers', type=int, default=2)
@click.option('--path2embedding', type=click.Path(exists=False, dir_okay=False), required=True)
@click.pass_context
def build_embedding(ctx:click.core.Context, path2image_dir:str, image_extension:str, model_name:str, cache_folder:str, device:str, batch_size:int, nb_workers:int, path2embedding:str):
    image_filepaths =  glob(path.join(path2image_dir, image_extension))
    
    if len(image_filepaths) == 0:
        logger.warning(f'no images were found at : {image_filepaths}')
        exit(0)

    tasks:List[str] = []
    for idx in range(0, len(image_filepaths), batch_size):
        batch = image_filepaths[idx:idx+batch_size]
        task:Dict[str, str] = {}
        for path2image_file in batch:
            task[path2image_file] = path2image_file
        tasks.append(json.dumps(task))

    embedding_system = VLMEmbedding(
        strategy_kwargs={
            'model_name': model_name,
            'cache_folder': cache_folder,
            'device': device
        }
    )

    accumulator:Dict[str, List[float]] = {}
    futures = embedding_system.submit_tasks(tasks=tasks, nb_workers=nb_workers, port=1200)
    with Progress() as progress:
        task_id = progress.add_task(description='embeddings generation', total=len(tasks))
        result:TaskResponse
        for result in futures:
            if result.status == True:
                accumulator.update(result.content)
            progress.advance(task_id=task_id, advance=1)
                
    logger.info(f'nb tasks : {len(tasks)} --- nb responses {len(accumulator)}')
    with open(path2embedding, mode='wb') as fp:
        pickle.dump(accumulator, fp)
        logger.info(f'embeddings were saved at {path2embedding}')

@handler.command()
@click.option('--path2embedding')
@click.option('--nb_neighbors', type=int, default=30)
@click.option('--min_dist', type=float, default=0.0)
@click.option('--n_components', type=int, default=2)
@click.option('--random_state', type=int, default=42)
@click.option('--metric', type=click.Choice(['euclidean', 'cosine']), default='cosine')
@click.pass_context
def reduce_dimension(ctx:click.core.Context, path2embedding:str, nb_neighbors:int, min_dist:float, n_components:int, random_state:int, metric:str):
    with open(path2embedding, mode='rb') as fp:
        accumulator:Dict[str, List[float]] = pickle.load(fp)
    
    nb_embeddings = len(accumulator)
    logger.info(f'nb loaded embeddings {nb_embeddings}')

    embeddings = np.vstack(list(accumulator.values()))
    embeddings = enhance_embedding(embeddings, k=11)

    logger.info('start dimension reduction')
    reductor = umap.UMAP(n_neighbors=nb_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state, metric=metric)
    reduced_embedding = reductor.fit_transform(embeddings)
    logger.info('dimension reduction done')
    
    basepath, filename = path.split(path2embedding)
    path2reduced_dim = path.join(basepath, f'reduced_{filename}')

    with open(path2reduced_dim, mode='wb') as fp:
        pickle.dump(reduced_embedding, fp)

@handler.command()
@click.option('--path2embedding')
@click.option('--threshold', type=float, default=0.80)
@click.option('--nb_iterations', type=int, default=3)
@click.option('--path2llava_model', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--path2llava_clip_model', type=click.Path(exists=True, dir_okay=False), required=True)
def explore_embedding(path2embedding:str, threshold:float, nb_iterations:int, path2llava_model:str, path2llava_clip_model:str):

    basepath, filename = path.split(path2embedding)
    path2reduced_dim = path.join(basepath, f'reduced_{filename}')

    with open(path2embedding, mode='rb') as fp:
        accumulator:Dict[str, List[float]] = pickle.load(fp)
    
    with open(path2reduced_dim, mode='rb') as fp:
        reduced_embedding:np.ndarray = pickle.load(fp)

    nb_embeddings = len(accumulator)
    logger.info(f'nb loaded embeddings {nb_embeddings}')

    embeddings = np.vstack(list(accumulator.values()))
    logger.info('embedding enhancing')
    embeddings = enhance_embedding(embeddings, k=11)
    clusters = find_communities(embeddings, threshold, nb_iterations)
    image_paths = list(accumulator.keys())
    displayer = Display(image_paths, reduced_embedding, clusters, path2llava_model, path2llava_clip_model)
    displayer.show()

if __name__ == '__main__':
    load_dotenv()
    handler(obj={})
