from typing import List, Tuple
from PIL import Image 
from PIL.Image import Image as PILImage 

import cv2 

from itertools import groupby

import operator as op 

import numpy as np
from os import path, mkdir  
from shutil import copyfile

from tqdm import tqdm 

from imagetopic.log import logger 

import base64

def encode_image(file_path) -> str:
    bgr_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    bgr_image = cv2.resize(bgr_image, dsize=(512, 512))
    _, vector = cv2.imencode('.jpg', bgr_image)
    binarystream = vector.tobytes()
    base64_data = base64.b64encode(s=binarystream).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_data}"

def enhance_embedding(embeddings:np.ndarray, k:int=5) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1)
    scores = (embeddings @ embeddings.T) / (norms[:, None] * norms[None, :] + 1e-8)
    top_ks = np.argsort(scores, axis=1)[:, -k:]
    accumulator:List[np.ndarray] = []
    for node_id, indices in enumerate(top_ks):
        neighbor_scores = scores[node_id][indices]
        _max = np.max(neighbor_scores)
        _min = np.min(neighbor_scores)
        neighbor_scores = (neighbor_scores - _min) / (_max - _min)

        neighbbor_embeddings = embeddings[indices, :]
        enhanced_embedding = neighbor_scores @ neighbbor_embeddings

        accumulator.append(enhanced_embedding)
    
    return np.vstack(accumulator)

def find_communities(embeddings:np.ndarray, threshold:float=0.85, iterations:int=1) -> List[List[int]]:
    """
    Identifies communities within a set of embeddings based on similarity.

    This function iteratively finds clusters (or communities) of similar nodes within the embeddings.
    It calculates cosine similarity scores between embeddings and groups them into communities
    based on a specified similarity threshold. During each iteration, embeddings are adjusted towards
    their community centroids, enhancing community structures.

    Args:
        embeddings (np.ndarray): A 2D array of embeddings, where each row represents an embedding.
        threshold (float, optional): The similarity threshold to determine if nodes belong to the same community. Defaults to 0.85.
        iterations (int, optional): The number of iterations to refine the community structure. Defaults to 1.

    Returns:
        List[List[int]]: A list of communities, each represented as a list of indices of nodes in that community.

    Note:
        The function uses cosine similarity to measure the closeness of embeddings. Nodes with similarities
        above the threshold are considered part of the same community. The communities are refined iteratively,
        allowing nodes to move towards the centroid of their communities.
    """

    nb_nodes = embeddings.shape[0]
    indices = np.arange(nb_nodes)

    accumulator:List[List[int]] = []
    for current_iter in range(iterations):
        accumulator = []
        norms = np.linalg.norm(embeddings, axis=1)
        scores = (embeddings @ embeddings.T) / (norms[:, None] * norms[None, :] + 1e-8)
        np.fill_diagonal(scores, 0.0)

        node_scores = np.zeros(nb_nodes)
        for idx, group in groupby(sorted(np.argmax(scores, axis=1).tolist())):
            node_scores[idx] = len(list(group))
 
        sorted_nodes_by_scores = sorted(zip(indices, node_scores), key=op.itemgetter(1), reverse=True)
            
        marked = np.zeros_like(node_scores, dtype=np.uint8) 
        
        for node_idx, _ in tqdm(sorted_nodes_by_scores):
            if marked[node_idx] == 1:
                continue

            neighbor_scores = scores[node_idx]
            
            reachables_neighbor_mask = neighbor_scores >= threshold
            reachables_neighbor_indices = indices[reachables_neighbor_mask]  
            availables_neighbor_indices = reachables_neighbor_indices[marked[reachables_neighbor_mask] == 0]
            
            if len(availables_neighbor_indices) == 0:
                accumulator.append([node_idx])
                continue

            availables_neighbor_scores_mat = scores[availables_neighbor_indices, :]
            availables_neighbor_max_scores = np.max(availables_neighbor_scores_mat, axis=1)
            current_node2_available_neighbor_scores = neighbor_scores[availables_neighbor_indices]

            neighborhood_size = np.abs(current_node2_available_neighbor_scores - availables_neighbor_max_scores) <= 0.1
            stroguest_attraction = current_node2_available_neighbor_scores >= availables_neighbor_max_scores

            selected_neighbor_mask = (neighborhood_size + stroguest_attraction) > 0
            selected_neighbor_indices = availables_neighbor_indices[selected_neighbor_mask].tolist()
            
            if len(selected_neighbor_indices) == 0:
                continue

            accumulator.append(selected_neighbor_indices)
                
            marked[selected_neighbor_indices] = 1
        
        logger.info(f'nb cluster {len(accumulator)} / iteration {current_iter}')
        for cluster in accumulator:
            population = embeddings[cluster]
            centroid = np.mean(population, axis=0)
            affinity = population @ centroid
            affinity = affinity / (np.linalg.norm(population, axis=1) * np.linalg.norm(centroid))
            embeddings[cluster] = population + affinity[:, None] * (centroid - population)  
    return accumulator

def save_image_clusters(clusters:List[List[str]], target_location:str):
    """
    Saves image clusters to a specified directory, organizing each cluster into a separate subdirectory.

    This function takes a list of image clusters, where each cluster is a list of image file paths. It saves
    each image in its respective cluster directory within the specified target location. Images that belong
    to clusters with only a single image (outliers) are grouped together in a separate directory.

    Args:
        clusters (List[List[str]]): A list of clusters, where each cluster is a list of image file paths.
        target_location (str): The directory where the clusters will be saved.

    Raises:
        AssertionError: If the target_location is not an existing directory.

    Note:
        Each cluster is saved in a separate subdirectory named by the cluster's index, padded with zeros. 
        For example, the first cluster is saved in a directory named '000'. Outliers are saved in a 
        directory named '###'. This organization facilitates easy identification and access to the images 
        in each cluster.
    """
    assert path.isdir(target_location)
    def _handle_cluster(cluster:List[str], cluster_id:str):
        for path2img in cluster:
            basepath, _= path.split(path2img)
            path2dir = path.join(target_location, cluster_id)
            if not path.isdir(path2dir):
                mkdir(path2dir)
            
            path2dst = path2img.replace(basepath, path2dir)
            copyfile(path2img, path2dst)

    idx = 0
    outliers:List[str] = []
    for cluster in clusters:
        if len(cluster) == 1:
            outliers.extend(cluster)
            continue
        
        _handle_cluster(cluster, f'{idx:03d}')
        idx = idx + 1
    

    _handle_cluster(outliers, '###')


def create_grid(image_paths:List[str], cell_size:Tuple[int, int], nb_cols:int, padding:int=0) -> PILImage:
    """
    Creates a grid of images with specified cell size and padding.

    Args:
        image_paths (List[str]): Paths to the images to be included in the grid.
        cell_size (Tuple[int, int]): The size (width, height) of each cell in the grid.
        nb_cols (int): The number of columns in the grid.
        padding (int, optional): The amount of padding between images in the grid. Defaults to 0.

    Returns:
        PIL.Image: A single PIL Image containing the grid of images.
    """
    images: List[PILImage] = []
    for path in image_paths:
        with Image.open(path) as img:
            resized_image = img.resize(cell_size)
            images.append(resized_image)

    nb_images = len(images)
    nb_cols = min(nb_images, nb_cols)
    nb_rows = max(1, (nb_images + nb_cols - 1) // nb_cols) 

    image_width, image_height = cell_size
    grid_width = nb_cols * (image_width + padding) - padding
    grid_height = nb_rows * (image_height + padding) - padding

    grid_image = Image.new("RGB", (grid_width, grid_height))

    for index, image in enumerate(images):
        row = index // nb_cols
        col = index % nb_cols
        x_pos = col * (image_width + padding)
        y_pos = row * (image_height + padding)
        grid_image.paste(image, (x_pos, y_pos))

    return grid_image