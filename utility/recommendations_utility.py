import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.notebook import tqdm
import os

import keras
from keras.preprocessing import image



def create_feature_extractor(model: keras.engine.functional.Functional, layer_name: str) -> keras.Model:
    """ 
    Create feature extractor according to the model and name of the output layer
    
    Parameters:
    ------------
    model : keras.engine.functional.Functional
        Keras model from applications.
    layer_name : str
        Name of layer with features.
        
    Returns:
    ------------
    keras.Model
        feature extractor
        
    Examples:
    ------------
    model = ResNet101V2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    feature_extractor = create_feature_extractor(model, 'fc2')
    
    # 'avg_pool' for ResNet101V2
    """

    feature_extractor = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(layer_name).output
    )
    return feature_extractor


def get_features(path: str, target_size: Tuple[int, int, int], feature_extractor: keras.Model) -> list:
    """ 
    Getting features from images in folder at 'path' using feature_extractor
    
    Parameters:
    ------------
    path : string
        Path to images folder.
        
    target_size : Tuple[int, int, int]
        tuple of images shape
    
    feature_extractor : keras.Model
        feature extractor created by feature_extractor function
        
    Returns:
    ------------
    image_features : list
        list of features
        
    """
    
    image_features = []
    iteration = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        iteration += 1
        if iteration%50 == 0:
            print("Iteration ", iteration)
            
        for file in filenames:
            img = image.load_img(os.path.join(dirpath, file), target_size=target_size)
            img_arr = image.img_to_array(img) / 255.
            image_features.append(feature_extractor(np.expand_dims(img_arr, axis=0)).numpy())

            
    return image_features
    
    
def get_viz(path: str, target_size: Tuple[int, int, int]) -> list:
    """ 
    Get visualization parameters of images
    
    Parameters:
    ------------
    path : string
        Path to images folder.
        
    target_size : tuple
        tuple of images shape
        
        
    Returns:
    ------------
    image_viz : list
        list of images for visualization
    """
    
    image_viz = []
    
    for (dirpath, dirnames, filenames) in tqdm(os.walk(path)):
        for file in filenames:
            img = image.load_img(os.path.join(dirpath, file), target_size=target_size)
            img_arr = image.img_to_array(img)
            image_viz.append(np.ubyte(img_arr)) # for further visualizing
    
    return image_viz


def get_feedback_path(path: str, total_items: int) -> Tuple[str, str, float]:
    """
    Creating feedback for look in path by folders
    
    Parameters:
    ------------
    path : string
        Path to images folder.
        
    total_items : int
        Number of outfits in folder.
        
        
    Returns:
    ------------
    feedback : Tuple[str, str, float]
        Tuple that looks like: (used_id, item_id, rating(1.0 for my model))
        
    """
    
    items_iterator = iter(range(total_items))
    feedback = []
    for user_id, look_number in enumerate(os.listdir(path)):
        f = os.path.join(path, look_number)
        for item in os.listdir(f):
            feedback.append(tuple([str(user_id), str(next(items_iterator)), 1.0]))
            
    return feedback


def get_feedback_df(df: pd.DataFrame) -> Tuple[str, str, float]:
    """
    Creating feedback using pandas dataframe
    
    Parameters:
    ------------
    df: pd.DataFrame
        DataFrame containing data about users and items.
        
        
    Returns:
    ------------
    feedback : Tuple[str, str, float]
        Tuple that looks like: (used_id, item_id, rating(1.0 for my model))
        
    """
    
    feedback = []
    for i in range(df.shape[0]):
        im_id = str(df.loc[i]['image_id'])
        look_id = df.loc[i]['look_id']
        rating = 1.0
        feedback.append(tuple([look_id, im_id, rating]))
        
    return feedback


