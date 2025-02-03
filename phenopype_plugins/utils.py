# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:27:50 2024

@author: mluerig
"""

#%% modules

# clean_namespace = dir()


import os
import sys
import numpy as np

import cv2
from importlib import util as import_util

from phenopype import config

# classes = [""]
# functions = ["parse_modelconfig", "get_global_modelconfig", "load_or_cache_model"]

# def __dir__():
#     return clean_namespace + classes + functions

#%% 

def model_loader_cacher(model_id, load_function, model_path=None, force_reload=False, **kwargs):
    """
    Loads or retrieves a cached model based on the provided model_id. If the model is not cached,
    it uses the provided load_function and model_path to load the model and caches it.

    Args:
        model_id (str): The identifier for the model.
        load_function (callable): Function to load the model if not cached.
        model_path (str, optional): Path to the model file. Required if not in config.models.

    Returns:
        model (object): The loaded or cached model.
    """
    global config  # Access to the global configuration dictionary

    # Ensure the model configuration exists for the given model_id
    if model_id in config.models:
        if not model_path:  # If no model_path provided, use the one in the configuration
            model_path = config.models[model_id].get("model_path")
    
    # Validate that a model_path is available
    if not model_path:
        raise ValueError(f"No model path provided for model_id {model_id}")
    else:
        current_model_basename = os.path.basename(model_path)

    # Check if the model is already loaded
    if model_id in config.models:
        if "model" in config.models[model_id]:
            cached_model_path = config.models[model_id].get("model_path")
            cached_model_basename = os.path.basename(cached_model_path) if cached_model_path else None
    
            # Check if the cached model's basename matches the current one
            if cached_model_basename != current_model_basename:
                print(f"- re-loading model '{current_model_basename}' into cache id '{model_id}'")
                config.models[model_id]["model"] = load_function(model_path, **kwargs)
            else:
                print(f"- using model '{current_model_basename}' cached under id '{model_id}' ")
        else:
            config.models[model_id]["model"] = load_function(model_path, **kwargs)
    else:
        config.models[model_id] = {}
        config.models[model_id]["model"] = load_function(model_path, **kwargs)
        print(f"- loading model '{current_model_basename}' into cache id '{model_id}'")

    # Cache the model path for future reference
    config.models[model_id]["model_path"] = model_path

    # Update the active model in the configuration
    config.active_model = config.models[model_id]["model"]
    return config.active_model

def model_path_resolver(func):
    def wrapper(*args, **kwargs):
        
        # Handle model_id and model_path
        model_id = kwargs.get('model_id', None)

        # Check if model_path is passed as a positional or keyword argument
        if len(args) > 1:  # Assuming image is the first argument
            model_path = args[1]
        else:
            model_path = kwargs.get('model_path', None)

        # Derive model_path from model_id if necessary
        if not model_path and model_id and model_id in config.models:
            model_path = config.models[model_id].get('model_path')

        # Raise an error if model_path is still None
        if not model_path:
            raise ValueError("model_path must be provided either directly or through model_id")

        # Avoid adding model_path redundantly if it's already passed as a positional argument
        if len(args) <= 1 and 'model_path' not in kwargs:
            kwargs['model_path'] = model_path
            
        # Call the decorated function with updated arguments
        return func(*args, **kwargs)

    return wrapper

def model_config_path_resolver(func):
    def wrapper(*args, **kwargs):
        
        # Access the model_id and model_config_path from kwargs
        model_id = kwargs.get('model_id', None)
                
        # Check if model_config_path is passed as a positional or keyword argument
        if len(args) > 2:  # Assuming image is the first argument
            model_config_path = args[2]
        else:
            model_config_path = kwargs.get('model_config_path', None)
            
        # Derive model_config_path from model_id if necessary
        if not model_config_path and model_id and model_id in config.models:
            model_config_path = config.models[model_id].get('model_config_path')

        if not model_config_path:
            raise ValueError("model_config_path must be provided either directly or through model_id")

        # Avoid adding model_config_path redundantly if it's already passed as a positional argument
        if len(args) <= 1 and 'model_config_path' not in kwargs:
            kwargs['model_config_path'] = model_config_path

        # Call the decorated function with updated kwargs
        return func(*args, **kwargs)

    return wrapper


def parse_model_config(model_config_path):
    
    # Load the module specified by the file path
    spec = import_util.spec_from_file_location(os.path.basename(model_config_path), model_config_path)
    module = import_util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check for the presence of 'load_model' and 'preprocess' functions
    load_model_fun = getattr(module, 'load_model', None)
    preprocess_fun = getattr(module, 'preprocess', None)

    # Ensure that the retrieved attributes are callable functions
    if not callable(load_model_fun) and callable(preprocess_fun):
        if not callable(load_model_fun):
            print("'load_model' function is missing.")
        if not callable(preprocess_fun):
            print("'preprocess' function is missing.")

    return load_model_fun, preprocess_fun

def modularize_model_config(module_name, file_path):
    """
    Dynamically loads a Python module from the specified file path and makes it available under the given module name.

    Args:
        module_name (str): The name under which the module will be registered in sys.modules.
        file_path (str): The path to the Python script to be loaded as a module.

    Returns:
        module: The loaded module, ready to use.
    """
    spec = import_util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = import_util.module_from_spec(spec)
    sys.modules[module_name] = module  # Optionally register the module globally
    spec.loader.exec_module(module)
    
    return module


def calculate_contour_hierarchy(contours):
    """Reproduce OpenCV's contour hierarchy calculation manually."""
    num_contours = len(contours)
    hierarchy = np.full((num_contours, 4), -1, dtype=np.int32)  # Initialize all to -1

    # Dictionary to store parent-child relationships
    parent_child_map = {}

    # Check for nesting relationships
    for i, cnt_outer in enumerate(contours):
        for j, cnt_inner in enumerate(contours):
            if i != j:  # Avoid self-checking
                inside = all(cv2.pointPolygonTest(cnt_outer, tuple(map(float, pt[0])), False) > 0 for pt in cnt_inner)
                if inside:
                    hierarchy[j][3] = i  # Set parent of j to i
                    hierarchy[i][2] = j  # Set child of i to j
                    if i not in parent_child_map:
                        parent_child_map[i] = []
                    parent_child_map[i].append(j)

    # Find previous and next relationships
    for i in range(num_contours):
        same_level_contours = [idx for idx in range(num_contours) if hierarchy[idx][3] == hierarchy[i][3]]
        same_level_contours.sort()
        idx_pos = same_level_contours.index(i)
        if idx_pos > 0:
            hierarchy[i][1] = same_level_contours[idx_pos - 1]  # Previous contour at the same level
        if idx_pos < len(same_level_contours) - 1:
            hierarchy[i][0] = same_level_contours[idx_pos + 1]  # Next contour at the same level

    return hierarchy

