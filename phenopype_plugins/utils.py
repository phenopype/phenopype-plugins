# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:27:50 2024

@author: mluerig
"""

#%% modules

clean_namespace = dir()


from importlib import util 
import os

classes = [""]
functions = ["parse_model_config"]

def __dir__():
    return clean_namespace + classes + functions

#%% 

def parse_model_config(model_config_path):
    
    # Load the module specified by the file path
    spec = util.spec_from_file_location(os.path.basename(model_config_path), model_config_path)
    module = util.module_from_spec(spec)
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

