#%% modules

import copy
import cv2
import numpy as np
import sys

from dataclasses import make_dataclass

from phenopype import _config
from phenopype import settings
from phenopype import utils_lowlevel
from phenopype.core import segmentation, visualization
from phenopype import utils_lowlevel as ul

from phenopype_plugins.utils import parse_model_config

try:
    import torch
except ImportError:
    pass

try:
    import keras
except ImportError:
    pass

try:
    import fastsam
except ImportError:
    pass

#%% functions
 
 
def predict_contour_fastSAM(
        image,
        model_id="a",
        prompt="everything",
        center=0.9,
        resize_roi=1024,
        confidence=0.8,
        iou=0.65,
        force_reload=False,
        **kwargs,
        ):
    
    # =============================================================================
    # setup
        
    ## set flags
    flags = make_dataclass(
        cls_name="flags",
        fields=[("prompt", str, prompt), 
                ("max_dim", str, kwargs.get("max_dim")), 
                ],
    )
    
    # =============================================================================
    # model management
    
    # Check if model_id is not None and exists in the configuration
    if model_id in _config.models:
        model_path = _config.models[model_id]["model_path"]
        
    # Check if the model hasn't been loaded yet
    if "model" not in _config.models[model_id]:
        print(f"- loading model  \"{model_id}\" into memory")
        _config.models[model_id]["model"] = fastsam.FastSAM(model_path)
    else:
        print(f"- using cached model \"{model_id}\"")
    _config.active_model = _config.models[model_id]["model"]

    # init model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _config.active_model
                            
    # =============================================================================
    # prepare image
    
    if flags.prompt == "everything":
        if center < 1:
            height, width = image.shape[:2]
            rx, ry = int(round((1 - center) * 0.5 * width)), int(round((1 - center) * 0.5 * height))
            rh, rw = int(round(center * height)), int(round(center * width))
            roi_orig = image[ry : ry + rh, rx : rx + rw]
        else:
            roi_orig = copy.deepcopy(image)

    elif flags.prompt in ["everything-box", "box"]:
        ## get mask from annotations
        annotations = kwargs.get("annotations", {})
        annotation_id_mask = kwargs.get(settings._mask_type + "_id", None)
        annotation_mask = utils_lowlevel._get_annotation(
            annotations,
            settings._mask_type,
            annotation_id_mask,
        )
        ## convert mask coords to ROI
        coords = annotation_mask["data"]["mask"]
        coords = utils_lowlevel._convert_tup_list_arr(coords)
        rx, ry, rw, rh = cv2.boundingRect(coords[0])  
        
        if flags.prompt == "everything-box":
            roi_orig = image[ry : ry + rh, rx : rx + rw]
        elif flags.prompt == "box":
            roi_orig = image
            resize_x = resize_roi / roi_orig.shape[1]
            resize_y = resize_roi / roi_orig.shape[0]
        
    ## resize roi
    roi_orig_height, roi_orig_width = roi_orig.shape[:2]
    roi = utils_lowlevel._resize_image(
        roi_orig, width=resize_roi, height=resize_roi)
    
    # =============================================================================
    # apply model
    
    print(f"- starting prompt on device {device}")
        
    ## encode roi 
    everything_results = model(
        roi,
        device=device,
        retina_masks=True,
        imgsz=[int(roi.shape[1]), int(roi.shape[0])],
        conf=confidence,
        iou=iou,
        verbose=False,
    )
    if not everything_results.__class__.__name__ == "NoneType":
        
        speed = everything_results[0].speed
        speed = {
            "preprocess": round(speed["preprocess"], 2),
            "inference": round(speed["inference"], 2),
            "postprocess": round(speed["postprocess"], 2),
            }
        print(f"- sucessfully processed image of shape {everything_results[0].orig_shape}")
        print(f"- speed: {speed}")

    else:
        return image

    # =============================================================================
    ## set up prompt
    
    prompt_process = fastsam.FastSAMPrompt(
        roi, everything_results, device=device)
    
    # =============================================================================
    ## get detected objects
    
    ## box-post
    if flags.prompt == "box":      
        mask_coords = ul._resize_mask([rx, ry, rw, rh], resize_x, resize_y)
        mask_coords_sam = utils_lowlevel._convert_box_xywh_to_xyxy(mask_coords)
        detections = prompt_process.box_prompt(bboxes=[mask_coords_sam])
    
    ## everything prompt 
    elif flags.prompt in ["everything","everything-box"]:
        detections = prompt_process.everything_prompt()
        
    ## tensor to array
    if detections.__class__.__name__ == "Tensor":    
        detections_array = np.asarray(detections.cpu(), "uint8")
    else: 
        detections_array = np.asarray(detections, "uint8")
        
    ## create binary mask
    roi_bin = np.bitwise_or.reduce(detections_array, axis=0)
    roi_bin[roi_bin==1] = 255
        
    ## resize to original dimensions
    roi_det = utils_lowlevel._resize_image(
        roi_bin, width=roi_orig_width, height=roi_orig_height)
    
    if flags.prompt in ["everything", "everything-box"]:
        if center == 1 and flags.prompt == "everything":
            image_bin = roi_det
        else:
            image_bin = np.zeros(image.shape[:2], "uint8")
            image_bin[ry : ry + rh, rx : rx + rw] = roi_det
        
    elif flags.prompt == "box":
        image_bin = roi_det
            
        
    return image_bin


def predict_contour_keras(
    image,
    model_id="a",
    binary_mask=False,
    threshold=True,
    threshold_method="otsu",
    threshold_value=127,
    threshold_blocksize=99,
    threshold_constant=5,
    force_reload=False,
    **kwargs,
):
    """
    Applies a trained deep learning model to an image and returns a grayscale mask 
    of foreground predictions, which can then be thresholded to return a binary mask.
    
    Three types of thresholding algorithms are supported: 
        - otsu: use Otsu algorithm to choose the optimal threshold value
        - adaptive: dynamic threshold values across image (uses arguments
          "blocksize" and "constant")
        - binary: fixed threshold value (uses argument "value")    
        
    Parameters
    ----------
    image : ndarray
        input image
    model_path : str
        path to a detection model (currently only keras h5 objects are supported)
    model_id : str, optional
        id for a model that has been added to a phenopype project (overrides model_path)
    threshold : bool, optional
        perform thresholding on returned grayscale segmentation mask to create binary image.
        default is True.
    threshold_method : {"otsu", "adaptive", "binary"} str, optional
        type of thresholding algorithm to be used on the model output
    threshold_blocksize: int, optional
        Size of a pixel neighborhood that is used to calculate a threshold 
        value for the model mask (has to be odd - even numbers will be ceiled; for
        "adaptive" method)
    threshold_constant : int, optional
        value to subtract from binarization output (for "adaptive" method)
    threshold_value : {between 0 and 255} int, optional
        thesholding value (for "binary" method)
    force_reload : bool, optional
        force a model reload every time the function is run (WARNING: this may 
        take a long time)     

    Returns
    -------
    image : ndarray
        binary image

    """
    # =============================================================================
    # setup
    
    fun_name = sys._getframe().f_code.co_name
    
    ## flags
    flags = make_dataclass(cls_name="flags", 
                           fields=[("binary_mask", bool, binary_mask)])
    
    # =============================================================================
    # annotation management
    if flags.binary_mask:
        
        annotations = kwargs.get("annotations", {})
        annotation_type = kwargs.get("annotation_type", settings._mask_type)
        annotation_id = kwargs.get(annotation_type + "_id", None)
            
    # =============================================================================
    # execute
    
    image_source = copy.deepcopy(image)
    
    if flags.binary_mask:
        binary_mask = np.zeros(image_source.shape, dtype="uint8")
        if annotation_type == settings._mask_type:
            print("mask")
            binary_mask = visualization.draw_mask(
                image=binary_mask, 
                annotations=annotations, 
                contour_id=annotation_id, 
                line_colour=255,
                line_width=0,
                fill=1)
        elif annotation_type == settings._contour_type:
            print("contour")
            binary_mask = visualization.draw_contour(
                image=binary_mask, 
                annotations=annotations, 
                contour_id=annotation_id, 
                line_colour=255,
                line_width=0,
                fill=1)

        image_source = cv2.bitwise_and(image_source, binary_mask)    
    if not model_id.__class__.__name__ == "NoneType":
        model_path = _config.models[model_id]["model_phenopype_path"]
        if not "model_loaded" in _config.models[model_id]:
            print("loading model " + model_id)
            _config.models[model_id]["model_loaded"] = keras.models.load_model(model_path)
        _config.active_model = _config.models[model_id]["model_loaded"]
        _config.active_model_path = model_path
    
    elif not _config.active_model_path == model_path or _config.active_model.__class__.__name__ == "NoneType" or force_reload==True:
        _config.active_model = keras.models.load_model(model_path)
        _config.active_model_path = model_path

    print("Using current model at " + _config.active_model_path)
    
    model = _config.active_model

    image_source = utils_lowlevel._resize_image(image_source, width=model.input.shape[1], height=model.input.shape[2])/255
    image_source = np.expand_dims(image_source, axis=0)
    pred = model.predict(image_source)
     
    mask_predicted = pred[0,:,:,1]*255
    mask_predicted = mask_predicted.astype(np.uint8)
    mask_predicted = utils_lowlevel._resize_image(mask_predicted, width=image.shape[1], height=image.shape[0], interpolation="linear")
    
    if threshold:
        mask_predicted = segmentation.threshold(
            mask_predicted, 
            invert=True,
            method=threshold_method,
            value=threshold_value, 
            blocksize=threshold_blocksize,
            constant=threshold_constant
            )
           

    # tf.keras.backend.clear_session()
    
    return mask_predicted


def predict_contour_torch(
        image,
        model_id="a",
        model_config_path=None,
        primer="contour",
        confidence=0.8,
        **kwargs,
        ):
    
    # =============================================================================
    # setup

    # =============================================================================
    # model management
    
    ## supply script with model specification 
    if model_config_path:
        fun_load_model, fun_preprocess = parse_model_config(model_config_path)
        _config.models[model_id]["fun_preprocess"] = fun_preprocess
        _config.models[model_id]["fun_load_model"] = fun_load_model
    
    # Check if model_id is not None and exists in the configuration
    if model_id in _config.models:
        preprocess = _config.models[model_id]["fun_preprocess"]
        load_model = _config.models[model_id]["fun_load_model"]
        model_path = _config.models[model_id]["model_path"]
        
    # Check if the model hasn't been loaded yet
    if "model" not in _config.models[model_id]:
        print(f"- loading model  \"{model_id}\" into memory")
        _config.models[model_id]["model"] = load_model(model_path)
    else:
        print(f"- using cached model \"{model_id}\"")
    _config.active_model = _config.models[model_id]["model"]

    # init model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _config.active_model
    
    # =============================================================================
    ## annotation management
    
    annotations = kwargs.get("annotations", {})
    
    if primer=="contour":
        annotation_id_input = kwargs.get(settings._contour_type + "_id", None)
        annotation = utils_lowlevel._get_annotation(
            annotations,
            settings._contour_type,
            annotation_id_input,
        )
        coords = annotation["data"][settings._contour_type][0]
    elif primer=="mask":
        annotation_id_input = kwargs.get(settings._mask_type + "_id", None)
        annotation = utils_lowlevel._get_annotation(
            annotations,
            settings._mask_type,
            annotation_id_input,
        )      
        coords = annotation["data"][settings._mask_type][0]
        
    # =============================================================================
    ## inference

    roi, roi_box = ul._extract_roi_center(image, coords, 512)
    image_tensor = preprocess(roi)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor.to(device)
    
    predict_masks = model(image_tensor)
    
    mask = predict_masks[0].clone().cpu()
    mask = mask > confidence
    mask = mask.squeeze(0).detach().numpy().astype(np.uint8)
    mask[mask==1] = 255
    
    image_bin = np.zeros(image.shape[:2], np.uint8)
    start_y, end_y,start_x,end_x = roi_box
    image_bin[start_y:end_y, start_x:end_x] = mask

    return image_bin
