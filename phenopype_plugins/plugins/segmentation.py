#%% modules

import copy
import cv2
import numpy as np
import sys
from math import inf
from rich.pretty import pretty_repr
from dataclasses import make_dataclass

from phenopype import __version__ as pp_ver
from phenopype import _vars
from phenopype import decorators
from phenopype import utils as pp_utils
from phenopype import utils_lowlevel as pp_ul
from phenopype.core import segmentation, visualization

from phenopype_plugins import __version__ as ppp_ver
from phenopype_plugins import utils

import warnings

try:
    import torch
except ImportError:
    warnings.warn("Failed to import PyTorch. Some functionalities may not work.", ImportWarning)

try:
    import keras
except ImportError:
    warnings.warn("Failed to import Keras. Some functionalities may not work.", ImportWarning)

try:
    from ultralytics import YOLO, SAM, FastSAM, engine
    from ultralytics.models.fastsam import FastSAMPredictor
    from ultralytics.models.sam import SAM2Predictor
except ImportError:
    warnings.warn("Failed to import Ultralytics. Some functionalities may not work.", ImportWarning)



#%% functions

@utils.model_path_resolver
@decorators.annotation_function
def predict_yolo_det(
        image,
        model_path,
        model_id="a",
        backend="YOLO",
        device="cpu",
        force_reload=False,
        size=640,
        confidence=0.8,
        iou=0.65,
        ret_roi=None,
        ret_pad=None,
        **kwargs,
        ):
    
    """
    Process an input image using the FastSAM model to detect and segment objects based on specified prompts.

    This function handles model loading (with caching capabilities), image preprocessing according to the prompt type,
    object detection, and segmentation, and returns an image with detections applied.

    Parameters
    ----------
    image : ndarray
        The input image to process.
    model_id : str, optional
        Identifier for the model configuration to use, defaults to 'a'.
    prompt : str, optional
        Type of detection and segmentation to perform. Options include 'everything', 'text', 'everything-box', or 'box', defaults to 'everything'.
    center : float, optional
        Fraction of the image center to consider for processing, relevant only for certain prompts, defaults to 0.9.
    resize_roi : int, optional
        The size to which regions of interest (ROIs) are resized before processing, defaults to 1024.
    confidence : float, optional
        Confidence threshold for the detection to consider a detection valid, defaults to 0.8.
    iou : float, optional
        Intersection over Union (IoU) threshold for determining object uniqueness, defaults to 0.65.
    **kwargs
        Additional keyword arguments for extended functionality, like 'max_dim' or specific annotations.

    Returns
    -------
    ndarray
        The processed image with detections and segmentations applied as specified by the prompt.

    Examples
    --------
    >>> processed_image = predict_fastSAM(input_image, model_id='b', prompt='box', resize_roi=512)
        Process 'input_image' using model configuration 'b', focusing on bounding box detections, resizing the ROI to 512x512 pixels.
    """
       
    # Load or retrieve the cached model
    model = utils.model_loader_cacher(model_id, eval(backend), model_path, force_reload)
    
    ## set device
    device = kwargs.get(
        device,
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
    ## resize roi
    image_height, image_width = image.shape[:2]
    roi = pp_utils.resize_image(
        image, width=size, height=size)
    
                
    ## encode roi 
    print(f"- starting object detection prompt on device {device}")
    results = model(
        roi,
        device=device,
        retina_masks=True,
        imgsz=(size,size),
        conf=confidence,
        iou=iou,
        verbose=False,
    )
    
    if results and len(results[0].boxes.xyxy) > 0:
        
        speed = results[0].speed
        speed = {
            "preprocess": round(speed["preprocess"], 2),
            "inference": round(speed["inference"], 2),
            "postprocess": round(speed["postprocess"], 2),
            }
        pp_ul._print(f"- sucessfully processed image of shape {results[0].orig_shape}")
        pp_ul._print(f"- speed: {pretty_repr(speed)}")

    else:
        pp_ul._print("- no objects found")
        return {}
    
    ## ultralytics.results to list of tuples
    coord_list = []
    if type(results[0]) == engine.results.Results:  
        results = results[0].cpu()
        boxes = results.boxes.xyxy[0]
        x_min, y_min, x_max, y_max = np.array(boxes)
        x_min = x_min * image_width / size
        x_max = x_max * image_width / size
        y_min = y_min * image_height / size
        y_max = y_max * image_height / size
        
        ## ROI from center coords
        if ret_roi:
            half_roi = ret_roi // 2
        
            # Calculate exact center
            center_x = round((x_min + x_max) / 2)
            center_y = round((y_min + y_max) / 2)
        
            # Ensure exact ROI size
            x_min = center_x - half_roi
            x_max = center_x + half_roi - 1 if ret_roi % 2 == 0 else center_x + half_roi
            y_min = center_y - half_roi
            y_max = center_y + half_roi - 1 if ret_roi % 2 == 0 else center_y + half_roi
                    
        # Apply padding around the bounding box
        elif ret_pad:
            x_min = max(0, x_min - ret_pad)
            x_max = min(image_width, x_max + ret_pad)
            y_min = max(0, y_min - ret_pad)
            y_max = min(image_height, y_max + ret_pad)
    
        # Add bounding box coordinates to coord_list
        top_left = (int(x_min), int(y_min))
        top_right = (int(x_max), int(y_min))
        bottom_right = (int(x_max), int(y_max))
        bottom_left = (int(x_min), int(y_max))
        coord_list.append([top_left, top_right, bottom_right, bottom_left, top_left])
                
    ## annotation management 
    annotation_type = kwargs["annotation_type"]
    label = kwargs.get("label")
    annotation = {
        "info": {
            "annotation_type": annotation_type,
            "phenopype_function": kwargs["fun_name"],
            "phenopype_version": pp_ver,
            "phenopype_plugins_version": ppp_ver,
        },
        "settings": {
            "image_size": size,
            "confidence": confidence,
            "IOU": iou,
            "tool": "polygon",
        },
        "data": {
            "speed": speed,
            "label": label,
            "n": len(results),
            "include": True,
            annotation_type: coord_list,
            },
    }
    
    return annotation


 
@utils.model_path_resolver
@decorators.annotation_function
def predict_yolo_seg(
        image,
        model_path,
        model_id="a",
        backend="YOLO",
        device="cpu",
        force_reload=False,
        size=640,
        confidence=0.8,
        iou=0.65,
        prompt="everything",
        trim=0.05,
        yolo_plot=False,
        yolo_ret=False,
        stats_mode="circle",
        min_nodes=3,
        max_nodes=inf,
        min_area=0,
        max_area=inf,
        min_diameter=0,
        max_diameter=inf,
        retrieval="ext",
        approximation="simple",
        **kwargs,
        ):
    
    """
    Performs segmentation on an input image using a pretrained YOLO model or a SAM-based foundation model.
    First loads the segmentation model to cache, processes the input image based on the specified 
    prompt, and extracts contours from detected masks. It also supports filtering based on object size, 
    area, and hierarchy structure, returning contours in a format compatible with `phenopype`.

    Parameters
    ----------
    image : ndarray
        The input image to be processed.
    model_path : str
        Path to the YOLO model weights file.
    model_id : str, optional
        Model identifier, default is 'a'.
    backend : str, optional
        The inference backend to use, default is 'YOLO'.
    device : str, optional
        The device for model inference ('cpu' or 'cuda'), default is 'cpu'.
    force_reload : bool, optional
        If True, forces the model to be reloaded instead of using a cached version, default is False.
    size : int, optional
        The image size to which the ROI is resized for YOLO inference, default is 640.
    conf : float, optional
        Confidence threshold for YOLO object detection, default is 0.8.
    iou : float, optional
        IoU threshold for non-maximum suppression, default is 0.65.
    prompt : str, optional
        Defines the segmentation approach. Options:
        - 'everything': Segment the entire image (trimming optional)
        - 'everything-in-mask': Only segment objects within a mask
        - 'mask-select': Apply segmentation on the whole image but return based on the mask
        - 'text': Use text-based segmentation
        - 'text-in-mask': Use text-based segmentation within a mask
        Default is 'everything'.
    trim : float, optional
        Fraction of the image edges to trim before segmentation (0.0 - 1.0), default is 0.05.
    yolo_plot : bool, optional
        If True, plots the segmentation results, default is False.
    yolo_ret : bool, optional
        If True, returns raw YOLO results instead of processed contours, default is False.
    stats_mode : str, optional
        Method for calculating contour statistics. Options: 'circle', 'box', etc., default is 'circle'.
    min_nodes : int, optional
        Minimum number of nodes for a valid contour, default is 3.
    max_nodes : int, optional
        Maximum number of nodes for a valid contour, default is infinity.
    min_area : float, optional
        Minimum area for a valid contour, default is 0.
    max_area : float, optional
        Maximum area for a valid contour, default is infinity.
    min_diameter : float, optional
        Minimum diameter for a valid contour, default is 0.
    max_diameter : float, optional
        Maximum diameter for a valid contour, default is infinity.
    retrieval : str, optional
        Contour retrieval mode ('ext', 'tree', etc.), default is 'ext'.
    approximation : str, optional
        Contour approximation method ('simple', 'none', etc.), default is 'simple'.
    **kwargs : dict
        Additional keyword arguments for model inference and annotation settings.

    Returns
    -------
    dict or None
        A dictionary containing extracted contours and metadata if successful, in the format:
        ```
        {
            "info": { ... },
            "settings": { ... },
            "data": {
                "n": int,  # Number of detected contours
                "contours": list,  # Extracted contours
                "support": list,  # Contour metadata
            }
        }
        ```
        If no contours are detected, returns None.

    """
        
    # Load or retrieve the cached model
    model = utils.model_loader_cacher(model_id, eval(backend), model_path, force_reload)
    
    ## set device
    device = kwargs.get(
        device,
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
    if prompt in ["everything", "text"]:
        if trim > 0:
            
            height, width = image.shape[:2]
            rx, ry = int(round(trim * 0.5 * width)), int(round(trim * 0.5 * width))
            rh, rw = int(round((1 - trim) * height)), int(round((1 - trim) * width))
            roi_orig = image[ry : ry + rh, rx : rx + rw]
        else:
            roi_orig = copy.deepcopy(image)

    elif prompt in ["mask-select", "everything-in-mask", "text-in-mask"]:
        
        ## get mask from annotations
        annotations = kwargs.get("annotations", {})
        annotation_id_mask = kwargs.get(_vars._mask_type + "_id", None)
        annotation_mask = pp_ul._get_annotation(
            annotations,
            _vars._mask_type,
            annotation_id_mask,
        )
        
        ## convert mask coords to ROI
        mask_coords = annotation_mask["data"]["mask"]
        mask_coords = pp_ul._convert_tup_list_arr(mask_coords)
        rx, ry, rw, rh = cv2.boundingRect(mask_coords[0])  
        if prompt in ["everything-in-mask", "text-in-mask"]:
            roi_orig = image[ry : ry + rh, rx : rx + rw]
        elif prompt == "mask-select":
            roi_orig = image
            resize_x = size / roi_orig.shape[1]
            resize_y = size / roi_orig.shape[0]       
        
    ## resize roi
    image_height, image_width = image.shape[:2]
    roi_orig_height, roi_orig_width = roi_orig.shape[:2]
    roi = pp_utils.resize_image(
        roi_orig, width=size, height=size)
            
    ## encode roi 
    pp_ul._print(f"- starting segmentation of image with size {size}px on device {device} ...")
    results = model(
        roi,
        device=device,
        retina_masks=True,
        imgsz=[int(roi.shape[1]), int(roi.shape[0])],
        conf=confidence,
        iou=iou,
        verbose=False,
    )
    if results:
        speed = results[0].speed
        speed = {
            "preprocess": round(speed["preprocess"], 2),
            "inference": round(speed["inference"], 2),
            "postprocess": round(speed["postprocess"], 2),
            }
        pp_ul._print(f"- sucessfully processed image. speed: {pretty_repr(speed)}")
    else:
        pp_ul._print("- segmentation not successful")
        return
    
    ## prompting
    if backend in ["SAM2", "FastSAM"]:
        if backend=="SAM2":
            prompt_process = SAM2Predictor()
        if backend=="FastSAM":
            prompt_process = FastSAMPredictor()
        if prompt == "mask-select":     
            mask_coords = pp_ul._resize_mask([rx, ry, rw, rh], resize_x, resize_y)
            mask_coords_sam = pp_ul._convert_box_xywh_to_xyxy(mask_coords)
            detections = prompt_process.prompt(results=results, bboxes=mask_coords_sam)
        elif prompt in ["everything","everything-in-mask"]:
            detections = results
        elif prompt in ["text", "text-in-mask"]:
            detections = prompt_process.prompt(results=results, texts=kwargs.get("text",""))
    else:
        detections = results
                
    ## ultralytics.results to array
    if type(detections[0]) == engine.results.Results:
        results = detections[0].cpu()

    ## plot
    if yolo_plot:
        out = results.plot()
        pp_utils.show_image([roi, out])
    if yolo_ret:
        return results 
               
    ## mask ==> contour
    if results.masks:
    
        ## compute hierarchies
        contours_det = []
        for mask in results.masks.data:
            binary_mask = (mask.data.cpu().numpy() > 0).astype(np.uint8) * 255  
            contours, hierarchy = cv2.findContours(
                image=binary_mask,
                mode=_vars.opencv_contour_flags["retrieval"][retrieval],
                method=_vars.opencv_contour_flags["approximation"][approximation],
            )
            contours_det.extend(contours)
        hierarchy_det = utils.calculate_contour_hierarchy(contours_det)

        ## phenopype-style contours
        contours, support = [], []
        for idx, (contour, hierarchy) in enumerate(zip(contours_det, hierarchy_det)):
            if len(contour) > min_nodes and len(contour) < max_nodes:
                center, area, diameter = pp_ul._calc_contour_stats(contour, stats_mode)
                if hierarchy[3] == -1:
                    hierarchy_level = "parent"
                else:
                    hierarchy_level = "child"
                if all([
                        diameter > min_diameter and diameter < max_diameter,
                        area > min_area and area < max_area,
                    ]):
                    
                    contour = pp_ul._resize_contour(contour, size, size, roi_orig_width, roi_orig_height)

                    ## Apply offset if edges were trimmed or ROI was used
                    if trim > 0 or prompt in ["everything-in-mask", "text-in-mask"]:
                        contour += np.array([rx, ry])  # Add ROI offset to all contour points

                    contours.append(contour)
                    support.append(
                        {
                            "center": center,
                            "area": area,
                            "diameter": diameter,
                            "hierarchy_level": hierarchy_level,
                            "hierarchy_idx_child": int(hierarchy[2]),
                            "hierarchy_idx_parent": int(hierarchy[3]),
                        }
                    )                      

        ## make annotation object
        annotation_type = kwargs["annotation_type"]
        annotation = {
            "info": {
                "annotation_type": annotation_type,
                "phenopype_function": kwargs["fun_name"],
                "phenopype_version": pp_ver,
            },
            "settings": {
                "size": size,
                "conf": confidence,
                "iou": iou,
                "prompt": prompt,
                "trim": trim,
                "approximation": approximation,
                "retrieval": retrieval,
                "min_nodes": min_nodes,
                "max_nodes": max_nodes,
                "min_area": min_area,
                "max_area": max_area,
                "min_diameter": min_diameter,
                "max_diameter": max_diameter,
            },
            "data": {
                "n": len(contours), 
                annotation_type: contours, 
                "support": support,},
        }
    
        if len(contours) == 0:
            pp_ul._print("- did not find any contours that match criteria", lvl=2)
        else:
            pp_ul._print("- found " + str(len(contours)) + " contours that match criteria", lvl=2)
            return annotation
    else:
        pp_ul._print("- did not find any contours", lvl=2)
        

@utils.model_path_resolver
@utils.model_config_path_resolver
@decorators.annotation_function
def predict_torch_seg(
        image,
        model_path,
        model_config_path,
        model_id="a",
        device="cpu",
        size=512,
        confidence=0.8,
        prompt="everything",
        force_reload=False,
        stats_mode="circle",
        min_nodes=3,
        max_nodes=inf,
        min_area=0,
        max_area=inf,
        min_diameter=0,
        max_diameter=inf,
        retrieval="ext",
        approximation="simple",
        **kwargs,
        ):
    """
    Perform image segmentation prediction using a pre-trained PyTorch model. This function handles
    model loading, preprocessing, and prediction, returning a binary mask of the segmented area 
    based on the specified prompt type. It also processes the segmentation mask to extract contours 
    and relevant statistics for further analysis.
    
    Parameters:
    ----------
    image : ndarray
        The input image array on which segmentation prediction is performed.
    model_path : str
        The path to the trained model file.
    model_config_path : str
        The path to the model's configuration file.
    model_id : str, optional
        Identifier for the model, used to cache or differentiate between models. Default is 'a'.
    device : str, optional
        The computation device to use for inference ('cpu' or 'cuda'). Default is 'cpu'.
    size : int, optional
        The size to which the region of interest (ROI) is resized before processing. Default is 512.
    confidence : float, optional
        Confidence threshold for converting model output to a binary mask. Default is 0.8.
    prompt : str, optional
        Determines the type of segmentation to perform. Options:
        - "everything": Applies segmentation to the entire image.
        - "mask": Applies segmentation only within an existing annotated mask region.
        Default is "everything".
    force_reload : bool, optional
        If True, forces reloading of the model even if it's cached. Default is False.
    stats_mode : str, optional
        Mode for calculating contour statistics. Options are "circle" or others, depending on the 
        contour's shape analysis method. Default is "circle".
    min_nodes : int, optional
        Minimum number of nodes required for a contour to be considered. Default is 3.
    max_nodes : int, optional
        Maximum number of nodes allowed for a contour to be considered. Default is `inf`.
    min_area : int, optional
        Minimum area of a contour to be considered. Default is 0.
    max_area : int, optional
        Maximum area of a contour to be considered. Default is `inf`.
    min_diameter : int, optional
        Minimum diameter of a contour to be considered. Default is 0.
    max_diameter : int, optional
        Maximum diameter of a contour to be considered. Default is `inf`.
    retrieval : str, optional
        Contour retrieval mode, corresponding to OpenCV's contour retrieval flags. Default is "ext".
    approximation : str, optional
        Contour approximation method, corresponding to OpenCV's approximation flags. Default is "simple".
    **kwargs : dict
        Additional keyword arguments, including:
        - annotations: Existing annotations for refining segmentation or masks.
        - annotation_type: The type of annotations being processed.
        - fun_name: Function name for tracking annotations.
        - other custom parameters for flexibility in processing.
    
    Returns:
    -------
    dict
        An annotation object containing:
        - "info": Metadata about the annotation type and processing details.
        - "settings": The settings used during segmentation and contour extraction.
        - "data": The extracted contours, their support information, and the total count (`n`).
    
    Notes:
    -----
    - If `prompt` is "mask", the function extracts a specific ROI based on an annotated mask and applies 
      segmentation only to that region.
    - Contours are filtered based on the provided criteria for nodes, area, and diameter.
    - The function supports device-agnostic computation and can leverage CUDA if available.
    
    Examples:
    --------
    >>> image = pp.load_image('path/to/image.jpg')
    >>> mask = pp.plugins.predict_torch_seg(
    ...     image, 
    ...     'path/to/model.pth', 
    ...     'path/to/model_config.py', 
    ...     prompt="everything", 
    ...     confidence=0.9
    ... )
    >>> print(mask["data"]["n"])  # Number of contours matching criteria
    >>> pp.show_image(mask)
    """
        
    ## load model config and checkpoint
    model_config = utils.modularize_model_config('model_config', model_config_path)
   
    # Load or retrieve the 
    model = utils.model_loader_cacher(model_id, model_config.load_model, model_path, force_reload)

    ## set device
    device = kwargs.get(
        device,
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    
    ## entire image
    if prompt == "everything":            
        annotations = kwargs.get("annotations", {})
        roi_orig = copy.deepcopy(image)
        
    ## mask only        
    elif prompt == "mask":
        
        ## get mask from annotations
        annotations = kwargs.get("annotations", {})
        annotation_id_mask = kwargs.get(_vars._mask_type + "_id", None)
        annotation_mask = pp_ul._get_annotation(
            annotations,
            _vars._mask_type,
            annotation_id_mask,
        )
        
        ## convert mask coords to ROI
        mask_coords = annotation_mask["data"]["mask"]
        mask_coords = pp_ul._convert_tup_list_arr(mask_coords)
        rx, ry, rw, rh = cv2.boundingRect(mask_coords[0])  
        roi_orig = image[ry : ry + rh, rx : rx + rw]
        
    ## resize roi
    roi_orig_height, roi_orig_width = roi_orig.shape[:2]
    roi = pp_utils.resize_image(
        roi_orig, width=size, height=size)
      
    ## preprocess
    roi_t = model_config.preprocess(roi)
    roi_t = roi_t.unsqueeze(0)
    roi_t.to(device)
    
    ## prediction
    predict_masks = model(roi_t)
   
    ## mask posprocessing
    mask = predict_masks[0].clone().cpu()
    mask = mask > confidence
    mask = mask.squeeze(0).detach().numpy().astype(np.uint8)
    mask[mask==1] = 255
    mask_resized = pp_utils.resize_image(
        mask, width=roi_orig_width, height=roi_orig_height)
    if prompt == "everything":            
        image_bin = mask_resized
    elif prompt == "mask":
        image_bin = np.zeros(image.shape[:2], np.uint8)
        image_bin[ry : ry + rh, rx : rx + rw] = mask_resized
    
    ## find contours in mask
    contours_det, hierarchy_det = cv2.findContours(
        image=image_bin,
        mode=_vars.opencv_contour_flags["retrieval"][retrieval],
        method=_vars.opencv_contour_flags["approximation"][approximation],
    )
    
    ## format contours
    contours, support = [], []
    for idx, (contour, hierarchy) in enumerate(zip(contours_det, hierarchy_det[0])):
        if len(contour) > min_nodes and len(contour) < max_nodes:
            center, area, diameter = pp_ul._calc_contour_stats(contour, stats_mode)
            if hierarchy[3] == -1:
                hierarchy_level = "parent"
            else:
                hierarchy_level = "child"
            if all([
                    diameter > min_diameter and diameter < max_diameter,
                    area > min_area and area < max_area,
                ]):
                
                contours.append(contour)
                support.append(
                    {
                        "center": center,
                        "area": area,
                        "diameter": diameter,
                        "hierarchy_level": hierarchy_level,
                        "hierarchy_idx_child": int(hierarchy[2]),
                        "hierarchy_idx_parent": int(hierarchy[3]),
                    }
                )                      

    ## make annotation object
    annotation_type = kwargs["annotation_type"]
    annotation = {
        "info": {
            "annotation_type": annotation_type,
            "phenopype_function": kwargs["fun_name"],
            "phenopype_version": pp_ver,
        },
        "settings": {
            "size": size,
            "conf": confidence,
            "prompt": prompt,
            "approximation": approximation,
            "retrieval": retrieval,
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "min_area": min_area,
            "max_area": max_area,
            "min_diameter": min_diameter,
            "max_diameter": max_diameter,
        },
        "data": {
            "n": len(contours), 
            annotation_type: contours, 
            "support": support,},
    }

    if len(contours) == 0:
        pp_ul._print("- did not find any contours that match criteria", lvl=2)
    else:
        pp_ul._print("- found " + str(len(contours)) + " contours that match criteria", lvl=2)
        return annotation

    
    
@utils.model_path_resolver
def predict_keras(
    image,
    model_path,
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
    Applies a pre-trained deep learning model to an image and returns a grayscale mask 
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
    # model management
    
    # Load or retrieve the cached model
    model = utils.model_loader_cacher(model_id, keras.models.load_model, model_path)
        
    ## set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    # =============================================================================
    
    image_source = copy.deepcopy(image)

    ## apply binary mask from annotations to image
    if flags.binary_mask:
        
        annotations = kwargs.get("annotations", {})
        annotation_type = kwargs.get("annotation_type", _vars._mask_type)
        annotation_id = kwargs.get(annotation_type + "_id", None)

        binary_mask = np.zeros(image_source.shape, dtype="uint8")
        if annotation_type == _vars._mask_type:
            print("mask")
            binary_mask = visualization.draw_mask(
                image=binary_mask, 
                annotations=annotations, 
                contour_id=annotation_id, 
                line_colour=255,
                line_width=0,
                fill=1)
        elif annotation_type == _vars._contour_type:
            print("contour")
            binary_mask = visualization.draw_contour(
                image=binary_mask, 
                annotations=annotations, 
                contour_id=annotation_id, 
                line_colour=255,
                line_width=0,
                fill=1)

        image_source = cv2.bitwise_and(image_source, binary_mask)    

    # =============================================================================
    ## inference

    image_source = pp_utils.resize_image(image_source, width=model.input.shape[1], height=model.input.shape[2])/255
    image_source = np.expand_dims(image_source, axis=0)
    
    pred = model.predict(image_source)
     
    mask_predicted = pred[0,:,:,1]*255
    mask_predicted = mask_predicted.astype(np.uint8)
    mask_predicted = pp_utils.resize_image(mask_predicted, width=image.shape[1], height=image.shape[0], interpolation="linear")
    
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