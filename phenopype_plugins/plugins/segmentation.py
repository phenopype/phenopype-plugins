#%% modules

import copy
import cv2
import numpy as np
import sys
from rich.pretty import pretty_repr
from dataclasses import make_dataclass

from phenopype import _vars
from phenopype import utils as pp_utils
from phenopype import utils_lowlevel as pp_ul
from phenopype.core import segmentation, visualization

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
def predict_fastSAM(
        image,
        model_path,
        model_id="a",
        backend="FastSAM",
        force_reload=False,
        prompt="everything",
        trim=0.05,
        resize_roi=1024,
        confidence=0.8,
        iou=0.65,
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
    
    # =============================================================================
    # setup
        
    ## set flags
    flags = make_dataclass(
        cls_name="flags",
        fields=[
            ("prompt", str, prompt), 
             ])
    
    # =============================================================================
    # model management
    
    # Load or retrieve the cached model
    model = utils.model_loader_cacher(model_id, eval(backend), model_path, force_reload)
        
    ## set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================================================================
    # prepare image
    
    if flags.prompt in ["everything", "text"]:
        if trim > 0:
            
            height, width = image.shape[:2]
            rx, ry = int(round(trim * 0.5 * width)), int(round(trim * 0.5 * width))
            rh, rw = int(round((1 - trim) * height)), int(round((1 - trim) * width))
            roi_orig = image[ry : ry + rh, rx : rx + rw]
        else:
            roi_orig = copy.deepcopy(image)

    elif flags.prompt in ["everything-box", "box"]:
        ## get mask from annotations
        annotations = kwargs.get("annotations", {})
        annotation_id_mask = kwargs.get(_vars._mask_type + "_id", None)
        annotation_mask = pp_ul._get_annotation(
            annotations,
            _vars._mask_type,
            annotation_id_mask,
        )
        ## convert mask coords to ROI
        coords = annotation_mask["data"]["mask"]
        coords = pp_ul._convert_tup_list_arr(coords)
        rx, ry, rw, rh = cv2.boundingRect(coords[0])  
        
        if flags.prompt == "everything-box":
            roi_orig = image[ry : ry + rh, rx : rx + rw]
        elif flags.prompt == "box":
            roi_orig = image
            resize_x = resize_roi / roi_orig.shape[1]
            resize_y = resize_roi / roi_orig.shape[0]       
        
    ## resize roi
    roi_orig_height, roi_orig_width = roi_orig.shape[:2]
    roi = pp_utils.resize_image(
        roi_orig, width=resize_roi, height=resize_roi)
            
    # =============================================================================
    # apply model
    
    print(f"- starting {flags.prompt} prompt on device {device}")
    
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
        print(f"- speed: {pretty_repr(speed)}")

    else:
        return image

    # =============================================================================
    ## set up prompt
        
    # overrides = dict(conf=0.25,  imgsz=1024)
    
    # task="segment", 
    # mode="predict", 
    # model=os.path.basename(model_path), 
    # save=False,
       
    # overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, save=False, 
    #                  imgsz=1024, verbose=False)
    if backend=="SAM":
        prompt_process = SAM2Predictor()
    if backend=="FastSAM":
        prompt_process = FastSAMPredictor()
        
    # everything_results = predictor(roi)
    results = everything_results
    out = results[0].plot()
    
    
    # from phenopype import show_image
    # show_image(out)    
    
    # =============================================================================
    ## get detected objects
    
    ## box-post
    if flags.prompt == "box":     
        
        mask_coords = pp_ul._resize_mask([rx, ry, rw, rh], resize_x, resize_y)
        mask_coords_sam = pp_ul._convert_box_xywh_to_xyxy(mask_coords)
        detections = prompt_process.prompt(results=everything_results, bboxes=mask_coords_sam)
    
    ## everything prompt 
    elif flags.prompt in ["everything","everything-box"]:
        detections = everything_results

    elif flags.prompt == "text":
        detections = prompt_process.prompt(results=everything_results, texts=kwargs.get("text",""))
        
    ## ultralytics.results to array
    if type(detections[0]) == engine.results.Results:
        results = detections[0].cpu()
        roi_bin = np.array(results.masks[0].data[0], dtype=np.uint8)
        for mask in results.masks[1:]:
            roi_bin |= np.array(mask.data[0], dtype=np.uint8)
        roi_bin[roi_bin==1] = 255

    ## resize to original dimensions
    roi_det = pp_utils.resize_image(
        roi_bin, width=roi_orig_width, height=roi_orig_height)
    
    if flags.prompt in ["everything", "everything-box", "text"]:
        if trim == 0 and flags.prompt in ["everything", "text"]:
            image_bin = roi_det
        else:
            image_bin = np.zeros(image.shape[:2], "uint8")
            image_bin[ry : ry + rh, rx : rx + rw] = roi_det
        
    elif flags.prompt in ["box"]:
        image_bin = roi_det
            
        
    return image_bin

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

@utils.model_path_resolver
@utils.model_config_path_resolver
def predict_torch(
        image,
        model_path,
        model_config_path,
        model_id="a",
        primer="contour",
        confidence=0.8,
        **kwargs,
        ):
    """
    Perform image segmentation prediction using a pre-trained PyTorch model. This function handles
    model loading, preprocessing, and prediction, returning a binary mask of the segmented area
    based on the specified 'primer' type.

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
    primer : str, optional
        Type of annotation to use for determining region of interest (ROI) in the image. Supported
        types are 'contour' and 'mask'. Default is 'contour'.
    confidence : float, optional
        Confidence threshold for converting model output to a binary mask. Default is 0.8.
    **kwargs : dict
        Additional keyword arguments which may include 'annotations' to specify existing 
        annotations for refining predictions or other custom parameters.

    Returns:
    -------
    ndarray
        A binary mask image of the segmented region, where the segmentation is based on the 
        primer type and specified region of interest within the image.

    Examples:
    --------
    >>> image = pp.load_image('path/to/image.jpg')
    >>> mask = pp.plugins.predict_torch(image, 'path/to/model.pth', 'path/to/model_config.py', primer="mask", confidence=0.9)
    >>> pp.show_image(mask)
    """
    
    # =============================================================================
    # model management
        
    ## load model config
    model_config = utils.modularize_model_config('model_config', model_config_path)

    # Load or retrieve the cached model
    model = utils.model_loader_cacher(model_id, model_config.load_model, model_path)

    ## set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================================================================
    ## annotation management
    
    annotations = kwargs.get("annotations", {})
    
    if primer=="contour":
        annotation_id_input = kwargs.get(_vars._contour_type + "_id", None)
        annotation = pp_ul._get_annotation(
            annotations,
            _vars._contour_type,
            annotation_id_input,
        )
        coords = annotation["data"][_vars._contour_type][0]
    elif primer=="mask":
        annotation_id_input = kwargs.get(_vars._mask_type + "_id", None)
        annotation = pp_ul._get_annotation(
            annotations,
            _vars._mask_type,
            annotation_id_input,
        )      
        coords = annotation["data"][_vars._mask_type][0]
        
    # =============================================================================
    ## inference

    roi, roi_box = pp_ul._extract_roi_center(image, coords, 512)
    image_tensor = model_config.preprocess(roi)
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
