#%% imports

import copy
import cv2
import numpy as np
import sys
from rich.pretty import pretty_repr
from dataclasses import make_dataclass
import time

from phenopype import _vars
from phenopype import utils as pp_utils
from phenopype import utils_lowlevel as pp_ul
from phenopype.core import segmentation, visualization

from phenopype_plugins import utils

import warnings

try:
    from radiomics import featureextractor
    import SimpleITK as sitk
except ImportError:
    warnings.warn("Failed to import pyradiomics. Some functionalities may not work.", ImportWarning)


#%% functions


def extract_radiomic_features(
    image,
    annotations,
    features=["firstorder"],
    channel_names=["blue", "green", "red"],
    min_diameter=5,
    **kwargs,
):
    """
    Collects 120 texture features using the pyradiomics feature extractor
    ( https://pyradiomics.readthedocs.io/en/latest/features.html ):

    - firstorder: First Order Statistics (19 features)
    - shape2d: Shape-based (2D) (16 features)
    - glcm: Gray Level Cooccurence Matrix (24 features)
    - gldm: Gray Level Dependence Matrix (14 features)
    - glrm: Gray Level Run Length Matrix (16 features)
    - glszm: Gray Level Size Zone Matrix (16 features)
    - ngtdm: Neighbouring Gray Tone Difference Matrix (5 features)

    Features are collected from every contour that is supplied along with the raw
    image, which, depending on the number of contours, may result in long computing 
    time and very large dataframes.

    The specified channels correspond to the channels that can be selected in
    phenopype.core.preprocessing.decompose_image.


    Parameters
    ----------
    image : ndarray
        input image
    annotation: dict
        phenopype annotation containing contours
    features: ["firstorder", "shape", "glcm", "gldm", "glrlm", "glszm", "ngtdm"] list, optional
        type of texture features to extract
    channels : list, optional
        image channel to extract texture features from. if none is given, will extract from all channels in image
    min_diameter: int, optional
        minimum diameter of the contour (shouldn't be too small for sensible feature extraction')

    Returns
    -------
    annotations: dict
        phenopype annotation containing texture features

    """

    # =============================================================================
    # annotation management

    ## get contours
    contour_id = kwargs.get(_vars._contour_type + "_id", None)
    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=_vars._contour_type,
        annotation_id=contour_id,
        kwargs=kwargs,
    )
    contours = annotation["data"][_vars._contour_type]
    contours_support = annotation["data"]["support"]
    
    ##  features
    fun_name = sys._getframe().f_code.co_name
    annotation_type = ul._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    # =============================================================================
    # setup

    tqdm_off = kwargs.get("tqdm_off",True)

    feature_activation = {}
    for feature in features:
        feature_activation[feature] = []
        
    # =============================================================================
    # execute

    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)

    ## create forgeround mask
    foreground_mask_inverted = np.zeros(image.shape[:2], np.uint8)
    
    # print(contours)
    for coords in contours:
        foreground_mask_inverted = cv2.fillPoly(foreground_mask_inverted, [coords], 255)
        
    texture_features = []
    if image.ndim == 2:
        layers = 1
        image = np.dstack((image,image)) 
    else:
        layers = image.shape[2]
        
    if len(channel_names) > layers:
        print("- Warning: more channels provided than in given image - skipping excess ones!")
    
    for idx1, (coords, support) in _tqdm(
            enumerate(zip(contours, contours_support)),
            "Computing texture features",
            total=len(contours),
            disable=tqdm_off
    ):

        output = {}
        
        if support["diameter"] > min_diameter:

            for idx2, channel in enumerate(channel_names):

                if (idx2 + 1) > image.shape[2]:
                    continue

                
                rx, ry, rw, rh = cv2.boundingRect(coords)
                data = image[ry : ry + rh, rx : rx + rw, idx2]
                mask = foreground_mask_inverted[ry : ry + rh, rx : rx + rw]
                sitk_data = sitk.GetImageFromArray(data)
                sitk_mask = sitk.GetImageFromArray(mask)
                
                if len(np.unique(mask)) > 1:
                
                    extractor = featureextractor.RadiomicsFeatureExtractor()
                    extractor.disableAllFeatures()
                    extractor.enableFeaturesByName(**feature_activation)
                    detected_features = extractor.execute(sitk_data, sitk_mask, label=255)

                else:
                    continue

                for key, val in detected_features.items():
                    if not "diagnostics" in key:
                        output[channel + "_" + key.split("_", 1)[1]] = float(val)

        texture_features.append(output)
        
        

    # =============================================================================
    # return

    annotation = {
        "info": {
            "phenopype_function": fun_name,
            "phenopype_version": __version__,
            "annotation_type": annotation_type,
        },
        "settings": {
            "features": features,
            "min_diameter": min_diameter,
            "channels_names": channel_names,
            "contour_id": contour_id,
        },
        "data": {annotation_type: texture_features,},
    }

    # =============================================================================
    # return

    return ul._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )

def detect_landmark(
    image,
    model_path,
    mask=True,
    **kwargs,
):
    """
    Place landmarks. Note that modifying the appearance of the points will only 
    be effective for the placement, not for subsequent drawing, visualization, 
    and export.
    
    Parameters
    ----------
    image : ndarray
        input image
    point_colour: str, optional
        landmark point colour (for options see pp.colour)
    point_size: int, optional
        landmark point size in pixels
    label : bool, optional
        add text label
    label_colour : str, optional
        landmark label colour (for options see pp.colour)
    label_size: int, optional
        landmark label font size (scaled to image)
    label_width: int, optional
        landmark label font width  (scaled to image)

    Returns
    -------
    annotations: dict
        phenopype annotation containing landmarks
    """

    # =============================================================================
    # annotation management

    fun_name = sys._getframe().f_code.co_name
    
    annotations = kwargs.get("annotations", {})
    annotation_type = ul._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)

    annotation = ul._get_annotation(
        annotations=annotations,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )
    
        
    # =============================================================================
    # execute
    
    landmark_tuple_list = []
        
    if mask:        
        if not annotations:
            print("- no mask coordinates provided - cannot detect within mask")
            pass
        else:
            annotation_id_mask = kwargs.get(_vars._mask_type + "_id", None)
            annotation_mask = ul._get_annotation(
                annotations,
                _vars._mask_type,
                annotation_id_mask,
                prep_msg="- masking regions in thresholded image:",
            )
            
            bbox_coords = cv2.boundingRect(np.asarray(annotation_mask["data"][_vars._mask_type], dtype="int32"))
    else:
        bbox_coords = None
        
    landmark_tuple_list = plugins.libraries.phenomorph.main.predict_image(
        img=image, model_path=model_path, bbox_coords=bbox_coords, plot=False)

    print("- found {} points".format(len(landmark_tuple_list)))

    annotation = {
        "info": {
            "annotation_type": annotation_type,
            "phenopype_function": "plugins.ml_morph.predict_image",
            "phenopype_version": __version__,
        },
        "settings": {
        },
        "data": {
            annotation_type: landmark_tuple_list
            },
    }


    return ul._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )
        
