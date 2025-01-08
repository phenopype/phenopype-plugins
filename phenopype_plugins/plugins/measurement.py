#%% imports

import cv2
import logging
import numpy as np
import sys
import warnings
from tqdm import tqdm as _tqdm
from PIL import Image

from phenopype import _vars, config
from phenopype.core import preprocessing
from phenopype import utils as pp_utils
from phenopype import utils_lowlevel as pp_ul
from phenopype_plugins import __version__

try:
    from radiomics import featureextractor
    import SimpleITK as sitk
except ImportError:
    warnings.warn("Failed to import pyradiomics. Some functionalities may not work.", ImportWarning)
    
    
try:
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
except ImportError:
    warnings.warn("Failed to import sckit-learn. Some functionalities may not work.", ImportWarning)




#%% functions


def extract_radiomic_features(
    image,
    annotations,
    features=["firstorder"],
    channel_names=["blue", "green", "red"],
    decompose=None,
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
    features: ["firstorder", "shape2D", "glcm", "gldm", "glrlm", "glszm", "ngtdm"] list, optional
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
    annotation = pp_ul._get_annotation(
        annotations=annotations,
        annotation_type=_vars._contour_type,
        annotation_id=contour_id,
        kwargs=kwargs,
    )
    contours = annotation["data"][_vars._contour_type]
    contours_support = annotation["data"]["support"]
    
    fun_name = sys._getframe().f_code.co_name
    annotation_type = pp_ul._get_annotation_type(fun_name)
    annotation_id = kwargs.get("annotation_id", None)
    
    
    # =============================================================================
    # run

    ## features
    features_use, texture_features = {}, []
    for feature in features:
        features_use[feature] = []
        
    ## capture errors
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)

    ## create forgeround mask
    foreground_mask_inverted = np.zeros(image.shape[:2], np.uint8)
    for coords in contours:
        foreground_mask_inverted = cv2.fillPoly(foreground_mask_inverted, [coords], 255)

    ## do internal decomposition
    if decompose:
        decomposed_images = []
        for dec in decompose:
            decomposed_image = preprocessing.decompose_image(image.copy(), dec)
            if len(decomposed_image.shape) == 2:  # If the image is grayscale, add a channel dimension
                decomposed_image = np.expand_dims(decomposed_image, axis=-1)
            decomposed_images.append(decomposed_image)
        stack = np.concatenate(decomposed_images, axis=-1)
    else:
        if image.ndim == 2:
            stack = np.expand_dims(image, axis=-1)
            
    ## checks
    print(channel_names, stack.shape[2])
    assert len(channel_names) == stack.shape[2], pp_ul._print("make sure N channel names match N image layers")
        
    
    for idx1, (coords, support) in _tqdm(
            enumerate(zip(contours, contours_support)),
            "Extracting radiomic features",
            total=len(contours),
            disable=not config.verbose
    ):

        output = {}
        
        if support["diameter"] > min_diameter:

            for idx2, channel in enumerate(channel_names):

                if (idx2 + 1) > stack.shape[2]:
                    continue

                
                rx, ry, rw, rh = cv2.boundingRect(coords)
                data = stack[ry : ry + rh, rx : rx + rw, idx2]
                mask = foreground_mask_inverted[ry : ry + rh, rx : rx + rw]
                sitk_data = sitk.GetImageFromArray(data)
                sitk_mask = sitk.GetImageFromArray(mask)
                
                if len(np.unique(mask)) > 1:
                
                    extractor = featureextractor.RadiomicsFeatureExtractor()
                    extractor.disableAllFeatures()
                    extractor.enableFeaturesByName(**features_use)
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

    return pp_ul._update_annotations(
        annotations=annotations,
        annotation=annotation,
        annotation_type=annotation_type,
        annotation_id=annotation_id,
        kwargs=kwargs,
    )



def recolorize_binclust(
        image, 
        mask=None,
        method='histogram', 
        blur=5,
        n_clusters=3, 
        bins_per_channel=3,
        resize=None, 
        plot=False, 
        ):
        
    # prep image for processing
    pixels = preprocessing.blur(image, kernel_size=blur, method="median")
    pixels = pixels.reshape(-1, 3)

    if method == 'histogram':
        bins = np.linspace(0, 255, bins_per_channel + 1)  # Bins from 0 to 255
        centers = 0.5 * (bins[:-1] + bins[1:])
        digitized = np.clip(np.digitize(pixels, bins) - 1, 0, bins_per_channel - 1)
        mean_colors = np.zeros((bins_per_channel ** 3, 3))
        for r in range(bins_per_channel):
            for g in range(bins_per_channel):
                for b in range(bins_per_channel):
                    bin_idx = r + bins_per_channel * g + (bins_per_channel ** 2) * b
                    bin_mask = (digitized[:, 0] == r) & (digitized[:, 1] == g) & (digitized[:, 2] == b)
                    if np.any(bin_mask):  # Only compute mean if bin is not empty
                        mean_colors[bin_idx] = np.mean(pixels[bin_mask], axis=0)
        indices = digitized[:, 0] + bins_per_channel * digitized[:, 1] + (bins_per_channel ** 2) * digitized[:, 2]
        recolored_pixels = mean_colors[indices]
        pixel_assignments = indices.reshape(image.shape[:2]).astype(np.uint8)
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_.astype(int)
        recolored_pixels = centers[kmeans.labels_]
        pixel_assignments = kmeans.labels_.reshape(image.shape[:2])
    else:
        raise ValueError("Invalid method")
        
    # Reshape recolored pixels to match the original image's RGB shape
    recolored_pixels = recolored_pixels.reshape(image.shape).astype(np.uint8)
    # recolored_pixels = recolored_pixels

    # If the original image includes an alpha channel, append it to the recolored image
    if mask is not None:
        recolored_pixels = np.dstack((recolored_pixels, mask))
        
    if plot:
        if mask is not None:
            image_rgba = np.dstack((image, mask))
            image_plot = Image.fromarray(image_rgba[..., [2,1,0,3]].astype('uint8'), 'RGBA')
            image_recolored = Image.fromarray(recolored_pixels[..., [2,1,0,3]].astype('uint8'), 'RGBA')
        else:
            image_plot = Image.fromarray(image[..., [2,1,0]].astype('uint8'),'RGB')
            image_recolored = Image.fromarray(recolored_pixels[..., [2,1,0]].astype('uint8'), 'RGB')

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image_plot)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image_recolored)
        plt.title(f'Recolored Image ({method})')
        plt.axis('off')
        plt.show()

    return {
        'centers': centers,
        'pixel_assignments': pixel_assignments,
        'recolored_image': recolored_pixels,
    }


#%% shelved

# def detect_landmark(
#     image,
#     model_path,
#     mask=True,
#     **kwargs,
# ):
#     """
#     Place landmarks. Note that modifying the appearance of the points will only 
#     be effective for the placement, not for subsequent drawing, visualization, 
#     and export.
    
#     Parameters
#     ----------
#     image : ndarray
#         input image
#     point_colour: str, optional
#         landmark point colour (for options see pp.colour)
#     point_size: int, optional
#         landmark point size in pixels
#     label : bool, optional
#         add text label
#     label_colour : str, optional
#         landmark label colour (for options see pp.colour)
#     label_size: int, optional
#         landmark label font size (scaled to image)
#     label_width: int, optional
#         landmark label font width  (scaled to image)

#     Returns
#     -------
#     annotations: dict
#         phenopype annotation containing landmarks
#     """

#     # =============================================================================
#     # annotation management

#     fun_name = sys._getframe().f_code.co_name
    
#     annotations = kwargs.get("annotations", {})
#     annotation_type = pp_ul._get_annotation_type(fun_name)
#     annotation_id = kwargs.get("annotation_id", None)

#     annotation = pp_ul._get_annotation(
#         annotations=annotations,
#         annotation_type=annotation_type,
#         annotation_id=annotation_id,
#         kwargs=kwargs,
#     )
    
        
#     # =============================================================================
#     # execute
    
#     landmark_tuple_list = []
        
#     if mask:        
#         if not annotations:
#             print("- no mask coordinates provided - cannot detect within mask")
#             pass
#         else:
#             annotation_id_mask = kwargs.get(_vars._mask_type + "_id", None)
#             annotation_mask = pp_ul._get_annotation(
#                 annotations,
#                 _vars._mask_type,
#                 annotation_id_mask,
#                 prep_msg="- masking regions in thresholded image:",
#             )
            
#             bbox_coords = cv2.boundingRect(np.asarray(annotation_mask["data"][_vars._mask_type], dtype="int32"))
#     else:
#         bbox_coords = None
        
#     landmark_tuple_list = plugins.libraries.phenomorph.main.predict_image(
#         img=image, model_path=model_path, bbox_coords=bbox_coords, plot=False)

#     print("- found {} points".format(len(landmark_tuple_list)))

#     annotation = {
#         "info": {
#             "annotation_type": annotation_type,
#             "phenopype_function": "plugins.ml_morph.predict_image",
#             "phenopype_version": __version__,
#         },
#         "settings": {
#         },
#         "data": {
#             annotation_type: landmark_tuple_list
#             },
#     }


#     return pp_ul._update_annotations(
#         annotations=annotations,
#         annotation=annotation,
#         annotation_type=annotation_type,
#         annotation_id=annotation_id,
#         kwargs=kwargs,
#     )
        
