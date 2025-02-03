# phenopype-plugins

AI-plugins for [phenopype](https://www.phenopype.org/) - still under early development. 

![FastSAM vs SAM](https://lh6.googleusercontent.com/G5AbLo-Pi1fmMZEuA9PFWVzsniMjbTj2GiJIeKiyGndjUVNxOFSyljIQi9C0i9oTkrgeuqpIInWe4jG2RYWw5iRjx3nCaFhqXQATWcgTEaN_GZonBM2eG9jJ7z_Re67LJD4F88ErvaTXREhKrxF3x5w)

Currently, plugins for three libraries are available:

`ultralytics`
 - predict_yolo_det (object detection): https://docs.ultralytics.com/tasks/detect/ (fine tuned models only)
 - predict_yolo_seg (segmentation): https://docs.ultralytics.com/tasks/segment/ (fine tuned models, and the SAM-family [SAM, SAM2, mobileSAM, FastSAM]) 

`pytorch`
 - predict_torch_seg (Torchvision segmentation models: https://pytorch.org/vision/main/models.html)

`keras`
 - predict_keras (Keras segmentation models https://keras.io/examples/vision/oxford_pets_image_segmentation/)

## Installation

Prior to using the functions with phenopype you need to install the respective dependencies:

1\. Install phenopype (see https://www.phenopype.org/docs/installation/phenopype/ for more details): 

    pip install phenopype

2\. Install the plugins module:

    pip install phenopype-plugins

3\. Install the dependencies

## Dependencies

If you have a GPU and the appropriate drivers install, make sure you install a fitting CUDA version first - e.g., v12.1:

    mamba install -c nvidia cuda-toolkit==12.1

### `torch`

1\. With GPU support:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

2\. Without GPU support:

    pip install torch torchvision

### `keras`

1\. With GPU support:

    pip install keras-gpu

2. Without GPU support:

    pip install keras-gpu


### `ultralytics`

1\. Install Ultralytics BEFORE phenopype due to conflicting opencv-python (ultralytics) and opencv-contrib-python (phenopype) versions (see step 2 for alternatives):

 
    pip install torch torchvision ## needed
    pip install ultralytics

2\. If you have already installed phenopype and can't or don't want to uninstall it, you can do the following:
    
    pip install ultralytics

    ## force reinstall opencv-contrib-python
    pip install opencv-contrib-python --force-reinstall
