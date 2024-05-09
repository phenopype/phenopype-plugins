# phenopype-plugins

AI-plugins for [phenopype](https://www.phenopype.org/) - currently under development, stay tuned!

## Installation

## Plugins

## Dependencies

### Pytorch

### Ultralytics

1\. Install Ultralytics BEFORE phenopype due to conflicting opencv-python (ultralytics) and opencv-contrib-python (phenopype) versions (see step 2 for alternatives):

    ## if your PC has a GPU, install cuda and the corresponding pytorch version first
    mamba install -c nvidia cuda-toolkit==12.1
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    pip install ultralytics
    pip install phenopype

2\. If you already have install phenopype and dan#t or don't want to uninstall it, you can do the following:
    
    ## install ultralytics into env where phenopype is already installed
    pip install ultralytics

    ## uninstall opencv-python
    pip uninstall opencv-python

    ## force reinstall opencv-contrib-python
    pip install opencv-contrib-python --force-reinstall

More information: https://docs.ultralytics.com/models/fast-sam/#available-models-supported-tasks-and-operating-modes

Please cite as:

    @misc{zhao2023fast,
        title={Fast Segment Anything},
        author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
        year={2023},
        eprint={2306.12156},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }