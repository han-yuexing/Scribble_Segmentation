# PTS Model Image Segmentation Program Description
This repository contains the code for our paper [A pseudo-labeling based weakly supervised segmentation method for few-shot texture images](https://www.sciencedirect.com/science/article/pii/S095741742302612X) .

## Purpose
To address the difficulty of annotating material images, line annotations are used instead of full annotations for learning. To compensate for the lack of supervision with line annotations, the PTS model was designed to generate pseudo-annotations for supplemental supervision. This software implements simple test inferences for models saved after training on a Windows system, displaying and saving segmentation results.

## Core Program Files
- `model/main_material.py`: Program entry point for model training and testing, some hyperparameters provided for runtime parameters.
- `model/attention.py`: Implementation code for the PTS auxiliary branch, which includes several other attention mechanism implementations used in early implementations (not called in the current implementation).
- `model/dataset.py`: Reads image data files, converting images stored under the `data` directory into training and testing inputs.
- `model/utils.py`: Contains implementations of various data augmentations (only some functions are used in the current implementation).
- `model/unetpp.py`: Implementation code for the PTS main branch, including various backbone replacements and different feature extraction returns.
- `model/metrics.py`: Custom loss function implementations and metric calculation functions.

## Data Directories
- `data/TiAl/train`: Line annotations and original images for titanium alloy single image training.
- `data/TiAl/train_two`: Line annotations and original images for titanium alloy two-image training.
- `data/TiAl/test_comparison`: Annotations and original images for titanium alloy testing.
- `data/ceramic/train_one`: Line annotations and original images for ceramic single image training.
- `data/ceramic/train`: Line annotations and original images for ceramic two-image training.
- `data/ceramic/test`: Annotations and original images for ceramic testing.

## Requirements
`requirements.txt`: The Python environment file (exported using Anaconda, should also be callable with venv, packages that fail to install can be replaced with pip).

The rest of the files are supplemental and do not affect the basic functionality of the PTS program. Considering potential subsequent derivative work, supplemental files are retained and need to be modified according to the current core file's interface format to be operational. For specific file functions and meanings, please refer to the comments in the `main_material.py` file.

## Program Operating Environment
- Operating System: ubuntu18.04
- Runtime Language: Python 3.6(.8)
- Python Libraries: pytorch 1.8.2, numpy 1.19.2, scikit-learn 0.24.2, pillow 8.2.0, pyqt 5.9.7 (used for software packaging), opencv 3.4.2.16. All libraries used during the training process are saved in the `requirements.txt` file, which can be used directly for environment installation. Since Conda is used and some libraries are installed using PyPI, libraries that fail to install according to the `requirements.txt` file can be installed using pip instead.

## Methods and Algorithms
The program is based on computer image processing and deep learning algorithms, including the following parts:
1. Data augmentation
2. Main branch (using a pretrained backbone unet++ network)
3. Auxiliary branch (feature embedding convolution, distance measurement and generation)
4. Prediction segmentation

The training input for the PTS model is the original image and its annotation (currently, line annotations are used, theoretically point annotations could replace them). After the input image and annotation undergo data augmentation such as rotation and flipping, they are input into the unet++ network for training. Based on the multi-order features extracted from the unet++ network, feature embedding and distance measurement of the adjacent batch are performed to generate pseudo-annotations for the current image, which are then used to supplementally supervise the main branch's unet++ network to obtain the final segmentation prediction result.

## Code Explanation
Software Interface: The PTS software implements the invocation of saved models for prediction on other images. The use of the software does not require a GPU, and the specific usage can be referred to in the software manual. The software only provides inference functionality based on known models.

Code: The program code part contains all the original code needed to run as well as some supplemental code files. The code provides training and testing entry points. The code runs on the GPU and requires GPU resources. Under the current setting of hyperparameters, this program code can successfully run on Server 4 (with a 3090 GPU), but the successful run is not guaranteed if any hyperparameters or contents are modified (memory, model convergence, etc.).

## Detailed Code Explanation
1. Reading runtime parameters:
   `def getArgs():`
   Function: Read runtime parameters
   Parameters:
   Returns: Runtime parameters `args`
2. Reading the main branch model:
   `def getModel(args):`
   Function: Read the main branch model
   Parameters: Runtime parameters `args`
   Returns: Main branch model `model`
3. Reading the auxiliary branch model
   `def getSelfAttnModel(args):`
   Function: Read the auxiliary branch model
   Parameters: Runtime parameters `args`
   Returns: Auxiliary branch model `model`

