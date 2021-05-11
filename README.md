
# Gesture recognition system
Author: Ladislav Ondris

This project performs gesture recognition from depth images. 
It consists of hand detection, hand pose estimation, and gesture classification.
Hands are detected using a Tiny YOLOv3 model.
The gesture recognition system then uses a JGR-P2O hand pose estimator
to determine the hands' skeleton, which is used for gesture classification.


See demonstration videos, which are located in the `docs/` directory.

## Prerequisites

Python 3.7.10  
Intel RealSense SR305 depth camera - for live capture

## Installation

Install the required packages with:  
```
pip install -r requirements.txt
```

In case TensorFlow has a wrong dependency of gast, which may result in warning
or error messages, install 0.3.3 version of gast, which downgrades the package from version 0.4.0.

```
pip install gast==0.3.3
```

## Usage examples

The following examples use mostly 'live' option as the source of images.
You can use the 'dataset' option instead. Although the custom dataset is not part of the 
repository, as its size too big, a few images were included for demonstration purposes.

### Hand detection

To detect both hands from images captured with SR305 camera (as the default option):  
```
python3 detect.py live --num-detections 2 --plot
```

<p float="left">
    <img src="./docs/readme/live_detection.png" alt="live_detection" width="220"/>
    <img src="./docs/readme/live_detection2.png" alt="live_detection2" width="220"/>
</p> 

```
usage: detect.py [-h] [--camera CAMERA] [--num-detections NUM_DETECTIONS]
                 [--plot]
                 source

positional arguments:
  source                the source of images (allowed options: live, dataset)

optional arguments:
  -h, --help            show this help message and exit
  --camera CAMERA       the camera model in use for live capture (default:
                        SR305)
  --num-detections NUM_DETECTIONS
                        the maximum number of bounding boxes for hand
                        detection (default: 1)
  --plot                plot the result of detection
```

### Hand pose estimation

To estimate hand poses from images captured with SR305 camera:  
```
python3 estimate.py live --plot
```

<p float="left">
    <img src="./docs/readme/live_estimation.png" alt="live_estimation" width="220"/>
    <img src="./docs/readme/live_estimation2.png" alt="live_estimation2" width="220"/>
</p>

```
usage: estimate.py [-h] [--camera CAMERA] [--plot] source

positional arguments:
  source           the source of images (allowed options: live, dataset)

optional arguments:
  -h, --help       show this help message and exit
  --camera CAMERA  the camera model in use for live capture (default: SR305)
  --plot           plot the result of estimation
```

### System's usage

The system requires that the user defines the gesture to be recognized, which
is described in Section *Preparation of gesture database*. For demonstration purposes,
the gesture database is already prepared for the gesture with an opened palm, 
fingers outstretched and apart.  

The usage of the real-time recognition from live images or from the custom dataset is shown in 
*Real-time gesture recognition*.


#### Preparation of gesture database

To capture a gesture with label `1` into a `gestures` directory with a scan
period of one second and SR305 camera:  
```
python3 database.py gestures 1 10
```

```
usage: database.py [-h] [--scan-period SCAN_PERIOD] [--camera CAMERA]
                   [--hide-plot]
                   directory label count

positional arguments:
  directory             the name of the directory that should contain the
                        user-captured gesture database
  label                 the label of the gesture that is to be captured
  count                 the number of samples to scan

optional arguments:
  -h, --help            show this help message and exit
  --scan-period SCAN_PERIOD
                        intervals between each capture in seconds (default: 1)
  --camera CAMERA       the camera model in use for live capture (default:
                        SR305)
  --hide-plot           hide plots of the captured poses - not recommended
```

#### Real-time gesture recognition

**For demonstration**, the directory named "gestures" is already present,
containing definitions for a gesture with an opened palm, fingers outstretched
and apart.  

To start the gesture recognition system using gesture database stored in 
the `gestures` directory:  
```
python3 recognize.py live gestures --plot
```

To start the gesture recognition from the evaluation dataset:  
```
python3 recognize.py dataset gestures --plot
```

The system plots figures similar to the following:  
<p float="left">
    <img src="./docs/readme/live_gesture1.png" alt="live_gesture1" width="220"/>
    <img src="./docs/readme/live_nongesture.png" alt="live_nongesture" width="220"/>
</p>

```
usage: recognize.py [-h] [--error-threshold ERROR_THRESHOLD]
                    [--orientation-threshold ORIENTATION_THRESHOLD]
                    [--camera CAMERA] [--plot] [--hide-feedback]
                    [--hide-orientation]
                    source directory

positional arguments:
  source                the source of images (allowed options: live, dataset)
  directory             the name of the directory containg the user-captured
                        gesture database

optional arguments:
  -h, --help            show this help message and exit
  --error-threshold ERROR_THRESHOLD
                        the pose (JRE) threshold (default: 120)
  --orientation-threshold ORIENTATION_THRESHOLD
                        the orientation threshold in angles (maximum: 90,
                        default: 90)
  --camera CAMERA       the camera model in use for live capture (default:
                        SR305)
  --plot                plot the result of gesture recognition
  --hide-feedback       hide the colorbar with JRE errors
  --hide-orientation    hide the vector depicting the hand's orientation
```

### Training of models

To train the Tiny YOLOv3 on the HandSeg dataset:  
```
python3 train_yolov3.py
```

To train the JGR-P2O model on the Bighand dataset from existing weights:  
```
python3 train_jgrp2o.py bighand --model logs/20210426-125059/train_ckpts/weights.25.h5
```

See `--help` for other optional arguments.

## Project structure

### Top-level structure

    .
    ├── datasets                # Datasets (including gesture database)
    ├── docs                    # Demonstration videos, readme files, and images 
    ├── text_source             # Latex source files of the thesis' text
    ├── src                     # Source files
    ├── LICENSE                 # MIT license
    ├── README.md               # Contents of this file
    ├── requirements.txt        # Package requirements 
    └── bachelors_thesis.pdf    # Text of the thesis

### Datasets

    datasets
    ├── bighand                     # Hand pose estimation dataset (not preprocessed)
    ├── cvpr15_MSRAHandGestureDB    # Hand pose estimatino dataset (is preprocessed)
    ├── handseg150k                 # Hand segmentation dataset (both hands)
    ├── simple_boxes                # Generated toy object detection dataset
    ├── custom                      # Created dataset for the evaluation of gesture recognition
    └── usecase                     # Contains gesture databases captured by the user 

### Source files

    src
    ├── acceptance               # Gesture acceptance module (gesture recognition algorithm)
    ├── datasets                 # Dataset related code (pipelines, plots, generation)
    ├── detection                # Detection methods - Tiny YOLOv3, RDF
    ├── estimation               # JGR-P2O estimation model and preprocessing
    ├── metacentrum              # Scripts for training models in Metacentrum
    ├── system                   # Access point to gesture recognition system 
    │                              (database_scanner, gesture_recognizer, hand_position_estimator)
    └── utils                    # Camera, logs, plots, live capture, config


## License

This project is licensed under the terms of the MIT license.
