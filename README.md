
# Gesture recognition system
Author: Ladislav Ondris

This project performs gesture recognition from depth images. 
It consists of hand detection, hand pose estimation, and gesture classification.
Hands are detected using a Tiny YOLOv3 model.
The gesture recognition system then uses a JGR-P2O hand pose estimator
to determine the hands' skeleton, which is used for gesture classification.


## Prerequisites

Python 3.7.10  
Intel RealSense depth camera (e.g., SR305)

## Installation

Install the required packages with:  
```pip install -r requirements.txt```

Install gast==0.3.3, which downgrades the package from version 0.4.0.
TensorFlow has a wrong dependency. It may otherwise print warnings and not function properly.  
`pip install gast==0.3.3`

## Usage examples

### System's usage

The system requires that the user defines the gesture to be recognized, which
is performed in *Preparation of gesture database*. The real-time recognition 
from live images or from the custom dataset is demonstrated in 
*Real-time gesture recognition*.

#### Preparation of gesture database

To capture a gesture with label `1` into a `gestures` directory with a scan period of one second and SR305 camera:  
```python3 src/system/database_scanner.py --dir gestures --label 1 --scan-period 1 --camera SR305```

#### Real-time gesture recognition

To start the gesture recognition system using gesture database stored in the `gestures` directory:  
`python3 src/system/gesture_recognizer.py --source live --dir gestures --error-threshold 120 --orientation-threshold 60 --camera SR305`

To start the gesture recognition from the evaluation dataset:  
`python3 src/system/gesture_recognizer.py --source dataset --dir gestures --error-threshold 120 --orientation-threshold 60 --camera SR305`

**For demonstration**, a directory named "test" is already present,
containing definitions for a gesture with opened palm with fingers outstretched
and apart.
`python3 src/system/gesture_recognizer.py --source live --dir test --error-threshold 120 --orientation-threshold 60 --camera SR305`

The system plots figures similar to the following:  
<img src="./docs/readme/live_gesture1.png" alt="live_gesture1" width="220"/>  
<img src="./docs/readme/live_nongesture.png" alt="live_nongesture" width="220"/>  

### Hand detection

To detect hands from images captured with SR305 camera, which is the default camera:
`python3 src/system/hand_position_estimator.py --detect --source live`

<img src="./docs/readme/live_detection.png" alt="live_detection" width="220"/>  
<img src="./docs/readme/live_detection2.png" alt="live_detection" width="220"/>  

### Hand pose estimation

To estimate hand poses from images captured with SR305 camera:
`python3 src/system/hand_position_estimator.py --estimate --source live`
  
<img src="./docs/readme/live_estimation.png" alt="live_estimation" width="220"/>  
<img src="./docs/readme/live_estimation2.png" alt="live_estimation" width="220"/>  

### Training of models

To train the JGR-P2O on the Bighand dataset:  
```python3 src/estimation/train.py --train bighand```

To train the JGR-P2O model on the Bighand dataset from existing weights:  
`python3 src/estimation/train.py --train bighand --model logs/20210426-125059/train_ckpts/weights.25.h5`

The number of features is set to 196 by default. Set the `--features` flag if required otherwise.

### Evaluation of models

To evaluate the trained JGR-P2O model on the MSRA dataset:  
`python3 src/estimation/evaluation.py --dataset msra --model logs/20210421-221853/train_ckpts/weights.22.h5 --features 128`


## Project structure

### Top-level structure

    .
    ├── datasets                # Datasets (including gesture database)
    ├── docs                    # Documentation files 
    ├── src                     # Source files
    ├── LICENSE                 # MIT license
    ├── README.md               # Contents of this file
    ├── requirements.txt        # Package requirements 
    └── upload_ibt_src.sh       # Script for uploading source files to metacentrum

### Datasets

    datasets
    ├── bighand                     # Hand pose estimation dataset (not preprocessed)
    ├── cvpr15_MSRAHandGestureDB    # Hand pose estimatino dataset (is preprocesed)
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
    ├── metacentrum              # Scripts for training in Metacentrum
    ├── system                   # Access point to gesture recognition system 
    │                              (database_scanner, gesture_recognizer, hand_position_estimator)
    └── utils                    # Camera, logs, plots, live capture, config


## License

This project is licensed under the terms of the MIT license.