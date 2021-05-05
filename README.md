
# Gesture recognition system
Author: Ladislav Ondris

## Prerequisites

## Project structure

## Usage examples

### System's usage

To capture a gesture with label `1` into a `gestures` directory with a scan period of one second and SR305 camera:  
`src/system/database_scanner.py --dir gestures --label 1 --scan-period 1 --camera SR305`

To start the gesture recognition system using gesture database stored in the `gestures` directory:  
`src/system/gesture_recognizer.py --live --dir gestures --error-threshold 120 --orientation-threshold 60 --camera SR305`

To start the gesture recognition from the custom dataset:  
`src/system/gesture_recognizer.py --custom_dataset --dir gestures --error-threshold 120 --orientation-threshold 60 --camera SR305`

### Training of models

To train the JGR-P2O on the Bighand dataset:  
`src/estimation/train.py --train bighand`

To train the JGR-P2O model on the Bighand dataset from existing weights:  
`src/estimation/train.py --train bighand --model logs/20210426-125059/train_ckpts/weights.25.h5`

The number of features is set to 196 by default. Set the `--features` flag if required otherwise.

### Evaluation of models

To evaluate the trained JGR-P2O model on the MSRA dataset:  
`src/estimation/evaluation.py --dataset msra --model logs/20210421-221853/train_ckpts/weights.22.h5 --features 128`


### 