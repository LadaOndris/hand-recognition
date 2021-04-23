#!/bin/bash
#PBS -N JGRP2O-MSRA
#PBS -q gpu
#PBS -l select=1:ncpus=32:ngpus=1:mem=42gb:cpu_flag=avx512dq:scratch_ssd=40gb
#PBS -l walltime=24:00:00
#PBS -m abe

DATADIR=/storage/brno6/home/ladislav_ondris/IBT
SCRATCHDIR="$SCRATCHDIR/IBT"
mkdir $SCRATCHDIR

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add conda-modules-py37
conda env remove -n ibt_msra
conda create -n ibt_msra python=3.7
conda activate ibt_msra
conda install matplotlib
conda install tensorflow
conda install scikit-learn
conda install scikit-image
pip install opencv-python
pip install gast==0.3.3
pip install tensorflow-addons
conda list

cp -r "$DATADIR/src" "$SCRATCHDIR/" || { echo >&2 "Couldnt copy srcdir to scratchdir."; exit 2; }

# Prepare directory
mkdir "$SCRATCHDIR/datasets"
# Copy cvpr15_MSRAHandGestureDB.tar
cp -r "$DATADIR/datasets/cvpr15_MSRAHandGestureDB.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy cvpr15_MSRAHandGestureDB.tar"; exit 2; }
# Extract it
tar -xf "$SCRATCHDIR/datasets/cvpr15_MSRAHandGestureDB.tar" -C "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt extract cvpr15_MSRAHandGestureDB.tar"; exit 2; }

# Load pretrained models
cp -r "$DATADIR/saved_models" "$SCRATCHDIR/" || { echo >&2 "Couldnt copy saved models"; exit 2; }

export PYTHONPATH=$SCRATCHDIR
python3 $SCRATCHDIR/src/estimation/train.py --train msra --evaluate msra --model "$SCRATCHDIR/saved_models/20210407-172551/weights.13.h5"

cp -r $SCRATCHDIR/logs $DATADIR/ || { echo >&2 "Couldnt copy logs to datadir."; exit 3; }
cp -r $SCRATCHDIR/saved_models $DATADIR/ || { echo >&2 "Couldnt copy saved_models to datadir."; exit 3; }
clean_scratch
