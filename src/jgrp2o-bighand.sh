#!/bin/bash
#PBS -N JGRP2O-BIGHAND
#PBS -q gpu
#PBS -l select=1:ncpus=32:ngpus=1:mem=42gb:cpu_flag=avx512dq:scratch_ssd=50gb
#PBS -l walltime=24:00:00
#PBS -m abe

DATADIR=/storage/brno6/home/ladislav_ondris/IBT
SCRATCHDIR="$SCRATCHDIR/IBT"
mkdir $SCRATCHDIR

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add conda-modules-py37
conda env remove -n ibt
conda create -n ibt python=3.7
conda activate ibt
conda install matplotlib
conda install tensorflow
conda install scikit-learn
pip install gast==0.3.3
pip install tensorflow-addons
conda list

# Copy source code
cp -r "$DATADIR/src" "$SCRATCHDIR/" || { echo >&2 "Couldnt copy srcdir to scratchdir."; exit 2; }

# Prepare datasets folder
mkdir "$SCRATCHDIR/datasets"
mkdir "$SCRATCHDIR/datasets/bighand"
# Copy full_annotations.tar
cp -r "$DATADIR/datasets/bighand/full_annotation.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy full_annotations.tar"; exit 2; }
# Extract it
tar -xf "$SCRATCHDIR/datasets/full_annotation.tar" -C "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt extract full_annotations.tar"; exit 2; }

# Copy Subject_1.tar
cp -r "$DATADIR/datasets/bighand/Subject_1.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy Subject_1.tar"; exit 2; }
# Extract it
tar -xf "$SCRATCHDIR/datasets/Subject_1.tar" -C "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt extract Subject_1.tar"; exit 2; }

cp -r "$DATADIR/datasets/bighand/Subject_2.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy Subject_2.tar"; exit 2; }
tar -xf "$SCRATCHDIR/datasets/Subject_2.tar" -C "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt extract Subject_2.tar"; exit 2; }

cp -r "$DATADIR/datasets/bighand/Subject_3.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy Subject_3.tar"; exit 2; }
tar -xf "$SCRATCHDIR/datasets/Subject_3.tar" -C "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt extract Subject_3.tar"; exit 2; }



export PYTHONPATH=$SCRATCHDIR
python3 $SCRATCHDIR/src/pose_estimation/train.py --train bighand

cp -r $SCRATCHDIR/logs $DATADIR/ || { echo >&2 "Couldnt copy logs to datadir."; exit 3; }
cp -r $SCRATCHDIR/saved_models $DATADIR/ || { echo >&2 "Couldnt copy saved_models to datadir."; exit 3; }
clean_scratch
