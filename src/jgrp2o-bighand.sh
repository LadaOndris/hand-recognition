#!/bin/bash
#PBS -N JGRP2O-MSRA
#PBS -q gpu
#PBS -l select=1:ncpus=24:ngpus=1:mem=32gb:cpu_flag=avx512dq:scratch_local=40gb
#PBS -l walltime=20:00:00
#PBS -m abe

DATADIR=/storage/brno6/home/ladislav_ondris/IBT
SCRATCHDIR="$SCRATCHDIR/IBT"
mkdir $SCRATCHDIR

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

mkdir "$DATADIR/datasets"
cp -r "$DATADIR/src" "$SCRATCHDIR/src" || { echo >&2 "Couldnt copy srcdir to scratchdir."; exit 2; }
cp -r "$DATADIR/datasets/bighand" "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt copy datasetdir to scratchdir."; exit 2; }


conda env remove -n ibt
conda create -n ibt python=3.7
conda activate ibt
conda install matplotlib
conda install tensorflow
pip install gast=0.3.3

python3 $SCRATCHDIR/src/pose_estimation/train.py --train msra --evaluate msra

cp -r $SCRATCHDIR/logs $DATADIR/logs || { echo >&2 "Couldnt copy logs to datadir."; exit 3; }
cp -r $SCRATCHDIR/saved_models $DATADIR/saved_models || { echo >&2 "Couldnt copy saved_models to datadir."; exit 3; }
clean_scratch
