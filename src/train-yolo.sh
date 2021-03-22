#!/bin/bash
#PBS -N Yolov3-Handseg
#PBS -q gpu
#PBS -l select=1:ncpus=24:ngpus=1:mem=24gb:cpu_flag=avx512dq:scratch_local=25gb
#PBS -l walltime=20:00:00
#PBS -m abe

DATADIR=/storage/brno6/home/ladislav_ondris/IBT
SCRATCHDIR="$SCRATCHDIR/IBT"
mkdir $SCRATCHDIR

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

mkdir "$DATADIR/datasets"
cp -r "$DATADIR/src" "$SCRATCHDIR/src" || { echo >&2 "Couldnt copy srcdir to scratchdir."; exit 2; }
cp -r "$DATADIR/datasets/handseg150k" "$SCRATCHDIR/datasets/handseg150k" || { echo >&2 "Couldnt copy datasetdir to scratchdir."; exit 2; }

module add python-3.6.2-gcc
module add python36-modules-gcc
module add opencv-3.4.5-py36
module add tensorflow-2.0.0-gpu-python3

python3 $SCRATCHDIR/src/detection/yolov3/train.py

cp -r $SCRATCHDIR/logs $DATADIR/logs || { echo >&2 "Couldnt copy logs to datadir."; exit 3; }
cp -r $SCRATCHDIR/saved_models $DATADIR/saved_models || { echo >&2 "Couldnt copy saved_models to datadir."; exit 3; }
clean_scratch
