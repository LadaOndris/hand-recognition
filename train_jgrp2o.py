import argparse
import glob
import os

import tensorflow as tf

import src.estimation.configuration as configs
from src.estimation.evaluation import evaluate
from src.estimation.train import train


def get_configs(dataset_name: str):
    if dataset_name == 'bighand':
        train_conf = configs.TrainBighandConfig()
        test_conf = configs.TestBighandConfig()
    elif dataset_name == 'msra':
        train_conf = configs.TrainMsraConfig()
        test_conf = configs.TestMsraConfig()
    else:
        raise ValueError(F"Invalid dataset: {dataset_name}")
    return train_conf, test_conf


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, action='store',
                    help='The dataset to be used for training (allowed options: msra, bighand)')
parser.add_argument('--evaluate', type=str, action='store', default=None,
                    help='The dataset to be used for evaluation (allowed options: msra, bighand)')
parser.add_argument('--model', type=str, action='store', default=None,
                    help='The weights to load the model from (default: none)')
parser.add_argument('--features', type=int, action='store', default=196,
                    help='The number of features (channels) throughout the network (default: 196)')
parser.add_argument('--batch-size', type=int, action='store', default=64,
                    help='The number of samples in a batch')
parser.add_argument('--learning-rate', type=float, action='store', default=0.0001,
                    help='Learning rate')
parser.add_argument('--learning-decay-rate', type=float, action='store', default=0.93,
                    help='A decay of learning rate after each epoch')
parser.add_argument('--ignore-otsus-threshold', type=float, action='store', default=0.01,
                    help='A theshold for ignoring Otsus thresholding method')
args = parser.parse_args()

train_cfg, test_cfg = get_configs(args.dataset)
train_cfg.learning_rate = args.learning_rate
train_cfg.learning_decay_rate = args.learning_decay_rate
train_cfg.batch_size = args.batch_size
test_cfg.batch_size = args.batch_size
train_cfg.ignore_threshold_otsus = args.ignore_otsus_threshold
test_cfg.ignore_threshold_otsus = args.ignore_otsus_threshold

log_dir, model_filepath = train(args.dataset, args.model, train_cfg, model_features=args.features)

if args.evaluate is not None:
    if model_filepath is not None and os.path.isfile(model_filepath):
        path = model_filepath
    else:
        ckpts_pattern = os.path.join(str(log_dir), 'train_ckpts/*')
        ckpts = glob.glob(ckpts_pattern)
        path = max(ckpts, key=os.path.getctime)
    if path is not None:
        thresholds, mje = evaluate(args.evaluate, path, args.features)
        tf.print("MJE:", mje)
    else:
        raise ValueError("No checkpoints available")
