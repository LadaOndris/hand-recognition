import argparse

from src.detection.yolov3.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, action='store', default=32,
                        help='Number of samples in a batch')
    parser.add_argument('--learning-rate', type=float, action='store', default=0.01,
                        help='Learning rate')
    parser.add_argument('--train-size', type=float, action='store', default=0.8,
                        help='The proportion of dataset to be used for training')
    args = parser.parse_args()
    train(args.batch_size, args.learning_rate, args.train_size)
