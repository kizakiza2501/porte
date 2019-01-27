from logging import getLogger

import numpy as np
import pandas as pd

logger = getLogger(__name__)

TRAIN_DATA = 'data/train.csv'
TEST_DATA = 'data/test.csv'


def readCsv(path):
    logger.info("enter")
    df = pd.read_csv(path)
    logger.info("exit")
    return df


def loadTrainData():
    logger.info("enter")
    df = pd.read_csv(TRAIN_DATA)
    logger.info("exit")
    return df


def loadTestData():
    logger.info("enter")
    df = pd.read_csv(TEST_DATA)
    logger.info("exit")
    return df


if __name__ == '__name__':
    print(loadTrainData.head())
    print(loadTestData.head())
