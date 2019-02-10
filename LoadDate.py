from logging import getLogger

import numpy as np
import pandas as pd

logger = getLogger(__name__)

TRAIN_DATA = 'data/train.csv'
TEST_DATA = 'data/test.csv'


def readCsv(path):
    logger.info("enter")
    df = pd.read_csv(path)

    # カテゴリ変数をダミー化
    # カラム名を取得し、「cat」の文字列が含まれるものは対象とする
    for col in df.columns.values:
        if 'cat' in col:
            logger.info('categorical:{}'.format(col))
            # Pandasのget_dumies でエンコード
            tmp = pd.get_dummies(df[col], col)
            # 生成した列と既存の列を入れ替える（既存列はInplaceで削除）
            for col2 in tmp.colums.values:
                df[col2] = tmp[col2].values
            df.drop(col, axis=1, inplace=True)

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
