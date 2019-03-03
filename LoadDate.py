from logging import getLogger

import numpy as np
import pandas as pd

import lightgbm as lgb

logger = getLogger(__name__)

TRAIN_DATA = 'data/train.csv'
TEST_DATA = 'data/test.csv'


def readCsv(path):
    logger.info("enter")
    df = pd.read_csv(path)

    # 重要度の高い項目を残す
    df = df[['ps_car_13', 'ps_ind_05_cat', 'ps_ind_17_bin', 'ps_reg_03', 'ps_car_07_cat']]

    # -1が存在する行を削除
    df = df.replace(-1, np.nan)
    df = df.dropna(how='any')

    df['ps_car_13'] = np.log(df['ps_car_13'])
    df['ps_reg_03'] = np.log(df['ps_reg_03'])

    for col in df.columns.values:
        # カテゴリ変数をダミー化
        # カラム名を取得し、「cat」の文字列が含まれるものは対象とする
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
