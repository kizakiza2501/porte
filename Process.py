import pandas as pd
import numpy as np
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
logger = getLogger(__name__)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

from LoadDate import loadTrainData, loadTestData

LogDir = 'logs/'
SubmitDir = 'data/'
SampleSubmitFile = SubmitDir + 'sample_submission.csv'


if __name__ == '__main__':

    # ロガーの設定
    logFmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

    # 標準出力
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(logFmt)
    logger.addHandler(handler)

    # ファイル出力
    handler = FileHandler(LogDir + 'log.txt', 'a')
    handler.setLevel('DEBUG')
    handler.setFormatter(logFmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('Traning data preparation start.')
    df = loadTrainData()

    # トレーニングデータ読み込み
    # （過学習を防ぐため）target列は削除しておく
    xTrain = df.drop('target', axis=1)
    yTrain = df['target'].values

    # 特徴列の並びが変わってしまうことがあるので列名を記録しておく
    useCols = xTrain.columns.values
    logger.info('train columns: {} {}'.format(useCols.shape, useCols))
    logger.info('Traning data preparation end.')

    # クロスバリデーション
    listAucScore = []
    listLoglossScore = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for trainIdx, validIdx in cv.split(xTrain, yTrain):
        trn_x = xTrain.iloc[trainIdx, :]
        val_x = xTrain.iloc[validIdx, :]

        trn_y = yTrain[trainIdx]
        val_y = yTrain[validIdx]

        clf = LogisticRegression(random_state=0)
        clf.fit(trn_x, trn_y)

        pred = clf.predict_proba(val_x)[:, 1]
        scLogloss = log_loss(val_y, pred)
        scAuc = roc_auc_score(val_y, pred)

        listAucScore.append(scAuc)
        listLoglossScore.append(scLogloss)
        logger.info('logloss: {}, auc: {}'.format(scLogloss, scAuc))

    logger.info('logloss: {}, auc: {}'.format(np.mean(listLoglossScore), np.mean(listAucScore)))

    # モデル作成・予測
    logger.info('data training start.')
    clf = LogisticRegression(random_state=0)
    clf.fit(xTrain, yTrain)
    logger.info('data training end.')

    # テストデータ読み込み
    df = loadTestData()
    xTest = df[useCols].sort_values('id')
    logger.info('Test data preparation start. {}'.format(xTest.shape))
    # テストデータからのターゲット予測
    # predictメソッドだとラベルが出力される
    # predict_probaメソッドは確率を出力する
    predTest = clf.predict_proba(xTest)
    logger.info('Test data preparation end. {}'.format(xTest.shape))

    logger.info('Submittion data preparation start.')
    # submit用のデータ作成
    dfSubmit = pd.read_csv(SampleSubmitFile).sort_values('id')
    # サンプルのデータにtarget列を追加し、予測結果を格納する
    dfSubmit['target'] = predTest
    dfSubmit.to_csv(SubmitDir + 'Submit.csv', index=False)
    logger.info('Submittion data preparation end.')
