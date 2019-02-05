from tqdm import tqdm
import pandas as pd
import numpy as np
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
logger = getLogger(__name__)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # パラメータサーチ
    # すべてのパラメータを記載して、GridprameterSearchにかける
    allParam = {
                'C': [10**i for i in range(-1, 2)],
                'fit_intercept': [True, False],
                'penalty': ['l2', 'l1'],
                'random_state': [0]
               }

    minScore = 100
    minParams = None

    for params in tqdm(list(ParameterGrid(allParam))):
        logger.info('params: {}'.format(params))

        listAucScore = []
        listLoglossScore = []
        # トレーニングデータと検証用データに分割する
        for trainIdx, validIdx in cv.split(xTrain, yTrain):
            # インデックスが返ってくるため、ilocで行を特定する
            # トレーニング用
            trn_x = xTrain.iloc[trainIdx, :]
            # 検証用
            val_x = xTrain.iloc[validIdx, :]

            # トレーニング用
            trn_y = yTrain[trainIdx]
            # 検証用
            val_y = yTrain[validIdx]

            # **変数名で、キーワードargsとして渡せる
            clf = LogisticRegression(**params)
            # トレーニングデータでモデル作成
            clf.fit(trn_x, trn_y)

            # 検証用データで予測する
            pred = clf.predict_proba(val_x)[:, 1]
            # 検証用データのYと突き合わせ
            scLogloss = log_loss(val_y, pred)
            scAuc = roc_auc_score(val_y, pred)

            listAucScore.append(scAuc)
            listLoglossScore.append(scLogloss)
            logger.info('logloss: {}, auc: {}'.format(scLogloss, scAuc))

        # 各検証結果の平均値を算出
        # auc を小さいほうに合わせて、マイナス符号をつける
        scLogloss = np.mean(listLoglossScore)
        scAuc = - np.mean(listAucScore)
        logger.info('logloss: {}, auc: {}'.format(np.mean(listLoglossScore), np.mean(listAucScore)))

        # aucが小さければ、パラメータを更新する
        if minScore > scAuc:
            minScore = scAuc
            minParams = params

    logger.info('minimum params: {}'.format(minParams))
    logger.info('minimum auc: {}'.format(minScore))

    # モデル作成・予測
    logger.info('data training start.')
    clf = LogisticRegression(**minParams)
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
