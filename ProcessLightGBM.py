from tqdm import tqdm
import pandas as pd
import numpy as np
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
logger = getLogger(__name__)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score
import lightgbm as lgb

from LoadDate import loadTrainData, loadTestData

LogDir = 'logs/'
SubmitDir = 'data/'
SampleSubmitFile = SubmitDir + 'sample_submission.csv'


def eval_gini(y_true, y_prob):
    '''
    gini係数を計算する関数
    '''
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


if __name__ == '__main__':

    try:
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
            'learning_rate': [0.005],
            'n_estimators': [40],
            'num_leaves': [6, 8, 12, 16],
            'boosting_type': ['gbdt'],
            'objective': ['binary'],
            'random_state': [501],  # Updated from 'seed'
            'colsample_bytree': [0.65, 0.66],
            'subsample': [0.7, 0.75],
            'reg_alpha': [1, 1.2],
            'reg_lambda': [1, 1.2, 1.4],
            }

        minScore = 100
        minParams = None

        for params in list(ParameterGrid(allParam)):
            logger.info('params: {}'.format(params))

            listGiniScore = []
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
                clf = lgb.LGBMClassifier(**params)
                # トレーニングデータでモデル作成
                clf.fit(trn_x, trn_y)

                # 検証用データで予測する
                pred = clf.predict_proba(val_x)[:, 1]
                # 検証用データのYと突き合わせ
                scLogloss = log_loss(val_y, pred)
                # gini を小さいほうに合わせて、マイナス符号をつける
                scGini = - eval_gini(val_y, pred)

                # listAucScore.append(scAuc)
                listGiniScore.append(scGini)
                listLoglossScore.append(scLogloss)
                logger.info('logloss: {}, gini: {}'.format(scLogloss, scGini))

            # 各検証結果の平均値を算出
            scLogloss = np.mean(listLoglossScore)
            scGini = np.mean(listGiniScore)
            logger.info('logloss Average: {}, gini Average: {}'.format(np.mean(listLoglossScore), np.mean(listGiniScore)))

            # giniが小さければ、パラメータを更新する
            if minScore > scGini:
                minScore = scGini
                minParams = params
                logger.info('Update params: {}'.format(minParams))

        logger.info('Eventually minimum params: {}'.format(minParams))
        logger.info('Eventually minimum gini: {}'.format(minScore))

        # モデル作成・予測
        logger.info('data training start.')
        # clf = xgb.sklearn.XGBClassifier(**minParams)
        clf = lgb.LGBMClassifier(**params)
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

    except Exception:
        import traceback
        traceback.print_exc()
