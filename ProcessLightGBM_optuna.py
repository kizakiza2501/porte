from tqdm import tqdm
import pandas as pd
import numpy as np
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
logger = getLogger(__name__)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score
import lightgbm as lgb
import optuna

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


def objective(trial):
    '''
    Optunaを使用してパラメータチューニング
    '''
    # クロスバリデーション
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    drop_rate = trial.suggest_uniform('drop_rate', 0, 1.0)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0, 1.0)
    learning_rate = trial.suggest_uniform('learning_rate', 0, 1.0)
    subsample = trial.suggest_uniform('subsample', 0.7, 1.0)
    num_leaves = trial.suggest_int('num_leaves', 5, 100)
    verbosity = trial.suggest_int('verbosity', -1, 1)

#   num_boost_round = trial.suggest_int('num_boost_round', 10, 100000)
#   min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 10, 1000)
#   min_child_samples = trial.suggest_int('min_child_samples', 5, 500)
#   min_child_weight = trial.suggest_int('min_child_weight', 5, 500)

    num_boost_round = trial.suggest_int('num_boost_round', 10, 100)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 10, 100)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 100)
    min_child_weight = trial.suggest_int('min_child_weight', 5, 100)

    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.65, 0.66)
    reg_alpha = trial.suggest_uniform('reg_alpha', 1, 1.2)
    reg_lambda = trial.suggest_uniform('reg_lambda', 1, 1.4)

    params = {"objective": "binary",
              "boosting_type": "gbdt",
              "learning_rate": learning_rate,
              "num_leaves": num_leaves,
              "max_bin": 256,
              "feature_fraction": feature_fraction,
              "verbosity": verbosity,
              "drop_rate": drop_rate,
              "is_unbalance": False,
              "max_drop": 50,
              "min_child_samples": min_child_samples,
              "min_child_weight": min_child_weight,
              "min_split_gain": 0,
              "num_boost_round": num_boost_round,
              "min_data_in_leaf": min_data_in_leaf,
              "subsample": subsample,
              "n_estimators": 40,
              "random_state": 501,
              "colsample_bytree": colsample_bytree,
              "reg_alpha": reg_alpha,
              "reg_lambda": reg_lambda
              }

    listGiniScore = []

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
        clf.fit(trn_x, trn_y, verbose=False)

        # 検証用データで予測する
        pred = clf.predict_proba(val_x)[:, 1]
        scGini = eval_gini(val_y, pred)
        listGiniScore.append(scGini)

    return(1 - np.mean(listGiniScore))


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

        logger.info('Prameter tuning with optuna start.')
        study = optuna.create_study()
        study.optimize(objective, n_trials=100)
        logger.info('Prameter tuning with optuna end. Best prameter are {}'.format(study.best_params))

        logger.info('model training start')
        clf = lgb.LGBMClassifier(**study.best_params)
        clf.fit(xTrain, yTrain, verbose=False)
        logger.info('model training end')

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
