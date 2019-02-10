# -*- coding: utf-8 -*-

import xgboost as xgb

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""XGBoost で多値分類するサンプルコード"""


def main():
    # Iris データセットを読み込む
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        random_state=42,
                                                        stratify=y)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    xgb_params = {
        # 多値分類問題
        'objective': 'multi:softmax',
        # クラス数
        'num_class': 3,
        # 学習用の指標 (Multiclass logloss)
        'eval_metric': 'mlogloss',
    }
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    evals_result = {}
    print('Traning start.')
    bst = xgb.train(xgb_params,
                    dtrain,
                    num_boost_round=1000,
                    early_stopping_rounds=10,
                    evals=evals,
                    evals_result=evals_result,
                    )
    print('Traning end.')

    y_pred = bst.predict(dtest)
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)

    train_metric = evals_result['train']['mlogloss']
    plt.plot(train_metric, label='train logloss')
    eval_metric = evals_result['eval']['mlogloss']
    plt.plot(eval_metric, label='eval logloss')
    plt.grid()
    plt.legend()
    plt.xlabel('rounds')
    plt.ylabel('logloss')
    plt.show()


if __name__ == '__main__':
    main()
