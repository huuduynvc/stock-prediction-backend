import os

from models.lstm import Lstm
from models.rnn import Rnn
from utils.utilities import get_data, preprocess
from models.xgb import Xgb


def load():
    if not os.path.isdir('save_model'):
        os.makedirs('save_model')

    files = os.listdir('save_model')

    models = {
        'lstm_close': Lstm(
            path='save_model/lstm_close.h5',
            features=['Close'],
            epochs=100),
        'lstm_close_poc': Lstm(
            path='save_model/lstm_close_poc.h5',
            features=['Close', 'poc'],
            epochs=10),
        'rnn_close': Rnn(
            path='save_model/rnn_close.h5',
            features=['Close'],
            epochs=64),
        'rnn_close_poc': Rnn(
            path='save_model/rnn_close_poc.h5',
            features=['Close', 'poc'],
            epochs=10),
        'xgb_close': Xgb('save_model/xgb_close.h5', ['Close']),
        'xgb_close_poc': Xgb('save_model/xgb_close_poc.h5', ['Close', 'poc']),

    }

    stocks = ['mcs']

    for stock in stocks:
        df = get_data(stock, start='2011-07-07', end='2021-07-07')
        for model in models:
            _, X, Y = preprocess(df, models[model].features, models[model].n_days)

            if not models[model].path.split('/')[-1] in files:
                models[model].fit(X[:-1], Y)

            models[model].predict(X[:models[model].n_days])

    return models
