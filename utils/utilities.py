from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
yf.pdr_override()

class ErrorAPI(Exception):
    def __init__(self, code, message):
        super().__init__()
        self.code = code
        self.message = message

    def detail(self):
        return {
                'error': {
                    'code': self.code,
                    'message': self.message
                }
        }


def get_data(
        stock='mcs',
        start=None,
        end=None
):
    df = pdr.get_data_yahoo(stock, start=start, end=end)
    if len(df) < 1:
        raise ErrorAPI(400, 'Failed to get data!')
    return df


def preprocess(df, features=None, n_days=60):
    if features is None:
        features = ['Close']

    data = df[['Close']]
    if 'poc' in features:
        data = data.assign(poc=data.pct_change())

    data = data.dropna()

    # Transform
    Y = data[n_days:][['Close']].to_numpy()
    data = data[features].to_numpy()

    n = len(data) - n_days + 1
    X = np.empty((n, len(features) * n_days))
    for i in range(n):
        X[i] = data[i:i + n_days].reshape(-1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    Y = scaler.fit_transform(Y)

    return scaler, X, Y
