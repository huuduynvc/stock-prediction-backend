from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import numpy as np

from utils.utilities import get_data, preprocess, ErrorAPI


def create_route(models: list):
    bp = Blueprint('bp', __name__)

    @bp.route('/predict')
    def predict():
        data = request.args
        # String
        stock = data.get('stock')
        # String
        model = data.get('model')
        # Start date
        start = data.get('start')

        startCheck = datetime.strptime(start, '%Y-%m-%d')
        if startCheck.timestamp() + models[model].n_days*86400 >= datetime.now().timestamp():
            raise ErrorAPI(400, 'ERROR')
        # End date
        end = data.get('end')

        if model not in models:
            raise ErrorAPI(400, 'Model not exist!')

        df = get_data(stock=stock, start=start, end=end)
        scaler, X, _ = preprocess(
            df,
            models[model].features,
            models[model].n_days
        )

        result = models[model].predict(X)
        result = scaler.inverse_transform(result)
        result = result.reshape(-1)

        empty = [None] * (len(df) - len(result) + 1)
        result = np.concatenate((empty, result))

        tomorrow = result[-1]

        df = df.assign(predict=result[:-1])
        df['predict'] = df['predict'].astype(float)
        df = df.dropna()
        df['date'] = df.index
        df.rename(columns={'Close': 'close'}, inplace=True)
        df.rename(columns={'Open': 'open'}, inplace=True)
        df.rename(columns={'Low': 'low'}, inplace=True)
        df.rename(columns={'High': 'high'}, inplace=True)
        df.rename(columns={'Volume': 'volume'}, inplace=True)
        return jsonify({
            'csv': df.to_dict(orient="records"),
            'tomorrow': tomorrow
        })

    return bp
