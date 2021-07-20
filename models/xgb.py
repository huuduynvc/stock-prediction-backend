import xgboost as xgb
import os
import numpy as np


class Xgb:
    def __init__(self, path, features, n_days=60):
        if not features:
            features = ['Close']

        self.path = path
        self.features = features
        self.n_days = n_days

        self.model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None,
            objective='reg:squarederror'
        )
        if os.path.isfile(path):
            self.model.load_model(path)

    def fit(self, X: np.ndarray, Y):
        X = X.reshape((X.shape[0], X.shape[1]))
        self.model.fit(X, Y)
        self.model.save_model(self.path)

    def predict(self, X: np.ndarray):
        X = X.reshape((X.shape[0], X.shape[1]))
        return self.model.predict(X).reshape(-1, 1)
