import numpy as np
from pyod.models.hbos import HBOS
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_array
from sklearn.utils.validation import check_X_y, check_is_fitted
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.abod import ABOD
from pyod.models.kde import KDE
from xgboost.sklearn import XGBClassifier
from pyod.models.base import BaseDetector
from pyod.utils.utility import check_parameter, check_detector, standardizer, precision_n_scores

class XGBOD_uOCSVM(BaseDetector):
    """
    XGBOD_uOCSVM is an unsupervised outlier detection algorithm that integrates the OCSVM
    anomaly detection model and uses XGBoost for classification.
    """

    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, objective="binary:logistic", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                 colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, **kwargs):
        """
        Initialize XGBOD_uOCSVM with hyperparameters for the XGBoost classifier.
        """
        super(XGBOD_uOCSVM, self).__init__()
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.booster = booster
        self.n_jobs = n_jobs
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.kwargs = kwargs

    def _init_detectors(self, X):
        """
        Initialize the OCSVM anomaly detector.
        """
        ocsvm = OCSVM()
        detectors = [ocsvm]
        standardization_flags = [True]  # Assuming OCSVM benefits from standardization.
        return detectors, standardization_flags

    def fit(self, X, y):
        """
        Fit the XGBOD_uOCSVM model to the training data X and labels y.
        """
        X, y = check_X_y(X, y)
        X = check_array(X)
        self._set_n_classes(y)
        self.detectors, self.standardization_flags = self._init_detectors(X)
        self.n_detectors = len(self.detectors)
        self.X_train_add = np.zeros([X.shape[0], self.n_detectors])

        X_norm, self.scaler = standardizer(X, keep_scalar=True)

        for i, detector in enumerate(self.detectors):
            if self.standardization_flags[i]:
                detector.fit(X_norm)
                self.X_train_add[:, i] = detector.decision_scores_
            else:
                detector.fit(X)
                self.X_train_add[:, i] = detector.decision_scores_

        self.X_train_new = np.concatenate((X, self.X_train_add), axis=1)

        self.clf = XGBClassifier(max_depth=self.max_depth, learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                                 objective=self.objective, booster=self.booster, n_jobs=self.n_jobs,
                                 nthread=self.nthread, gamma=self.gamma, min_child_weight=self.min_child_weight,
                                 max_delta_step=self.max_delta_step, subsample=self.subsample,
                                 colsample_bytree=self.colsample_bytree, colsample_bylevel=self.colsample_bylevel,
                                 reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
                                 scale_pos_weight=self.scale_pos_weight, base_score=self.base_score,
                                 random_state=self.random_state, **self.kwargs)

        self.clf.fit(self.X_train_new, y)
        self.decision_scores_ = self.clf.predict_proba(self.X_train_new)[:, 1]
        self.labels_ = self.clf.predict(self.X_train_new).ravel()

        return self

    def _get_detector_scores(self, X):
        """
        Get the decision score from the OCSVM detector.
        """
        X_add = np.zeros([X.shape[0], self.n_detectors])
        X_norm = self.scaler.transform(X)

        for i, detector in enumerate(self.detectors):
            if self.standardization_flags[i]:
                X_add[:, i] = detector.decision_function(X_norm)
            else:
                X_add[:, i] = detector.decision_function(X)

        return X_add

    def decision_function(self, X):
        """
        Compute decision scores for the input data X.
        """
        check_is_fitted(self, ['clf', 'decision_scores_', 'labels_', 'scaler'])
        X = check_array(X)
        X_add = self._get_detector_scores(X)
        X_new = np.concatenate((X, X_add), axis=1)
        pred_scores = self.clf.predict_proba(X_new)[:, 1]
        return pred_scores.ravel()

    def predict(self, X):
        """
        Predict labels for the input data X.
        """
        check_is_fitted(self, ['clf', 'decision_scores_', 'labels_', 'scaler'])
        X = check_array(X)
        X_add = self._get_detector_scores(X)
        X_new = np.concatenate((X, X_add), axis=1)
        pred_labels = self.clf.predict(X_new)
        return pred_labels.ravel()

    def predict_proba(self, X):
        """
        Return the decision scores as probabilities.
        """
        return self.decision_function(X)

    def fit_predict(self, X, y):
        """
        Fit the model and return predicted labels for the input data X.
        """
        self.fit(X, y)
        return self.labels_

    def fit_predict_score(self, X, y, scoring='roc_auc_score'):
        """
        Fit the model and return predicted labels along with a performance score (ROC AUC or Precision @ rank n).
        """
        self.fit(X, y)

        if scoring == 'roc_auc_score':
            score = roc_auc_score(y, self.decision_scores_)
        elif scoring == 'prc_n_score':
            score = precision_n_scores(y, self.decision_scores_)
        else:
            raise NotImplementedError('Only ROC and Precision @ rank n are supported.')

        print("{metric}: {score}".format(metric=scoring, score=score))
        return score
