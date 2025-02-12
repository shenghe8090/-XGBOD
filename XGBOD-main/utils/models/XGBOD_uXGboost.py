import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_array
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_is_fitted
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.abod import ABOD
from pyod.models.kde import KDE
from xgboost.sklearn import XGBClassifier
from pyod.models.base import BaseDetector
from pyod.utils.utility import check_parameter
from pyod.utils.utility import check_detector
from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores


class XGBOD_uXGBoost(BaseDetector):
    """
    XGBOD_uXGBoost is an anomaly detection model that does not combine any unsupervised outlier detection
    methods  with an XGBoost classifier for final classification.
    """

    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, objective="binary:logistic", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                 colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0,
                 **kwargs):
        """
        Initializes the XGBOD_uXGBoost model with hyperparameters for the XGBoost classifier.

        Args:
            max_depth: Maximum depth of the decision tree.
            learning_rate: Step size shrinking to prevent overfitting.
            n_estimators: Number of boosting rounds.
            objective: Loss function used in training.
            booster: Type of boosting model.
            n_jobs: Number of parallel jobs to run.
            nthread: Number of threads to use for training (deprecated, use n_jobs).
            gamma: Regularization parameter.
            min_child_weight: Minimum sum of instance weight (hessian) needed in a child.
            max_delta_step: Step size used to prevent overfitting.
            subsample: Fraction of samples used for each tree.
            colsample_bytree: Fraction of features used for each tree.
            colsample_bylevel: Fraction of features used for each level.
            reg_alpha: L1 regularization term on weights.
            reg_lambda: L2 regularization term on weights.
            scale_pos_weight: Controls the balance of positive and negative weights.
            base_score: The starting score for each instance.
            random_state: Seed for random number generation.
        """
        super(XGBOD_uXGBoost, self).__init__()  # Inherit from BaseDetector
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

    def identity_mapping(self, ):
        """
        Returns the identity mapping for detectors (no modification).
        """
        return self

    def _init_detectors(self, X):
        """
        Initializes the unsupervised anomaly detection methods (detectors) and
        whether standardization is needed for each detector.

        Args:
            X: The input feature matrix.

        Returns:
            detectors: List of initialized detectors.
            standardization_flags: Flags indicating if standardization is required for each detector.
        """
        id = self.identity_mapping()  # Identity mapping detector for no transformation
        detectors = [id]  # List of detectors, just using identity mapping in this case
        standardization_flags = [False]  # No standardization required for identity mapping

        return detectors, standardization_flags

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Args:
            X: Feature matrix.
            y: Labels (ground truth).

        Returns:
            self: The fitted model instance.
        """
        X, y = check_X_y(X, y)  # Check the validity of the input arrays
        X = check_array(X)  # Ensure the feature matrix is valid
        self._set_n_classes(y)  # Set the number of classes based on labels
        self.detectors, self.standardization_flags = self._init_detectors(X)  # Initialize detectors
        self.n_detectors = len(self.detectors)  # Number of detectors
        self.X_train_add = np.zeros([X.shape[0], self.n_detectors])  # Create empty array for additional features

        # Standardize the input features
        X_norm, self.scaler = standardizer(X, keep_scalar=True)

        # Fit each detector and calculate decision scores
        for i, detector in enumerate(self.detectors):
            if self.standardization_flags[i]:
                detector.fit(X_norm)  # Fit with standardized data
                self.X_train_add[:, i] = detector.decision_scores_  # Store decision scores
            else:
                detector.fit(X)  # Fit with original data
                self.X_train_add[:, i] = detector.decision_scores_  # Store decision scores

        # Combine the original features with the anomaly detection scores
        self.X_train_new = np.concatenate((X, self.X_train_add), axis=1)

        # Train an XGBoost classifier using the combined feature matrix
        self.clf = XGBClassifier(
            max_depth=self.max_depth, learning_rate=self.learning_rate, n_estimators=self.n_estimators,
            objective=self.objective, booster=self.booster, n_jobs=self.n_jobs, nthread=self.nthread,
            gamma=self.gamma, min_child_weight=self.min_child_weight, max_delta_step=self.max_delta_step,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree, colsample_bylevel=self.colsample_bylevel,
            reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda, scale_pos_weight=self.scale_pos_weight,
            base_score=self.base_score, random_state=self.random_state, **self.kwargs
        )

        # Fit the classifier and store the decision scores and predictions
        self.clf.fit(self.X_train_new, y)
        self.decision_scores_ = self.clf.predict_proba(self.X_train_new)[:, 1]
        self.labels_ = self.clf.predict(self.X_train_new).ravel()

        return self

    def decision_function(self, X):
        """
        Predict the anomaly scores for the given input data X.

        Args:
            X: Feature matrix to predict on.

        Returns:
            pred_scores: Anomaly scores for the input data.
        """
        check_is_fitted(self, ['clf', 'decision_scores_', 'labels', 'scaler'])  # Ensure model is fitted

        X = check_array(X)  # Validate input data
        X_add = np.zeros([X.shape[0], self.n_detectors])  # Initialize additional feature array
        X_norm = self.scaler.transform(X)  # Normalize the input features

        # Compute decision scores for each detector
        for i, detector in enumerate(self.detectors):
            if self.standardization_flags[i]:
                X_add[:, i] = detector.decision_function(X_norm)  # Standardized data
            else:
                X_add[:, i] = detector.decision_function(X)  # Original data

        # Concatenate the original features with the decision scores
        X_new = np.concatenate((X, X_add), axis=1)

        # Predict probabilities using the trained XGBoost classifier
        pred_scores = self.clf.predict_proba(X_new)[:, 1]
        return pred_scores.ravel()

    def predict(self, X):
        """
        Predict the anomaly labels for the given input data X.

        Args:
            X: Feature matrix to predict on.

        Returns:
            pred_labels: Predicted labels for the input data (1 for anomaly, 0 for normal).
        """
        check_is_fitted(self, ['clf', 'decision_scores_', 'labels', 'scaler'])  # Ensure model is fitted

        X = check_array(X)  # Validate input data
        X_add = np.zeros([X.shape[0], self.n_detectors])  # Initialize additional feature array
        X_norm = self.scaler.transform(X)  # Normalize the input features

        # Compute decision scores for each detector
        for i, detector in enumerate(self.detectors):
            if self.standardization_flags[i]:
                X_add[:, i] = detector.decision_function(X_norm)  # Standardized data
            else:
                X_add[:, i] = detector.decision_function(X)  # Original data

        # Concatenate the original features with the decision scores
        X_new = np.concatenate((X, X_add), axis=1)

        # Predict labels using the trained XGBoost classifier
        pred_labels = self.clf.predict(X_new)
        return pred_labels.ravel()

    def predict_proba(self, X):
        """
        Predict the anomaly scores (probabilities) for the given input data X.

        Args:
            X: Feature matrix to predict on.

        Returns:
            pred_scores: Predicted probabilities for the input data.
        """
        return self.decision_function(X)

    def fit_predict(self, X, y):
        """
        Fit the model and predict the labels for the input data.

        Args:
            X: Feature matrix.
            y: Labels (ground truth).

        Returns:
            labels: Predicted labels for the input data.
        """
        self.fit(X, y)
        return self.labels_

    def fit_predict_score(self, X, y, scoring='roc_auc_score'):
        """
        Fit the model and compute a specific scoring metric on the predictions.

        Args:
            X: Feature matrix.
            y: Labels (ground truth).
            scoring: The scoring metric to compute, e.g., 'roc_auc_score' or 'prc_n_score'.

        Returns:
            score: The computed score for the specified metric.
        """
        self.fit(X, y)  # Fit the model

        # Compute the score based on the specified scoring metric
        if scoring == 'roc_auc_score':
            score = roc_auc_score(y, self.decision_scores_)  # ROC AUC score
        elif scoring == 'prc_n_score':
            score = precision_n_scores(y, self.decision_scores_)  # Precision @ n score
        else:
            raise NotImplementedError('PyOD built-in scoring only supports ROC and Precision @ rank n')

        print("{metric}: {score}".format(metric=scoring, score=score))  # Print the score
        return score
