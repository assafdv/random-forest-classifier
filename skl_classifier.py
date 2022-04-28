import os
import logging
import itertools
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from splitter import StratifiedKFoldSplitter
from scorer import Scorer

# logger
logging.getLogger().setLevel('INFO')


class SklClassifier:
    def __init__(self, name, model, hparams_grid, model_selection_score_metric='f1_macro', random_state=0):
        """
        init
        :param model: sk-learn classification model.
        :param name: a string with the classifier name.
        :param hparams_grid: a dictionary represent hyper-parameter grid for model selection.
        :param model_selection_score_metric: string, model selection score metric
        :param random_state: random state for reproducibility
        """
        self.name = name
        self.random_state = random_state
        self.model = model
        self.hyparams_grid = hparams_grid
        self.model_selection_score_metric = model_selection_score_metric
        self.trained = False

    def train(self, X, y, model_selection_method='KF-CV', **kwargs):
        """
        train model with optimal hyper-parameters
        :param X: array (shape=[n_samples, n_dims], float). Features.
        :param y: array (shape=[n_samples,_], float). Targets.
        :param model_selection_method: str, model selection method.
        """
        # init
        best_params = kwargs.get('best_params', None)

        # model selection
        if best_params is None:
            best_params = self.model_selection(X=X, y=y, model_selection_method=model_selection_method)
        self.model.set_params(**best_params)

        # train
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X):
        """
        predict targets.
        :param X: array (shape=[n_samples, n_dims], float). Features.
        :return: array (shape=[n_samples,_], int). predicted label per data sample
        """
        if not self.trained:
            raise ValueError("Model is not trained.")

        post_prob, cls_labels = self.predict_proba(X)

        # predict using MAP criterion
        y = cls_labels[np.argmax(post_prob, axis=1)]
        return y

    def predict_proba(self, X):
        """
        compute posterior probabilities
        :param X: array (shape=[n_samples, n_dims], float). Features.
        :return: tuple (prob, classes) where prob is array (shape=[n_samples, n_classes]) with probabilities and
        classes is array (shape = [_nclasses, _] with classes labels
        """
        if not self.trained:
            raise ValueError("Model is not trained.")

        cls_labels = self.model.classes_
        post_prob = self.model.predict_proba(X)

        return post_prob, cls_labels

    def model_selection(self, X,
                        y,
                        X_u=None,
                        model_selection_method='KF-CV',
                        n_splits=5,
                        model_selection_score_metric='f1_macro'):
        """
        model selection
        :param X: array (shape=[n_samples, n_dims], float). Features.
        :param y: array (shape=[n_samples,_], float). Targets.
        :param X_u: array (shape=[n_pool_samples,_], float). pool of unlabeled samples (features). (relevant only for
        'CEM' model selection).
        :param model_selection_method: str, model selection method.
        :param n_splits: int, number of CV splits (relevant only for KF-CV model selection).
        :param model_selection_score_metric: a string indicates the score metric for model selection
        (relevant only for KF-CV model selection).
        :return: dict, optimal parameters.
        """
        if model_selection_method == 'KF-CV' or X_u is None:
            best_params, _ = self.model_selection_kfcv(X,
                                                       y,
                                                       n_splits=n_splits,
                                                       model_selection_score_metric=model_selection_score_metric)
        elif model_selection_method == 'CEM':
            best_params, _ = self.model_selection_cem(X, y, X_u)
        else:
            raise ValueError("unsupported model selection method")
        return best_params

    def model_selection_kfcv(self,
                             X,
                             y,
                             n_splits=5,
                             model_selection_score_metric='f1_macro',
                             per_class_score_weights=None):
        """
        model selection using K-folds cross-validation.
        :param X: array (shape=[n_samples, n_dims], float). Features.
        :param y: array (shape=[n_samples,_], float). Targets.
        :param n_splits: number of splits
        :param model_selection_score_metric: a string indicates the score metric for model selection
        :param per_class_score_weights: a numpy array with weights per class for score metric (for model selection)
        :return: tuple (best_param, cv_results). best_params is a dict with the optimal parameters. cv results is a dict
         with cv train\test results.
        """
        logging.info('Running Model Selection: KF-CV')
        # input validation
        if self.model is None:
            raise ValueError("model was not initialized")
        elif np.size(X, axis=0) != np.size(y, axis=0):
            raise ValueError("number of features not equal number of targets")

        # configure CV
        splitter = StratifiedKFoldSplitter(n_splits, random_state=self.random_state)
        scorer = Scorer(model_selection_score_metric)
        grid = GridSearchCV(self.model,
                            self.hyparams_grid,
                            cv=splitter.split(y),
                            scoring=scorer.make_skl_scorer(per_class_score_weights),
                            refit=False,
                            return_train_score=True,
                            error_score=0,
                            iid=False,
                            n_jobs=-1)

        # train
        grid.fit(X, y)

        # get best parameters
        best_params = grid.best_params_

        # cv scores
        cv_results = dict()
        cv_results['scores'] = {}
        cv_results['scores'][model_selection_score_metric] = {}
        cv_results['scores'][model_selection_score_metric]['test'] = \
            {'mean': grid.cv_results_['mean_test_score'][grid.best_index_],
             'std': grid.cv_results_['std_test_score'][grid.best_index_]}
        cv_results['scores'][model_selection_score_metric]['train'] = \
            {'mean': grid.cv_results_['mean_train_score'][grid.best_index_],
             'std': grid.cv_results_['std_train_score'][grid.best_index_]}

        return best_params, cv_results

    def model_selection_cem(self, X_train, y_train, X_u):
        """
        model selection using Classification Entropy Maximization (CEM).
        :param X_train: array (shape=[n_samples, n_dims], float). Features.
        :param y_train: array (shape=[n_samples,_], float). Targets.
        :param X_u: array (shape=[n_pool_samples,_], float). pool of unlabeled samples (features).
        :return: tuple (best_param, cem_results). best_params is a dict with the optimal parameters. cem_results is a
        dict with the CEM results.
        """
        logging.info('Running Model Selection: CEM')
        # input validation
        if self.model is None:
            raise ValueError("model was not initialized")
        elif np.size(X_train, axis=0) != np.size(y_train, axis=0):
            raise ValueError("number of features not equal number of targets")

        # clone model
        model_ = clone(self.model)

        # prepare experiment
        keys, values = zip(*self.hyparams_grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        n_params = len(experiments)

        # loop over experiments
        ce_vals = np.ones(n_params)*np.nan
        for i, param in enumerate(experiments):
            # set params
            model_.set_params(**param)
            # train
            model_.fit(X_train, y_train)
            # predict
            y_u = model_.predict(X_u)
            # compute CEM
            ce_vals[i] = self.compute_ce(y_u)

        # find set parameters for which CE is maximized
        best_index = np.argmax(ce_vals)
        best_ce = ce_vals[best_index]
        best_params = experiments[int(best_index)]

        # store results in dict (for compatibility with other model selection methods)
        ce_results = dict()
        ce_results['scores'] = {}
        ce_results['scores']['CE'] = {}
        ce_results['scores']['CE']['max'] = best_ce
        return best_params, ce_results


    def get_params(self):
        """
        get model params
        :return: dict with parameter name (key) and parameter value (value).
        """
        params = self.model.get_params()
        return params

    def save_model_to_file(self, path):
        """
        save sklearn model in joblib format
        :param path: output path.
        """
        with open(os.path.join(path, '{}_model.joblib'.format(self.name)), "wb") as f:
            joblib.dump(self.model, f)

    def is_trained(self):
        """
        check if model is trained.
        :return: bool, True for tained model. False otherwise.
        """
        return self.trained

    @staticmethod
    def compute_ce(y):
        """
        compute Cross Entropy (CE). see
        :param y: array (shape=[n_samples,_], int). realization of a discrete random variable y.
        :return: float, CE
        """
        unq_y, counts = np.unique(y, return_counts=True)
        n_y = np.size(unq_y)
        p_y = counts/np.sum(counts)
        ce = -np.mean(p_y*np.log(p_y, out=np.zeros(n_y), where=p_y != 0))
        return ce
