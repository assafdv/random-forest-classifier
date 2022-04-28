import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skl_classifier import SklClassifier


class RFClassifier(SklClassifier):
    def __init__(self,
                 nb_estimators=200,
                 cv_params_preset='cv_params_default',
                 model_selection_score_metric='f1_score',
                 random_state=0):
        """
        init.
        :param nb_estimators:number of estimators to be used by the classifier
        :param cv_params_preset: string indicates the parameters preset. Can be either 'cv_params_linear_kernel',
         'cv_params_rbf_kernel' or 'cv_params_linear_and_rbf_kernel' (default)
        :param random_state: random state for reproducibility (default value is 0).
        """
        name = 'rf'
        params = {'n_estimators': nb_estimators,
                  'max_depth': 3,
                  'min_samples_leaf': 5,
                  'max_features': 'sqrt',
                  'class_weight': 'balanced_subsample',
                  'random_state': random_state}
        model = RandomForestClassifier(**params)
        super().__init__(model=model,
                         name=name,
                         hparams_grid=self.create_cv_params_grid(cv_params_preset),
                         model_selection_score_metric=model_selection_score_metric,
                         random_state=random_state)

    def get_feature_importances(self):
        """
        return the feature importances (the hight the more important the feature)
        :return: array, shape = [n_features]
        """
        return self.model.feature_importances_

    def get_params(self):
        """
        returns a dict with (important) model params.
        """
        # get model params
        params = self.model.get_params()

        # to dict
        d = dict()
        d.setdefault('model_name', self.name)
        d.setdefault('n_estimators', params['n_estimators'])
        d.setdefault('max_depth', params['max_depth'])
        d.setdefault('min_samples_leaf', params['min_samples_leaf'])
        d.setdefault('max_features', params['max_features'])
        d.setdefault('class_weight', params['class_weight'])
        return d

    @staticmethod
    def create_cv_params_grid(preset):
        """
        create grid of parameters for cross-validation according to given preset.
        :param preset: a string indicates the parameters preset. Can be either 'cv_params_linear_kernel',
         'cv_params_rbf_kernel' or 'cv_params_linear_and_rbf_kernel'
        :return: a list of dictionaries with all valid combinations
        """
        if preset == 'cv_params_default':
            # Maximum number of levels in tree
            min_depth = 3
            max_depth = 5
            max_depth_range = [int(x) for x in np.arange(min_depth, max_depth + 1)]

            # Minimum number of samples required to split a node
            # min_samples_split = [5]

            # Minimum number of samples required at each leaf node. A split point at any depth will only be considered
            # if it leaves at least min_samples_leaf training samples in each of the left and right branches
            min_samples_leaf = [5, 7, 10]

            # the number of features to consider when looking for the best split
            max_features = ["sqrt"]

            # create parameters grid
            cv_params_grid = {'max_depth': max_depth_range,
                              # 'min_samples_split': min_samples_split,
                              'min_samples_leaf': min_samples_leaf,
                              'max_features': max_features}
        else:
            raise NotImplementedError

        return cv_params_grid
