import numpy as np
from sklearn import metrics


class Scorer:
    def __init__(self, score_metric=None):
        """
        init
        :param score_metric: a string indicates the score metric to be used
        """
        self.score_metric = score_metric

    def make_skl_scorer(self, weights=None):
        """
        create sklearn scorer
        :param weights: weights: None or list of weights.
        :return sklearn scorer
        """
        if self.score_metric == 'accuracy':
            skl_scorer = metrics.make_scorer(metrics.accuracy_score)
        elif self.score_metric == 'f1_macro':
            skl_scorer = metrics.make_scorer(self.f1_score, weights=weights)
        elif self.score_metric == 'recall_macro':
            skl_scorer = metrics.make_scorer(self.recall_score, weights=weights)
        else:
            return ValueError("unsupported metrics")
        return skl_scorer

    def evaluate_score(self, y_true, y_pred, weights):
        """
        evaluate score
        :param y_true: array (shape=[n_samples,_], float). True values.
        :param y_pred: array (shape=[n_samples,_], float). Predicted values.
        :param weights: None or list of weights.
        :return: a single score value
        """
        if self.score_metric == 'accuracy':
            score = metrics.accuracy_score(y_true, y_pred)
        elif self.score_metric == 'f1_macro':
            score = self.f1_score(y_true, y_pred, weights)
        elif self.score_metric == 'recall_macro':
            score = self.recall_score(y_true, y_pred, weights)
        else:
            return ValueError("unsupported metrics")
        return score

    def scores_to_dict(self, y_true, y_pred, weights=None, labels=None):
        """
        evaluate all scores.
        :param y_true: array (shape=[n_samples,_], float). True values.
        :param y_pred: array (shape=[n_samples,_], float). Predicted values.
        :param weights: None or list of weights.
        :param labels: array (shape=[n_classes,_], optional). Lisy of labels to index the confusion matrix. This might
         be used to reorder or select subset of labels.
        :return: dict with score name (keys) and score value.
        """
        # init
        n_samples = np.size(y_true)
        unq_y = np.unique(y_true)

        scores_dict = dict()
        scores_dict['confusion_matrix'] = dict()
        scores_dict['confusion_matrix']['matrix'] = []
        scores_dict['confusion_matrix']['columns'] = []
        scores_dict['scores'] = dict()
        scores_dict['scores']['accuracy'] = 0
        if unq_y.dtype.type != np.str_:
            scores_dict['scores']['mae'] = 0
        scores_dict['scores']['recall_macro'] = 0
        scores_dict['scores']['precision_macro'] = 0
        scores_dict['scores']['f1_score'] = 0

        if n_samples > 0:
            # compute confusion matrix
            cnf_matrix, cnf_labels = self.get_confusion_matrix(y_true, y_pred, labels=labels)
            scores_dict['confusion_matrix']['matrix'] = cnf_matrix.tolist()
            scores_dict['confusion_matrix']['columns'] = cnf_labels.tolist()

            # compute scores
            scores_dict['scores']['accuracy'] = metrics.accuracy_score(y_true, y_pred)
            scores_dict['scores']['recall_macro'] = self.recall_score(y_true, y_pred, weights)
            scores_dict['scores']['precision_macro'] = self.precision_score(y_true, y_pred, weights)
            scores_dict['scores']['f1_score'] = self.f1_score(y_true, y_pred, weights)
            if unq_y.dtype.type != np.str_:
                scores_dict['scores']['mae'] = self.mean_absolute_err_score(y_true, y_pred, weights)

        return scores_dict

    def f1_score(self, y_true, y_pred, weights):
        """
        compute the F1 score per class and return the weighted average (macro average).
        :param y_true: array (shape=[n_samples,_], float). True values.
        :param y_pred: array (shape=[n_samples,_], float). Predicted values.
        :param weights: None or list of weights.
        :return: float, weighted average f1 score
        """
        # compute precision and recall
        recall_per_class = self.get_recall_per_class(y_true, y_pred)
        precision_per_class = self.get_precision_per_class(y_true, y_pred)

        # compute f1-score (macro averaging)
        f1_num = 2*recall_per_class*precision_per_class
        f1_den = recall_per_class + precision_per_class
        f1_per_class = np.divide(f1_num, f1_den, out=np.zeros_like(f1_den), where=f1_den != 0)

        # compute weighted mean
        if weights is None:
            f1_weighted = np.mean(f1_per_class)
        else:
            f1_weighted = np.average(f1_per_class, weights)
        return f1_weighted

    def recall_score(self, y_true, y_pred, weights):
        """
        compute the recall per class (balanced accuracy) and return the weighted average (macro average)
        :param y_true: array (shape=[n_samples,_], float). True values.
        :param y_pred: array (shape=[n_samples,_], float). Predicted values.
        :param weights: None or list of weights.
        :return: float, weighted average recall score
        """
        # compute recall
        recall_per_class = self.get_recall_per_class(y_true, y_pred)

        # compute weighted mean
        if weights is None:
            recall_weighted = np.mean(recall_per_class)
        else:
            recall_weighted = np.average(recall_per_class, weights)
        return recall_weighted

    def precision_score(self, y_true, y_pred, weights):
        """
        compute the precision per class (balanced accuracy) and return the weighted average (macro average)
        :param y_true: array (shape=[n_samples,_], float). True values.
        :param y_pred: array (shape=[n_samples,_], float). Predicted values.
        :param weights: list (float), weight per class
        :return: float, weighted average precision score
        """
        # compute precision
        precision_per_class = self.get_precision_per_class(y_true, y_pred)

        # compute weighted mean
        if weights is None:
            precision_weighted = np.mean(precision_per_class)
        else:
            precision_weighted = np.average(precision_per_class, weights)
        return precision_weighted

    def mean_absolute_err_score(self, y_true, y_pred, weights):
        """
        compute the mean absolute error loss
        :param y_true: array (shape=[n_samples,_], int). True values.
        :param y_pred: array (shape=[n_samples,_], int). Predicted values.
        :param weights: list (float), weight per class
        :return: float, mean absolute error
        """
        if y_true.dtype.type == np.str_:
            raise ValueError("labels must be int or float")

        mae = metrics.mean_absolute_error(y_true, y_pred, sample_weight=weights)
        return mae

    def get_recall_per_class(self, y_true, y_pred):
        """
        compute recall per class: TP/(TP+FN)
        :param y_true: array (shape=[n_samples,_], float). True values.
        :param y_pred: array (shape=[n_samples,_], float). Predicted values.
        :return: array (shape=[n_class,_], float). recall values per class
        """
        cnf_matrix, _ = self.get_confusion_matrix(y_true, y_pred)
        TP, FN, _, _ = self.get_stats_from_cnf_matrix(cnf_matrix)
        recall_per_class = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
        return recall_per_class

    def get_precision_per_class(self, y_true, y_pred):
        """
        compute precision per class: TP/(TP+FP)
        :param y_true: array (shape=[n_samples,_], float). True values.
        :param y_pred: array (shape=[n_samples,_], float). Predicted values.
        :return: array (shape=[n_class,_], float). precision values per class.
        """
        cnf_matrix, _ = self.get_confusion_matrix(y_true, y_pred)
        TP, _, FP, _ = self.get_stats_from_cnf_matrix(cnf_matrix)
        precision_per_class = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
        return precision_per_class

    @staticmethod
    def get_stats_from_cnf_matrix(cnf_matrix):
        """
        get the TP, FN, FP and TN statistics per class from confusion matrix
        :param cnf_matrix: a numpy array with a confusion matrix
        :return: tuple (TP, FN, FP, TN). Each element is array (shape=[n_classes,_], float).
        """
        TP = np.diag(cnf_matrix)
        FP = np.sum(cnf_matrix, axis=0) - TP
        FN = np.sum(cnf_matrix, axis=1) - TP
        TN = np.sum(cnf_matrix) - (FP + FN + TP)
        return TP, FN, FP, TN

    @staticmethod
    def get_confusion_matrix(y_true, y_pred, labels=None):
        """
        compute confusion matrix
        :param y_true: array (shape=[n_samples,_], float). True values.
        :param y_pred: array (shape=[n_samples,_], float). Predicted values.
        :param labels: array (shape=[n_classes,_], optional). Lisy of labels to index the confusion matrix. This might
         be used to reorder or select subset of labels.
        :return: tuple (cnf_matrix, cnf_cols). cnf_matrix is array (shape=[n_class,n_classes], int) with un-normalized
        values. cnf_cols is array (shape=[n_class,_], string) with columns name.
        """
        cnf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=labels)
        if labels is None:
            cnf_cols = np.unique(np.append(y_true, y_pred))
        else:
            cnf_cols = labels
        return cnf_matrix, cnf_cols

