import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut


# logger
logging.getLogger().setLevel('INFO')


class StratifiedKFoldSplitter:
    def __init__(self, n_splits=5, min_class_instances_in_train_set=1, random_state=0):
        """
        init.
        :param n_splits: int, number of test\train splits
        :param min_class_instances_in_train_set: int, minimum number of class instances in the train set.
        :param random_state: int, seed to used by the random number generator.
        """
        self.n_splits = n_splits
        self.min_class_instances_in_train_set = min_class_instances_in_train_set
        self.random_state = random_state

    def split(self, y):
        """
        Provides train/test indices to split data in train/test sets. A variation of stratified-KFold that returns
        stratified folds is used. The folds are made by preserving the percentage of
        samples for each class but ensure at least one instance per class.
        :param y: array-like, shape (n_samples,). The target variable.
        :return: generator of tuples (train_index, test_index)
        """
        # init
        n_samples = np.size(y, axis=0)
        min_class_instances = self.min_class_instances_in_train_set + 1  # +1 for test set
        np.random.seed(self.random_state)

        # get class frequencies
        unq_y, counts_y = np.unique(y, return_counts=True)
        valid_y = unq_y[counts_y >= min_class_instances]
        invalid_y = unq_y[counts_y < min_class_instances]

        # loop over classes
        test_fold = np.zeros([n_samples, self.n_splits])
        test_fold[np.isin(y, invalid_y), :] = -1
        for c in valid_y.tolist():
            # shuffle class indices
            ind_c = np.nonzero(y == c)[0]
            ind_c = np.random.permutation(ind_c)
            n_c = np.size(ind_c)
            n_c_test = int(np.ceil(n_c / self.n_splits))
            # mark test indices for each split (allow duplicates)
            for n in range(self.n_splits):
                test_ind = ind_c[:n_c_test]
                test_fold[test_ind, n] = 1
                ind_c = np.roll(ind_c, -n_c_test)  # shift array

        # generate (train_set, test_set) tuple
        for n in range(self.n_splits):
            train_index = np.nonzero(test_fold[:, n] == 0)[0]
            test_index = np.nonzero(test_fold[:, n] == 1)[0]
            yield (train_index, test_index)
