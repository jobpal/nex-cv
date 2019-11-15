import numpy as np
from collections import Counter


class TestingDatasetProvider:  # pylint: disable=too-few-public-methods
    """
    Use .get_datasets(X,y) to get a list of tuples with training and texting X, y
    """

    def get_datasets(self, X, y):  # pylint: disable=invalid-name
        """
        Given X, y, provide a list of (X'_train, y'_train, X'_test, y'_test) tuples for testing
        :param X:
        :param y:
        :return: List
        """
        raise NotImplementedError

    def __str__(self):
        """
        May be included in, for example, an evaluation results dump:
        so should be minimal, but useful
        :return:
        """
        return self.__class__.__name__


class NegativeExamplesDatasetProvider(TestingDatasetProvider):  # pylint: disable=too-few-public-methods
    """
    Evaluates performance of the FAQ classifier.
    It acts similarly to the cross validation, but estimates the high-level classifier,
    which includes knowledge of "other" or other-classification.
    As the result, expected classes can't be the ground-truth ones from the model.

    Evaluation includes:
    - Split questions from larger categories into training and test parts
    - Split small categories into training and test parts.
      Expected results for them would be "other".
    """

    def __init__(self,
                 min_category_size=0,
                 other_min_prop=0.15,
                 n_retries=10,
                 test_fraction=0.2):

        self.min_category_size = min_category_size
        self.other_min_prop = other_min_prop

        self.splitter = self._split_labels_by_size if min_category_size > 0 else\
            self._split_labels_by_prop

        self.test_fraction = test_fraction
        self.n_retries = n_retries

    def _split_labels_by_size(self, X, y):  # pylint: disable=invalid-name, unused-argument
        label_counter = Counter(y)
        counts = label_counter.most_common()

        highpop_labels = [
            label for label, count in counts if count >= self.min_category_size
        ]
        lowpop_labels = [
            label for label, count in counts if count < self.min_category_size
        ]

        return lowpop_labels, highpop_labels

    def _split_labels_by_prop(self, X, y):  # pylint: disable=invalid-name
        label_counter = Counter(y)
        counts = label_counter.most_common()
        min_other_size = self.other_min_prop * int(len(X))

        other_pop = 0
        lowpop_labels = []
        highpop_labels = []

        for label, count in reversed(counts):
            if other_pop >= min_other_size:
                highpop_labels.append(label)
            else:
                other_pop += count
                lowpop_labels.append(label)

        return lowpop_labels, highpop_labels

    @staticmethod
    def _random_list_split(l, fraction):
        split_index = int(len(l) * fraction)

        if fraction > 0 and split_index == 0:
            split_index = 1
        np.random.shuffle(l)
        return l[:split_index], l[split_index:]

    def get_datasets(self, X, y):  # pylint: disable=invalid-name

        lowpop_labels, highpop_labels = self.splitter(X, y)

        # split questions from large categories - each category split into test and training

        result = []

        for retry in range(0, self.n_retries):

            X_train, y_train, X_test, y_test = [], [], [], []

            for label in highpop_labels:
                new_test_index, new_train_index = \
                    NegativeExamplesDatasetProvider._random_list_split(
                        [i for i, y_i in enumerate(y) if y_i == label],
                        self.test_fraction
                    )

                X_train += [X[i] for i in new_train_index]
                y_train += [y[i] for i in new_train_index]
                X_test += [X[i] for i in new_test_index]
                y_test += [y[i] for i in new_test_index]

            # split small categories - some caregories are going into the test set and
            # will be all tested with the ground truth == "other"

            test_small_labels, train_small_labels = \
                NegativeExamplesDatasetProvider._random_list_split(
                    lowpop_labels,
                    self.test_fraction
                )

            for i, y_i in enumerate(y):
                if y_i in test_small_labels:
                    X_test += [X[i]]
                    y_test += [None]
                if y_i in train_small_labels:
                    X_train += [X[i]]
                    y_train += [y_i]

            result.append((X_train, y_train, X_test, y_test))

        return result

    def __str__(self):
        return super().__str__() + "(%s;%s)" % (self.min_category_size,
                                                self.other_min_prop)

