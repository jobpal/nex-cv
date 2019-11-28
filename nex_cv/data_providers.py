import numpy as np
from collections import Counter
from datetime import datetime


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

        self.splitter = self._split_labels_by_size if min_category_size > 0 else \
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


class TimeSeqDatasetProvider(TestingDatasetProvider):
    """
    The training data is grown gradually using the creation date for sorting the data
    "created" is expected to be in the "intent" annotation dict
    """
    DATE_KEY = "created"
    INTENT_KEY = "intent_id"

    def __init__(self, min_examples_ratio=0.3, max_added_examples=None, n_steps=5, date_key=DATE_KEY,
                 test_on_all=False, test_ratio=0.33):
        """

        :param min_examples_ratio: (float) ratio of minimum number of training examples
        :param max_added_examples: (int) maximum number of added examples
        :param n_steps: (int) number of steps in the learning curve
        :param date_key: (str) key used for sorting samples
        :param test_on_all: (bool) if False test set is sampled only from the time covered by training set
                                  if True test set is sampled from all data
        :param test_ratio: (float) test data ratio
        """
        self.min_examples_ratio = min(min_examples_ratio, 0.8)
        self.max_added_examples = max_added_examples
        self.n_steps = max(n_steps, 1)

        self.test_on_all = test_on_all
        self.test_step = int(1. / test_ratio)
        self.date_key = date_key

    @staticmethod
    def resolve_date(a, date_key):
        """

        :param a: list(dict) with annotations
        :param date_key: str key used to extract date
        :return: timestamp extracted from date
        """

        try:
            return datetime.timestamp(a.get(date_key))
        except:
            return None

    @staticmethod
    def resolve_intent(a, intent_key):
        """

        :param a:
        :param intent_key:
        :return:
        """
        try:
            return a.get(intent_key)
        except:
            return None

    @staticmethod
    def sort_dataset_by_date(X, y, date_key, return_timestamps=False):
        """

        :param X: list(str): training sentences
        :param y: list(dict): with intent_id and created (and updated) dates
        :param date_key: key used for the date used in sorting the data
        :param return_timestamps: if the timestamps are returned (e.g. used by the data provider)
        :return: sorted dataset: X, y[, time_stamps]
        """

        time_stamps = [TimeSeqDatasetProvider.resolve_date(a, date_key) for a in y]
        kept_indices = [i for i in range(len(time_stamps)) if time_stamps[i] is not None]
        if len(kept_indices) != len(time_stamps):
            X = np.take(X, kept_indices).tolist()
            y = np.take(y, kept_indices).tolist()
            time_stamps = np.take(time_stamps, kept_indices).tolist()

        # Return empty datasets
        if len(y) != len(time_stamps) != len(X) or len(y) == 0:
            if return_timestamps:
                return [], [], []
            else:
                return [], []

        y = [TimeSeqDatasetProvider.resolve_intent(y_i, TimeSeqDatasetProvider.INTENT_KEY) for y_i in y]

        sorted_indices = np.argsort(time_stamps)

        X = np.array(X)[sorted_indices].tolist()
        y = np.array(y)[sorted_indices].tolist()

        if return_timestamps:
            time_stamps = np.array(time_stamps)[sorted_indices].tolist()
            return X, y, time_stamps

        return X, y

    def get_datasets(self, X, y):  # pylint: disable=invalid-name
        """

        :param X: list: training text
        :param y: list: intents
        :return: list: training and test datasets
        """

        X, y, time_stamps = self.sort_dataset_by_date(X, y, self.date_key, return_timestamps=True)

        datasets = []
        full_train_set = {"X": [], "y": [], "t": []}
        full_test_set = {"X": [], "y": [], "t": []}

        for sample_idx in range(len(X)):
            if not sample_idx % self.test_step:
                full_test_set["X"].append(X[sample_idx])
                full_test_set["y"].append(y[sample_idx])
                full_test_set["t"].append(time_stamps[sample_idx])
            else:
                full_train_set["X"].append(X[sample_idx])
                full_train_set["y"].append(y[sample_idx])
                full_train_set["t"].append(time_stamps[sample_idx])

        n_training_examples = len(full_train_set["X"])
        n_test_examples = len(full_test_set["X"])

        min_examples = int(self.min_examples_ratio * n_training_examples)

        test_cursor = 0
        train_cursor = min_examples

        # step_size: how many examples we add / evaluation
        if self.max_added_examples and min_examples + self.max_added_examples < n_training_examples:
            step_size = self.max_added_examples // self.n_steps
        else:
            step_size = (n_training_examples - min_examples) // self.n_steps
        if step_size == 0:
            # in case n_steps > number of training examples
            step_size = 1

        # Generate datasets
        while train_cursor <= n_training_examples:  # and prev_train_cursor < n_training_examples:
            sub_train_x = full_train_set["X"][:train_cursor]
            sub_train_y = full_train_set["y"][:train_cursor]

            if self.test_on_all:
                datasets.append((sub_train_x, sub_train_y, full_test_set["X"], full_test_set["y"]))
            else:
                last_time_stamp = full_train_set["t"][train_cursor - 1]
                while test_cursor < n_test_examples and full_test_set["t"][test_cursor] < last_time_stamp:
                    test_cursor += 1
                datasets.append((sub_train_x, sub_train_y,
                                 full_test_set["X"][:test_cursor], full_test_set["y"][:test_cursor]))

            if train_cursor >= n_training_examples or (self.max_added_examples is not None and
                                                       train_cursor >= min_examples + self.max_added_examples):
                break

            train_cursor = min(n_training_examples, train_cursor + step_size)
        return datasets

    def __str__(self):
        return super().__str__() + "(%s;%s,%s,%s)" % (self.min_examples_ratio, self.n_steps,
                                                      self.test_on_all, self.test_step)
