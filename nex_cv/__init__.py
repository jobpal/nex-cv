import json
import requests
import os
import time
import numpy as np
from collections import Counter
from datetime import datetime
from sklearn.metrics import f1_score


class APIClassifier:
    """
    An abstract definition expected by the evaluator; see README for 
    additional guidance on implementation, testing,  and running.
    """

    def all_intents(self):
        """
        Return list of all intents; list should contain whatever can
        be used as the "intent" param to delete_intent. Should contain
        the list of everthing that needs to be deleted!
        """
        raise NotImplementedError

    def create_intent(self, intent, examples):
        """
        Creates specified intent and trans it with provided examples
        """
        raise NotImplementedError

    def delete_intent(self, intent):
        """
        Deletes the associated intent
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def fit(self, X, y):
        """
        Deletes all the existing intents and creates new ones. If any custom waiting -
        including sleeping and/or making a status API call - should be done by
        overriding this method in your implementation. When using external APIs, sleeping
        before using the classifier may be necessary, as well as throttling calls to
        conform to API limitations.
        """

        for intent in self.all_intents():
            self.delete_intent(intent)
        for label in set(y):
            self.create_intent(label,
                               [x for i, x in enumerate(X) if y[i] == label])

    def classify(self, input_text):
        """
        Returns the classification result in the format:
        [dict(intent_id=..., confidence=...)]
        If multiple intents are provided, the [0]th one should be the best / highest-confidence.
        
        The intent_id field should not provide intents that were not trained with create_intent();
        so, for example, if implementing a connector for DialogFlow, note that it provides a
        "Default Fallback Intent." If that kind of default-fallback is the outcome of classification,
        return an empty list from this function. Otherwise, the EvaluationStrategy will not be able
        to distinguish fallback behavior from misclassification.luationStrategy will not be able
        to distinguish fallback behavior from misclassification.
        """
        raise NotImplementedError

    def test(self):
        """
        A small utility that tests two trivially distinguishable intents
        """

        violations = []

        tests = {}
        qs_by_cat = dict(
            A=["hello", "hi", "hey", "greeting"],
            B=["bye", "goodbye", "see you", "ok bye"])

        X, y = [], []
        for cat, qs in qs_by_cat.items():
            tests[cat] = "%s %s" % (qs[0], qs[0])
            X.extend(qs)
            y.extend([cat for _ in qs])

        def run_test(which, X_, y_, tests_):
            vs = []
            self.fit(X_, y_)
            for c, t in tests_.items():
                pred = self.classify(t)
                if c != pred[0]["intent_id"]:
                    vs.append("%s: encountered %s (%s) but expected %s (%s)" %
                              (which, pred[0]["intent_id"],
                               round(pred[0]["confidence"], 2), c, t))
            return vs

        violations.extend(run_test("basic", X, y, tests))

        y_switched = ["A" if y_i == "B" else "B" for y_i in y]
        tests_switched = dict(A=tests["B"], B=tests["A"])
        violations.extend(run_test("switched", X, y_switched, tests_switched))

        violations.extend(run_test("basic repeated", X, y, tests))

        return violations


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


class EvaluationStrategy:  # pylint: disable=too-few-public-methods
    """
    Defines how to intepret prediction results
    """

    def get_metrics(self, X, y, classify_results):  # pylint: disable=invalid-name
        raise NotImplementedError


class IntentEvaluationStrategy(EvaluationStrategy):  # pylint: disable=too-few-public-methods
    """
    Interprets prediction results as for a standard multi-class case, plus
    accounting for no-guesses (too low confidence) and  negative
    examples.
    """

    # Possible outcomes
    TRUE_LABEL = "T"
    FALSE_LABEL = "F"
    FALSE_OTHER = "FO"
    TRUE_OTHER = "TO"
    FALSE_CONSERVATIVE_OTHER = "FCO"

    def __init__(self,
                 classification_threshold,
                 others_label="__Other",
                 include_samples=None,
                 max_samples=10):
        """
        :param confidence_threshold:
        :param others_label: In the case where there is some label which should
        be interpreted as "other," it can be specified
        :param include_samples: A list of cases for which to include samples.
        For example, to include only wrongly-classified samples - excluding
        low-confidence None's - pass [IntentEvaluationStrategy.FALSE_LABEL]
        :param max_samples: Will select a random provided number of samples to include;
        0 -> none all; -1 -> none
        """
        self.classification_threshold = classification_threshold
        self.others_label = others_label
        self.include_samples = include_samples
        self.max_samples = max_samples
        self.annotation_resolver = lambda x: x

    def get_outcome(self, y_i, annotation):
        """
        Account for the "other" category and interaction with the classification threshold
        """

        true_label = y_i
        pred_label = annotation[0][
            'intent_id'] if annotation and annotation[0] else None
        pred_conf = annotation[0]['confidence']

        outcome = None
        if true_label == pred_label:
            if pred_label is None:
                # there was no right answer, and none was given
                outcome = IntentEvaluationStrategy.TRUE_OTHER
            elif pred_conf >= self.classification_threshold:
                # the answer was right and the confidence was high enough
                outcome = IntentEvaluationStrategy.TRUE_LABEL
            else:
                # the answer would have been right, but the confidence was too low
                outcome = IntentEvaluationStrategy.FALSE_CONSERVATIVE_OTHER
        elif true_label is None and pred_conf <= self.classification_threshold:
            # there was no right answer, and none was given
            outcome = IntentEvaluationStrategy.TRUE_OTHER
        elif pred_conf <= self.classification_threshold:
            # no answer was given, the guess was wrong anyway, but there WAS a good answer
            outcome = IntentEvaluationStrategy.FALSE_OTHER
        else:
            outcome = IntentEvaluationStrategy.FALSE_LABEL

        pred_label = pred_label if pred_conf > self.classification_threshold else self.others_label
        return true_label, pred_label, pred_conf, outcome

    def get_metrics(self, X, y, classify_results):  # pylint: disable=invalid-name
        """
        The three parameters are expected to be lists of the same length. Metrics - all rounded
        to 3 decimal points - include:

        * accuracy: based on interpreting the "None/Other" label as an additional label
        * f1_weighted: equivalent to the old F1 metric; uses weighting by classes
        * f1_macro: does not weight by classes
        * good_carefulness: Of the low-confidence responses suppressed, X were good guesses
        * total: number of
        * confusion_matrix: sparse-ish: every label has a row but may not have a column if it's 0
        * metrics_by_label: dict, containing additional metrics for each
        * samples: list of tuples

        :return: Dict of metrics as described above.
        """

        if len(X) != len(y) != len(classify_results):
            return None

        confusion_matrix = {self.others_label: {}}
        confidence_adjusted_pred_label_list = []  # pylint: disable=invalid-name
        outcome_counts = {}
        samples = []
        all_labels = set()
        potential_samples = []

        for i, x in enumerate(X):  # pylint: disable=invalid-name
            true_label, pred_label, pred_conf, outcome = self.get_outcome(
                y[i], classify_results[i])

            # Keep track of the outcome and confidence-threshold-adjusted labels
            confidence_adjusted_pred_label_list.append(pred_label)
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            all_labels.add(true_label)
            all_labels.add(pred_label)

            if true_label:
                confusion_matrix[true_label] = confusion_matrix.get(
                    true_label, {})
                confusion_matrix[true_label][pred_label] =\
                    confusion_matrix[true_label].get(pred_label if pred_label else self.others_label, 0) + 1
            else:
                confusion_matrix[self.others_label][pred_label] =\
                    confusion_matrix[self.others_label].get(pred_label if pred_label else self.others_label, 0) + 1

            if not self.include_samples or outcome in self.include_samples:
                potential_samples.append((X[i], true_label, pred_label,
                                          pred_conf, outcome))
        if self.max_samples >= 0:
            indexes = np.arange(len(potential_samples))
            indexes = indexes[:self.max_samples] if len(
                indexes) > self.max_samples > 0 else indexes
            np.random.shuffle(indexes)  # pylint: disable=no-member
            samples = [potential_samples[i] for i in indexes]

        metrics_by_label = {}

        confusion_matrix.update({
            label: {}
            for label in all_labels
            if label and not confusion_matrix.get(label)
        })
        for true_label, predictions in confusion_matrix.items():
            total = sum(predictions.values())

            count_fc_a = predictions.get(self.others_label, 0)
            count_tp_a = predictions.get(true_label, 0)
            count_fn_a = total - count_tp_a - count_fc_a
            count_fp_a = sum([other_predictions.get(true_label, 0) for\
                        other_label, other_predictions in confusion_matrix.items() if\
                        not other_label == true_label])

            precision = (count_tp_a / (count_tp_a + count_fp_a)) if (
                count_tp_a + count_fp_a) else 0
            recall = (count_tp_a / (count_tp_a + count_fn_a)) if (
                count_tp_a + count_fn_a) else 0
            f1_for_label = (2 * ((precision * recall) / (precision + recall))
                            ) if (precision + recall) > 0 else 0
            metrics_by_label[true_label] = dict(
                precision=round(precision, 3),
                recall=round(recall, 3),
                f1=round(f1_for_label, 3),
                no_answer=round(count_fc_a / total, 3) if total else 0,
                total=total)
        total = sum(outcome_counts.values())
        accuracy = (outcome_counts.get(IntentEvaluationStrategy.TRUE_LABEL, 0) +\
            outcome_counts.get(IntentEvaluationStrategy.TRUE_OTHER, 0)) / total

        good_carefulness = outcome_counts.get(
            IntentEvaluationStrategy.FALSE_CONSERVATIVE_OTHER, 0)

        no_answer = outcome_counts.get(IntentEvaluationStrategy.FALSE_CONSERVATIVE_OTHER, 0) + \
                    outcome_counts.get(IntentEvaluationStrategy.TRUE_OTHER, 0) + \
                    outcome_counts.get(IntentEvaluationStrategy.FALSE_OTHER, 0)
        no_none_adjusted_y_label_list = [
            label if label else self.others_label for label in y
        ]

        return dict(
            accuracy=round(accuracy, 3),
            f1_weighted=round(
                f1_score(
                    no_none_adjusted_y_label_list,
                    confidence_adjusted_pred_label_list,
                    average='weighted'), 3),
            f1_macro=round(
                f1_score(
                    no_none_adjusted_y_label_list,
                    confidence_adjusted_pred_label_list,
                    average='macro'), 3),
            good_carefulness=round(good_carefulness / no_answer, 3)
            if good_carefulness < no_answer else 0,
            total=total,
            confusion_matrix=confusion_matrix,
            metrics_by_label=metrics_by_label,
            samples=samples)


class EvaluationResult:
    """
    Adds timing information to any other metric.
    """

    def __init__(self):
        self.info = {}
        self.metrics = []
        self.start_time = datetime.utcnow()

    def asdict(self):
        total_time = datetime.utcnow() - self.start_time  # pylint: disable=invalid-name

        self.info.update(
            dict(
                total_running_time_sec=total_time.total_seconds(),
                avg_running_time_sec=total_time.total_seconds() / len(
                    self.metrics)))

        return dict(info=self.info, metrics=self.metrics)


class Evaluator:
    """
    Executes evaluation; if the dataset provider provides folds as in $k$-fold
    cross-validation, this would be like CV.
    """

    def __init__(self, classifier, dataset_provider, evaluation_strategy):
        self.classifier = classifier
        self.dataset_provider = dataset_provider
        self.evaluation_strategy = evaluation_strategy

    def __str__(self):
        return self.__class__.__name__

    def evaluate(self, X, y):  # pylint: disable=invalid-name
        evaluation_result = EvaluationResult()
        for X_train, y_train, X_test, y_test in self.dataset_provider.get_datasets(
                X, y):  # pylint: disable=invalid-name

            if not evaluation_result.info.get("classifier"):
                evaluation_result.info["classifier"] = str(self.classifier)

            self.classifier.fit(X_train, y_train)
            classification_results = []
            for i, text in enumerate(X_test):
                cr = self.classifier.classify(text)
                classification_results.append(cr)

            evaluation_result.metrics.append(
                self.evaluation_strategy.get_metrics(X_test, y_test,
                                                     classification_results))

        evaluation_result.info["evaluator"] = str(self)
        evaluation_result.info["dataset_provider"] = str(self.dataset_provider)

        return evaluation_result.asdict()


class NexCVEvaluator(Evaluator):
    """
    nex-cv is similar to k-fold cross-validation, but extends the dataset partition
    to include negative examples (using the DatasetProvider implementation) and
    the results interpretation (using the EvaluationStrategy)
    """

    def __init__(self, classifier, min_category_size, other_min_prop,\
                 classification_threshold, max_samples, n_retries, *args, **kwargs):

        super().__init__(
            classifier=classifier,
            dataset_provider=NegativeExamplesDatasetProvider(
                min_category_size=min_category_size,
                other_min_prop=other_min_prop,
                n_retries=n_retries),
            evaluation_strategy=IntentEvaluationStrategy(
                classification_threshold=classification_threshold,
                include_samples=[],  #include all examples
                max_samples=max_samples),
            *args,
            **kwargs)

