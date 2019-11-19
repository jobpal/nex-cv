import numpy as np
from sklearn.metrics import f1_score


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

