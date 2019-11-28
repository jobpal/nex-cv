from datetime import datetime
from nex_cv.data_providers import NegativeExamplesDatasetProvider
from nex_cv.data_providers import TimeSeqDatasetProvider
from nex_cv.evaluation_strategies import IntentEvaluationStrategy


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


class SequentialEvaluator(Evaluator):

    def __init__(self, classifier, classification_threshold, max_samples, *args, **kwargs):
        super().__init__(
            classifier=classifier,
            dataset_provider=TimeSeqDatasetProvider(0.5, max_added_examples=500),
            evaluation_strategy=IntentEvaluationStrategy(
                classification_threshold=classification_threshold,
                include_samples=[],  # include all examples
                max_samples=max_samples),
            *args,
            **kwargs)
