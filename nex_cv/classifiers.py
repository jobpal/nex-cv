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
