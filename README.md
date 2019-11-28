# nex-cv

Please refer to this paper for motivation, description, and validation: https://www.aclweb.org/anthology/W19-4110/

This blog post may be a quicker read: https://medium.com/jobpal-dev/plausible-negative-examples-for-better-multi-class-classifier-evaluation-d8e8cb2422db

**Abstract.** We describe and validate  a metric for estimating multi-class classifier performance based on cross-validation and adapted for improvement of small, unbalanced natural-language datasets used in chatbot design. Our experiences draw upon building recruitment chatbots that mediate communication between job-seekers and recruiters by exposing the ML/NLP dataset to the recruiting team. Evaluation approaches must be understandable to various stakeholders, and useful for improving chatbot performance. The metric, `nex-cv`, uses negative examples in the evaluation of text classification, and fulfils three requirements. First, it is useful: we describe how this metric can be used and acted upon by non-developer staff. Second, it is not overly optimistic. Third, it allows model-agnostic comparison. We validate the metric based on seven recruitment-domain-specific datasets in English and German over the course of one year of chatbot monitoring and improvements.



## Testing Support

This repository provides the absrtract code needed to apply the evaluation with any classifier. The README document also describes how to test an implementation:

* Functional testing of the entire evaluation workflow: as described in the "Overvidw" section,  the results of `NexCVEvaluator(other_min_prop=0, min_category_size=0, ...)` should match the results of standard 5-fold cross-validation.
* Integration testing of the implementation: as described in the "Implementation" section below, the provided `MyClassifier(...).test()` helps to check that your wrapper behaves as the rest of the package expects.

## Overview

An evaluator produces a dict, which is managed by EvaluationResult. The evaluator uses a DatasetProvider instance to generate different testing datesets (in this case, it randomly assigns some low-population classes to use as negative examples, based on the `other_min_prop` and `other_min_prop` parameters), and an EvaluationStrategy to interpret the outcome of training a Classifier and testing it.


```python
results = []

classifier = MyClassifier() # See below for implementation guidance
for min_category_size, other_min_prop in [(0, 0),(0, 0.15), (5, 0)]: # Three recommended settings
                                                                     # (see paper for description and justification)
    mcvd = NexCVEvaluator(
        classifier=classifier,
        other_min_prop=other_min_prop,
        min_category_size=min_category_size,
        max_samples=10, # Number of examples to include in the results for inspection
        n_retries=5, # Number of times to run this
        classification_threshold=0.5 # When confidence is below this, any guess is suppressed
    )
    results.append(mcvd.evaluate(X, y))

# Write results to a file
import simplejson
pp2f = open("my_classifier_results.json", "w")
pp2f.write(simplejson.dumps(results, indent=4, sort_keys=True))
pp2f.close()
```

**Functional testing.** Although the `nex-cv` is based on CV, and in the case where both `other_min_prop` nor `min_category_size` are `0`, the evaluation result (with `n_retries=5`) should be almost the same as the result of 5-fold cross-validation.

You can also use sequential evaluation using:

```python
seq_eval = SequentialEvaluator(
    classifier=classifier,
    max_samples=10, # Number of examples to include in the results for inspection
    classification_threshold=0.5 # When confidence is below this, any guess is suppressed
)
``` 

The results track the model performance (run time, accuracy, f1) over time. The dataset should be prepared 
in the following form `y = [{"intent_id": intent_id, "created": datetime.datetime}, ...]`.

## Implementation

All necessary evaluation code is in `nex_cv`, but only an abstract client definition is provided. Wrap any classification API as shown belowl; a small test utility is also provided:

```python
my_classifier = MyClassifier(...)
violations = my_classifier.test()
if violations:
    print("Implementation violates consistency constraints:\n-", "\n- ".join(violations))
else:
    print("Implementation passes basic test!")
```

A complete implementation requires only the following definitions - everything else is already provided:

```python
class MyClassifier(APIClassifier):

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
    
    def classify(self, text):
        """
        Returns the classification result in the format:
        [dict(intent_id=..., confidence=...)]
        If multiple intents are provided, the [0]th one should be the best / highest-confidence.
        
        The intent_id field should not provide intents that were not trained with create_intent();
        so, for example, if implementing a connector for DialogFlow, note that it provides a
        "Default Fallback Intent." If that kind of default-fallback is the outcome of classification,
        return an empty list from this function. Otherwise, the EvaluationStrategy will not be able
        to distinguish fallback behavior from misclassification.
        """
        raise NotImplementedError

```

