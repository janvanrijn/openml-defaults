import sklearn


def openml_sklearn_metric_mapping(openml_metric):
    mapping = {
        'predictive_accuracy': sklearn.metrics.accuracy_score,
        'f_measure': sklearn.metrics.f1_score,
    }
    if openml_metric not in mapping:
        raise ValueError('Could not find sklearn metric for openml measure: %s'
                         % openml_metric)
    return mapping[openml_metric]
