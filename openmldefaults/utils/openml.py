import sklearn


def openml_sklearn_metric_mapping(openml_metric):
    # maps from OpenML name to sklearn name
    mapping = {
        'predictive_accuracy': 'accuracy',
        'f_measure': 'f1_weighted'
    }
    if openml_metric not in mapping:
        raise ValueError('Could not find sklearn metric for openml measure: %s'
                         % openml_metric)
    return mapping[openml_metric]
