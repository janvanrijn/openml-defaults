from sklearn.model_selection._search import BaseSearchCV


class DefaultSearchCV(BaseSearchCV):

    def __init__(self, estimator, defaults, scoring=None,
                 fit_params=None, n_jobs=1, iid='warn', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score='raise', return_train_score="warn"):
        DefaultSearchCV._verify_defaults(defaults)
        self.defaults = defaults
        self.param_distributions = DefaultSearchCV._determine_param_distributions(defaults)
        super(DefaultSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    @staticmethod
    def _verify_defaults(defaults):
        if not isinstance(defaults, list):
            raise ValueError()
        expected_keys = None
        for default in defaults:
            if not isinstance(default, dict):
                raise ValueError()
            if expected_keys is None:
                expected_keys = default.keys()
            else:
                if expected_keys != default.keys():
                    raise ValueError()

    @staticmethod
    def _determine_param_distributions(defaults):
        result = {}
        for default in defaults:
            for param, value in default.items():
                if param not in result:
                    result[param] = list()
                if value not in result[param]:
                    result[param].append(value)
        return result

    # For sklearn version 0.19.0 and up
    # def _get_param_iterator(self):
    #     """Return ParameterSampler instance for the given distributions"""
    #     return self.defaults

    def fit(self, X, y=None, groups=None):
        return self._fit(X, y, groups, self.defaults)
