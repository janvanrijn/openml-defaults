from sklearn.model_selection._search import BaseSearchCV


class DefaultSearchCV(BaseSearchCV):

    def __init__(self, estimator, defaults, scoring=None,
                 fit_params=None, n_jobs=1, iid='warn', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score="warn"):
        if not isinstance(defaults, list):
            raise ValueError()
        for default in defaults:
            if not isinstance(default, dict):
                raise ValueError()

        self.defaults = defaults
        self.random_state = random_state
        super(DefaultSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def _get_param_iterator(self):
        """Return ParameterSampler instance for the given distributions"""
        return self.defaults
