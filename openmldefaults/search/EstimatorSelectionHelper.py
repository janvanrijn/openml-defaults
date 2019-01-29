import logging
import numpy as np
import pandas as pd
import sklearn.base
import sklearn.model_selection


# code adjusted from: http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
class EstimatorSelectionHelper(sklearn.base.BaseEstimator):

    def __init__(self, models, params, cv, n_jobs, verbose, scoring, maximize):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.maximize = maximize
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.scoring = scoring

    def fit(self, X, y):
        for idx, key in enumerate(self.keys):
            model = self.models[key]
            params = self.params[key]
            if self.verbose:
                logging.info('fitting %s with grid size %d (%d/%d)' % (key, len(params), idx + 1, len(self.keys)))
            # unfortunately, we have to refit all ..
            gs = sklearn.model_selection.GridSearchCV(model, params,
                                                      cv=self.cv,
                                                      n_jobs=self.n_jobs,
                                                      verbose=self.verbose,
                                                      scoring=self.scoring,
                                                      refit=True,
                                                      return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def predict(self, X):
        return self.grid_searches[self._best_estimator(self.maximize)].best_estimator_.predict(X)

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

    def _best_estimator(self, maximize):
        best_estimator = None
        best_score = -np.inf if maximize else np.inf
        for key in self.keys:
            if (maximize and self.grid_searches[key].best_score_ > best_score) or \
                    (not maximize and self.grid_searches[key].best_score_ < best_score):
                best_estimator = key
        return best_estimator
