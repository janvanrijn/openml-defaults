%load_ext autoreload
%autoreload
import arff
import argparse
import ConfigSpace
import logging
import numpy as np
import openmlcontrib
import openmldefaults
import os
import pandas as pd
import pickle
import sklearn
import typing
import pdb
import sys
import logging


""""
Objective:
hyp1 = log(mf['p']);   valid = (0, 1]
hyp2 = 3;            valid = {1,...,100}
hyp3 = 1/mf['n'] * 10; valid = [0, 10]
"""
def objective(meta_features):
    hyp1 = np.log(meta_features['ClassEntropy']) * 0.2
    hyp2 = 3
    hyp3 = 1 #/ meta_features['NumberOfFeatures'] * 10
    return(np.array([hyp1, hyp2, hyp3]))

def eval_objective(params, meta_features):
    if params.get("h1") <= -1 or params.get("h1") > 1:
        return(1000)
    if params.get("h2") < 1 or  params.get("h2") > 100:
        return(1000)
    if params.get("h3") < 0 or  params.get("h3") > 10:
        return(1000)
    tru = objective(meta_features)
    prd = np.array([params.get("h1"),  params.get("h2"), params.get("h3")])
    return ((tru-prd)**2).sum()

def eval_avg_objective(cfg, metafeatures):
    out = 0
    for i in range(4):
        rw = metafeatures.iloc[i]
        out += eval_objective(cfg.get_dictionary(rw), rw)
    return(out / 4)


def get_cs():
    cs = ConfigSpace.ConfigurationSpace('classif_svm', 42)
    h1 = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='h1', lower=-1, upper=1)
    h2 = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='h2', lower=1, upper=100)
    h3 = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='h3', lower=0, upper=10)
    cs.add_hyperparameters([h1, h2, h3])
    return(cs)


if __name__ == '__main__':
    transform_fns = openmldefaults.symbolic.all_transform_fns()
    metafeatures_ranges = os.path.expanduser('~/Documents/projects/openml-defaults/data/metafeatures_ranges.pkl')
    cs = get_cs()
    search_hps = [hp.name for hp in cs.get_hyperparameters()]

    metafeatures = pd.DataFrame(data = {
        'ClassEntropy' : [0.3, 1, 4, 6],
        'MajorityClassPercentage' : [0.1, 0.3, 0.7, 0.99],
        'NumberOfFeatures' : [4, 18, 100, 1000]
    })

    candidate = None
    best_score = 10e3
    for i in range(10000):
        for hyp in ['h1', 'h2', 'h3']:
            scs = openmldefaults.symbolic.SymbolicConfigurationSpaceSampler(cs,
                transform_fns,
                ['ClassEntropy', 'MajorityClassPercentage', 'NumberOfFeatures'],
                metafeatures_ranges, hyp, candidate, 1000)
            # print("Optimizing %s" % hyp)
            cfgs = scs.sample_configurations(500)
            results = [eval_avg_objective(cfg, metafeatures) for cfg in cfgs]
            if (np.min(results) < best_score):
                print("Better score found using %s" % hyp)
                candidate = cfgs[np.argmin(results)]
                best_score = np.min(results)
                print(candidate.get_repr())
                print(best_score)




