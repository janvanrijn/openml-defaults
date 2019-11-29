%load_ext autoreload

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
import openml
import abc
import ConfigSpace
import typing

from openmldefaults.config_spaces import get_config_space
from openmldefaults.symbolic import SymbolicConfigurationSpaceSampler, all_transform_fns

if __name__ == "__main__":
    # %autoreload 2
    # logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    # cs = get_config_space("svc", 122, "micro")
    # tfs = all_transform_fns()
    # scv = SymbolicConfigurationSpaceSampler(cs, tfs, meta_features = ['NumberOfInstances', 'NumberOfFeatures'], meta_feature_ranges_file = "/home/flo/Documents/projects/openml-defaults/data/metafeatures_ranges.pkl")
    # val = scv.sample_configurations(1)
    # val[0].get_repr()

    # # Dump metafeature ranges to pickle.
    # arff_file = "/home/flo/Documents/projects/openml-defaults/data/metafeatures_openml100.arff"
    # with open(arff_file, 'r') as fp:
    #     metadata_quality_frame = openmlcontrib.meta.arff_to_dataframe(arff.load(fp), None)

    # dct = {}
    # for name, col in metadata_quality_frame.iteritems():
    #     dct[name] = {"min": col.min(), "max": col.max()}

    # with open("/home/flo/Documents/projects/openml-defaults/data/metafeatures_ranges.pkl", 'wb') as fp:
    #     pickle.dump(dct, file = fp)

    # with open("/home/flo/Documents/projects/openml-defaults/data/metafeatures_ranges.pkl", 'rb') as fp:
    #     dict = pickle.load(fp)

    with open("/home/flo/experiments/openml-defaults/symbolic_defaults_vs_rs/svc/3.0/average_rank_max_predictive_accuracy/32/1024/1/sum/1/MinMaxScaler/StandardScaler/defaults_32_0.pkl", 'rb') as fp:
         dict = pickle.load(fp)

    [x.get_repr() for x in dict[2]]

