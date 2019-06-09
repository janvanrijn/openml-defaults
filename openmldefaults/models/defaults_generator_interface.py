import ConfigSpace
import pandas as pd
import typing


class DefaultsGenerator(object):

    def generate_defaults_discretized(self, df: pd.DataFrame, num_defaults: int,
                                      minimize: bool, aggregate: typing.Callable,
                                      config_space: ConfigSpace.ConfigurationSpace,
                                      raise_no_improvement: bool):
        raise NotImplementedError('Interface Method, pleae subclass!')
