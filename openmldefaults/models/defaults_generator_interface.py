import ConfigSpace
import pandas as pd
import typing


class DefaultsGenerator(object):

    def __init__(self):
        self.name = None

    def generate_defaults_discretized(self, df: pd.DataFrame, num_defaults: int,
                                      minimize: bool, aggregate: typing.Callable,
                                      config_space: ConfigSpace.ConfigurationSpace,
                                      raise_no_improvement: bool) \
            -> typing.Tuple[typing.List, typing.Dict[str, typing.Any]]:
        raise NotImplementedError('Interface Method, please subclass!')
