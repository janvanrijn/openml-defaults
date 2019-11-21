import abc
import ConfigSpace
import typing
import random
import numpy as np
import pickle
import logging

from openmldefaults.symbolic import ABCTransformer

class SymbolicConfigurationValue(object):

    def __init__(self,
                 value: typing.Any,
                 transformer: typing.Optional[ABCTransformer],
                 meta_feature: typing.Optional[str]):
        self.value = value
        self.transformer = transformer
        self.meta_feature = meta_feature
        if (self.transformer is None) != (self.meta_feature is None):
            raise ValueError('transformer and meta-feature should both be set or initialized to None')

    def get_value(self, meta_features: typing.Dict[str, float]) -> typing.Union[str, float, int]:
        if self.transformer is None:
            return self.value
        elif meta_features is None:
            raise ValueError('Meta-features is None, while transformer fn is in place. ')
        return self.transformer.transform(self.value, meta_features[self.meta_feature])

    def get_repr(self) -> typing.Dict:
        if self.transformer is None:
            return {"Constant": self.value, "Transformer": "None"}
        else:
            return {"Constant": self.value,
                    "Transformer": self.transformer.__str__(),
                    "Metafeature": self.meta_feature}


class SymbolicConfiguration(object):

    def __init__(self, configuration: typing.Dict[str, SymbolicConfigurationValue]):
        self.configuration = configuration

    def get_dictionary(self, meta_features: typing.Optional[typing.Dict[str, float]]) -> typing.Dict[str, typing.Union[str, float, int]]:
        return {
            param: symbolic_val.get_value(meta_features) for param, symbolic_val in self.configuration.items()
        }

    def get_repr(self) -> typing.Dict:
        return {
            param: symbolic_val.get_repr() for param, symbolic_val in self.configuration.items()
        }


class ConfigurationSampler(abc.ABC):

    def sample_configurations(self, n_configurations: int) -> typing.List[SymbolicConfiguration]:
        raise NotImplementedError()

    def get_hyperparameter_names(self) -> typing.List[str]:
        raise NotImplementedError()


class VanillaConfigurationSpaceSampler(ConfigurationSampler):

    def __init__(self, configuration_space: ConfigSpace.ConfigurationSpace):
        self.configuration_space = configuration_space

    def sample_configuration(self) -> SymbolicConfiguration:
        result = dict()
        for p, v in self.configuration_space.sample_configuration().get_dictionary().items():
            result[p] = SymbolicConfigurationValue(v, None, None)
        return SymbolicConfiguration(result)

    def sample_configurations(self, n_configurations: int) -> typing.List[SymbolicConfiguration]:
        return [self.sample_configuration() for _ in range(n_configurations)]

    def get_hyperparameter_names(self) -> typing.List[str]:
        return self.configuration_space.get_hyperparameter_names()


class SymbolicConfigurationSpaceSampler(ConfigurationSampler):

    def __init__(self, configuration_space: ConfigSpace.ConfigurationSpace,
        transform_fns: typing.Dict, meta_features: str, meta_feature_ranges_file: str):
        self.configuration_space = configuration_space
        self.transform_fns = transform_fns
        self.meta_features = meta_features
        with open(meta_feature_ranges_file, 'rb') as fp:
            self.meta_feature_ranges = pickle.load(fp)

    def sample_configuration(self) -> SymbolicConfiguration:
        result = dict()
        # For now we only substitute a single symbolic
        hyperparameter = random.choice(self.configuration_space.get_hyperparameter_names())
        hyperparameter = self.configuration_space.get_hyperparameter(hyperparameter)

        for p, v in self.configuration_space.sample_configuration().get_dictionary().items():

            if p == hyperparameter.name and \
                not isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter)  and\
                not isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant):
                transform = random.choice(list(self.transform_fns.values()))
                meta_feature = random.choice(self.meta_features)
                limits = self.get_meta_feature_range(meta_feature)
                valid_range = self.overlaps(
                    [transform.inverse(hyperparameter.lower, meta_feature_value = limits["min"]),
                     transform.inverse(hyperparameter.upper, meta_feature_value = limits["min"])],
                    [transform.inverse(hyperparameter.lower, meta_feature_value = limits["max"]),
                     transform.inverse(hyperparameter.upper, meta_feature_value = limits["max"])])
                if (np.sign(valid_range[0]) == np.sign(valid_range[1])):
                    v = random.choice(np.geomspace(valid_range[0], valid_range[1], 50))
                else: v = random.choice(np.linspace(valid_range[0], valid_range[1], 50))
                # if (valid_range[0] > valid_range[1]):
                #    logging.warning('No valid alpha found for hyperparameter: %s and transformer %s.' % (hyperparameter.name, transform.__str__()))
                #    result[p] = SymbolicConfigurationValue(v, None, None)
                #    continue
                if not np.isnan(v):
                    result[p] = SymbolicConfigurationValue(v, transform, meta_feature)
                else:
                    result[p] = SymbolicConfigurationValue(v, None, None)
            else:
                result[p] = SymbolicConfigurationValue(v, None, None)

        return SymbolicConfiguration(result)

    def sample_configurations(self, n_configurations: int) -> typing.List[SymbolicConfiguration]:
        return [self.sample_configuration() for _ in range(n_configurations)]

    def get_hyperparameter_names(self) -> typing.List[str]:
        return self.configuration_space.get_hyperparameter_names()

    def get_meta_feature_range(self, name: str) -> typing.Dict:
        return self.meta_feature_ranges[name]

    @staticmethod
    def overlaps(a, b):
        return [min(a[1], b[1]), max(a[0], b[0])]
