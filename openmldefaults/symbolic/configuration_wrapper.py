import abc
import ConfigSpace
import typing

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


class SymbolicConfiguration(object):

    def __init__(self, configuration: typing.Dict[str, SymbolicConfigurationValue]):
        self.configuration = configuration

    def get_dictionary(self, meta_features: typing.Optional[typing.Dict[str, float]]) -> typing.Dict[str, typing.Union[str, float, int]]:
        return {
            param: symbolic_val.get_value(meta_features) for param, symbolic_val in self.configuration.items()
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
