import abc
import numpy as np
import typing

"""
ABCTransformer is the base class for a basic class of transformers that
takes in
- param_value (numeric) : A number that ensures that the transformation project into a valid range.
- meta_feature_value (numeric | int) : The value of the meta_feature.

Additional methods:
- __str__ for nicer prints
- inverse: Allows to calculate a range for param_value, such that transform() maps into
  the valid range of the hyperparameter.
"""
class ABCTransformer(abc.ABC):
    def __str__(self):
        return('abc_transformer')
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        raise NotImplementedError()
    @staticmethod
    def inverse(y: float, meta_feature_value: float) -> float:
        raise NotImplementedError()
    __call__ = transform


class InverseTransformer(ABCTransformer):
    def __str__(self):
        return('inverse_transformer')
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        # raising is ok
        if meta_feature_value == 0.0:
            raise ZeroDivisionError()
        result = param_value / meta_feature_value
        if np.isinf(result):
            raise OverflowError()
        return result
    @staticmethod
    def inverse(y: float, meta_feature_value: float) -> float:
        return y * meta_feature_value
    __call__ = transform


class PowerTransformer(ABCTransformer):
    def __str__(self):
        return('power_transformer')
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        return param_value ** meta_feature_value
    @staticmethod
    def inverse(y: float, meta_feature_value: float) -> float:
        if meta_feature_value == 0.0:
            raise ZeroDivisionError()
        if y < 0.0: # Produces complex numbers
            raise ValueError()
        return y ** (1/float(meta_feature_value))
    __call__ = transform


class LogTransformer(ABCTransformer):
    def __str__(self):
        return('log_transformer')
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        return param_value * np.log(meta_feature_value)
    @staticmethod
    def inverse(y: float, meta_feature_value: float) -> float:
        if np.log(meta_feature_value) == 0.0:
            raise ZeroDivisionError()
        return y / np.log(meta_feature_value)
    __call__ = transform


class RootTransformer(ABCTransformer):
    def __str__(self):
        return('log_transformer')
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        return param_value * np.sqrt(meta_feature_value)
    @staticmethod
    def inverse(y: float, meta_feature_value: float) -> float:
        if np.log(meta_feature_value) == 0.0:
            raise ZeroDivisionError()
        return y / np.sqrt(meta_feature_value)
    __call__ = transform


class ValuePowerTransformer(ABCTransformer):
    def __str__(self):
        return('value_power_transformer')
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        if meta_feature_value < 0.0 and param_value % 1.0 != 0: # Produces complex numbers
            raise ValueError()
        return meta_feature_value ** param_value
    @staticmethod
    def inverse(y: float, meta_feature_value: float) -> float:
        if meta_feature_value <= 0.0 or y <= 0.0:
            raise ValueError()
        if meta_feature_value == 1.0:
            raise ZeroDivisionError()
        return np.log(y) / np.log(meta_feature_value)
    __call__ = transform

class ConstantTransformer(ABCTransformer):
    def __str__(self):
        return('constant')
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        return param_value
    @staticmethod
    def inverse(y: float, meta_feature_value: float) -> float:
        return y
    __call__ = transform

def all_transform_fns() -> typing.Dict[str, ABCTransformer]:
    transform_fns = {
        'inverse_transformer': InverseTransformer(),
        'power_transformer': PowerTransformer(),
        'log_transformer': LogTransformer(),
        'root_transformer': RootTransformer(),
        'value_power_transformer': ValuePowerTransformer(),
        'constant_transformer': ConstantTransformer()
    }
    return transform_fns
