import abc
import numpy as np
import typing


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

def all_transform_fns() -> typing.Dict[str, ABCTransformer]:
    transform_fns = {
        'inverse_transformer': InverseTransformer(),
        'power_transformer': PowerTransformer(),
        'log_transformer': LogTransformer(),
        'root_transformer': RootTransformer(),
    }
    return transform_fns
