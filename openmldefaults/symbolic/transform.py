import abc
import numpy as np
import typing


class ABCTransformer(abc.ABC):
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        raise NotImplementedError()


class InverseTransformer(ABCTransformer):
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        # raising is ok
        if meta_feature_value == 0.0:
            raise ZeroDivisionError()
        result = param_value / meta_feature_value
        if np.isinf(result):
            raise OverflowError()
        return result


class PowerTransformer(ABCTransformer):
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        return param_value ** meta_feature_value


class LogTransformer(ABCTransformer):
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        return param_value * np.log(meta_feature_value)


class RootTransformer(ABCTransformer):
    @staticmethod
    def transform(param_value: float, meta_feature_value: float) -> float:
        return param_value * np.sqrt(meta_feature_value)


def all_transform_fns() -> typing.Dict[str, ABCTransformer]:
    transform_fns = {
        'inverse_transformer': InverseTransformer(),
        'power_transformer': PowerTransformer(),
        'log_transformer': LogTransformer(),
        'root_transformer': RootTransformer(),
    }
    return transform_fns
