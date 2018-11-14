import numpy as np
import typing


def inverse_transform_fn(param_value: float, meta_feature_value: float) -> float:
    # raising is ok
    if meta_feature_value == 0.0:
        raise ZeroDivisionError()
    result = param_value / meta_feature_value
    if np.isinf(result):
        raise OverflowError()
    return result


def power_transform_fn(param_value: float, meta_feature_value: float) -> float:
    return param_value ** meta_feature_value


def multiply_transform_fn(param_value: float, meta_feature_value: float) -> float:
    return param_value * meta_feature_value


# def sigmoid_transform_fn(param_value: float, meta_feature_value: float) -> float:
#     return 1 / (1 + np.e ** (-1 * meta_feature_value))


def log_transform_fn(param_value: float, meta_feature_value: float) -> float:
    return param_value * np.log(meta_feature_value)


def root_transform_fn(param_value: float, meta_feature_value: float) -> float:
    return param_value * np.sqrt(meta_feature_value)


def all_transform_fns() -> typing.Dict[str, typing.Callable]:
    transform_fns = {
        'inverse_transform_fn': inverse_transform_fn,
        'power_transform_fn': power_transform_fn,
        'multiply_transform_fn': multiply_transform_fn,
        'log_transform_fn': log_transform_fn,
        'root_transform_fn': root_transform_fn
    }
    return transform_fns
