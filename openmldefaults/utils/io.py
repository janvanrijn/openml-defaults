import arff
import ConfigSpace
import json
import numpy as np
import openmldefaults
import pandas as pd

from typing import List


def get_setup_dirname(resized_grid_size, num_defaults):
    return 'c%d_d%d' % (resized_grid_size, num_defaults)


def print_columns(df, params):
    for param in params:
        unique = np.array(df[param].unique())
        print(openmldefaults.utils.get_time(), '%s (%s) unique values: %s (%d)' % (param, df[param].dtype, unique,
                                                                                   len(unique)))


def get_meta_data_config_space(meta_data):
    cs_fn = getattr(openmldefaults.config_spaces,
                    'get_%s_%s_search_space' % (meta_data['classifier'], meta_data['config_space']))
    return cs_fn()


def get_component_mapping(config_space: ConfigSpace.ConfigurationSpace):
    """
    Each hyperparameter has both a name and a meta-field, containing an component prefix.
    This function returns a mapping from the concatenated component prefix and hyperparameter
    name to the hyperparameter name (by which it can be obtained from the config space)
    """
    result = dict()
    for param in config_space.get_hyperparameters():
        component_name = param.meta['component'] + '__' + param.name
        result[component_name] = param.name
    return result


def cast_columns_of_dataframe(df: pd.DataFrame, params: List, config_space: ConfigSpace.ConfigurationSpace):
    for param in params:
        hyperparameter = config_space.get_hyperparameter(param)

        if isinstance(hyperparameter, ConfigSpace.UniformIntegerHyperparameter) or \
                (isinstance(hyperparameter, ConfigSpace.Constant) and isinstance(hyperparameter.value, int)) or \
                (isinstance(hyperparameter, ConfigSpace.UnParametrizedHyperparameter) and isinstance(hyperparameter.value, int)):
            # limitation of pandas: can't mix nan and integer
            df[param] = df[param].dropna().apply(lambda x: str(int(x)))
    return df


def get_dataset_metadata(dataset_path):
    with open(dataset_path) as fp:
        first_line = fp.readline()
        if first_line[0] != '%':
            raise ValueError('arff data file should start with comment for meta-data')
    meta_data = json.loads(first_line[1:])
    return meta_data


def load_dataset(dataset_path, params, resized_grid_size, flip_performances, condition_on=None):
    if dataset_path.endswith('.feather'):
        import feather
        df = feather.read_dataframe(dataset_path)
    elif dataset_path.endswith('.arff'):
        with open(dataset_path, 'r') as fp:
            dataset = arff.load(fp)
            # see if there is meta_data
            fp.seek(0)
            try:
                first_line = fp.readline()
                meta_data = json.loads(first_line[1:])
            except json.decoder.JSONDecodeError:
                meta_data = None
        columns = [column_name for column_name, colum_type in dataset['attributes']]
        df = pd.DataFrame(data=dataset['data'], columns=columns)
        if meta_data is not None:
            config_space = get_meta_data_config_space(meta_data)
            df = cast_columns_of_dataframe(df, params, config_space)
    else:
        raise ValueError()
    print(openmldefaults.utils.get_time(), 'Original data frame dimensions:', df.shape)

    for param in params:
        if param not in df.columns.values:
            raise ValueError('Param column not found. Columns %s, illegal: %s' % (df.columns.values, param))

    if resized_grid_size is not None:
        df = openmldefaults.utils.reshape_configs(df, params, resized_grid_size)

    print_columns(df, params)

    # remove values that are not according to the condition
    if condition_on is not None:
        for column, value in condition_on.items():
            df = df.loc[df[column] == value]

    # always set the index
    df = df.set_index(params)

    num_obs, num_tasks = df.shape
    if flip_performances:
        for i in range(num_obs):
            for j in range(num_tasks):
                df.iloc[i, j] = -1 * df.iloc[i, j]

    return df
