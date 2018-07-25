import arff
import ConfigSpace
import feather
import json
import numpy as np
import openmldefaults
import pandas as pd


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


def cast_columns_of_dataframe(df, params, meta_data):
    config_space = get_meta_data_config_space(meta_data)
    for param in params:
        hyperparameter = config_space.get_hyperparameter(param)

        if isinstance(hyperparameter, ConfigSpace.UniformIntegerHyperparameter) or \
                (isinstance(hyperparameter, ConfigSpace.Constant) and isinstance(hyperparameter.value, int)) or \
                (isinstance(hyperparameter, ConfigSpace.UnParametrizedHyperparameter) and isinstance(hyperparameter.value, int)):
            df[param] = df[param].astype(int)
    return df


def load_dataset(dataset_path, params, resized_grid_size, flip_performances, condition_on=None):
    if dataset_path.endswith('.feather'):
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
            df = cast_columns_of_dataframe(df, params, meta_data)
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
