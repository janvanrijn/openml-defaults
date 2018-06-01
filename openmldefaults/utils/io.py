import arff
import feather
import numpy as np
import openmldefaults
import pandas as pd


def get_setup_dirname(resized_grid_size, num_defaults):
    return 'c%d_d%d' % (resized_grid_size, num_defaults)


def print_columns(df, params):
    for param in params:
        unique = np.array(df[param].unique())
        print(openmldefaults.utils.get_time(), '%s unique values: %s (%d)' % (param, unique, len(unique)))


def load_dataset(dataset_path, params, resized_grid_size, flip_performances):
    if dataset_path.endswith('.feather'):
        df = feather.read_dataframe(dataset_path)
    elif dataset_path.endswith('.arff'):
        with open(dataset_path, 'r') as fp:
            dataset = arff.load(fp)
        columns = [column_name for column_name, colum_type in dataset['attributes']]
        df = pd.DataFrame(data=dataset['data'], columns=columns)
    else:
        raise ValueError()
    print(openmldefaults.utils.get_time(), 'Original data frame dimensions:', df.shape)

    for param in params:
        if param not in df.columns.values:
            raise ValueError('Param column not found. Columns %s, illegal: %s' % (df.columns.values, param))

    if resized_grid_size is not None:
        df = openmldefaults.utils.reshape_configs(df, params, resized_grid_size)

    print_columns(df, params)

    # always set the index
    df = df.set_index(params)

    num_obs, num_tasks = df.shape
    if flip_performances:
        for i in range(num_obs):
            for j in range(num_tasks):
                df.iloc[i, j] = -1 * df.iloc[i, j]

    return df
