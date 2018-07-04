
def selected_row_to_config_dict(df, row_idx):
    values = df.index[row_idx]
    if len(values) > 1:
        keys = df.index.names
    else:
        keys = [df.index.name]
    if not isinstance(keys, list):
        raise ValueError('data frame index not interpreted properly')
    if len(keys) != len(values):
        raise ValueError()
    result = {keys[i]: values[i] for i in range(len(values))}
    return result
