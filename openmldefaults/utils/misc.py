import ConfigSpace


def remove_hyperparameter(config_space: ConfigSpace.ConfigurationSpace,
                          hyperparameter_name: str) -> ConfigSpace.ConfigurationSpace:
    config_space_prime = ConfigSpace.ConfigurationSpace(meta=config_space.meta)
    for hyperparameter in config_space.get_hyperparameters():
        if hyperparameter.name != hyperparameter_name:
            config_space_prime.add_hyperparameter(hyperparameter)
    for condition in config_space.get_conditions():
        if condition.parent.name != hyperparameter_name and condition.child.name != hyperparameter_name:
            config_space_prime.add_condition(condition)
        else:
            raise ValueError()
    return config_space_prime
