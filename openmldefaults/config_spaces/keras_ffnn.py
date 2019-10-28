import ConfigSpace


def get_hyperparameter_search_space(seed):
    """
    Ranger config space

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurationsrpart.py

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('mlr.ranger', seed)

    # fixed to deviance, as exponential requires two classes
    imputer = ConfigSpace.CategoricalHyperparameter(
        name='num.impute.selected.cpo', choices=['impute.hist', 'impute.median', 'impute.mean'], default_value='impute.hist')
    epochs = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=128)
    optimizer = ConfigSpace.CategoricalHyperparameter(
        name = "optimizer", choices = ['sgd', 'adam','rmsprop'])
    lr = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='lr', lower=0.0001, upper=1, log=True)
    decay = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='decay', lower= 0, upper=1, log=True)
    momentum = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='momentum', lower= 0, upper=1, log=True)
    layers = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='layers', lower=1, upper=4)
    batchnorm_dropout = ConfigSpace.CategoricalHyperparameter(
        name="batchnorm_dropout", choices=['batchnorm', 'dropout'])
    dropout_rate = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='dropout_rate', lower=0, upper=1, log=True)
    input_dropout_rate = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='input_dropout_rate', lower=0, upper=1, log=True)
    units_layer_1 = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='units_layer_1', lower=8, upper=512)
    units_layer_2 = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='units_layer_2', lower=8, upper=512)
    units_layer_3 = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='units_layer_3', lower=8, upper=512)
    units_layer_4 = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='units_layer_4', lower=8, upper=512)
    act_layer = ConfigSpace.CategoricalHyperparameter(
        name="activation", choices=['relu', 'tanh'])
    init_layer = ConfigSpace.CategoricalHyperparameter(
        name="init_layer", choices=['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])
    l1_reg_layer = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='l1_reg_layer', lower= 0, upper=0.04, log=True)
    l2_reg_layer = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='l2_reg_layer', lower= 0, upper=0.04, log=True)
    learning_rate_scheduler = ConfigSpace.CategoricalHyperparameter(
        name = "learning_rate_scheduler", choices = ['TRUE', 'FALSE'])
    init_seed = ConfigSpace.CategoricalHyperparameter(
        name="init_seed", choices=[1, 11, 101, 131, 499])

    cs.add_hyperparameters([
        imputer,
        epochs,
        optimizer,
        lr,
        decay,
        momentum,
        layers,
        batchnorm_dropout,
        input_dropout_rate,
        dropout_rate,
        units_layer_1,
        units_layer_2,
        units_layer_3,
        units_layer_4,
        act_layer,
        init_layer,
        l1_reg_layer,
        l2_reg_layer,
        learning_rate_scheduler,
        init_seed
        ])

    opt_moment = ConfigSpace.InCondition(momentum, optimizer, ['sgd'])
    cs.add_condition(opt_moment)
    l2 = ConfigSpace.GreaterThanCondition(units_layer_2, layers, 1)
    l3 = ConfigSpace.GreaterThanCondition(units_layer_3, layers, 2)
    l4 = ConfigSpace.GreaterThanCondition(units_layer_4, layers, 3)
    cs.add_condition(l2)
    cs.add_condition(l3)
    cs.add_condition(l4)
    return cs
