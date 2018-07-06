import ConfigSpace
import numpy as np
import openmldefaults
import scipy

from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete


class OpenMLDistributionHelper(object):
    def _cdf(self, x, *args):
        raise NotImplementedError()

    def _sf(self, x, *args):
        raise NotImplementedError()

    def _ppf(self, q, *args):
        raise NotImplementedError()

    def _isf(self, q, *args):
        raise NotImplementedError()

    def _stats(self, *args, **kwds):
        raise NotImplementedError()

    def _munp(self, n, *args):
        raise NotImplementedError()

    def _entropy(self, *args):
        raise NotImplementedError()


class loguniform_gen(OpenMLDistributionHelper, rv_continuous):
    def _pdf(self, x, base, low, high):
        raise NotImplementedError()

    def _argcheck(self, base, low, high):
        self.base = base
        self.a = low
        self.b = high
        return (high > low) and low > 0 and high > 0 and base >= 2

    def logspace(self, num):
        start = np.log(self.a) / np.log(self.base)
        stop = np.log(self.b) / np.log(self.base)
        return np.logspace(start, stop, num=num, endpoint=True, base=self.base)

    def _rvs(self, base, low, high):
        low = np.log(low) / np.log(base)
        high = np.log(high) / np.log(base)
        return np.power(self.base,
                        self._random_state.uniform(low=low, high=high,
                                                   size=self._size))
loguniform = loguniform_gen(name='loguniform')


class loguniform_int_gen(OpenMLDistributionHelper, rv_discrete):
    def _pmf(self, x, base, low, high):
        raise NotImplementedError()

    def _argcheck(self, base, low, high):
        self.base = base
        self.a = low
        self.b = high
        return (high > low) and low >= 1 and high >= 1 and base >= 2

    def _rvs(self, base, low, high):
        assert self.a >= 1
        low = np.log(low - 0.4999) / np.log(base)
        high = np.log(high + 0.4999) / np.log(base)
        return np.rint(np.power(base, self._random_state.uniform(
            low=low, high=high, size=self._size))).astype(int)
loguniform_int = loguniform_int_gen(name='loguniform_int')


def config_space_to_dist(ConfigurationSpace):
    """
    Turns config space object into a dict of distributions
    """
    result = dict()
    for hyperparameter in ConfigurationSpace.get_hyperparameters():
        if isinstance(hyperparameter, ConfigSpace.hyperparameters.CategoricalHyperparameter):
            result[hyperparameter.name] = [openmldefaults.config_spaces.reinstantiate_parameter_value(openmldefaults.config_spaces.post_process(val)) for val in hyperparameter.choices]
        elif isinstance(hyperparameter, ConfigSpace.hyperparameters.UniformFloatHyperparameter):
            if hyperparameter.log is True:
                result[hyperparameter.name] = loguniform(2, hyperparameter.lower, hyperparameter.upper)
            else:
                result[hyperparameter.name] = scipy.stats.uniform(loc=hyperparameter.lower, scale=hyperparameter.upper - hyperparameter.lower)
        elif isinstance(hyperparameter, ConfigSpace.hyperparameters.UniformIntegerHyperparameter):
            if hyperparameter.log is True:
                raise NotImplementedError()
            else:
                result[hyperparameter.name] = scipy.stats.randint(hyperparameter.lower, hyperparameter.upper + 1)
        elif isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter):
            result[hyperparameter.name] = [openmldefaults.config_spaces.reinstantiate_parameter_value(openmldefaults.config_spaces.post_process(hyperparameter.value))]
        elif isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant):
            result[hyperparameter.name] = [openmldefaults.config_spaces.reinstantiate_parameter_value(openmldefaults.config_spaces.post_process(hyperparameter.value))]
        else:
            raise NotImplementedError()
    return result
