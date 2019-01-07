from . import svc
from . import gradient_boosting

import openmldefaults
import sklearnbot
import typing


def get_config_space(classifier_name: str, random_seed: int, space_type: typing.Optional[str]):
    if space_type is None:
        sklearnbot.config_spaces.get_config_space(classifier_name, random_seed)
    else:
        cs = getattr(getattr(openmldefaults.config_spaces, classifier_name), 'get_hyperparameter_search_space_%s' % space_type)
        return cs(random_seed)
