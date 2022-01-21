from numbers import Number

import pandas as pd
from typing import Dict, Optional, Callable, List, Tuple, Union
import numpy as np


class Blackbox:

    def __init__(
            self,
            configuration_space: Dict,
            fidelity_space: Optional[Dict] = None,
            fidelity_values: Optional[np.array] = None,
            objectives_names: Optional[List[str]] = None,
    ):
        """
        Interface aiming at following [HPOBench](https://github.com/automl/HPOBench) for compatibility.
        """
        self.configuration_space = configuration_space
        self.fidelity_space = fidelity_space
        self.fidelity_values = fidelity_values
        self.objectives_names = objectives_names

    def objective_function(
            self,
            configuration: Dict,
            fidelity: Union[Dict, Number] = None,
            seed: Optional[int] = None
    ) -> Dict:
        """
        Returns an evaluation of the blackbox, first perform data check and then call `_objective_function` that should
        be overriden in the child class.
        :param configuration: configuration to be evaluated, should belong to `self.configuration_space`
        :param fidelity: not passing a fidelity is possible if either the blackbox does not have a fidelity space
        or if it has a single fidelity in its fidelity space. In the latter case, all fidelities are returned in form
        of a tensor with shape (num_fidelities, num_objectives).
        :param seed:
        :return: dictionary of objectives evaluated or tensor with shape (num_fidelities, num_objectives) if no fidelity
        was given.
        """
        if self.fidelity_space is None:
            assert fidelity is None
        else:
            if fidelity is None:
                assert len(self.fidelity_space) == 1, \
                    "not passing a fidelity is only supported when only one fidelity is present."

        if isinstance(fidelity, Number):
            # allows to call
            # `objective_function(configuration=..., fidelity=2)`
            # instead of
            # `objective_function(configuration=..., {'num_epochs': 2})`
            fidelity_names = list(self.fidelity_space.keys())
            assert len(fidelity_names) == 1, \
                "passing numeric value is only possible when there is a single fidelity in the fidelity space."
            fidelity = {fidelity_names[0]: fidelity}

        # todo check configuration/fidelity matches their space
        return self._objective_function(
            configuration=configuration,
            fidelity=fidelity,
            seed=seed,
        )

    def _objective_function(
            self,
            configuration: Dict,
            fidelity: Optional[Dict] = None,
            seed: Optional[int] = None
    ) -> Dict:
        """
        Override this function to provide your benchmark function.
        """
        pass

    def __call__(self, *args, **kwargs) -> Dict:
        """
        Allows to call blackbox directly as a function rather than having to call the specific method.
        :return:
        """
        return self.objective_function(*args, **kwargs)

    def hyperparameter_objectives_values(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :return: a tuple of two dataframes, the first one contains hyperparameters values and the second
        one contains objective values, this is used when fitting a surrogate model.
        """
        pass


def from_function(
        configuration_space: Dict,
        eval_fun: Callable,
        fidelity_space: Optional[Dict] = None,
        fidelity_values = None,
        objectives_names: Optional[List[str]] = None,
):
    """
    Helper to create a blackbox from a function, useful for test or to wrap-up real blackbox functions.
    :param configuration_space:
    :param eval_fun: function that returns dictionary of objectives given configuration and fidelity
    :param fidelity_space:
    :return:
    """
    class BB(Blackbox):
        def __init__(self):
            super(BB, self).__init__(
                configuration_space=configuration_space,
                fidelity_space=fidelity_space,
                fidelity_values=fidelity_values,
                objectives_names=objectives_names,
            )

        def objective_function(
            self,
            configuration: Dict,
            fidelity: Optional[Dict] = None,
            seed: Optional[int] = None
    ) -> Dict:
            return eval_fun(configuration, fidelity, seed)

    return BB()