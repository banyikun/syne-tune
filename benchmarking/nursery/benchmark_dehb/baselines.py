from dataclasses import dataclass
from typing import Dict, Optional

from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.synchronous import (
    SynchronousGeometricHyperbandScheduler,
    GeometricDifferentialEvolutionHyperbandScheduler,
)
from syne_tune.config_space import (
    Categorical,
    ordinal,
)


@dataclass
class MethodArguments:
    config_space: Dict
    metric: str
    mode: str
    random_seed: int
    max_resource_attr: str
    resource_attr: str
    num_brackets: Optional[int] = None


class Methods:
    ASHA = "ASHA"
    SYNC_HB = "SYNC_HB"
    DEHB = "DEHB"
    BOHB = "BOHB"
    ASHA_ORD = "ASHA_ORD"
    SYNC_HB_ORD = "SYNC_HB_ORD"
    DEHB_ORD = "DEHB_ORD"
    BOHB_ORD = "BOHB_ORD"


def _convert_categorical_to_ordinal(args: MethodArguments) -> dict:
    config_space = args.config_space
    return {
        name: (
            ordinal(domain.categories) if isinstance(domain, Categorical) else domain
        )
        for name, domain in config_space.items()
    }


methods = {
    Methods.ASHA: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        type="promotion",
        # search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.SYNC_HB: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        # search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.DEHB: lambda method_arguments: GeometricDifferentialEvolutionHyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random_encoded",
        # search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.BOHB: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        # search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.ASHA_ORD: lambda method_arguments: HyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="random",
        type="promotion",
        # search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.SYNC_HB_ORD: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="random",
        # search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.DEHB_ORD: lambda method_arguments: GeometricDifferentialEvolutionHyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="random_encoded",
        # search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.BOHB_ORD: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="kde",
        # search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
}


# TODO
if __name__ == "__main__":
    # Run a loop that initializes all schedulers on all benchmark to see if they all work

    benchmarks = ["fcnet-protein", "nas201-cifar10", "lcbench-Fashion-MNIST"]
    for benchmark_name in benchmarks:
        benchmark = benchmark_definitions[benchmark_name]
        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            time_this_resource_attr=benchmark.time_this_resource_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
        )
        for method_name, method_fun in methods.items():
            print(f"checking initialization of: {method_name}, {benchmark_name}")
            scheduler = method_fun(
                MethodArguments(
                    config_space=backend.blackbox.configuration_space,
                    metric=benchmark.metric,
                    mode=benchmark.mode,
                    random_seed=0,
                    max_t=max(backend.blackbox.fidelity_values),
                    resource_attr=next(iter(backend.blackbox.fidelity_space.keys())),
                    transfer_learning_evaluations=get_transfer_learning_evaluations(
                        blackbox_name=benchmark.blackbox_name,
                        test_task=benchmark.dataset_name,
                        datasets=benchmark.datasets,
                    ),
                    use_surrogates=benchmark_name == "lcbench-Fashion-MNIST",
                )
            )
            scheduler.suggest(0)
            scheduler.suggest(1)
