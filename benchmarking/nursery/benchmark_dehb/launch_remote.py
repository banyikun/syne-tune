from argparse import ArgumentParser
from pathlib import Path

from sagemaker.pytorch import PyTorch

from benchmarking.nursery.benchmark_dehb.baselines import methods, Methods
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
)
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag",
        type=str,
        required=True,
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    hash = random_string(4)

    for method in methods.keys():
        sm_args = dict(
            entry_point="benchmark_main.py",
            source_dir=str(Path(__file__).parent),
            checkpoint_s3_uri=s3_experiment_path(
                tuner_name=method, experiment_name=experiment_tag
            ),
            instance_type="ml.c5.4xlarge",
            instance_count=1,
            py_version="py38",
            framework_version="1.10.0",
            max_run=3600 * 72,
            role=get_execution_role(),
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
        )

        print(f"{experiment_tag}-{method}")
        sm_args["hyperparameters"] = {
            "experiment_tag": experiment_tag,
            "num_seeds": 30,
            "method": method,
        }
        est = PyTorch(**sm_args)
        est.fit(job_name=f"{experiment_tag}-{method}-{hash}", wait=False)
