from pathlib import Path
import itertools
from tqdm import tqdm

from sagemaker.pytorch import PyTorch

from benchmarking.nursery.benchmark_dehb.benchmark_main import parse_args
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
)
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string


if __name__ == "__main__":
    args, method_names, benchmark_names, _ = parse_args()
    experiment_tag = args.experiment_tag
    suffix = random_string(4)

    combinations = list(itertools.product(method_names, benchmark_names))
    for method, benchmark_name in tqdm(combinations):
        name = method + "-" + benchmark_name
        sm_args = dict(
            entry_point="benchmark_main.py",
            source_dir=str(Path(__file__).parent),
            checkpoint_s3_uri=s3_experiment_path(
                tuner_name=name, experiment_name=experiment_tag
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

        print(f"{experiment_tag}-{name}")
        sm_args["hyperparameters"] = {
            "experiment_tag": experiment_tag,
            "num_seeds": args.num_seeds,
            "method": method,
            "benchmark": benchmark_name,
        }
        est = PyTorch(**sm_args)
        est.fit(job_name=f"{experiment_tag}-{method}-{suffix}", wait=False)
