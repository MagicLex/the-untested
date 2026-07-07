"""Clone the project envs. Idempotent.

    python3 tools/build_envs.py [chem|train|all]   (default all)

  untested-chem-env    python-feature-pipeline + RDKit   (F3 featurize)
  untested-train-env   pandas-training-pipeline + RDKit  (T train: scaffold
                       split needs RDKit, sklearn/matplotlib come from the base)
"""
import sys
from pathlib import Path

import hopsworks

_here = Path(__file__).resolve()
ROOT_REL = str(Path(str(_here).split("/hopsfs/", 1)[1]).parent.parent)

ENVS = {
    "chem": ("untested-chem-env", "python-feature-pipeline",
             f"{ROOT_REL}/requirements-featurize.txt"),
    "train": ("untested-train-env", "pandas-training-pipeline",
              f"{ROOT_REL}/requirements-train.txt"),
}


def build(env_api, name, base, reqs):
    env = env_api.get_environment(name)
    if env is None:
        env = env_api.create_environment(name, base_environment_name=base)
        print(f"cloned {name} from {base}", flush=True)
    env.install_requirements(reqs, await_installation=True)
    print(f"installed deps into {name}", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = list(ENVS) if which == "all" else [which]
    env_api = hopsworks.login().get_environment_api()
    for k in keys:
        build(env_api, *ENVS[k])


if __name__ == "__main__":
    main()
