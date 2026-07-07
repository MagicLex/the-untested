"""Clone the featurize env: python-feature-pipeline base + RDKit.

F3 (featurize-molecules) turns SMILES into fingerprints via chem_features.py,
which needs RDKit; the base feature-pipeline env has none. Idempotent.

    python3 tools/build_envs.py
"""
from pathlib import Path

import hopsworks

NAME = "untested-chem-env"
BASE = "python-feature-pipeline"
_here = Path(__file__).resolve()
ROOT_REL = str(Path(str(_here).split("/hopsfs/", 1)[1]).parent.parent)


def main():
    proj = hopsworks.login()
    env_api = proj.get_environment_api()
    env = env_api.get_environment(NAME)
    if env is None:
        env = env_api.create_environment(NAME, base_environment_name=BASE)
        print(f"cloned {NAME} from {BASE}", flush=True)
    env.install_requirements(f"{ROOT_REL}/requirements-featurize.txt",
                             await_installation=True)
    print(f"installed featurize deps into {NAME}", flush=True)


if __name__ == "__main__":
    main()
