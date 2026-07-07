"""Deploy the amrscorer online inference endpoint: on-demand QSAR for any SMILES.

Points at the amr_qsar champion bundle, uploads predictor.py next to it, and
deploys a custom PYTHON predictor on untested-serve-env. Recreates over a stale
deployment. Smoke-tests with quinine (should score high on P. falciparum).
"""
import os

import hopsworks
from hsml.resources import PredictorResources, Resources
from hsml.scaling_config import PredictorScalingConfig, ScaleMetric

NAME = "amrscorer"
ENV = "untested-serve-env"
_here = os.path.dirname(os.path.abspath(__file__))


def main():
    proj = hopsworks.login()
    mr = proj.get_model_registry()
    try:
        model = mr.get_best_model("amr_qsar", "mean_amr_auc", "max")
    except Exception:
        model = mr.get_model("amr_qsar", version=1)
    print(f"serving amr_qsar v{model.version}", flush=True)

    script_dir = f"/Projects/{proj.name}/Models/{model.name}/{model.version}/Files"
    proj.get_dataset_api().upload(f"{_here}/predictor.py", script_dir, overwrite=True)

    ms = proj.get_model_serving()
    for d in ms.get_deployments():
        if d.name == NAME:
            d.delete()
            print("removed stale deployment", flush=True)

    deployment = model.deploy(
        name=NAME,
        description="On-demand AMR QSAR: SMILES -> per-target activity + "
                    "applicability-domain confidence.",
        script_file=f"{script_dir}/predictor.py",
        environment=ENV,
        resources=PredictorResources(
            requests=Resources(cores=1, memory=2048, gpus=0),
            limits=Resources(cores=2, memory=4096, gpus=0)),
        scaling_configuration=PredictorScalingConfig(
            min_instances=1, max_instances=2,
            scale_metric=ScaleMetric.CONCURRENCY, target=10))
    deployment.start(await_running=600)
    print(f"deployment {NAME} running: {deployment.is_running()}", flush=True)

    res = deployment.predict(inputs=[{"smiles": "COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1"}])
    print("SMOKE quinine ->", res, flush=True)


if __name__ == "__main__":
    main()
