# the-untested -- FTI on Hopsworks: predict bioactivity of never-tested natural products
# Feature (LOTUS map + ChEMBL labels + molecule fingerprints) -> Training (multi-task QSAR + applicability domain) -> Inference (batch map + KServe + certainty-map app)
FEAT_ENV = python-feature-pipeline
CHEM_ENV = untested-chem-env
TRAIN_ENV = untested-train-env

envs:                ## clone the RDKit featurize env (feature base + rdkit)
	python3 tools/build_envs.py

lotus-job:           ## deploy + schedule LOTUS ingest (plant<->molecule + structures + taxonomy)
	hops job deploy untested-lotus pipelines/lotus_pipeline.py --env $(FEAT_ENV) --overwrite
	python3 tools/schedule.py untested-lotus "0 0 3 1 * ?" --run

chembl-job:          ## deploy + schedule ChEMBL bioactivity-label ingest (bulk SQLite)
	hops job deploy untested-chembl pipelines/chembl_pipeline.py --env $(FEAT_ENV) --overwrite
	python3 tools/schedule.py untested-chembl "0 0 4 1 * ?" --run

features-job:        ## deploy the molecule featurizer (RDKit fingerprints -> molecule_features)
	hops job deploy untested-features pipelines/features_pipeline.py --env $(CHEM_ENV) --overwrite
	python3 tools/schedule.py untested-features "0 0 5 2 * ?" --run

labels-job:          ## deploy the panel label pivot (compound_activity -> wide compound_labels)
	hops job deploy untested-labels pipelines/labels_pipeline.py --env $(FEAT_ENV) --overwrite

train-job:           ## deploy + schedule the AMR QSAR training (stage-1 bar; every run registered)
	hops job deploy untested-train pipelines/train.py --env $(TRAIN_ENV) --overwrite
	python3 tools/schedule.py untested-train "0 40 2 ? * *"

map-job:             ## deploy + schedule the batch map (score untested naturals -> plant_property_map)
	hops job deploy untested-map pipelines/map_pipeline.py --env pandas-training-pipeline --overwrite
	python3 tools/schedule.py untested-map "0 20 3 ? * *"

smoke-lotus:         ## run the LOTUS ingest from the terminal pod
	python3 pipelines/lotus_pipeline.py
smoke-chembl:        ## run the ChEMBL ingest from the terminal pod
	python3 pipelines/chembl_pipeline.py

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/  --/'
.PHONY: envs lotus-job chembl-job labels-job features-job train-job map-job smoke-lotus smoke-chembl help
