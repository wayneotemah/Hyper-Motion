PYTHON ?= .venv/bin/python
EXPERIMENT ?= demo_sign_test
CONFIG ?= experiments/$(EXPERIMENT)/config.yaml

.PHONY: bootstrap verify prep package remote

bootstrap:
	./scripts/bootstrap_mac.sh

verify:
	$(PYTHON) scripts/verify_env.py

prep:
	$(PYTHON) scripts/run_local_prep.py --config $(CONFIG)

package:
	$(PYTHON) scripts/package_experiment.py --config $(CONFIG)

remote:
	$(PYTHON) scripts/run_remote_stub.py --config $(CONFIG) --package
