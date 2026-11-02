PYTHON ?= python3
VENV ?= .venv
DEMO_DIR ?= outputs/demo

.PHONY: venv install install-d455 install-t265 install-reporting git-check test demo package-check check

venv:
	$(PYTHON) -m venv "$(VENV)"
	@echo "run: source $(VENV)/bin/activate && make install"

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-d455:
	$(PYTHON) -m pip install -e ".[dev,d455]"

install-t265:
	$(PYTHON) -m pip install -e ".[dev,t265]"

install-reporting:
	$(PYTHON) -m pip install -e ".[dev,reporting]"

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q

git-check:
	$(PYTHON) scripts/check_git_hygiene.py

demo:
	$(PYTHON) -m marker6dof.commands.demo.generate_demo --config configs/demo_calibration.json --output-dir "$(DEMO_DIR)"
	$(PYTHON) -m marker6dof.commands.tracking.run_video --config configs/demo_calibration.json --video "$(DEMO_DIR)/demo_input.avi" --output-dir "$(DEMO_DIR)/result" --no-annotated-video
	$(PYTHON) -m marker6dof.commands.demo.evaluate_demo --ground-truth "$(DEMO_DIR)/ground_truth.csv" --estimates "$(DEMO_DIR)/result/tcp_poses.csv" --output "$(DEMO_DIR)/result/evaluation.json"

package-check:
	$(PYTHON) tools/check_package.py

check: git-check test demo package-check
