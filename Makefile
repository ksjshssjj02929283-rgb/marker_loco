.PHONY: venv install test demo

venv:
	python3 -m venv .venv
	@echo "run: source .venv/bin/activate && make install"

install:
	python3 -m pip install -e ".[dev,d455]"

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q

demo:
	python3 scripts/generate_demo.py --output-dir outputs/demo
	python3 scripts/run_video.py --config configs/demo_calibration.json --video outputs/demo/demo_input.avi --output-dir outputs/demo/result
	python3 scripts/evaluate_demo.py --ground-truth outputs/demo/ground_truth.csv --estimates outputs/demo/result/tcp_poses.csv --output outputs/demo/result/evaluation.json
