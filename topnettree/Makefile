.PHONY: test cv-fast cv-full

test:
	python -m pytest -q

cv-fast:
	python scripts/run_cv.py --config configs/fast.yaml

cv-full:
	python scripts/run_cv.py --config configs/default.yaml
