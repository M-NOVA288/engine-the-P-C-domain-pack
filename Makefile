.PHONY: setup data train evals test demo roi api all

setup:
	pip install -r requirements.txt

data:
	python -m data.synth.generate --cases 60 --policies 6000 --seed 7

train:
	python -m models.train

evals:
	python -m evals.run_evals --gate

test:
	pytest -q

demo:
	python -m copilot.cli data/submissions/CASE-0030

roi:
	python -m roi.roi --referral-rate 0.35

api:
	uvicorn api.main:app --reload

all: data train evals test
