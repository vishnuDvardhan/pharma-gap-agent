.PHONY: install test eval demo serve docker

install:
	pip install -e ".[anthropic,openai,bedrock,dev]"

test:
	pytest -q

eval:
	python eval/eval_retrieval.py
	python eval/eval_gaps.py --min-gap-recall 0.95

demo:
	gapagent check zelvorin_global --markets US,IN,JP
	gapagent ask "What blocks a launch in India and Japan?" --trace

serve:
	gapagent serve --port 8000

docker:
	docker compose up --build
