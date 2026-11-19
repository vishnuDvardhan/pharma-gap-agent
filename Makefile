.PHONY: install test eval demo demo-multi index serve docker

install:
	pip install -e ".[anthropic,openai,bedrock,vectordb,dev]"

test:
	pytest -q

eval:
	python eval/eval_retrieval.py
	python eval/eval_gaps.py --min-gap-recall 0.95

demo:
	gapagent check zelvorin_global --markets US,IN,JP
	gapagent ask "What blocks a launch in India and Japan?" --trace

demo-multi:
	gapagent ask --multi --trace "What blocks a launch in India and Japan?"

index:
	gapagent index
	gapagent search "translation into Japanese" --dossier zelvorin_global -k 3

serve:
	gapagent serve --port 8000

docker:
	docker compose up --build
