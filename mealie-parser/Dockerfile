# Web UI + run pods for k3s: the UI (webui/app.py) starts Jobs that run mealie_parser.py --jsonl.
# Built on Mealie's own image, so runs use the parser code of the Mealie being tuned: keep the
# tag in step with dean-tofu k3s/apps/prod/mealie. Build from the repo root:
#   docker build -f mealie-parser/Dockerfile .
FROM ghcr.io/mealie-recipes/mealie:v3.30.0@sha256:90f83da6f3ab0a421efc827118a556168573930105f13fa3321768f42163d784 AS base
# DATA_DIR: Mealie creates its data directories on import; send them to the writable /tmp.
# LOG_LEVEL: Mealie's logger would print every HTTP request the runner makes.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 DATA_DIR=/tmp/mealie-data LOG_LEVEL=warning
WORKDIR /stress
COPY stresskit/ /stress/stresskit/
COPY mealie-parser/mealie_parser.py mealie-parser/engine.py mealie-parser/evalstore.py /stress/
# The labelled lines are built from a pinned dataset, not committed: both checksums are checked.
COPY mealie-parser/gold/make_gold.py /stress/gold/
RUN curl -fsSL -o /tmp/training.sqlite3 \
      https://raw.githubusercontent.com/strangetom/ingredient-parser/7c4c47a8cd8fbf3e13873fd1996b99985167b844/train/data/training.sqlite3 \
    && echo "f4bef553ddd9ae734fd8a0c8ee9b299606071cc5e9ad9fa9aa96fac5f3339a99  /tmp/training.sqlite3" | sha256sum -c - \
    && python /stress/gold/make_gold.py /tmp/training.sqlite3 /stress/gold/strangetom.jsonl \
    && echo "e510944658d69da6fda498f64fa66dd05d2157fef0f3ca76ff2a1601ba8ed30d  /stress/gold/strangetom.jsonl" | sha256sum -c - \
    && rm /tmp/training.sqlite3
COPY mealie-parser/webui/ /stress/webui/

FROM base AS test
COPY mealie-parser/tests/ /stress/tests/
RUN python -m unittest discover -s tests -v && touch /tests-passed

FROM base
COPY --from=test /tests-passed /tests-passed
USER 65534:65534
EXPOSE 8080
ENTRYPOINT []
CMD ["python", "/stress/webui/app.py"]
