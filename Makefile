remove-venv:
	rm -rf .venv

create-venv:
	python -m venv .venv; \
	. .venv/bin/activate; \
    python -m pip install -U pip; \

dependencies:
	. .venv/bin/activate; \
	pip install -r requirements.txt --no-cache-dir

dependencies-dev:
	. .venv/bin/activate; \
	pip install -r requirements-dev.txt --no-cache-dir

install: remove-venv create-venv dependencies

build:
	. .venv/bin/activate; \
	python -m build

build-install:
	. .venv/bin/activate; \
	pip install . --no-cache-dir; \
	rm -rf build dist ./**/*.egg-info