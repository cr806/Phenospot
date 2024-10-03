ENV = ./venv
PYTHON = ${ENV}/bin/python3
PIP = ${ENV}/bin/pip

.PHONY: run clean

run: venv setup
	${PYTHON} app.py

setup: venv requirements.txt
	${PIP} install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf ${ENV}

venv:
	python3 -m venv ${ENV}

freeze:
	${PIP} freeze > requirements.txt

# Source: https://earthly.dev/blog/python-makefile/
