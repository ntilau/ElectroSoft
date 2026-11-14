PY_VENV := .venv

.PHONY: all help setup test

all: setup

help:
	@echo "=== ElectroSoft — Python port of FES ==="
	@echo "  make setup       Create .venv and install the package"
	@echo "  make test        Run the pytest suite"
	@echo ""

setup:
	./configure

test: $(PY_VENV)
	$(abspath $(PY_VENV))/bin/python -m pytest tests/ -v

$(PY_VENV):
	./configure
