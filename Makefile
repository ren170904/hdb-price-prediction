#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = hdb_price_prediction
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	pipenv --python $(PYTHON_VERSION)
	@echo ">>> New pipenv created. Activate with:\npipenv shell"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Fetch raw HDB data from data.gov.sg
.PHONY: fetch
fetch:
	$(PYTHON_INTERPRETER) -m sg_hdb_price_analysis.data.fetch

## Fetch MRT stations & bus stops from OpenStreetMap
.PHONY: transit
transit:
	$(PYTHON_INTERPRETER) -m sg_hdb_price_analysis.data.transit

## Geocode unique HDB addresses via OneMap
.PHONY: geocode
geocode:
	$(PYTHON_INTERPRETER) -m sg_hdb_price_analysis.data.geocode

## Train LightGBM and RandomForest models
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m sg_hdb_price_analysis.models.train

## Execute all analysis notebooks in order (writes outputs in place)
.PHONY: notebooks
notebooks:
	for nb in notebooks/0*.ipynb; do \
		jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$$nb" || exit 1; \
	done

## Launch Streamlit dashboard
.PHONY: dashboard
dashboard:
	streamlit run app/dashboard.py

## Launch FastAPI web app (price check + deal finder) on http://localhost:8000
.PHONY: webapp
webapp:
	$(PYTHON_INTERPRETER) -m uvicorn app.api:app --port 8000

## Full pipeline: fetch → transit → geocode → train → dashboard
.PHONY: all
all: fetch transit geocode train dashboard



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
