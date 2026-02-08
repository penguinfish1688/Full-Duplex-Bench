.PHONY: help install prepare-asr run-personaplex run-moshi evaluate clean

PYTHON := python3
EVAL_DIR := ./evaluation
ASR_DIR := ./get_transcript
MODEL_DIR := ./model_inference
DATA_DIR := ./data

help:
	@echo "Full-Duplex-Bench Workflow"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. Edit model_inference/personaplex/config.yaml with your settings"
	@echo "  2. make install"
	@echo "  3. make run-personaplex"
	@echo "  4. make evaluate"
	@echo ""
	@echo "Targets:"
	@echo "  make install           Install dependencies (PyYAML required)"
	@echo "  make prepare-asr       Generate ASR-aligned transcripts"
	@echo "  make run-personaplex   Run Personaplex inference (reads config.yaml)"
	@echo "  make run-moshi         Run Moshi inference"
	@echo "  make evaluate          Run all evaluations (behavior, features, timing)"
	@echo "  make clean             Remove outputs and ASR files"
	@echo ""


# ──────────────────────────────────────────────────────────────────────────────
# Setup & Installation
# ──────────────────────────────────────────────────────────────────────────────

install:
	$(PYTHON) -m pip install --upgrade pip
 	$(PYTHON) -m pip install -r requirements.txt

prepare-asr:
	$(PYTHON) $(ASR_DIR)/asr.py --root_dir $(DATA_DIR)

run-personaplex:
	cd $(MODEL_DIR)/personaplex && $(PYTHON) inference.py

run-moshi:
	cd $(MODEL_DIR)/moshi && $(PYTHON) inference.py

evaluate:
	$(PYTHON) $(EVAL_DIR)/evaluate.py --task behavior
	$(PYTHON) $(EVAL_DIR)/evaluate.py --task general_before_after
	$(PYTHON) $(EVAL_DIR)/get_timing.py

clean:
	find . -name "output.wav" -delete
	find . -name "output.json" -delete

