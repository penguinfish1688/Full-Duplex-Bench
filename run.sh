#!/bin/bash
# Full-Duplex-Bench workflow
# Assumes venv is already activated before running this script
# Usage: bash run.sh <target>

set -e

EVAL_DIR="./evaluation"
ASR_DIR="./get_transcript"
MODEL_DIR="./model_inference"
DATA_DIR="./data"
PYTHON="python"

show_help() {
    echo "Full-Duplex-Bench Workflow"
    echo ""
    echo "Usage: source ~/.venv_fdb/bin/activate && bash run.sh <target>"
    echo ""
    echo "Targets:"
    echo "  help               Show this help message"
    echo "  install            Install dependencies"
    echo "  prepare-asr        Generate ASR-aligned transcripts"
    echo "  run-personaplex    Run Personaplex inference (reads config.yaml)"
    echo "  run-moshi          Run Moshi inference"
    echo "  evaluate           Run all evaluations"
    echo "  clean              Remove outputs and ASR files"
    echo ""
}

install_deps() {
    echo "Installing dependencies..."
    $PYTHON -m pip install --upgrade pip
    $PYTHON -m pip install -r requirements-inference.txt
    echo "✓ Installation complete"
}

prepare_asr() {
    echo "Preparing ASR-aligned transcripts..."
    $PYTHON "$ASR_DIR/asr.py" --root_dir "$DATA_DIR"
    echo "✓ ASR preparation complete"
}

run_personaplex() {
    echo "Running Personaplex inference..."
    cd "$MODEL_DIR/personaplex"
    $PYTHON inference.py
    cd - > /dev/null
    echo "✓ Personaplex inference complete"
}

run_moshi() {
    echo "Running Moshi inference..."
    cd "$MODEL_DIR/moshi"
    $PYTHON inference.py
    cd - > /dev/null
    echo "✓ Moshi inference complete"
}

evaluate_all() {
    echo "Running all evaluations..."
    $PYTHON "$EVAL_DIR/evaluate.py" --task behavior
    $PYTHON "$EVAL_DIR/evaluate.py" --task general_before_after
    $PYTHON "$EVAL_DIR/get_timing.py"
    echo "✓ All evaluations complete"
}

clean_outputs() {
    echo "Cleaning outputs and ASR files..."
    find . -name "output.wav" -delete
    find . -name "output.json" -delete
    echo "✓ Cleanup complete"
}

TARGET="${1:-help}"

case "$TARGET" in
    help)
        show_help
        ;;
    install)
        install_deps
        ;;
    prepare-asr)
        prepare_asr
        ;;
    run-personaplex)
        run_personaplex
        ;;
    run-moshi)
        run_moshi
        ;;
    evaluate)
        evaluate_all
        ;;
    clean)
        clean_outputs
        ;;
    *)
        echo "Unknown target: $TARGET"
        show_help
        exit 1
        ;;
esac
