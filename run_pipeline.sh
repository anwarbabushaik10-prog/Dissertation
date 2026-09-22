#!/usr/bin/env bash
# End-to-end reproduction of the Explainable Plant Disease Detection prototype.
# Run from the project root: ./run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
cd src
export PYTHONUNBUFFERED=1

echo "=== 1. Prepare stratified splits + class imbalance report ==="
python prepare_splits.py 2>&1 | tee ../outputs/metrics/prepare_splits.log

echo "=== 2. Audit rotation/flip-invariant perceptual duplicates ==="
python audit_splits.py 2>&1 | tee ../outputs/metrics/audit_splits.log

echo "=== 3. Train CNN (MobileNetV2 transfer learning) ==="
python train.py --model cnn 2>&1 | tee ../outputs/metrics/cnn_train.log

echo "=== 4. Train ViT (ViT-Tiny) ==="
python train.py --model vit 2>&1 | tee ../outputs/metrics/vit_train.log

echo "=== 5. Evaluate CNN on test set ==="
python evaluate.py --model cnn 2>&1 | tee ../outputs/metrics/cnn_evaluate.log

echo "=== 6. Evaluate ViT on test set ==="
python evaluate.py --model vit 2>&1 | tee ../outputs/metrics/vit_evaluate.log

echo "=== 7.M + LIME explanations + quantitative quality (CNN) ==="
python explain.py --model cnn 2>&1 | tee ../outputs/metrics/cnn_explain.log

echo "=== 8. Grad-CAM + LIME explanations + quantitative quality (ViT) ==="
python explain.py --model vit 2>&1 | tee ../outputs/metrics/vit_explain.log

echo "=== 9. Export compact model comparison ==="
python summarize_results.py 2>&1 | tee ../outputs/metrics/summarize_results.log

echo "=== Done. See outputs/ for figures, metrics and explanations. ==="
