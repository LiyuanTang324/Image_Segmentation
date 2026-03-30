#!/bin/bash
# Run all 4 models × 7 datasets for Group A Image Segmentation.
# Usage:  bash run_all.sh [SEED]
#   e.g.  bash run_all.sh          (seed=42)
#         bash run_all.sh 123      (seed=123)

SEED=${1:-42}

DATASETS=("BUS-BRA" "BUSI" "BUSIS" "CAMUS" "DDTI" "Fetal_HC" "KidneyUS")
MODELS=("U_Net" "R2U_Net" "AttU_Net" "R2AttU_Net")

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ""
        echo "=============================================="
        echo "  ${model}  on  ${dataset}  |  seed=${SEED}"
        echo "=============================================="
        python3 main.py \
            --dataset "$dataset" \
            --model_type "$model" \
            --mode train \
            --seed "$SEED" \
            --output_root "./output/seed_${SEED}"
    done
done

echo ""
echo "All experiments finished."
