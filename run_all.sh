#!/bin/bash
# Run all 4 models × 7 datasets for Group A Image Segmentation.
# Usage:  bash run_all.sh            (train + test all)
#         bash run_all.sh test       (test-only, requires trained weights)

MODE=${1:-train}

DATASETS=("BUS-BRA" "BUSI" "BUSIS" "CAMUS" "DDTI" "Fetal_HC" "KidneyUS")
MODELS=("U_Net" "R2U_Net" "AttU_Net" "R2AttU_Net")

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ""
        echo "=============================================="
        echo "  ${MODE}  |  ${model}  on  ${dataset}"
        echo "=============================================="
        python3 main.py \
            --dataset "$dataset" \
            --model_type "$model" \
            --mode "$MODE"
    done
done

echo ""
echo "All experiments finished."
