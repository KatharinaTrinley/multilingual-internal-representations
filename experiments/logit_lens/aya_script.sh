#!/bin/bash

MODEL_ID="CohereForAI/aya-23-8B"
SCRIPT="logit_lens.py"
DATA_DIR="./saved_data"

for FILE in "$DATA_DIR"/*.json; do
    # Extract filename only, e.g., "translation2_en2zh.json"
    BASENAME=$(basename "$FILE")
    
    # Remove "translation2_" prefix and ".json" suffix to get e.g. "en2zh"
    LANG_PAIR=${BASENAME#translation2_}
    LANG_PAIR=${LANG_PAIR%.json}
    
    # Create task name: e.g., "transen2zh_aya"
    TASK_NAME="trans${LANG_PAIR}_aya"

    echo "Processing $FILE with task name $TASK_NAME"

    python "$SCRIPT" \
        --model_id "$MODEL_ID" \
        --task_name "$TASK_NAME" \
        --json_path "$FILE"
done