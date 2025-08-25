#!/bin/bash

pip install datasets tiktoken sentencepiece protobuf
pip install -U transformers
#pip install transformers==4.49.0

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1028'

SAVE_BASE_DIR="./save_activations_chinesellama"
MODEL_NAME="hfl/chinese-llama-2-7b"
# validation split
SPLIT="train"
MAX_SAMPLES=30000
FP16_FLAG="--fp16"

mkdir -p $SAVE_BASE_DIR



#for codemixing
PAIRS=(
    #'en-ar_EG' 'en-ar_SA' 'en-bg_BG' 'en-bn_IN' 'en-ca_ES' 'en-cs_CZ' 'en-da_DK' 'en-de_DE' 'en-el_GR' 'en-es_MX' 'en-et_EE' 'en-fa_IR' 'en-fi_FI' 'en-fil_PH' 'en-fr_CA' 'en-fr_FR' 'en-gu_IN' 'en-he_IL' 'en-hi_IN' 'en-hr_HR' 'en-hu_HU' 'en-id_ID' 'en-is_IS' 'en-it_IT' 'en-ja_JP' 'en-kn_IN' 'en-ko_KR' 'en-lt_LT' 'en-lv_LV' 'en-ml_IN' 'en-mr_IN' 'en-nl_NL' 'en-no_NO' 'en-pa_IN' 'en-pl_PL' 'en-pt_BR' 'en-pt_PT' 'en-ro_RO' 'en-ru_RU' 'en-sk_SK' 'en-sl_SI' 'en-sr_RS' 'en-sv_SE' 'en-sw_KE' 'en-sw_TZ' 'en-ta_IN' 'en-te_IN' 'en-th_TH' 'en-tr_TR' 'en-uk_UA' 'en-ur_PK' 'en-vi_VN' 'en-zh_CN' 'en-zh_TW' 'en-zu_ZA'
    #codemixing
    'fr_en_0.25-en' 'fr_en_0.5-en' 'fr_en_0.75-en' 'fr_es_0.25-en' 'fr_es_0.5-en' 'fr_es_0.75-en' 'fr_it_0.25-en' 'fr_it_0.5-en' 'fr_it_0.75-en' 'fr_ja_0.25-en' 'fr_ja_0.5-en' 'fr_ja_0.75-en' 'fr_ko_0.25-en' 'fr_ko_0.5-en' 'fr_ko_0.75-en' 'zh_en_0.25-en' 'zh_en_0.5-en' 'zh_en_0.75-en' 'zh_ja_0.25-en' 'zh_ja_0.5-en' 'zh_ja_0.75-en' 'zh_ko_0.25-en' 'zh_ko_0.5-en' 'zh_ko_0.75-en' 'zh_es_0.25-en' 'zh_es_0.5-en' 'zh_es_0.75-en' 'zh_it_0.25-en' 'zh_it_0.5-en' 'zh_it_0.75-en'
    #non codemixing
    'fr_FR-en' 'it_IT-en' 'ja_JP-en' 'ko_KR-en' 'es_MX-en' 'zh_CN-en'
    )

# all output can be found here in the log file
LOG_FILE="activations_collection.log"

echo "Starting neuron activation collection for ${#PAIRS[@]} language pairs" | tee -a $LOG_FILE
echo "Results will be saved to: $SAVE_BASE_DIR" | tee -a $LOG_FILE
echo "=====================================================" | tee -a $LOG_FILE

# we loop through all the pairs and run the script for each one
for pair in "${PAIRS[@]}"; do
    echo "$(date): Processing language pair: $pair" | tee -a $LOG_FILE
    
    # Run the script for this language pair
    python get_neurons_wmt24_codemixed.py \
        --model_name "$MODEL_NAME" \
        --save_path "$SAVE_BASE_DIR" \
        --language_pair "$pair" \
        --split "$SPLIT" \
        --max_samples "$MAX_SAMPLES" \
        $FP16_FLAG 2>&1 | tee -a "$LOG_FILE"
    
    # error handling
    if [ $? -eq 0 ]; then
        echo "$(date): Successfully processed $pair" | tee -a $LOG_FILE
    else
        echo "$(date): ERROR processing $pair" | tee -a $LOG_FILE
    fi
    
    echo "-------------------------------------------------" | tee -a $LOG_FILE
    sleep 5
done

echo "$(date): All language pairs processed. Check $LOG_FILE for details." | tee -a $LOG_FILE
