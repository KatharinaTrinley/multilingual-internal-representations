#!/bin/bash

pip install datasets tiktoken sentencepiece protobuf
pip install -U transformers
#pip install transformers==4.49.0

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1028'

#SAVE_BASE_DIR="./save_activations_chinesellama"
#MODEL_NAME="hfl/chinese-llama-2-7b"
MODEL_NAME="CohereForAI/aya-23-8B"
# validation split
SPLIT="train"
MAX_SAMPLES=100
FP16_FLAG="--fp16"

# All EC40 language pairs (both directions with English)
"""
PAIRS=(
    # High resource (5M+): de, nl, fr, es, ru, cs, hi, bn, ar, he
    #'en-de' 'en-nl' 'en-fr' 'en-es' 'en-ru' 'en-cs' 'en-hi' 'en-bn' 'en-ar' 'en-he'
    'de-en' 'nl-en' 'fr-en' 'es-en' 'ru-en' 'cs-en' 'hi-en' 'bn-en' 'ar-en' 'he-en'
    
    # Medium resource (1M): sv, da, it, pt, pl, bg, kn, mr, mt, ha
    #'en-sv' 'en-da' 'en-it' 'en-pt' 'en-pl' 'en-bg' 'en-kn' 'en-mr' 'en-mt' 'en-ha'
    'sv-en' 'da-en' 'it-en' 'pt-en' 'pl-en' 'bg-en' 'kn-en' 'mr-en' 'mt-en' 'ha-en'
    
    # Low resource (100k): af, lb, ro, oc, uk, sr, sd, gu, ti, am
    #'en-af' 'en-lb' 'en-ro' 'en-oc' 'en-uk' 'en-sr' 'en-sd' 'en-gu' 'en-ti' 'en-am'
    'af-en' 'lb-en' 'ro-en' 'oc-en' 'uk-en' 'sr-en' 'sd-en' 'gu-en' 'ti-en' 'am-en'
    
    # Extremely low resource (50k): no, is, ast, ca, be, bs, ne, ur, kab, so
    #'en-no' 'en-is' 'en-ast' 'en-ca' 'en-be' 'en-bs' 'en-ne' 'en-ur' 'en-kab' 'en-so'
    'no-en' 'is-en' 'ast-en' 'ca-en' 'be-en' 'bs-en' 'ne-en' 'ur-en' 'kab-en' 'so-en'
)
"""

#for wmt24
"""
PAIRS=(
    #'en-ar_EG' 'en-ar_SA' 'en-bg_BG' 'en-bn_IN' 'en-ca_ES' 'en-cs_CZ' 'en-da_DK' 'en-de_DE' 'en-el_GR' 'en-es_MX' 'en-et_EE' 'en-fa_IR' 'en-fi_FI' 'en-fil_PH' 'en-fr_CA' 'en-fr_FR' 'en-gu_IN' 'en-he_IL' 'en-hi_IN' 'en-hr_HR' 'en-hu_HU' 'en-id_ID' 'en-is_IS' 'en-it_IT' 'en-ja_JP' 'en-kn_IN' 'en-ko_KR' 'en-lt_LT' 'en-lv_LV' 'en-ml_IN' 'en-mr_IN' 'en-nl_NL' 'en-no_NO' 'en-pa_IN' 'en-pl_PL' 'en-pt_BR' 'en-pt_PT' 'en-ro_RO' 'en-ru_RU' 'en-sk_SK' 'en-sl_SI' 'en-sr_RS' 'en-sv_SE' 'en-sw_KE' 'en-sw_TZ' 'en-ta_IN' 'en-te_IN' 'en-th_TH' 'en-tr_TR' 'en-uk_UA' 'en-ur_PK' 'en-vi_VN' 'en-zh_CN' 'en-zh_TW' 'en-zu_ZA'
    'ar_EG-en' 'ar_SA-en' 'bg_BG-en' 'bn_IN-en' 'ca_ES-en' 'cs_CZ-en' 'da_DK-en' 'de_DE-en' 'el_GR-en' 'es_MX-en' 'et_EE-en' 'fa_IR-en' 'fi_FI-en' 'fil_PH-en' 'fr_CA-en' 'fr_FR-en' 'gu_IN-en' 'he_IL-en' 'hi_IN-en' 'hr_HR-en' 'hu_HU-en' 'id_ID-en' 'is_IS-en' 'it_IT-en' 'ja_JP-en' 'kn_IN-en' 'ko_KR-en' 'lt_LT-en' 'lv_LV-en' 'ml_IN-en' 'mr_IN-en' 'nl_NL-en' 'no_NO-en' 'pa_IN-en' 'pl_PL-en' 'pt_BR-en' 'pt_PT-en' 'ro_RO-en' 'ru_RU-en' 'sk_SK-en' 'sl_SI-en' 'sr_RS-en' 'sv_SE-en' 'sw_KE-en' 'sw_TZ-en' 'ta_IN-en' 'te_IN-en' 'th_TH-en' 'tr_TR-en' 'uk_UA-en' 'ur_PK-en' 'vi_VN-en' 'zh_CN-en' 'zh_TW-en' 'zu_ZA-en'
)
"""

#for codemixing
PAIRS=(
    #'en-ar_EG' 'en-ar_SA' 'en-bg_BG' 'en-bn_IN' 'en-ca_ES' 'en-cs_CZ' 'en-da_DK' 'en-de_DE' 'en-el_GR' 'en-es_MX' 'en-et_EE' 'en-fa_IR' 'en-fi_FI' 'en-fil_PH' 'en-fr_CA' 'en-fr_FR' 'en-gu_IN' 'en-he_IL' 'en-hi_IN' 'en-hr_HR' 'en-hu_HU' 'en-id_ID' 'en-is_IS' 'en-it_IT' 'en-ja_JP' 'en-kn_IN' 'en-ko_KR' 'en-lt_LT' 'en-lv_LV' 'en-ml_IN' 'en-mr_IN' 'en-nl_NL' 'en-no_NO' 'en-pa_IN' 'en-pl_PL' 'en-pt_BR' 'en-pt_PT' 'en-ro_RO' 'en-ru_RU' 'en-sk_SK' 'en-sl_SI' 'en-sr_RS' 'en-sv_SE' 'en-sw_KE' 'en-sw_TZ' 'en-ta_IN' 'en-te_IN' 'en-th_TH' 'en-tr_TR' 'en-uk_UA' 'en-ur_PK' 'en-vi_VN' 'en-zh_CN' 'en-zh_TW' 'en-zu_ZA'
    #codemixing
    'fr_en_0.25-en' 'fr_en_0.5-en' 'fr_en_0.75-en' 'fr_es_0.25-en' 'fr_es_0.5-en' 'fr_es_0.75-en' 'fr_it_0.25-en' 'fr_it_0.5-en' 'fr_it_0.75-en' 'fr_ja_0.25-en' 'fr_ja_0.5-en' 'fr_ja_0.75-en' 'fr_ko_0.25-en' 'fr_ko_0.5-en' 'fr_ko_0.75-en' 'zh_en_0.25-en' 'zh_en_0.5-en' 'zh_en_0.75-en' 'zh_ja_0.25-en' 'zh_ja_0.5-en' 'zh_ja_0.75-en' 'zh_ko_0.25-en' 'zh_ko_0.5-en' 'zh_ko_0.75-en' 'zh_es_0.25-en' 'zh_es_0.5-en' 'zh_es_0.75-en' 'zh_it_0.25-en' 'zh_it_0.5-en' 'zh_it_0.75-en'
    #non codemixing
    'fr_FR-en' 'zh_CN-en'
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
    python machine_translation.py \
        --model_name "$MODEL_NAME" \
        --output_dir "aya_output" \
        --language_pair "$pair" \
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

#second time

for pair in "${PAIRS[@]}"; do
    echo "$(date): Processing language pair: $pair" | tee -a $LOG_FILE
    
    # Run the script for this language pair
    python machine_translation.py \
        --model_name "hfl/chinese-llama-2-7b" \
        --output_dir "chinesellama_output" \
        --language_pair "$pair" \
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

#third

for pair in "${PAIRS[@]}"; do
    echo "$(date): Processing language pair: $pair" | tee -a $LOG_FILE
    
    # Run the script for this language pair
    python machine_translation.py \
        --model_name "meta-llama/Llama-3.1-8B" \
        --output_dir "llama3_output" \
        --language_pair "$pair" \
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
