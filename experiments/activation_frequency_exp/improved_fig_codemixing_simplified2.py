import os
import pickle
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.patches as mpatches
import copy
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations

#source: https://github.com/Smu-Tan/Neuron-Specialization/tree/main

def load_activations(base_path, layer='layer_0'):
    """Loads layer activations from all .pkl files in subfolders matching 'en-xx'."""
    activations = {}
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir() and os.path.basename(f.path).endswith('-en')]
    #print("debug subfolders:",subfolders)
    for subfolder in subfolders:
        lang_dir = os.path.basename(subfolder)
        pkl_files = glob(os.path.join(subfolder, "*.pkl"))
        #print("debug pkl_files:", pkl_files)
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                if layer in data:
                    activations[lang_dir] = data[layer].cpu().numpy()
    
    return activations

def get_specialized_neurons(activations, threshold=0.9):
    """Identifies specialized neurons per language direction based on cumulative activation threshold."""
    specialized_neurons = {}
    for lang, activation in activations.items():
        sorted_indices = np.argsort(-np.abs(activation))  # Sort by absolute activation magnitude
        cumsum = np.cumsum(np.abs(activation[sorted_indices]))
        total_sum = cumsum[-1]
        selected_neurons = sorted_indices[cumsum / total_sum <= threshold]
        specialized_neurons[lang] = set(selected_neurons)
    
    return specialized_neurons

def remove_shared_neurons(specialized_neurons):
    """Removes neurons that are shared across all languages from each language's specialized neuron set."""
    all_langs = list(specialized_neurons.values())
    if not all_langs:
        return specialized_neurons  # Edge case: empty input

    shared_neurons = set.intersection(*all_langs)
    
    filtered_neurons = {
        lang: neurons - shared_neurons
        for lang, neurons in specialized_neurons.items()
    }
    
    return filtered_neurons

def compute_iou(specialized_neurons):
    specialized_neurons = remove_shared_neurons(specialized_neurons)
    """Computes IoU scores between specialized neuron sets of different language directions."""
    languages = list(specialized_neurons.keys())
    num_langs = len(languages)
    iou_matrix = np.zeros((num_langs, num_langs))
    
    for i in range(num_langs):
        for j in range(num_langs):
            set_i = specialized_neurons[languages[i]]
            set_j = specialized_neurons[languages[j]]
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    
    languages = [lang.replace('-en', '') for lang in list(specialized_neurons.keys())]

    return languages, iou_matrix

def compute_intersection_union_table(specialized_neurons):
    specialized_neurons = remove_shared_neurons(specialized_neurons)
    languages = list(specialized_neurons.keys())
    num_langs = len(languages)
    table = np.empty((num_langs, num_langs), dtype=object)

    for i in range(num_langs):
        for j in range(num_langs):
            set_i = specialized_neurons[languages[i]]
            set_j = specialized_neurons[languages[j]]
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            table[i, j] = f"{intersection}/{union}" if union > 0 else "0/0"

    languages = [lang.replace('-en', '') for lang in languages]
    return languages, table

def save_intersection_union_table(languages, table, output_path="intersection_union_table.csv"):
    df = pd.DataFrame(table, index=languages, columns=languages)
    df.to_csv(output_path)
    print(f"Table saved to {output_path}")


language_families = {
    # Afro-Asiatic
    'ar_EG': 'Afro-Asiatic', 'ar_SA': 'Afro-Asiatic', 'he_IL': 'Afro-Asiatic',

    # Slavic
    'bg_BG': 'Slavic', 'cs_CZ': 'Slavic', 'hr_HR': 'Slavic', 'hu_HU': 'Slavic',
    'pl_PL': 'Slavic', 'ro_RO': 'Slavic', 'ru_RU': 'Slavic', 'sk_SK': 'Slavic',
    'sl_SI': 'Slavic', 'sr_RS': 'Slavic', 'uk_UA': 'Slavic',

    # Indo-Aryan
    'bn_IN': 'Indo-Aryan', 'gu_IN': 'Indo-Aryan', 'hi_IN': 'Indo-Aryan',
    'kn_IN': 'Indo-Aryan', 'ml_IN': 'Indo-Aryan', 'mr_IN': 'Indo-Aryan',
    'pa_IN': 'Indo-Aryan', 'ta_IN': 'Indo-Aryan', 'te_IN': 'Indo-Aryan',
    'ur_PK': 'Indo-Aryan',

    # Romance
    'ca_ES': 'Romance', 'es_MX': 'Romance', 'fr_CA': 'Romance', 'fr_FR': 'Romance',
    'it_IT': 'Romance', 'pt_BR': 'Romance', 'pt_PT': 'Romance', 'ro_RO': 'Romance',

    # Germanic
    'da_DK': 'Germanic', 'de_DE': 'Germanic', 'is_IS': 'Germanic', 'nl_NL': 'Germanic',
    'no_NO': 'Germanic', 'sv_SE': 'Germanic',

    # Uralic
    'et_EE': 'Uralic', 'fi_FI': 'Uralic', 'hu_HU': 'Uralic',

    # Indo-Iranian
    'fa_IR': 'Indo-Iranian',

    # Austroasiatic
    'vi_VN': 'Austroasiatic',

    # Austronesian
    'fil_PH': 'Austronesian',

    # Japonic
    'ja_JP': 'Japonic',

    # Koreanic
    'ko_KR': 'Koreanic',

    # Turkic
    'tr_TR': 'Turkic',

    # Sino-Tibetan
    'zh_CN': 'Sino-Tibetan', 'zh_TW': 'Sino-Tibetan',

    # Tai-Kadai
    'th_TH': 'Tai-Kadai',

    # Bantu
    'sw_KE': 'Bantu', 'sw_TZ': 'Bantu', 'zu_ZA': 'Bantu',

    # Code-Mixing
    **{lang: 'codemixing' for lang in [
        'fr_en_0.25', 'fr_it_0.25', 'fr_ko_0.25', 'zh_es_0.25', 'zh_ja_0.25',
        'fr_en_0.5', 'fr_it_0.5', 'fr_ko_0.5', 'zh_es_0.5', 'zh_ja_0.5',
        'fr_en_0.75', 'fr_it_0.75', 'fr_ko_0.75', 'zh_es_0.75', 'zh_ja_0.75',
        'fr_es_0.25', 'fr_ja_0.25', 'zh_en_0.25', 'zh_it_0.25', 'zh_ko_0.25',
        'fr_es_0.5', 'fr_ja_0.5', 'zh_en_0.5', 'zh_it_0.5', 'zh_ko_0.5',
        'fr_es_0.75', 'fr_ja_0.75', 'zh_en_0.75', 'zh_it_0.75', 'zh_ko_0.75'
    ]}
}

ec40_resourcedness = {
    # High resource (5M)
    'de_DE': 'High', 'nl_NL': 'High', 'fr_FR': 'High', 'es_MX': 'High', 'ru_RU': 'High',
    'cs_CZ': 'High', 'hi_IN': 'High', 'bn_IN': 'High', 'ar_EG': 'High', 'ar_SA': 'High', 'he_IL': 'High',

    # Medium resource (1M)
    'sv_SE': 'Medium', 'da_DK': 'Medium', 'it_IT': 'Medium', 'pt_BR': 'Medium', 'pt_PT': 'Medium',
    'pl_PL': 'Medium', 'bg_BG': 'Medium', 'kn_IN': 'Medium', 'mr_IN': 'Medium',

    # Low resource (100k)
    'ro_RO': 'Low', 'uk_UA': 'Low', 'sr_RS': 'Low', 'gu_IN': 'Low',

    # Extremely-Low resource (50k)
    'no_NO': 'Extremely-Low', 'is_IS': 'Extremely-Low', 'ca_ES': 'Extremely-Low',
    'ur_PK': 'Extremely-Low',

    # Code-Mixing languages at High
    **{lang: 'High' for lang in [
        'fr_en_0.25', 'fr_it_0.25', 'fr_ko_0.25', 'zh_es_0.25', 'zh_ja_0.25',
        'fr_en_0.5', 'fr_it_0.5', 'fr_ko_0.5', 'zh_es_0.5', 'zh_ja_0.5',
        'fr_en_0.75', 'fr_it_0.75', 'fr_ko_0.75', 'zh_es_0.75', 'zh_ja_0.75',
        'fr_es_0.25', 'fr_ja_0.25', 'zh_en_0.25', 'zh_it_0.25', 'zh_ko_0.25',
        'fr_es_0.5', 'fr_ja_0.5', 'zh_en_0.5', 'zh_it_0.5', 'zh_ko_0.5',
        'fr_es_0.75', 'fr_ja_0.75', 'zh_en_0.75', 'zh_it_0.75', 'zh_ko_0.75'
    ]}
}

aya_languages = {
    'ar_EG', 'ar_SA',
    'zh_CN', 'zh_TW',
    'cs_CZ',
    'nl_NL',
    'fr_FR', 'fr_CA',
    'de_DE',
    'el_GR',
    'he_IL',
    'hi_IN',
    'id_ID',
    'it_IT',
    'ja_JP',
    'ko_KR',
    'fa_IR',
    'pl_PL',
    'pt_PT', 'pt_BR',
    'ro_RO',
    'ru_RU',
    'es_ES', 'es_MX',
    'tr_TR',
    'uk_UA',
    'vi_VN',
    # Code-Mixing
    *[lang for lang in [
        'fr_en_0.25', 'fr_it_0.25', 'fr_ko_0.25', 'zh_es_0.25', 'zh_ja_0.25',
        'fr_en_0.5', 'fr_it_0.5', 'fr_ko_0.5', 'zh_es_0.5', 'zh_ja_0.5',
        'fr_en_0.75', 'fr_it_0.75', 'fr_ko_0.75', 'zh_es_0.75', 'zh_ja_0.75',
        'fr_es_0.25', 'fr_ja_0.25', 'zh_en_0.25', 'zh_it_0.25', 'zh_ko_0.25',
        'fr_es_0.5', 'fr_ja_0.5', 'zh_en_0.5', 'zh_it_0.5', 'zh_ko_0.5',
        'fr_es_0.75', 'fr_ja_0.75', 'zh_en_0.75', 'zh_it_0.75', 'zh_ko_0.75'
    ]]
}

simpler_languages = {
    'zh_CN',
    'fr_FR',
    'it_IT',
    'ja_JP',
    'ko_KR',
    'es_ES', 'es_MX',
    # Code-Mixing
    *[lang for lang in [
        'fr_en_0.25', 'fr_it_0.25', 'fr_ko_0.25', 'zh_es_0.25', 'zh_ja_0.25',
        'fr_en_0.5', 'fr_it_0.5', 'fr_ko_0.5', 'zh_es_0.5', 'zh_ja_0.5',
        'fr_en_0.75', 'fr_it_0.75', 'fr_ko_0.75', 'zh_es_0.75', 'zh_ja_0.75',
        'fr_es_0.25', 'fr_ja_0.25', 'zh_en_0.25', 'zh_it_0.25', 'zh_ko_0.25',
        'fr_es_0.5', 'fr_ja_0.5', 'zh_en_0.5', 'zh_it_0.5', 'zh_ko_0.5',
        'fr_es_0.75', 'fr_ja_0.75', 'zh_en_0.75', 'zh_it_0.75', 'zh_ko_0.75'
    ]]
}


hardcoded_language_order = [
    'ar_EG', 'ar_SA', 'he_IL',
    'ru_RU', 'cs_CZ',
    'bg_BG', 'uk_UA', 'sr_RS', 'sk_SK', 'sl_SI', 'hr_HR', 'hu_HU', 'pl_PL',
    'hi_IN', 'bn_IN',
    'gu_IN', 'kn_IN', 'ml_IN', 'mr_IN', 'pa_IN', 'ta_IN', 'te_IN', 'ur_PK',
    'fr_FR', 'es_MX',
    'ca_ES', 'fr_CA', 'it_IT', 'pt_BR', 'pt_PT', 'ro_RO',
    'de_DE', 'nl_NL',
    'sv_SE', 'da_DK', 'no_NO', 'is_IS',
    'fi_FI', 'et_EE',
    'fa_IR',
    'zh_CN', 'zh_TW',
    'ja_JP', 'ko_KR', 'tr_TR',
    'vi_VN',
    'fil_PH',
    'th_TH',
    'sw_KE', 'sw_TZ', 'zu_ZA',
    # Code-Mixing
    *[lang for lang in [
    'fr_en_0.25', 'fr_en_0.5', 'fr_en_0.75',
    'fr_es_0.25', 'fr_es_0.5', 'fr_es_0.75',
    'fr_it_0.25', 'fr_it_0.5', 'fr_it_0.75',
    'fr_ja_0.25', 'fr_ja_0.5', 'fr_ja_0.75',
    'fr_ko_0.25', 'fr_ko_0.5', 'fr_ko_0.75',

    'zh_en_0.25', 'zh_en_0.5', 'zh_en_0.75',
    'zh_es_0.25', 'zh_es_0.5', 'zh_es_0.75',
    'zh_it_0.25', 'zh_it_0.5', 'zh_it_0.75',
    'zh_ja_0.25', 'zh_ja_0.5', 'zh_ja_0.75',
    'zh_ko_0.25', 'zh_ko_0.5', 'zh_ko_0.75'
    ]]
]



def reorder_iou_matrix_by_hardcoded_order(languages, iou_matrix):
    """Reorders the IoU matrix to match the hardcoded language order."""
    # Find which languages from the hardcoded order are available in the data
    available_languages = []
    for lang in hardcoded_language_order:
        if lang in languages:
            available_languages.append(lang)
    
    # If no languages match, return the original data
    if not available_languages:
        return languages, iou_matrix
    
    # Create a mapping of language to index in the original matrix
    lang_to_index = {lang: idx for idx, lang in enumerate(languages)}
    
    # Get indices of available languages in the original matrix
    indices = [lang_to_index[lang] for lang in available_languages if lang in lang_to_index]
    
    # Reorder the matrix
    reordered_iou_matrix = iou_matrix[np.ix_(indices, indices)]
    
    return available_languages, reordered_iou_matrix

def plot_iou_heatmap(languages, iou_matrix, layer):
    """Plots a heatmap of the IoU scores with color-coded language families and resourcedness."""
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Process values for better visualization
    df = iou_matrix * 100  # Convert to percentage
    df = df.astype(int)
    df_ori = copy.deepcopy(df)
    #df = df ** 1.25 #original intensity
    df = df_ori ** 0.5
    df = (df-df.min())/(df.max()-df.min()) if df.max() != df.min() else df
    df = df * 100
    
    # Create heatmap with darker colors
    sns.heatmap(df, cmap='Blues', ax=ax, xticklabels=languages, yticklabels=languages, 
                linewidths=1, vmin=0, vmax=50)
    
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_title(f"{layer} - Language Neuron Specialization IoU (wmt24pp and codemixing)", fontsize=18, pad=20)
    
    # Set tick labels with proper rotation and color based on resourcedness
    x_tick_labels = ax.get_xticklabels()
    y_tick_labels = ax.get_yticklabels()
    
    for label in x_tick_labels:
        lang = label.get_text()
        # Set color based on presence in Aya and resourcedness
        if lang in aya_languages:
            # All languages in Aya are marked as high-resource (green)
            label.set_color('green')
            label.set_fontweight('bold')
        elif lang in ec40_resourcedness:
            resourcedness = ec40_resourcedness[lang]
            if resourcedness == 'High':
                label.set_color('green')
                label.set_fontweight('bold')
            elif resourcedness == 'Medium':
                label.set_color('orange')
            elif resourcedness == 'Low':
                label.set_color('red')
            elif resourcedness == 'Extremely-Low':
                label.set_color('red')
                label.set_alpha(0.7)  # Slightly more transparent for Extremely-Low
    
    for label in y_tick_labels:
        lang = label.get_text()
        # Set color based on presence in Aya and resourcedness
        if lang in aya_languages:
            # All languages in Aya are marked as high-resource (green)
            label.set_color('green')
            label.set_fontweight('bold')
        elif lang in ec40_resourcedness:
            resourcedness = ec40_resourcedness[lang]
            if resourcedness == 'High':
                label.set_color('green')
                label.set_fontweight('bold')
            elif resourcedness == 'Medium':
                label.set_color('orange')
            elif resourcedness == 'Low':
                label.set_color('red')
            elif resourcedness == 'Extremely-Low':
                label.set_color('red')
                label.set_alpha(0.7)  # Slightly more transparent for Extremely-Low
    
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(y_tick_labels, rotation=0, fontsize=10)

    ax.set_xlim(-1, len(languages))
    ax.set_ylim(len(languages), -1)
    
    # Define language family colors
    germanic_color = '#93c47d'
    romance_color = '#46bdc6'
    slavic_color = '#8e7cc3'
    aryan_color = '#c27ba0'
    afro_asiatic_color = '#ffd966'
    uralic_color = '#f6b26b'
    indo_iranian_color = '#e06666'
    austroasiatic_color = '#6d9eeb'
    austronesian_color = '#cfe2f3'
    japonic_color = '#d5a6bd'
    koreanic_color = '#a2c4c9'
    turkic_color = '#b4a7d6'
    sino_tibetan_color = '#b6d7a8'
    tai_kadai_color = '#ffe599'
    bantu_color = '#f9cb9c'
    codemixing_color = '#999999'  # Added color for code-mixing

    # Create family indices
    family_indices = {
        'Germanic': [],
        'Romance': [],
        'Slavic': [],
        'Indo-Aryan': [],
        'Afro-Asiatic': [],
        'Uralic': [],
        'Indo-Iranian': [],
        'Austroasiatic': [],
        'Austronesian': [],
        'Japonic': [],
        'Koreanic': [],
        'Turkic': [],
        'Sino-Tibetan': [],
        'Tai-Kadai': [],
        'Bantu': [],
        'codemixing': []  # Added codemixing
    }

    # Populate indices based on language family
    for i, lang in enumerate(languages):
        if lang in language_families:
            family = language_families[lang]
            family_indices[family].append(i)

    # Map families to colors
    family_colors = {
        #'Germanic': germanic_color,
        'Romance': romance_color,
        #'Slavic': slavic_color,
        #'Indo-Aryan': aryan_color,
        #'Afro-Asiatic': afro_asiatic_color,
        #'Uralic': uralic_color,
        #'Indo-Iranian': indo_iranian_color,
        #'Austroasiatic': austroasiatic_color,
        #'Austronesian': austronesian_color,
        'Japonic': japonic_color,
        'Koreanic': koreanic_color,
        #'Turkic': turkic_color,
        'Sino-Tibetan': sino_tibetan_color,
        #'Tai-Kadai': tai_kadai_color,
        #'Bantu': bantu_color,
        'Codemixing': codemixing_color  # Added codemixing
    }

    # Add colored rectangles for language families
    for family, indices in family_indices.items():
        color = family_colors.get(family, '#cccccc')
        for i in indices:
            ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=True, color=color))
            ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=True, color=color))
            ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=False, edgecolor='white'))
            ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=False, edgecolor='white'))

    # Family legend
    family_legend_labels = list(family_colors.keys())
    family_legend_colors = [family_colors[family] for family in family_legend_labels]
    family_legend_handles = [mpatches.Patch(color=color, label=label)
                             for label, color in zip(family_legend_labels, family_legend_colors)]

    # resourcedness legend
    resourcedness_legend_labels = [
        'High resource (HRL)', 
        'Medium resource (LRL)', 
        'Low resource (LRL)',
        'Extremely-Low resource (LRL)'
    ]
    resourcedness_legend_colors = ['green', 'orange', 'red', 'red']
    
    resourcedness_legend_handles = []
    for i, (color, label) in enumerate(zip(resourcedness_legend_colors, resourcedness_legend_labels)):
        if i == 3:  # Extremely-Low
            handle = plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=10, alpha=0.7, label=label)
        else:
            handle = plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=10, label=label)
        resourcedness_legend_handles.append(handle)
    
    # 2 legends
    ax.legend(handles=family_legend_handles, loc='upper left', 
                              bbox_to_anchor=(0.0, 1.15), ncol=5, 
                              title="Language Families", fontsize=10, title_fontsize=12)
    

    #plt.figtext(0.5, 0.01, ha='center', fontsize=10, 
                #bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'IoU_{layer}_wmt24pp_HRL_LRL.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def extract_indices(lang_list):
    langs = ['en', 'es', 'it', 'ja', 'ko']
    props = ['0.25', '0.5', '0.75']
    targets = [f"fr_{l}_{p}" for l in langs for p in props] + \
              [f"zh_{l}_{p}" for l in langs for p in props]
    # Use None for any missing tag to avoid ValueError
    indices = [lang_list.index(t) if t in lang_list else None for t in targets]
    return targets, indices

#paird t
def paired_by_layer_test(fr_lists, zh_lists):
    fr = np.squeeze(np.stack(fr_lists))   # (5, 32)
    zh = np.squeeze(np.stack(zh_lists))   # (5, 32)

    # Aggregate replicates -> per-layer means
    fr_mean = fr.mean(axis=0)   # (32,)
    zh_mean = zh.mean(axis=0)   # (32,)
    diff = fr_mean - zh_mean

    # Normality check of paired differences (n=32)
    shapiro_p = stats.shapiro(diff).pvalue

    if shapiro_p > 0.05:
        # Paired t-test
        t, p = stats.ttest_rel(fr_mean, zh_mean)
        # Paired Cohen's d
        d = diff.mean() / diff.std(ddof=1)
        test_used = "paired t-test"
        effect = d
    else:
        # Wilcoxon signed-rank
        w, p = stats.wilcoxon(fr_mean, zh_mean, zero_method="wilcox", alternative="two-sided")
        test_used = "Wilcoxon signed-rank"
        effect = None  # (optionally report median(diff))

    return {
        "test": test_used,
        "p_value": float(p),
        "normality_p": float(shapiro_p),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "paired_cohens_d": None if effect is None else float(effect)
    }

# ---- Robust regression option (counts; controls for layer & replicate) ----
def gee_count_model(fr_lists, zh_lists, family="nb"):
    # Build long DataFrame: y, lang, layer, replicate
    rows = []
    for r, arr in enumerate(fr_lists):
        for layer, y in enumerate(np.squeeze(arr)):
            rows.append({"y": int(y), "lang": "fr", "layer": layer, "rep": r})
    for r, arr in enumerate(zh_lists):
        for layer, y in enumerate(np.squeeze(arr)):
            rows.append({"y": int(y), "lang": "zh", "layer": layer, "rep": r})
    df = pd.DataFrame(rows)

    fam = sm.families.Poisson() if family == "poisson" else sm.families.NegativeBinomial()
    # Cluster by replicate; control for layer via fixed effects
    model = smf.gee("y ~ C(lang) + C(layer)", groups="rep", data=df, family=fam)
    res = model.fit()

    # Effect of language (rate ratio)
    beta = res.params.get("C(lang)[T.zh]", np.nan)
    rr = float(np.exp(beta)) if np.isfinite(beta) else np.nan
    return res.summary().as_text(), rr

#base_path = "./save_activations_llama3/save_activations_llama3"
#base_path = "./save_activations_chinesellama/save_activations_chinesellama"
base_path = "./save_activations_aya/save_activations_aya"

Lang_list = ['fr_FR', 'es_MX', 'it_IT', 'zh_CN', 'ja_JP', 'ko_KR', 'fr_en_0.25', 'fr_en_0.5', 'fr_en_0.75', 'fr_es_0.25', 'fr_es_0.5', 'fr_es_0.75', 'fr_it_0.25', 'fr_it_0.5', 'fr_it_0.75', 'fr_ja_0.25', 'fr_ja_0.5', 'fr_ja_0.75', 'fr_ko_0.25', 'fr_ko_0.5', 'fr_ko_0.75', 'zh_en_0.25', 'zh_en_0.5', 'zh_en_0.75', 'zh_es_0.25', 'zh_es_0.5', 'zh_es_0.75', 'zh_it_0.25', 'zh_it_0.5', 'zh_it_0.75', 'zh_ja_0.25', 'zh_ja_0.5', 'zh_ja_0.75', 'zh_ko_0.25', 'zh_ko_0.5', 'zh_ko_0.75']
iou_list = []
for layer_index in range(32):
    layer_name = f'layer_{layer_index}'  
    print(f"Processing {layer_name}...")
    
    # Load all activations
    activations = load_activations(base_path, layer=layer_name)
    #print("debug:",activations)
    # Filter to only include Aya languages
    activations = {lang: act for lang, act in activations.items() if lang.replace("-en", "") in simpler_languages}

    # Skip layer if no Aya languages are found
    if not activations:
        print(f"No Aya languages found in {layer_name}, skipping...")
        continue

    # Compute specialized neurons
    specialized_neurons = get_specialized_neurons(activations)

    # Compute IoU
    languages, iou_matrix = compute_iou(specialized_neurons)
    #debug plot_iou_heatmap(languages, iou_matrix, "debug"+layer_name)
    #languages, table = compute_intersection_union_table(specialized_neurons)
    
    # Reorder IoU matrix based on hardcoded order (if needed)
    languages, iou_matrix = reorder_iou_matrix_by_hardcoded_order(languages, iou_matrix)
    #languages, table = reorder_iou_matrix_by_hardcoded_order(languages, table)
    #iou_dict[layer_index] = iou_matrix
    iou_list.append(iou_matrix)
    # Plot IoU heatmap
    #plot_iou_heatmap(languages, iou_matrix, layer_name)
    #save_intersection_union_table(languages, table, output_path=f"{layer_name}.csv")
    #rint(f"Completed {layer_name}")

targets, indices = extract_indices(Lang_list)
fr_indices = indices[:15]
zh_indices = indices[15:]

# unique unordered pairs within each group
fr_pairs = list(combinations(fr_indices, 2))  # C(15,2)=105
zh_pairs = list(combinations(zh_indices, 2))  # C(15,2)=105

# Build arrays: shape = (num_pairs, num_layers)
# where each row is the iou across layers for a given (i, j) pair
fr_lists = np.array([[e[i][j] for e in iou_list] for (i, j) in fr_pairs])
zh_lists = np.array([[e[i][j] for e in iou_list] for (i, j) in zh_pairs])

print("fr_lists shape:", fr_lists.shape)  # (105, num_layers)
print("zh_lists shape:", zh_lists.shape)  # (105, num_layers)

# Now you can feed these into your test (expects shape (replicates, layers))
results = paired_by_layer_test(fr_lists, zh_lists)
print(results)