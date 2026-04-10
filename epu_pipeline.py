"""
=============================================================================
Greek Economic Policy Uncertainty (EPU) Index — Full Pipeline
=============================================================================

Constructs a monthly EPU index for Greece by analyzing ~1.5M newspaper
articles (2010–2026) and validates it against the Greece–Germany 10-year
government bond spread.

Methodology based on Baker, Bloom & Davis (2016), extended with:
  - Greek morphological stem matching
  - Context-window weighting (negators, diminishers, amplifiers)
  - Bigram detection for domain-specific compound phrases
  - Loughran-McDonald financial sentiment scoring (Greek-adapted)

Authors: QuantBros (2026)
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — update these paths to match your local setup
# =============================================================================
DATA_PATH = "."  # directory containing input CSVs and spread Excel file
CSV_FILES = ["Newspapers_1.csv", "Newspapers_2.csv"]
SPREAD_FILE = "Greece_Germany_Bond_Spread_2010_2026.xlsx"
OUTPUT_DIR = "./output"
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  GREEK EPU INDEX — PIPELINE START")
print("=" * 70)

try:
    from wordcloud import WordCloud

    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

import statsmodels.api as sm
from scipy import stats
from numpy.polynomial import polynomial as P
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# KEYWORD DICTIONARIES
# =============================================================================
# Greek morphological stems — each list captures all inflected forms via
# prefix matching (e.g. "αβεβαι" matches αβεβαιότητα, αβεβαιότητας, etc.)

print("\n  Loading keyword dictionaries...")

# --- Uncertainty stems ---
uncertainty_stems = [
    "αβεβαι", "αβεβαίωτ", "ανησυχ", "αμφιβολ", "κίνδυν", "κινδύν",
    "απρόβλεπτ", "αστάθει", "ασταθ", "αναταραχ", "αδιέξοδ", "επισφαλ",
    "αποσταθεροπο", "απειλ", "κρίσ", "κριση", "χρεοκοπ", "πτώχευσ",
    "ύφεσ", "υφεσ", "κλυδωνισμ", "πανικ", "μεταβλητότητ", "επιδείνωσ",
    "δυσμεν", "αναστάτωσ", "ανασφάλει", "κατάρρευσ", "καταρρέ", "ευάλωτ",
    "αντιξοότητ",
]

# --- Economy stems ---
economy_stems = [
    "οικονομ", "οικονομικ", "πληθωρισμ", "αποπληθωρισμ", "ανεργ", "χρέο",
    "χρεο", "δημοσιονομικ", "τράπεζ", "τραπεζ", "επιτόκι", "επιτοκ",
    "ομόλογ", "ομολογ", "δανει", "δάνει", "φορολογ", "προϋπολογισμ",
    "επένδυσ", "επενδύ", "επενδυτ", "χρηματιστήρι", "χρηματοπιστωτικ",
    "νομισματικ", "ρευστότητ", "έλλειμμ", "ελλειμμ", "πλεόνασμ",
    "ιδιωτικοποί", "αεπ",
]

# --- Policy stems ---
policy_stems = [
    "κυβέρνησ", "κυβερνησ", "κυβερνητικ", "πρωθυπουργ", "νομοσχέδι",
    "νομοσχεδ", "νομοθεσ", "μεταρρύθμισ", "μεταρρυθμ", "υπουργ", "εκλογ",
    "κοινοβούλι", "κοινοβουλευτικ", "ευρωζών", "ευρωζωνη", "κομισιόν",
    "κομισιον", "μνημόνι", "μνημονι", "τρόικα", "τροικα", "ρυθμιστικ",
    "eurogroup", "διαπραγματεύ", "διαπραγματευ", "λιτότητ", "περικοπ",
    "δημοψήφισμ", "δημοψηφισμ", "grexit", "συμφων",
]

# --- Domain-specific bigrams (high-signal compound phrases) ---
uncertainty_bigrams = [
    "οικονομική κρίση", "οικονομικη κριση", "οικονομική αβεβαιότητα",
    "οικονομικη αβεβαιοτητα", "πολιτική αστάθεια", "πολιτικη ασταθεια",
    "πολιτική κρίση", "πολιτικη κριση", "δημοσιονομική κρίση",
    "δημοσιονομικη κριση", "τραπεζική κρίση", "τραπεζικη κριση",
    "κρίση χρέους", "κριση χρεους", "κίνδυνος χρεοκοπίας",
    "κινδυνος χρεοκοπιας", "capital controls", "κεφαλαιακοί έλεγχοι",
    "κούρεμα ομολόγων", "κουρεμα ομολογων", "αναδιάρθρωση χρέους",
    "αναδιαρθρωση χρεους", "δημοσιονομικό έλλειμμα",
    "δημοσιονομικο ελλειμμα", "πακέτο διάσωσης", "πακετο διασωσης",
    "μέτρα λιτότητας", "μετρα λιτοτητας", "spread ομολόγων",
    "spread ομολογων", "εκλογική αβεβαιότητα", "εκλογικη αβεβαιοτητα",
    "πολιτική αβεβαιότητα", "πολιτικη αβεβαιοτητα", "πιστωτικό γεγονός",
    "πιστωτικο γεγονος",
]

# --- Context modifiers ---
negators = [
    "δεν", "δε", "μην", "μη", "χωρίς", "χωρις", "ούτε", "ουτε",
    "κανένα", "κανενα", "καμία", "καμια", "κανείς", "κανεις",
    "δίχως", "διχως", "ποτέ", "ποτε", "τίποτα", "τιποτα",
    "not", "no", "neither", "nor", "without", "never",
]

diminishers = [
    "τελείωσ", "τελειωσ", "τελειών", "τελειων", "λύθηκ", "λυθηκ",
    "λύν", "λυν", "λύσ", "λυσ", "ξεπεράστηκ", "ξεπεραστηκ", "ξεπερν",
    "αποφεύχθηκ", "αποφευχθηκ", "αποφεύγ", "αποφευγ", "βελτιώθηκ",
    "βελτιωθηκ", "βελτίωσ", "βελτιωσ", "σταθεροποιήθηκ",
    "σταθεροποιηθηκ", "σταθεροποί", "σταθεροποι", "ανέκαμψ", "ανεκαμψ",
    "ανακάμπτ", "ανακαμπτ", "υποχώρησ", "υποχωρησ", "υποχωρ",
    "εξαλείφθηκ", "εξαλειφθηκ", "αντιμετωπίστηκ", "αντιμετωπιστηκ",
    "απομακρύνθηκ", "απομακρυνθηκ", "εκτονώθηκ", "εκτονωθηκ", "εκτόνωσ",
    "εκτονωσ", "περιορίστηκ", "περιοριστηκ",
    "ended", "resolved", "overcame", "avoided", "improved",
]

amplifiers = [
    "βαθαίν", "βαθαιν", "βάθυν", "βαθυν", "επιδεινών", "επιδεινων",
    "επιδεινώνετ", "επιδεινωνετ", "εξαπλών", "εξαπλων", "εξαπλώνετ",
    "εξαπλωνετ", "κλιμακών", "κλιμακων", "κλιμάκωσ", "κλιμακωσ",
    "εντείν", "εντειν", "εντάσ", "εντασ", "χειροτερεύ", "χειροτερευ",
    "οξύν", "οξυν", "οξεί", "οξει", "διευρύν", "διευρυν", "μεγαλών",
    "μεγαλων", "αυξάν", "αυξαν", "εκτινάσσ", "εκτινασσ", "εκτοξεύ",
    "εκτοξευ",
    "deepens", "worsens", "escalat", "intensif", "surges",
]

# --- Loughran-McDonald sentiment (Greek-adapted) ---
lm_negative = [
    "καταστροφ", "χειρότερ", "χειροτερ", "αποτυχ", "αρνητικ", "πτώσ",
    "πτωσ", "βύθισ", "βυθισ", "κραχ", "ζημί", "ζημι", "απώλει", "απωλει",
    "υποβάθμισ", "υποβαθμισ", "αδύναμ", "αδυναμ", "επιβράδυνσ",
    "επιβραδυνσ", "στασιμότητ", "μείωσ", "μειωσ", "συρρίκνωσ",
    "συρρικνωσ", "αθέτησ", "αθετησ", "παραβίασ", "παραβιασ", "κυρώσε",
    "κυρωσε", "αποκλεισμ", "φτώχει", "φτωχει", "τοξικ", "προβληματικ",
    "κατάρρευσ", "καταρρέ", "χρεοκοπ", "πτώχευσ", "επισφαλ", "ανείσπρακτ",
    "σκάνδαλ", "σκανδαλ", "διαφθορ", "φοροδιαφυγ",
    "default", "downgrade", "bankruptcy", "recession", "crisis", "deficit",
]

lm_positive = [
    "ανάπτυξ", "αναπτυξ", "ανάκαμψ", "ανακαμψ", "βελτίωσ", "βελτιωσ",
    "πρόοδ", "προοδ", "κέρδ", "κερδ", "πλεόνασμ", "σταθερ", "αναβάθμισ",
    "αναβαθμισ", "αισιοδοξ", "εμπιστοσύν", "εμπιστοσυν", "αύξησ", "αυξησ",
    "ισχυρ", "επιτυχ", "ελάφρυνσ", "ελαφρυνσ",
    "growth", "recovery", "improve", "profit", "surplus",
]

all_stems = uncertainty_stems + economy_stems + policy_stems


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def contains_any(text_lower: str, stems: list) -> bool:
    """Check if text contains any of the given stems."""
    return any(s.lower() in text_lower for s in stems)


def count_matches(text_lower: str, stems: list) -> int:
    """Count total occurrences of all stems in text."""
    return sum(text_lower.count(s.lower()) for s in stems)


def count_bigrams(text_lower: str, bigrams: list) -> int:
    """Count occurrences of domain-specific bigrams."""
    return sum(text_lower.count(b.lower()) for b in bigrams)


def context_score(text: str, uncertainty_stems: list, negators: list,
                  diminishers: list, amplifiers: list, window: int = 5):
    """
    Context-aware uncertainty scoring.
    
    For each uncertainty keyword found, examines a ±window word neighborhood:
      - Negator before keyword → score 0 (suppressed)
      - Diminisher in context → score 0 (resolved uncertainty)
      - Negated diminisher → score 1.5 (uncertainty persists despite attempt)
      - Amplifier in context → score 2.0 (intensified uncertainty)
      - Otherwise → score 1.0 (neutral mention)
    
    Returns: (effective_score, total_found, negated, diminished, amplified)
    """
    words = text.lower().split()
    effective, total, negated, diminished, amplified = 0, 0, 0, 0, 0

    for i, word in enumerate(words):
        if not any(stem.lower() in word for stem in uncertainty_stems):
            continue

        total += 1
        start = max(0, i - window)
        end = min(len(words), i + window + 1)
        context = words[start:end]
        preceding = words[start:i]

        has_negator_before = any(neg in preceding for neg in negators)

        # Check for diminishers and whether they themselves are negated
        has_diminisher = False
        diminisher_negated = False
        for j, cword in enumerate(context):
            for dim in diminishers:
                if dim.lower() in cword:
                    has_diminisher = True
                    dim_pos = start + j
                    dim_preceding = words[max(0, dim_pos - 3):dim_pos]
                    if any(neg in dim_preceding for neg in negators):
                        diminisher_negated = True
                    break

        has_amplifier = any(
            any(amp.lower() in cword for amp in amplifiers)
            for cword in context
        )

        # Apply scoring hierarchy
        if has_negator_before and not has_diminisher:
            negated += 1
        elif has_diminisher and not diminisher_negated:
            diminished += 1
        elif has_diminisher and diminisher_negated:
            effective += 1.5
            amplified += 1
        elif has_amplifier:
            effective += 2.0
            amplified += 1
        else:
            effective += 1.0

    return effective, total, negated, diminished, amplified


def compute_lm_sentiment(text_lower: str) -> float:
    """
    Compute Loughran-McDonald sentiment score.
    Returns value in [-1, 1]: negative = bearish, positive = bullish.
    """
    pos = count_matches(text_lower, lm_positive)
    neg = count_matches(text_lower, lm_negative)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


# =============================================================================
# ARTICLE PROCESSOR
# =============================================================================

def process_article(content, title):
    """
    Classify a single article and compute all NLP features.
    
    Returns a dict with:
      - is_epu_content: True if article body meets EPU criteria
      - is_epu_title: True if title alone meets EPU criteria (≥2 of 3 categories + uncertainty)
      - context_score, bigram_count, lm_sentiment, etc.
    """
    result = {
        "is_epu_content": False,
        "is_epu_title": False,
        "context_score": 0, "context_total": 0, "context_negated": 0,
        "context_diminished": 0, "context_amplified": 0,
        "bigram_count": 0, "lm_sentiment": 0.0,
        "word_count": 0,
    }

    # --- Title-based classification ---
    if pd.notna(title) and len(str(title)) > 10:
        title_lower = str(title).lower()
        has_u = contains_any(title_lower, uncertainty_stems)
        has_e = contains_any(title_lower, economy_stems)
        has_p = contains_any(title_lower, policy_stems)
        title_score = int(has_u) + int(has_e) + int(has_p)
        if title_score >= 2 and has_u:
            result["is_epu_title"] = True

    # --- Content-based classification ---
    if pd.isna(content) or len(str(content)) < 50:
        return result

    text_lower = str(content).lower()
    result["word_count"] = len(text_lower.split())

    # Require all three categories present in body text
    if not (contains_any(text_lower, uncertainty_stems)
            and contains_any(text_lower, economy_stems)
            and contains_any(text_lower, policy_stems)):
        return result

    result["is_epu_content"] = True
    eff, tot, neg, dim, amp = context_score(
        text_lower, uncertainty_stems, negators, diminishers, amplifiers
    )

    if eff <= 0:
        result["is_epu_content"] = False
        return result

    result.update({
        "context_score": eff, "context_total": tot, "context_negated": neg,
        "context_diminished": dim, "context_amplified": amp,
        "bigram_count": count_bigrams(text_lower, uncertainty_bigrams),
        "lm_sentiment": compute_lm_sentiment(text_lower),
    })

    return result


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

print("\n" + "=" * 70)
print("  PROCESSING ARTICLES...")
print("=" * 70)

CHUNK_SIZE = 10_000
monthly_data = {}
epu_words_all = Counter()

for csv_file in CSV_FILES:
    filepath = os.path.join(DATA_PATH, csv_file)
    if not os.path.exists(filepath):
        print(f"  WARNING: {csv_file} not found — skipping")
        continue

    print(f"\n  Reading {csv_file}")
    chunk_count = 0

    for chunk in pd.read_csv(
        filepath, chunksize=CHUNK_SIZE,
        usecols=["date", "content", "article_title"], dtype=str,
    ):
        chunk_count += 1
        if chunk_count % 20 == 0:
            print(f"    ...chunk {chunk_count} ({chunk_count * CHUNK_SIZE:,} rows)")

        chunk["date_parsed"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.dropna(subset=["date_parsed"])
        chunk["year_month"] = chunk["date_parsed"].dt.to_period("M")

        for ym, group in chunk.groupby("year_month"):
            ym_str = str(ym)
            if ym_str not in monthly_data:
                monthly_data[ym_str] = {
                    "total": 0, "epu_content": 0, "epu_title": 0,
                    "context_score_sum": 0, "context_total_sum": 0,
                    "context_negated_sum": 0, "context_diminished_sum": 0,
                    "context_amplified_sum": 0, "bigram_sum": 0,
                    "lm_sentiment_sum": 0.0, "epu_word_sum": 0,
                }

            md = monthly_data[ym_str]
            md["total"] += len(group)

            for _, row in group.iterrows():
                res = process_article(row.get("content"), row.get("article_title"))

                if res["is_epu_title"]:
                    md["epu_title"] += 1
                if res["is_epu_content"]:
                    md["epu_content"] += 1
                    md["context_score_sum"] += res["context_score"]
                    md["context_total_sum"] += res["context_total"]
                    md["context_negated_sum"] += res["context_negated"]
                    md["context_diminished_sum"] += res["context_diminished"]
                    md["context_amplified_sum"] += res["context_amplified"]
                    md["bigram_sum"] += res["bigram_count"]
                    md["lm_sentiment_sum"] += res["lm_sentiment"]
                    md["epu_word_sum"] += res["word_count"]

                    # Collect top keywords for word cloud (first 50 EPU articles/month)
                    if md["epu_content"] <= 50:
                        content = row.get("content")
                        if pd.notna(content):
                            words = str(content).lower().split()
                            filtered = [
                                w for w in words
                                if len(w) > 3 and any(s in w for s in all_stems)
                            ]
                            epu_words_all.update(filtered)


# =============================================================================
# INDEX CONSTRUCTION
# =============================================================================

print("\n  Building indices...")

rows = []
for ym_str, md in monthly_data.items():
    total = md["total"]
    if total == 0:
        continue

    epu_c = md["epu_content"]
    epu_t = md["epu_title"]

    # Eight alternative index formulations
    basic = epu_c / total
    title_based = epu_t / total
    context_weighted = md["context_score_sum"] / total if md["epu_word_sum"] > 0 else 0
    bigram = basic + (md["bigram_sum"] / max(total, 1)) * 5
    avg_lm = md["lm_sentiment_sum"] / epu_c if epu_c > 0 else 0
    lm_weight = 1 + max(0, -avg_lm) * 2
    sentiment_lm = basic * lm_weight
    title_content_combined = title_based * 0.4 + basic * 0.6
    context_intensity = (
        md["context_score_sum"] / max(md["context_total_sum"], 1)
        if epu_c > 0 else 0
    )
    ultimate = (
        basic * (1 + context_intensity) * lm_weight
        + (md["bigram_sum"] / max(total, 1)) * 3
    )
    title_bigram = title_based + (md["bigram_sum"] / max(total, 1)) * 3

    rows.append({
        "year_month": ym_str, "total_articles": total,
        "epu_content": epu_c, "epu_title": epu_t,
        "basic": basic, "title_based": title_based,
        "context_weighted": context_weighted, "bigram": bigram,
        "sentiment_lm": sentiment_lm, "title_content": title_content_combined,
        "ultimate": ultimate, "title_bigram": title_bigram,
        "avg_lm_sentiment": avg_lm,
    })

epu_df = pd.DataFrame(rows)
epu_df["period"] = pd.PeriodIndex(epu_df["year_month"], freq="M")
epu_df = epu_df.sort_values("period").reset_index(drop=True)

# Normalize to mean=100 and apply linear detrending
method_cols = [
    "basic", "title_based", "context_weighted", "bigram",
    "sentiment_lm", "title_content", "ultimate", "title_bigram",
]

for col in method_cols:
    mean_val = epu_df[col].mean()
    epu_df[col + "_index"] = (epu_df[col] / mean_val) * 100 if mean_val > 0 else 100
    x = np.arange(len(epu_df))
    coeffs = P.polyfit(x, epu_df[col + "_index"].values, 1)
    epu_df[col + "_dt"] = epu_df[col + "_index"] - P.polyval(x, coeffs) + 100

epu_df["date"] = epu_df["period"].dt.to_timestamp()


# =============================================================================
# MERGE WITH BOND SPREAD
# =============================================================================

print("  Merging with bond spread data...")

spread_df = pd.read_excel(os.path.join(DATA_PATH, SPREAD_FILE))
spread_df["date"] = pd.to_datetime(spread_df["Date"], format="mixed")
spread_df["period"] = spread_df["date"].dt.to_period("M")
spread_monthly = spread_df.groupby("period")["Spread"].mean().reset_index()

merge_cols = [
    "period", "date", "total_articles", "epu_content", "epu_title",
    "avg_lm_sentiment",
]
for col in method_cols:
    merge_cols += [col + "_index", col + "_dt"]

merged = (
    pd.merge(epu_df[merge_cols], spread_monthly, on="period", how="inner")
    .dropna(subset=["Spread"])
    .reset_index(drop=True)
)
merged = merged.rename(columns={"Spread": "spread"})

# Auxiliary variables for regressions
merged["spread_lag1"] = merged["spread"].shift(1)
merged["spread_change"] = merged["spread"].diff()
merged["trend"] = range(len(merged))
merged["log_spread"] = np.log(merged["spread"])

for col in method_cols:
    merged[col + "_dt_lag1"] = merged[col + "_dt"].shift(1)
    merged[col + "_dt_change"] = merged[col + "_dt"].diff()


# =============================================================================
# CORRELATION ANALYSIS & METHOD SELECTION
# =============================================================================

print("  Computing correlations...")

corr_results = {}
for col in method_cols:
    for suffix, label in [("_dt", "detrended"), ("_index", "original")]:
        full_col = col + suffix
        data = merged[[full_col, "spread"]].dropna()
        if len(data) > 10:
            pr, pp = stats.pearsonr(data[full_col], data["spread"])
            sr, sp = stats.spearmanr(data[full_col], data["spread"])
            corr_results[f"{col} ({label})"] = {
                "r": pr, "p": pp, "sr": sr, "sp": sp, "col": full_col,
            }

# Select best method (highest positive Pearson r among detrended)
best_name = max(
    {k: v for k, v in corr_results.items() if v["r"] > 0}.keys(),
    key=lambda k: corr_results[k]["r"],
)
best_col = corr_results[best_name]["col"]

print(f"  Best method: {best_name} (r = {corr_results[best_name]['r']:.3f})")

# Cross-correlation to find optimal lead/lag
cross_corrs = {}
for lag in range(-12, 13):
    if lag >= 0:
        epu_s = merged[best_col].iloc[:len(merged) - lag].reset_index(drop=True)
        spr_s = merged["spread"].iloc[lag:].reset_index(drop=True)
    else:
        epu_s = merged[best_col].iloc[-lag:].reset_index(drop=True)
        spr_s = merged["spread"].iloc[:len(merged) + lag].reset_index(drop=True)
    if len(epu_s) > 10:
        min_len = min(len(epu_s), len(spr_s))
        cross_corrs[lag] = stats.pearsonr(epu_s[:min_len], spr_s[:min_len])[0]

optimal_lag = max(cross_corrs.keys(), key=lambda k: cross_corrs[k])
print(f"  Optimal lag: {optimal_lag} months (r = {cross_corrs[optimal_lag]:.3f})")


# =============================================================================
# OLS REGRESSIONS (HAC / Newey-West standard errors)
# =============================================================================

print("  Running OLS regressions...")

models = {}

# Reg 1: Bivariate — EPU alone
d = merged[["spread", best_col]].dropna()
models["Reg1 (Bivariate)"] = sm.OLS(
    d["spread"], sm.add_constant(d[best_col])
).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

# Reg 2: Multivariate — EPU + lagged spread + trend
d = merged[["spread", best_col, "spread_lag1", "trend"]].dropna()
models["Reg2 (Multivariate)"] = sm.OLS(
    d["spread"], sm.add_constant(d[[best_col, "spread_lag1", "trend"]])
).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

# Reg 3: Predictive — lagged EPU
if optimal_lag > 0:
    lagged_epu_col = f"{best_col}_lag{optimal_lag}"
    merged[lagged_epu_col] = merged[best_col].shift(optimal_lag)
    d = merged[["spread", lagged_epu_col, "trend"]].dropna()
    models[f"Reg3 (Predictive: EPU Lag {optimal_lag})"] = sm.OLS(
        d["spread"], sm.add_constant(d[[lagged_epu_col, "trend"]])
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})


# =============================================================================
# VISUALIZATION
# =============================================================================

print("  Generating figures...")

dates = merged["period"].dt.to_timestamp()
events = {
    "2010-05": "First Bailout",
    "2012-03": "PSI / 2nd Bailout",
    "2015-07": "Referendum",
    "2018-08": "End of Bailouts",
    "2020-03": "COVID-19",
    "2022-02": "Ukraine War",
}


def add_events(ax, ypos=None):
    """Annotate key historical events on a time series chart."""
    for ds, label in events.items():
        try:
            ed = pd.Timestamp(ds)
            if dates.min() <= ed <= dates.max():
                ax.axvline(x=ed, color="gray", linestyle="--", alpha=0.3)
                if ypos is not None:
                    ax.annotate(
                        label, xy=(ed, ypos), fontsize=7,
                        rotation=45, ha="left", va="top",
                    )
        except Exception:
            pass


# --- Figure 1: EPU vs Bond Spread time series ---
fig, ax1 = plt.subplots(figsize=(15, 6))
ax1.plot(dates, merged[best_col], color="#1f77b4", linewidth=1.3,
         label=f"EPU ({best_name})")
ax1.set_ylabel("EPU Index", color="#1f77b4", fontsize=11)
ax2 = ax1.twinx()
ax2.plot(dates, merged["spread"], color="#d62728", linewidth=1.3, alpha=0.8,
         label="Bond Spread")
ax2.set_ylabel("Spread (%)", color="#d62728", fontsize=11)
add_events(ax1, ax1.get_ylim()[1] * 0.95)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
plt.title(
    f'Greek EPU vs Bond Spread (r = {corr_results[best_name]["r"]:.3f})',
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "01_epu_vs_spread.png"),
    dpi=300, bbox_inches="tight",
)
plt.close()

# --- Figure 2: Scatter plot ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(merged[best_col], merged["spread"], alpha=0.5, s=20, color="#1f77b4")
z = np.polyfit(
    merged[best_col].dropna(),
    merged.loc[merged[best_col].notna(), "spread"],
    1,
)
p = np.poly1d(z)
xl = np.linspace(merged[best_col].min(), merged[best_col].max(), 100)
ax.plot(xl, p(xl), color="red", linewidth=2)
ax.set_xlabel("EPU Index", fontsize=11)
ax.set_ylabel("Spread (%)", fontsize=11)
ax.set_title(
    f'{best_name}: r = {corr_results[best_name]["r"]:.3f}',
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "02_scatter_best.png"),
    dpi=300, bbox_inches="tight",
)
plt.close()

# --- Figure 3: Cross-correlation ---
fig, ax = plt.subplots(figsize=(12, 5))
lags_list = sorted(cross_corrs.keys())
corrs_list = [cross_corrs[lag] for lag in lags_list]
colors_bar = ["#d62728" if lag == optimal_lag else "#1f77b4" for lag in lags_list]
ax.bar(lags_list, corrs_list, color=colors_bar, alpha=0.7)
ax.axhline(y=0, color="gray", linestyle="-")
ax.set_xlabel("Lag (months, positive = EPU leads)", fontsize=11)
ax.set_ylabel("Pearson r", fontsize=11)
ax.set_title(
    f"Cross-Correlation: Optimal lag = {optimal_lag} months "
    f"(r = {cross_corrs[optimal_lag]:.3f})",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "03_cross_correlation.png"),
    dpi=300, bbox_inches="tight",
)
plt.close()

# --- Optional: Word cloud ---
if HAS_WORDCLOUD and epu_words_all:
    fig, ax = plt.subplots(figsize=(10, 6))
    wc = WordCloud(
        width=800, height=400, background_color="white",
        max_words=100, colormap="Reds",
    ).generate_from_frequencies(dict(epu_words_all.most_common(150)))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title("Most Frequent Keywords in EPU Articles",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "04_wordcloud.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# EXPORT RESULTS
# =============================================================================

print("  Saving outputs...")

# Time series CSV
best_ts = epu_df[[
    "year_month",
    best_col.replace("_dt", "_index"),
    best_col.replace("_dt", "") + "_dt",
]].copy()
best_ts.columns = ["Year_Month", "EPU_Index", "EPU_Index_Detrended"]
best_ts.to_csv(os.path.join(OUTPUT_DIR, "epu_timeseries.csv"), index=False)

# Full dataset
epu_df.to_csv(os.path.join(OUTPUT_DIR, "epu_all_methods.csv"), index=False)
merged.to_csv(os.path.join(OUTPUT_DIR, "merged_data.csv"), index=False)

# Results summary
with open(os.path.join(OUTPUT_DIR, "results.txt"), "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("GREEK EPU INDEX — RESULTS SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Best Method: {best_name}\n")
    f.write(f"Optimal Lag: {optimal_lag} months\n\n")
    f.write("REGRESSIONS\n")
    f.write("=" * 80 + "\n\n")
    for name, model in models.items():
        f.write(f"--- {name} ---\n{model.summary().as_text()}\n\n")

print("\n" + "=" * 70)
print(f"  DONE — all outputs saved to: {OUTPUT_DIR}")
print("=" * 70)
