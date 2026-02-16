# ═══ Cell 2 (id=config) ═══
# ============================================================
# CONFIGURATION — edit these before running
# ============================================================

CSV_PATH = "rotten_tomatoes_critic_reviews.csv"
IMDB_GENRES_PATH = "imdb_genres.csv"  # from hf://datasets/jquigl/imdb-genres

# Column name mapping
COLUMN_MAP = {
    "movie_id":        "rotten_tomatoes_link",
    "critic_name":     "critic_name",
    "outlet":          "publisher_name",
    "review_date":     "review_date",
    "review_type":     "review_type",
    "is_fresh":        None,
    "review_score":    "review_score",
    "is_top_critic":   "top_critic",
    "movie_genres":    None,
    "wide_release_date": None,
    "runtime":         None,
    "mpaa_rating":     None,
    "studio":          None,
    "director":        None,
    "cast":            None,
}

# Modeling parameters
NEGATIVE_RATIO = 10
MIN_REVIEWS_PER_MOVIE = 20
MIN_REVIEWS_PER_CRITIC = 5
CUTOFF_DATE = "2019-01-01"
N_SIMS = 10_000
THRESHOLDS = [0.60, 0.70, 0.80, 0.90]
N_LATENT_GENRES = 15         # NMF components (supplement for movies without IMDB match)
RANDOM_SEED = 42

BACKTEST_CHECKPOINTS = [5, 10, 25, 50]
MAX_CRITICS_UNIVERSE = 3000

# ═══ Cell 4 (id=imports) ═══
import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    mean_absolute_error, mean_squared_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF

try:
    import lightgbm as lgb
    USE_LGBM = True
    print("Using LightGBM")
except (ImportError, OSError):
    USE_LGBM = False
    print("LightGBM not found — falling back to LogisticRegression")

np.random.seed(RANDOM_SEED)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

print("Imports OK")

# ═══ Cell 5 (id=load-data) ═══
# ── Load CSV ──────────────────────────────────────────────
raw = pd.read_csv(CSV_PATH)
print(f"Loaded {len(raw):,} rows, {raw.shape[1]} columns")
print(f"Columns: {list(raw.columns)}")
raw.head(3)

# ═══ Cell 6 (id=map-columns) ═══
# ── Map columns to standard names ────────────────────────
def safe_rename(df, mapping):
    """Rename columns based on mapping dict, skipping Nones and missing cols."""
    rename_dict = {}
    for std_name, csv_name in mapping.items():
        if csv_name is not None and csv_name in df.columns and csv_name != std_name:
            rename_dict[csv_name] = std_name
    return df.rename(columns=rename_dict)

df = safe_rename(raw.copy(), COLUMN_MAP)

# Ensure required columns exist
required = ["movie_id", "review_date"]
for col in required:
    mapped = COLUMN_MAP.get(col)
    actual = col if col in df.columns else mapped
    assert actual in df.columns, f"Missing required column: {col} (mapped to {mapped})"

print("Standardized columns:", list(df.columns))

# ═══ Cell 8 (id=clean-data) ═══
# ── Parse dates ───────────────────────────────────────────
df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
df = df[df["review_date"].notna()].copy()
df = df[df["review_date"] >= "1990-01-01"].copy()
df = df[df["review_date"] <= "2021-01-01"].copy()

# ── Standardize critic_name ───────────────────────────────
if "critic_name" in df.columns:
    df["critic_name"] = df["critic_name"].astype(str).str.strip().str.lower()
    df = df[df["critic_name"] != "nan"].copy()
else:
    raise ValueError("No critic_name column found")

# ── Create is_fresh ───────────────────────────────────────
if "is_fresh" not in df.columns or df["is_fresh"].isna().all():
    if "review_type" in df.columns:
        df["is_fresh"] = (df["review_type"].str.strip().str.lower() == "fresh").astype(int)
    else:
        raise ValueError("Need either is_fresh or review_type column")
df["is_fresh"] = df["is_fresh"].astype(int)

# ── Standardize is_top_critic ─────────────────────────────
if "is_top_critic" in df.columns:
    df["is_top_critic"] = df["is_top_critic"].map(
        {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
    ).fillna(0).astype(int)
else:
    df["is_top_critic"] = 0

# ── Create movie_key ──────────────────────────────────────
df["movie_key"] = df["movie_id"].astype(str)

# ══════════════════════════════════════════════════════════
# IMDB GENRE MATCHING
# ══════════════════════════════════════════════════════════
def normalize_title(t):
    t = t.lower().strip()
    t = re.sub(r'[^a-z0-9 ]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    for article in ['the ', 'a ', 'an ']:
        if t.startswith(article):
            t = t[len(article):]
    return t

def slug_to_title(slug):
    s = slug.replace('m/', '')
    s = re.sub(r'^\d+[-_]', '', s)
    return s.replace('_', ' ').strip().lower()

import os
if os.path.exists(IMDB_GENRES_PATH):
    imdb_raw = pd.read_csv(IMDB_GENRES_PATH)
    imdb_raw["title_clean"] = imdb_raw["movie title - year"].str.rsplit(" - ", n=1).str[0].str.strip().str.lower()
    imdb_raw["title_norm"] = imdb_raw["title_clean"].apply(normalize_title)
    imdb_dedup = imdb_raw.drop_duplicates("title_norm", keep="first")

    # Build RT title lookup
    rt_titles = df[["movie_key"]].drop_duplicates().copy()
    rt_titles["title_raw"] = rt_titles["movie_key"].apply(slug_to_title)
    rt_titles["title_norm"] = rt_titles["title_raw"].apply(normalize_title)

    # Merge
    genre_merge = rt_titles.merge(
        imdb_dedup[["title_norm", "genre", "expanded-genres"]],
        on="title_norm", how="left"
    )

    # Parse expanded-genres into multi-hot
    all_expanded = genre_merge["expanded-genres"].dropna().str.split(", ").explode().str.strip().str.lower()
    top_genres = all_expanded.value_counts().head(16).index.tolist()
    GENRE_COLS = [f"genre_{g.replace(' ', '_').replace('-', '_')}" for g in top_genres]

    for g, col in zip(top_genres, GENRE_COLS):
        genre_merge[col] = genre_merge["expanded-genres"].str.lower().str.contains(g, na=False).astype(int)

    genre_merge["primary_genre"] = genre_merge["genre"].fillna("unknown").str.lower()
    genre_merge = genre_merge[["movie_key", "primary_genre"] + GENRE_COLS]

    # Merge into df
    df = df.merge(genre_merge, on="movie_key", how="left")
    for col in GENRE_COLS:
        df[col] = df[col].fillna(0).astype(int)
    df["primary_genre"] = df["primary_genre"].fillna("unknown")

    match_rate = (df.groupby("movie_key")["primary_genre"].first() != "unknown").mean()
    HAS_GENRES = True
    print(f"IMDB genres loaded: {len(imdb_raw):,} movies")
    print(f"  Matched {match_rate:.1%} of RT movies to IMDB genres")
    print(f"  {len(GENRE_COLS)} genre columns: {top_genres}")
else:
    HAS_GENRES = False
    GENRE_COLS = []
    print(f"IMDB genres file not found at {IMDB_GENRES_PATH} — genre features disabled")

# ── Summary ───────────────────────────────────────────────
print(f"\nCleaned dataset: {len(df):,} reviews")
print(f"  Movies:  {df['movie_key'].nunique():,}")
print(f"  Critics: {df['critic_name'].nunique():,}")
print(f"  Date range: {df['review_date'].min().date()} to {df['review_date'].max().date()}")
print(f"  Fresh rate: {df['is_fresh'].mean():.3f}")

# ═══ Cell 10 (id=movie-stats) ═══
# ── Movie-level stats ─────────────────────────────────────
movie_stats = df.groupby("movie_key").agg(
    n_reviews=("is_fresh", "size"),
    n_fresh=("is_fresh", "sum"),
    first_review=("review_date", "min"),
    last_review=("review_date", "max"),
    pct_top_critic=("is_top_critic", "mean"),
).reset_index()
movie_stats["tomatometer"] = movie_stats["n_fresh"] / movie_stats["n_reviews"]

# Filter to movies with enough reviews
movie_stats = movie_stats[movie_stats["n_reviews"] >= MIN_REVIEWS_PER_MOVIE].copy()
valid_movies = set(movie_stats["movie_key"])
df = df[df["movie_key"].isin(valid_movies)].copy()

print(f"Movies with >= {MIN_REVIEWS_PER_MOVIE} reviews: {len(valid_movies):,}")
print(f"Reviews retained: {len(df):,}")

# ═══ Cell 11 (id=critic-universe) ═══
# ── Build critic universe ─────────────────────────────────
critic_counts = df.groupby("critic_name").size().reset_index(name="n_movies_reviewed")
critic_counts = critic_counts[critic_counts["n_movies_reviewed"] >= MIN_REVIEWS_PER_CRITIC]

# Cap universe size for memory
if MAX_CRITICS_UNIVERSE and len(critic_counts) > MAX_CRITICS_UNIVERSE:
    critic_counts = critic_counts.nlargest(MAX_CRITICS_UNIVERSE, "n_movies_reviewed")

CRITIC_UNIVERSE = sorted(critic_counts["critic_name"].tolist())
critic_to_idx = {c: i for i, c in enumerate(CRITIC_UNIVERSE)}
n_critics = len(CRITIC_UNIVERSE)

print(f"Critic universe: {n_critics:,} critics")
print(f"  Min reviews: {critic_counts['n_movies_reviewed'].min()}")
print(f"  Median reviews: {critic_counts['n_movies_reviewed'].median():.0f}")
print(f"  Max reviews: {critic_counts['n_movies_reviewed'].max()}")

# ═══ Cell 13 (id=time-split) ═══
# ── Split movies by first review date ────────────────────
cutoff = pd.Timestamp(CUTOFF_DATE)

train_movies = set(movie_stats[movie_stats["first_review"] < cutoff]["movie_key"])
val_movies = set(movie_stats[movie_stats["first_review"] >= cutoff]["movie_key"])

df_train = df[df["movie_key"].isin(train_movies)].copy()
df_val = df[df["movie_key"].isin(val_movies)].copy()

# Only keep reviews from universe critics
df_train_univ = df_train[df_train["critic_name"].isin(critic_to_idx)].copy()
df_val_univ = df_val[df_val["critic_name"].isin(critic_to_idx)].copy()

print(f"Train: {len(train_movies):,} movies, {len(df_train):,} reviews ({len(df_train_univ):,} from universe critics)")
print(f"Val:   {len(val_movies):,} movies, {len(df_val):,} reviews ({len(df_val_univ):,} from universe critics)")

# ═══ Cell 15 (id=hpbvlpmqt2g) ═══
# ══════════════════════════════════════════════════════════
# NMF LATENT GENRES (supplement real genres — captures hidden structure)
# ══════════════════════════════════════════════════════════

train_rev_pairs = df_train[
    df_train["critic_name"].isin(critic_to_idx)
][["movie_key", "critic_name"]].drop_duplicates()

train_movie_list = sorted(train_movies)
movie_to_row = {m: i for i, m in enumerate(train_movie_list)}

row_idx = train_rev_pairs["movie_key"].map(movie_to_row).dropna().astype(int)
col_idx = train_rev_pairs.loc[row_idx.index, "critic_name"].map(critic_to_idx)
valid = row_idx.notna() & col_idx.notna()
row_idx, col_idx = row_idx[valid].values, col_idx[valid].values

review_matrix = csr_matrix(
    (np.ones(len(row_idx)), (row_idx, col_idx)),
    shape=(len(train_movie_list), n_critics)
)
print(f"Review matrix: {review_matrix.shape} ({review_matrix.nnz:,} non-zero)")

k = N_LATENT_GENRES
print(f"Fitting NMF with {k} components...")
nmf_model = NMF(n_components=k, init="nndsvda", max_iter=300, random_state=RANDOM_SEED)
W_train = nmf_model.fit_transform(review_matrix)
H = nmf_model.components_

W_train_norm = W_train / (W_train.sum(axis=1, keepdims=True) + 1e-8)
LATENT_GENRE_COLS = [f"lg_{i}" for i in range(k)]

movie_genre_df = pd.DataFrame(W_train_norm, columns=LATENT_GENRE_COLS)
movie_genre_df["movie_key"] = train_movie_list

critic_genre_loadings_norm = H.T / (H.T.sum(axis=1, keepdims=True) + 1e-8)
critic_genre_df = pd.DataFrame(critic_genre_loadings_norm, columns=[f"critic_{c}" for c in LATENT_GENRE_COLS])
critic_genre_df["critic_name"] = CRITIC_UNIVERSE

# Val movies — project into NMF space
val_rev_pairs = df_val[df_val["critic_name"].isin(critic_to_idx)][["movie_key", "critic_name"]].drop_duplicates()
val_movie_list = sorted(val_movies)
val_movie_to_row = {m: i for i, m in enumerate(val_movie_list)}
row_v = val_rev_pairs["movie_key"].map(val_movie_to_row).dropna().astype(int)
col_v = val_rev_pairs.loc[row_v.index, "critic_name"].map(critic_to_idx)
valid_v = row_v.notna() & col_v.notna()
review_matrix_val = csr_matrix(
    (np.ones(valid_v.sum()), (row_v[valid_v].values, col_v[valid_v].values)),
    shape=(len(val_movie_list), n_critics)
)
W_val = nmf_model.transform(review_matrix_val)
W_val_norm = W_val / (W_val.sum(axis=1, keepdims=True) + 1e-8)
movie_genre_val_df = pd.DataFrame(W_val_norm, columns=LATENT_GENRE_COLS)
movie_genre_val_df["movie_key"] = val_movie_list
movie_genre_all_df = pd.concat([movie_genre_df, movie_genre_val_df], ignore_index=True)

# ══════════════════════════════════════════════════════════
# CRITIC GENRE-SPECIFIC FRESH RATES (from real IMDB genres)
# ══════════════════════════════════════════════════════════

print("\nComputing critic-genre fresh rates...")

# For real genres: compute each critic's fresh rate per genre
critic_genre_fresh_cols = []
if HAS_GENRES and GENRE_COLS:
    train_revs = df_train[df_train["critic_name"].isin(critic_to_idx)].copy()
    for gc in GENRE_COLS:
        col_name = f"critic_fresh_{gc}"
        critic_genre_fresh_cols.append(col_name)
        g_revs = train_revs[train_revs[gc] == 1]
        g_stats = g_revs.groupby("critic_name")["is_fresh"].mean().reset_index()
        g_stats.columns = ["critic_name", col_name]
        critic_genre_df = critic_genre_df.merge(g_stats, on="critic_name", how="left")

    # Also: critic genre coverage propensity (what fraction of their reviews are in each genre)
    critic_genre_prop_cols = []
    for gc in GENRE_COLS:
        prop_col = f"critic_prop_{gc}"
        critic_genre_prop_cols.append(prop_col)
        g_counts = train_revs.groupby("critic_name")[gc].mean().reset_index()
        g_counts.columns = ["critic_name", prop_col]
        critic_genre_df = critic_genre_df.merge(g_counts, on="critic_name", how="left")
    
    print(f"  {len(critic_genre_fresh_cols)} genre-specific fresh rate features")
    print(f"  {len(critic_genre_prop_cols)} genre coverage propensity features")
else:
    critic_genre_prop_cols = []

# For NMF genres: critic fresh rate weighted by latent genre
CRITIC_NMF_FRESH_COLS = [f"critic_fresh_lg_{i}" for i in range(k)]
train_revs_lg = df_train[
    df_train["critic_name"].isin(critic_to_idx) &
    df_train["movie_key"].isin(movie_to_row)
][["movie_key", "critic_name", "is_fresh"]].drop_duplicates(subset=["movie_key", "critic_name"], keep="first")
train_revs_lg = train_revs_lg.merge(movie_genre_df[["movie_key"] + LATENT_GENRE_COLS], on="movie_key", how="left")

critic_nmf_fresh = {}
for c_name in CRITIC_UNIVERSE:
    c_revs = train_revs_lg[train_revs_lg["critic_name"] == c_name]
    if len(c_revs) == 0:
        critic_nmf_fresh[c_name] = {col: 0.5 for col in CRITIC_NMF_FRESH_COLS}
        continue
    row = {}
    for i in range(k):
        w = c_revs[f"lg_{i}"].values
        ws = w.sum()
        row[f"critic_fresh_lg_{i}"] = float((c_revs["is_fresh"].values * w).sum() / ws) if ws > 0.01 else float(c_revs["is_fresh"].mean())
    critic_nmf_fresh[c_name] = row

critic_nmf_fresh_df = pd.DataFrame.from_dict(critic_nmf_fresh, orient="index").reset_index()
critic_nmf_fresh_df.columns = ["critic_name"] + CRITIC_NMF_FRESH_COLS

# ══════════════════════════════════════════════════════════
# OUTLET FEATURES (from training data)
# ══════════════════════════════════════════════════════════

print("Computing outlet features...")
if "outlet" in df_train.columns:
    outlet_stats = df_train.groupby("outlet").agg(
        outlet_n_reviews=("is_fresh", "size"),
        outlet_fresh_rate=("is_fresh", "mean"),
        outlet_n_critics=("critic_name", "nunique"),
    ).reset_index()
    # Map to critics via their most common outlet
    critic_outlet = df_train.groupby("critic_name")["outlet"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown").reset_index()
    critic_outlet.columns = ["critic_name", "primary_outlet"]
    critic_outlet = critic_outlet.merge(outlet_stats, left_on="primary_outlet", right_on="outlet", how="left")
    critic_outlet = critic_outlet.drop(columns=["outlet", "primary_outlet"])
    OUTLET_FEAT_COLS = ["outlet_n_reviews", "outlet_fresh_rate", "outlet_n_critics"]
    print(f"  Outlet features: {OUTLET_FEAT_COLS}")
else:
    critic_outlet = pd.DataFrame({"critic_name": CRITIC_UNIVERSE})
    OUTLET_FEAT_COLS = []

# ══════════════════════════════════════════════════════════
# CRITIC CONTRARIAN SCORE (deviation from consensus)
# ══════════════════════════════════════════════════════════

print("Computing critic contrarian scores...")
# For each review, compute how much the critic disagreed with the movie's overall score
train_movie_avg = df_train.groupby("movie_key")["is_fresh"].mean().reset_index()
train_movie_avg.columns = ["movie_key", "movie_avg_fresh"]
train_with_avg = df_train[df_train["critic_name"].isin(critic_to_idx)].merge(train_movie_avg, on="movie_key")
train_with_avg["deviation"] = (train_with_avg["is_fresh"] - train_with_avg["movie_avg_fresh"]).abs()

critic_contrarian = train_with_avg.groupby("critic_name")["deviation"].mean().reset_index()
critic_contrarian.columns = ["critic_name", "critic_contrarian_score"]
# Also: signed bias (does critic tend to be more positive or negative than consensus?)
critic_bias = train_with_avg.groupby("critic_name").apply(
    lambda x: (x["is_fresh"] - x["movie_avg_fresh"]).mean()
).reset_index()
critic_bias.columns = ["critic_name", "critic_positivity_bias"]
print(f"  critic_contrarian_score, critic_positivity_bias")

print(f"\nAll supplemental features ready.")

# ═══ Cell 17 (id=critic-features) ═══
def build_critic_features(reviews_df, critic_list):
    """
    Build critic-level features including genre interactions, outlet, and contrarian score.
    """
    rev = reviews_df[reviews_df["critic_name"].isin(critic_list)].copy()
    
    # Basic aggregates
    cstats = rev.groupby("critic_name").agg(
        critic_n_reviews=("is_fresh", "size"),
        critic_n_fresh=("is_fresh", "sum"),
        critic_n_movies=("movie_key", "nunique"),
        critic_first_review=("review_date", "min"),
        critic_last_review=("review_date", "max"),
        critic_is_top=("is_top_critic", "max"),
    ).reset_index()
    
    cstats["critic_fresh_rate"] = cstats["critic_n_fresh"] / cstats["critic_n_reviews"]
    career_days = (cstats["critic_last_review"] - cstats["critic_first_review"]).dt.days.clip(lower=30)
    cstats["critic_review_rate_per_year"] = cstats["critic_n_reviews"] / (career_days / 365.25)
    
    # Fill missing
    all_critics_df = pd.DataFrame({"critic_name": critic_list})
    cstats = all_critics_df.merge(cstats, on="critic_name", how="left")
    global_fresh = rev["is_fresh"].mean() if len(rev) > 0 else 0.5
    cstats = cstats.fillna({
        "critic_n_reviews": 0, "critic_n_fresh": 0, "critic_n_movies": 0,
        "critic_is_top": 0, "critic_fresh_rate": global_fresh, "critic_review_rate_per_year": 0,
    })
    cstats = cstats.drop(columns=["critic_first_review", "critic_last_review"], errors="ignore")
    
    # ── Merge NMF latent genre loadings ───────────────────
    cstats = cstats.merge(critic_genre_df, on="critic_name", how="left")
    cstats = cstats.merge(critic_nmf_fresh_df, on="critic_name", how="left")
    
    # ── Merge outlet features ─────────────────────────────
    cstats = cstats.merge(critic_outlet, on="critic_name", how="left")
    
    # ── Merge contrarian / positivity bias ────────────────
    cstats = cstats.merge(critic_contrarian, on="critic_name", how="left")
    cstats = cstats.merge(critic_bias, on="critic_name", how="left")
    
    # Fill all NaN
    for col in cstats.columns:
        if col == "critic_name":
            continue
        if cstats[col].dtype in [float, np.float64]:
            if "fresh" in col or "rate" in col or "bias" in col:
                cstats[col] = cstats[col].fillna(global_fresh if "fresh" in col else 0.0)
            elif "lg_" in col:
                cstats[col] = cstats[col].fillna(1.0 / N_LATENT_GENRES)
            else:
                cstats[col] = cstats[col].fillna(0.0)
    
    return cstats


def build_movie_features(reviews_df, movie_keys):
    """
    Build movie-level features including real genres and NMF latent genres.
    """
    rev = reviews_df[reviews_df["movie_key"].isin(movie_keys)].copy()
    
    mstats = rev.groupby("movie_key").agg(
        movie_n_reviews=("is_fresh", "size"),
        movie_fresh_rate=("is_fresh", "mean"),
        movie_pct_top_critic=("is_top_critic", "mean"),
        movie_n_unique_outlets=("outlet", "nunique") if "outlet" in rev.columns else ("critic_name", "nunique"),
        movie_first_review=("review_date", "min"),
        movie_last_review=("review_date", "max"),
    ).reset_index()
    
    mstats["movie_release_month"] = mstats["movie_first_review"].dt.month
    mstats["movie_release_quarter"] = mstats["movie_first_review"].dt.quarter
    mstats["movie_release_year"] = mstats["movie_first_review"].dt.year
    # Review velocity: reviews per day during the review window
    review_span = (mstats["movie_last_review"] - mstats["movie_first_review"]).dt.days.clip(lower=1)
    mstats["movie_review_velocity"] = mstats["movie_n_reviews"] / review_span
    mstats = mstats.drop(columns=["movie_first_review", "movie_last_review"])
    
    # ── Real genre multi-hot (from IMDB) ──────────────────
    if HAS_GENRES and GENRE_COLS:
        genre_info = rev.groupby("movie_key")[GENRE_COLS].max().reset_index()
        mstats = mstats.merge(genre_info, on="movie_key", how="left")
        for col in GENRE_COLS:
            mstats[col] = mstats[col].fillna(0).astype(int)
    
    # ── NMF latent genre loadings ─────────────────────────
    mstats = mstats.merge(movie_genre_all_df, on="movie_key", how="left")
    for col in LATENT_GENRE_COLS:
        mstats[col] = mstats[col].fillna(1.0 / N_LATENT_GENRES)
    
    return mstats


# ══════════════════════════════════════════════════════════
# BUILD FEATURES
# ══════════════════════════════════════════════════════════
print("Building critic features...")
critic_features = build_critic_features(df_train, CRITIC_UNIVERSE)
print(f"  Shape: {critic_features.shape}")

print("Building movie features...")
movie_features = build_movie_features(df_train, train_movies)
movie_features_val = build_movie_features(df_val, val_movies)
movie_features_all = pd.concat([movie_features, movie_features_val], ignore_index=True)
print(f"  Shape: {movie_features_all.shape}")

# Summarize feature groups
genre_c = [c for c in critic_features.columns if "genre_" in c and "lg_" not in c]
nmf_c = [c for c in critic_features.columns if "lg_" in c]
outlet_c = [c for c in critic_features.columns if "outlet" in c]
other_c = [c for c in critic_features.columns if c not in genre_c + nmf_c + outlet_c + ["critic_name"]]
print(f"\nCritic features breakdown:")
print(f"  Core:    {len(other_c)} — {other_c}")
print(f"  Genre:   {len(genre_c)} — {genre_c[:5]}...")
print(f"  NMF:     {len(nmf_c)}")
print(f"  Outlet:  {len(outlet_c)} — {outlet_c}")

genre_m = [c for c in movie_features_all.columns if "genre_" in c and "lg_" not in c]
nmf_m = [c for c in movie_features_all.columns if "lg_" in c]
other_m = [c for c in movie_features_all.columns if c not in genre_m + nmf_m + ["movie_key"]]
print(f"\nMovie features breakdown:")
print(f"  Core:    {len(other_m)} — {other_m}")
print(f"  Genre:   {len(genre_m)}")
print(f"  NMF:     {len(nmf_m)}")

# ═══ Cell 19 (id=selection-data) ═══
def build_selection_dataset(reviews_df, movie_keys, critic_list, neg_ratio, seed=42):
    """
    Build a binary dataset: did critic c review movie m?
    Uses negative sampling to keep size manageable.
    """
    rng = np.random.RandomState(seed)
    critic_set = set(critic_list)
    
    # Build observed (positive) pairs
    rev = reviews_df[
        reviews_df["movie_key"].isin(movie_keys) &
        reviews_df["critic_name"].isin(critic_set)
    ][["movie_key", "critic_name"]].drop_duplicates()
    
    pos_pairs = set(zip(rev["movie_key"], rev["critic_name"]))
    print(f"  Positive pairs: {len(pos_pairs):,}")
    
    # Build positive rows
    rows = [(m, c, 1) for m, c in pos_pairs]
    
    # Negative sampling: for each movie, sample neg_ratio * n_pos critics who did NOT review
    movie_to_reviewers = rev.groupby("movie_key")["critic_name"].apply(set).to_dict()
    critic_arr = np.array(critic_list)
    
    neg_count = 0
    for mk in movie_keys:
        reviewed_by = movie_to_reviewers.get(mk, set())
        n_pos = len(reviewed_by)
        if n_pos == 0:
            continue
        n_neg = min(n_pos * neg_ratio, len(critic_list) - n_pos)
        if n_neg <= 0:
            continue
        
        # Find non-reviewers
        non_reviewers = [c for c in critic_list if c not in reviewed_by]
        if len(non_reviewers) > n_neg:
            chosen = rng.choice(non_reviewers, size=n_neg, replace=False)
        else:
            chosen = non_reviewers
        
        for c in chosen:
            rows.append((mk, c, 0))
            neg_count += 1
    
    result = pd.DataFrame(rows, columns=["movie_key", "critic_name", "reviewed"])
    print(f"  Negative pairs: {neg_count:,}")
    print(f"  Total selection dataset: {len(result):,} rows")
    return result


print("Building TRAIN selection dataset...")
sel_train = build_selection_dataset(df_train, list(train_movies), CRITIC_UNIVERSE, NEGATIVE_RATIO, seed=RANDOM_SEED)

print("\nBuilding VAL selection dataset...")
sel_val = build_selection_dataset(df_val, list(val_movies), CRITIC_UNIVERSE, NEGATIVE_RATIO, seed=RANDOM_SEED + 1)

# ═══ Cell 20 (id=merge-features) ═══
def merge_features(sel_df, critic_feats, movie_feats):
    """Merge critic and movie features into a selection dataset."""
    merged = sel_df.merge(critic_feats, on="critic_name", how="left")
    merged = merged.merge(movie_feats, on="movie_key", how="left")
    return merged

sel_train_feat = merge_features(sel_train, critic_features, movie_features_all)
sel_val_feat = merge_features(sel_val, critic_features, movie_features_all)

# Define feature columns (everything except identifiers and target)
META_COLS = ["movie_key", "critic_name", "reviewed"]
FEATURE_COLS = [c for c in sel_train_feat.columns if c not in META_COLS]

# Fill any remaining NaN with 0
sel_train_feat[FEATURE_COLS] = sel_train_feat[FEATURE_COLS].fillna(0)
sel_val_feat[FEATURE_COLS] = sel_val_feat[FEATURE_COLS].fillna(0)

print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Train shape: {sel_train_feat.shape}")
print(f"Val shape:   {sel_val_feat.shape}")

# ═══ Cell 22 (id=train-selection) ═══
X_sel_train = sel_train_feat[FEATURE_COLS].values
y_sel_train = sel_train_feat["reviewed"].values
X_sel_val = sel_val_feat[FEATURE_COLS].values
y_sel_val = sel_val_feat["reviewed"].values

print(f"Selection model — Train: {len(y_sel_train):,} (pos rate: {y_sel_train.mean():.4f})")
print(f"Selection model — Val:   {len(y_sel_val):,} (pos rate: {y_sel_val.mean():.4f})")

if USE_LGBM:
    sel_base = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        scale_pos_weight=(1 - y_sel_train.mean()) / y_sel_train.mean(),
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    print("Training selection model (LightGBM + isotonic calibration, cv=3)...")
    sel_model = CalibratedClassifierCV(sel_base, cv=3, method="isotonic")
    sel_model.fit(X_sel_train, y_sel_train)
else:
    scaler_sel = StandardScaler()
    X_sel_train = scaler_sel.fit_transform(X_sel_train)
    X_sel_val = scaler_sel.transform(X_sel_val)
    sel_base = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=300,
        solver="saga", random_state=RANDOM_SEED
    )
    # Use cv=2 + sigmoid for speed with large sklearn datasets
    print("Training selection model (LogisticRegression + sigmoid calibration, cv=2)...")
    sel_model = CalibratedClassifierCV(sel_base, cv=2, method="sigmoid")
    sel_model.fit(X_sel_train, y_sel_train)

# Evaluate
sel_proba_train = sel_model.predict_proba(X_sel_train)[:, 1]
sel_proba_val = sel_model.predict_proba(X_sel_val)[:, 1]

print(f"\n{'='*50}")
print(f"SELECTION MODEL RESULTS")
print(f"{'='*50}")
print(f"  Train AUC:      {roc_auc_score(y_sel_train, sel_proba_train):.4f}")
print(f"  Val AUC:        {roc_auc_score(y_sel_val, sel_proba_val):.4f}")
print(f"  Train Log Loss: {log_loss(y_sel_train, sel_proba_train):.4f}")
print(f"  Val Log Loss:   {log_loss(y_sel_val, sel_proba_val):.4f}")

# ═══ Cell 23 (id=sel-calibration) ═══
# ── Calibration curve for selection model ─────────────────
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
prob_true, prob_pred = calibration_curve(y_sel_val, sel_proba_val, n_bins=10)
ax.plot(prob_pred, prob_true, "o-", label="Selection model")
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Selection Model — Calibration Curve (Val)")
ax.legend()
plt.tight_layout()
plt.show()

# ═══ Cell 25 (id=outcome-data) ═══
# ── Build outcome training data (only observed reviews) ───
def build_outcome_dataset(reviews_df, movie_keys, critic_list, critic_feats, movie_feats):
    """Build dataset for outcome model: only rows where critic actually reviewed."""
    critic_set = set(critic_list)
    rev = reviews_df[
        reviews_df["movie_key"].isin(movie_keys) &
        reviews_df["critic_name"].isin(critic_set)
    ][["movie_key", "critic_name", "is_fresh"]].drop_duplicates(
        subset=["movie_key", "critic_name"], keep="first"
    )
    rev = rev.merge(critic_feats, on="critic_name", how="left")
    rev = rev.merge(movie_feats, on="movie_key", how="left")
    return rev

out_train = build_outcome_dataset(df_train, list(train_movies), CRITIC_UNIVERSE, critic_features, movie_features_all)
out_val = build_outcome_dataset(df_val, list(val_movies), CRITIC_UNIVERSE, critic_features, movie_features_all)

OUT_FEATURE_COLS = [c for c in out_train.columns if c not in ["movie_key", "critic_name", "is_fresh"]]

out_train[OUT_FEATURE_COLS] = out_train[OUT_FEATURE_COLS].fillna(0)
out_val[OUT_FEATURE_COLS] = out_val[OUT_FEATURE_COLS].fillna(0)

print(f"Outcome train: {len(out_train):,} reviews (fresh rate: {out_train['is_fresh'].mean():.3f})")
print(f"Outcome val:   {len(out_val):,} reviews (fresh rate: {out_val['is_fresh'].mean():.3f})")

# ═══ Cell 26 (id=train-outcome) ═══
X_out_train = out_train[OUT_FEATURE_COLS].values
y_out_train = out_train["is_fresh"].values
X_out_val = out_val[OUT_FEATURE_COLS].values
y_out_val = out_val["is_fresh"].values

if USE_LGBM:
    out_base = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    print("Training outcome model (LightGBM + isotonic calibration, cv=3)...")
    out_model = CalibratedClassifierCV(out_base, cv=3, method="isotonic")
    out_model.fit(X_out_train, y_out_train)
else:
    scaler_out = StandardScaler()
    X_out_train = scaler_out.fit_transform(X_out_train)
    X_out_val = scaler_out.transform(X_out_val)
    out_base = LogisticRegression(
        C=1.0, max_iter=300, solver="saga", random_state=RANDOM_SEED
    )
    print("Training outcome model (LogisticRegression + sigmoid calibration, cv=2)...")
    out_model = CalibratedClassifierCV(out_base, cv=2, method="sigmoid")
    out_model.fit(X_out_train, y_out_train)

out_proba_train = out_model.predict_proba(X_out_train)[:, 1]
out_proba_val = out_model.predict_proba(X_out_val)[:, 1]

print(f"\n{'='*50}")
print(f"OUTCOME MODEL RESULTS")
print(f"{'='*50}")
print(f"  Train AUC:      {roc_auc_score(y_out_train, out_proba_train):.4f}")
print(f"  Val AUC:        {roc_auc_score(y_out_val, out_proba_val):.4f}")
print(f"  Train Log Loss: {log_loss(y_out_train, out_proba_train):.4f}")
print(f"  Val Log Loss:   {log_loss(y_out_val, out_proba_val):.4f}")

# ═══ Cell 27 (id=out-calibration) ═══
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
prob_true_o, prob_pred_o = calibration_curve(y_out_val, out_proba_val, n_bins=10)
ax.plot(prob_pred_o, prob_true_o, "o-", label="Outcome model")
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Outcome Model — Calibration Curve (Val)")
ax.legend()
plt.tight_layout()
plt.show()

# ═══ Cell 29 (id=combine) ═══
def predict_selection_proba(movie_key, critic_list, critic_feats, movie_feats):
    """
    For a movie, predict P(review) for each critic in critic_list.
    Returns array of shape (len(critic_list),).
    """
    mf = movie_feats[movie_feats["movie_key"] == movie_key]
    if len(mf) == 0:
        return np.full(len(critic_list), 0.01)  # fallback
    
    cf = critic_feats[critic_feats["critic_name"].isin(critic_list)].copy()
    cf = cf.set_index("critic_name").loc[critic_list].reset_index()  # maintain order
    
    # Cross join
    cf["_key"] = 1
    mf_row = mf.iloc[[0]].copy()
    mf_row["_key"] = 1
    merged = cf.merge(mf_row.drop(columns=["movie_key"]), on="_key").drop(columns=["_key"])
    
    X = merged.drop(columns=["critic_name"]).fillna(0).values
    
    if not USE_LGBM and hasattr(sel_model, 'estimators_'):
        # If we used scaler, need to transform
        try:
            X = scaler_sel.transform(X)
        except:
            pass
    
    return sel_model.predict_proba(X)[:, 1]


def predict_outcome_proba(movie_key, critic_list, critic_feats, movie_feats):
    """
    For a movie, predict P(fresh | reviewed) for each critic.
    Returns array of shape (len(critic_list),).
    """
    mf = movie_feats[movie_feats["movie_key"] == movie_key]
    if len(mf) == 0:
        return np.full(len(critic_list), 0.5)  # fallback
    
    cf = critic_feats[critic_feats["critic_name"].isin(critic_list)].copy()
    cf = cf.set_index("critic_name").loc[critic_list].reset_index()
    
    cf["_key"] = 1
    mf_row = mf.iloc[[0]].copy()
    mf_row["_key"] = 1
    merged = cf.merge(mf_row.drop(columns=["movie_key"]), on="_key").drop(columns=["_key"])
    
    feat_cols = [c for c in merged.columns if c != "critic_name"]
    X = merged[feat_cols].fillna(0).values
    
    if not USE_LGBM:
        try:
            X = scaler_out.transform(X)
        except:
            pass
    
    return out_model.predict_proba(X)[:, 1]


def expected_tomatometer(movie_key, critic_feats, movie_feats):
    """Compute E[Tomatometer] = sum(p*q) / sum(p) over all critics."""
    p = predict_selection_proba(movie_key, CRITIC_UNIVERSE, critic_feats, movie_feats)
    q = predict_outcome_proba(movie_key, CRITIC_UNIVERSE, critic_feats, movie_feats)
    denom = p.sum()
    if denom < 1e-8:
        return 0.5
    return (p * q).sum() / denom


# Quick test on one movie
test_mk = list(val_movies)[0]
et = expected_tomatometer(test_mk, critic_features, movie_features_all)
actual = movie_stats[movie_stats["movie_key"] == test_mk]["tomatometer"].values[0]
print(f"Movie: {test_mk}")
print(f"  Expected Tomatometer: {et:.3f}")
print(f"  Actual Tomatometer:   {actual:.3f}")

# ═══ Cell 31 (id=nowcast) ═══
def nowcast_movie(
    movie_key,
    as_of_datetime,
    reviews_df,
    critic_feats,
    movie_feats,
    n_sims=N_SIMS,
    thresholds=THRESHOLDS,
    seed=RANDOM_SEED,
):
    """
    Nowcast the final Tomatometer for a movie given reviews up to as_of_datetime.
    
    Returns dict with:
      - point_estimate: mean of simulated final tomatometers
      - median, p5, p25, p75, p95: percentiles
      - threshold_probs: dict {threshold: P(final_T >= threshold)}
      - observed: dict with S_obs, N_obs, current_T
      - simulations: array of n_sims final tomatometer values
    """
    rng = np.random.RandomState(seed)
    as_of = pd.Timestamp(as_of_datetime)
    
    # ── Observed reviews ──────────────────────────────────
    movie_revs = reviews_df[
        (reviews_df["movie_key"] == movie_key) &
        (reviews_df["review_date"] <= as_of)
    ].copy()
    
    # De-dup by critic (keep first review)
    movie_revs = movie_revs.sort_values("review_date").drop_duplicates(
        subset=["critic_name"], keep="first"
    )
    
    observed_critics = set(movie_revs["critic_name"])
    S_obs = int(movie_revs["is_fresh"].sum())
    N_obs = len(movie_revs)
    current_T = S_obs / N_obs if N_obs > 0 else 0.0
    
    # ── Remaining critics ─────────────────────────────────
    remaining = [c for c in CRITIC_UNIVERSE if c not in observed_critics]
    
    if len(remaining) == 0:
        # All critics have reviewed
        sims = np.full(n_sims, current_T)
    else:
        # Get p (selection) and q (outcome) for remaining critics
        p = predict_selection_proba(movie_key, remaining, critic_feats, movie_feats)
        q = predict_outcome_proba(movie_key, remaining, critic_feats, movie_feats)
        
        # Monte Carlo simulation
        # Vectorized: shape (n_sims, n_remaining)
        n_rem = len(remaining)
        
        # Sample review decisions
        U_review = rng.random((n_sims, n_rem))
        R = (U_review < p[np.newaxis, :]).astype(int)  # 1 if will review
        
        # Sample fresh decisions (only matters where R=1)
        U_fresh = rng.random((n_sims, n_rem))
        F = (U_fresh < q[np.newaxis, :]).astype(int) * R  # 1 if fresh AND reviewed
        
        sim_new_reviews = R.sum(axis=1)   # additional reviews per sim
        sim_new_fresh = F.sum(axis=1)     # additional fresh per sim
        
        total_reviews = N_obs + sim_new_reviews
        total_fresh = S_obs + sim_new_fresh
        
        # Handle edge case: no reviews at all
        sims = np.where(
            total_reviews > 0,
            total_fresh / total_reviews,
            0.5  # fallback
        )
    
    # ── Compute summary statistics ────────────────────────
    result = {
        "movie_key": movie_key,
        "as_of": as_of,
        "point_estimate": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "p5": float(np.percentile(sims, 5)),
        "p25": float(np.percentile(sims, 25)),
        "p75": float(np.percentile(sims, 75)),
        "p95": float(np.percentile(sims, 95)),
        "std": float(np.std(sims)),
        "threshold_probs": {t: float((sims >= t).mean()) for t in thresholds},
        "observed": {
            "S_obs": S_obs,
            "N_obs": N_obs,
            "current_T": current_T,
            "n_remaining_critics": len(remaining),
        },
        "simulations": sims,
    }
    return result


def print_nowcast(result):
    """Pretty-print a nowcast result."""
    obs = result["observed"]
    print(f"\n{'='*60}")
    print(f"  NOWCAST: {result['movie_key']}")
    print(f"  As of:   {result['as_of']}")
    print(f"{'='*60}")
    print(f"  Observed: {obs['S_obs']}/{obs['N_obs']} Fresh  "
          f"(current {obs['current_T']:.1%})")
    print(f"  Remaining critics in universe: {obs['n_remaining_critics']}")
    print(f"")
    print(f"  Point estimate (mean): {result['point_estimate']:.1%}")
    print(f"  Median:                {result['median']:.1%}")
    print(f"  90% CI:  [{result['p5']:.1%}, {result['p95']:.1%}]")
    print(f"  50% CI:  [{result['p25']:.1%}, {result['p75']:.1%}]")
    print(f"  Std dev:               {result['std']:.3f}")
    print(f"")
    print(f"  Threshold probabilities:")
    for t, p in result["threshold_probs"].items():
        print(f"    P(T >= {t:.0%}): {p:.1%}")


print("Nowcast function defined.")

# ═══ Cell 32 (id=nowcast-demo) ═══
# ── Quick demo on one validation movie ────────────────────
demo_mk = list(val_movies)[0]
demo_revs = df_val[df_val["movie_key"] == demo_mk].sort_values("review_date")
demo_actual_T = movie_stats[movie_stats["movie_key"] == demo_mk]["tomatometer"].values[0]

print(f"Demo movie: {demo_mk}")
print(f"Total reviews: {len(demo_revs)}, Actual final T: {demo_actual_T:.1%}")

# Nowcast after 5 reviews
if len(demo_revs) >= 5:
    as_of = demo_revs.iloc[4]["review_date"]
    result = nowcast_movie(demo_mk, as_of, df, critic_features, movie_features_all)
    print_nowcast(result)
    print(f"\n  ** Actual final Tomatometer: {demo_actual_T:.1%} **")

# ═══ Cell 34 (id=backtest) ═══
def run_backtest(val_movie_keys, reviews_df, critic_feats, movie_feats,
                 movie_stats_df, checkpoints=BACKTEST_CHECKPOINTS,
                 n_sims=2000, max_movies=100):
    """
    Run backtest: for each val movie, nowcast at each checkpoint and compare to actual.
    Uses fewer sims for speed.
    """
    results = []
    movie_list = list(val_movie_keys)[:max_movies]
    
    for i, mk in enumerate(movie_list):
        if (i + 1) % 20 == 0:
            print(f"  Backtesting movie {i+1}/{len(movie_list)}...")
        
        # Get actual tomatometer
        actual_row = movie_stats_df[movie_stats_df["movie_key"] == mk]
        if len(actual_row) == 0:
            continue
        actual_T = actual_row["tomatometer"].values[0]
        total_reviews = actual_row["n_reviews"].values[0]
        
        # Sort reviews by date
        m_revs = reviews_df[reviews_df["movie_key"] == mk].sort_values("review_date")
        
        for cp in checkpoints:
            if cp > len(m_revs):
                continue
            
            as_of = m_revs.iloc[cp - 1]["review_date"]
            
            try:
                nc = nowcast_movie(
                    mk, as_of, reviews_df, critic_feats, movie_feats,
                    n_sims=n_sims, seed=RANDOM_SEED + i
                )
            except Exception as e:
                continue
            
            row = {
                "movie_key": mk,
                "checkpoint": cp,
                "pct_reviews_seen": cp / total_reviews,
                "actual_T": actual_T,
                "predicted_T": nc["point_estimate"],
                "median_T": nc["median"],
                "p5": nc["p5"],
                "p95": nc["p95"],
                "current_naive_T": nc["observed"]["current_T"],
                "error": nc["point_estimate"] - actual_T,
                "naive_error": nc["observed"]["current_T"] - actual_T,
            }
            # Threshold event actuals
            for t in THRESHOLDS:
                row[f"actual_above_{int(t*100)}"] = int(actual_T >= t)
                row[f"pred_prob_{int(t*100)}"] = nc["threshold_probs"][t]
            
            results.append(row)
    
    return pd.DataFrame(results)


print("Running backtest on validation movies...")
bt = run_backtest(
    val_movies, df, critic_features, movie_features_all,
    movie_stats, max_movies=100, n_sims=2000
)
print(f"Backtest complete: {len(bt)} predictions across {bt['movie_key'].nunique()} movies")

# ═══ Cell 35 (id=backtest-metrics) ═══
# ── Backtest Metrics ──────────────────────────────────────
print(f"{'='*70}")
print(f"BACKTEST RESULTS")
print(f"{'='*70}")

for cp in BACKTEST_CHECKPOINTS:
    subset = bt[bt["checkpoint"] == cp]
    if len(subset) == 0:
        continue
    
    mae_model = mean_absolute_error(subset["actual_T"], subset["predicted_T"])
    rmse_model = np.sqrt(mean_squared_error(subset["actual_T"], subset["predicted_T"]))
    mae_naive = mean_absolute_error(subset["actual_T"], subset["current_naive_T"])
    rmse_naive = np.sqrt(mean_squared_error(subset["actual_T"], subset["current_naive_T"]))
    
    print(f"\n  After {cp} reviews (n={len(subset)} movies):")
    print(f"    MODEL  — MAE: {mae_model:.4f}  RMSE: {rmse_model:.4f}")
    print(f"    NAIVE  — MAE: {mae_naive:.4f}  RMSE: {rmse_naive:.4f}")
    print(f"    Improvement:   MAE {(1 - mae_model/mae_naive)*100:+.1f}%  RMSE {(1 - rmse_model/rmse_naive)*100:+.1f}%")

# Brier scores for threshold events
print(f"\n{'='*70}")
print(f"BRIER SCORES (lower is better)")
print(f"{'='*70}")

brier_rows = []
for cp in BACKTEST_CHECKPOINTS:
    subset = bt[bt["checkpoint"] == cp]
    if len(subset) == 0:
        continue
    row = {"checkpoint": cp, "n": len(subset)}
    for t in THRESHOLDS:
        t_int = int(t * 100)
        y_true = subset[f"actual_above_{t_int}"].values
        y_prob = subset[f"pred_prob_{t_int}"].values
        if len(np.unique(y_true)) > 1:
            brier = brier_score_loss(y_true, y_prob)
            row[f"Brier_>={t_int}%"] = f"{brier:.4f}"
        else:
            row[f"Brier_>={t_int}%"] = "N/A"
    brier_rows.append(row)

brier_df = pd.DataFrame(brier_rows)
print(brier_df.to_string(index=False))

# ═══ Cell 36 (id=backtest-plots) ═══
# ── Backtest visualization ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: MAE by checkpoint
ax = axes[0]
mae_data = []
for cp in BACKTEST_CHECKPOINTS:
    s = bt[bt["checkpoint"] == cp]
    if len(s) == 0:
        continue
    mae_data.append({
        "checkpoint": cp,
        "Model": mean_absolute_error(s["actual_T"], s["predicted_T"]),
        "Naive": mean_absolute_error(s["actual_T"], s["current_naive_T"]),
    })
mae_df = pd.DataFrame(mae_data)
ax.plot(mae_df["checkpoint"], mae_df["Model"], "o-", label="Model")
ax.plot(mae_df["checkpoint"], mae_df["Naive"], "s--", label="Naive (current avg)")
ax.set_xlabel("Reviews seen")
ax.set_ylabel("MAE")
ax.set_title("MAE vs. Reviews Seen")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Predicted vs Actual scatter
ax = axes[1]
for cp in BACKTEST_CHECKPOINTS:
    s = bt[bt["checkpoint"] == cp]
    if len(s) == 0:
        continue
    ax.scatter(s["actual_T"], s["predicted_T"], alpha=0.4, s=20, label=f"{cp} reviews")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_xlabel("Actual Tomatometer")
ax.set_ylabel("Predicted Tomatometer")
ax.set_title("Predicted vs Actual")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ═══ Cell 37 (id=calibration-backtest) ═══
# ── Threshold calibration plots ───────────────────────────
fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(4 * len(THRESHOLDS), 4))
if len(THRESHOLDS) == 1:
    axes = [axes]

for ax, t in zip(axes, THRESHOLDS):
    t_int = int(t * 100)
    y_true = bt[f"actual_above_{t_int}"].values
    y_prob = bt[f"pred_prob_{t_int}"].values
    
    if len(np.unique(y_true)) < 2:
        ax.set_title(f">={t_int}% (insufficient data)")
        continue
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8)
    ax.plot(prob_pred, prob_true, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Predicted P")
    ax.set_ylabel("Observed P")
    ax.set_title(f"T >= {t_int}%")
    ax.grid(True, alpha=0.3)

fig.suptitle("Threshold Calibration (All Checkpoints Pooled)", y=1.02)
plt.tight_layout()
plt.show()

# ═══ Cell 39 (id=worked-example) ═══
# ── Pick a well-reviewed validation movie for the worked example ──
val_stats = movie_stats[movie_stats["movie_key"].isin(val_movies)].copy()
val_stats = val_stats.sort_values("n_reviews", ascending=False)

# Pick top movie by review count
example_mk = val_stats.iloc[0]["movie_key"]
example_actual_T = val_stats.iloc[0]["tomatometer"]
example_n = int(val_stats.iloc[0]["n_reviews"])

print(f"Worked example movie: {example_mk}")
print(f"Total reviews: {example_n}")
print(f"Actual final Tomatometer: {example_actual_T:.1%}")
print()

# Get sorted reviews
ex_revs = df[df["movie_key"] == example_mk].sort_values("review_date")

# Define checkpoints
ex_checkpoints = [5, 10, 25, 50, 100]
ex_checkpoints = [cp for cp in ex_checkpoints if cp <= len(ex_revs)]

timeline = []
for cp in ex_checkpoints:
    as_of = ex_revs.iloc[cp - 1]["review_date"]
    nc = nowcast_movie(example_mk, as_of, df, critic_features, movie_features_all, n_sims=5000)
    print_nowcast(nc)
    print(f"  ** Actual final Tomatometer: {example_actual_T:.1%} **")
    
    timeline.append({
        "reviews_seen": cp,
        "date": as_of,
        "current_T": nc["observed"]["current_T"],
        "predicted_T": nc["point_estimate"],
        "p5": nc["p5"],
        "p95": nc["p95"],
    })

# ═══ Cell 40 (id=timeline-plot) ═══
# ── Timeline visualization ────────────────────────────────
if len(timeline) > 1:
    tl = pd.DataFrame(timeline)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.fill_between(tl["reviews_seen"], tl["p5"], tl["p95"],
                    alpha=0.2, color="C0", label="90% CI")
    ax.plot(tl["reviews_seen"], tl["predicted_T"], "o-", color="C0",
            label="Model prediction", linewidth=2)
    ax.plot(tl["reviews_seen"], tl["current_T"], "s--", color="C1",
            label="Naive (current avg)", linewidth=1.5)
    ax.axhline(y=example_actual_T, color="red", linestyle=":",
               linewidth=2, label=f"Actual ({example_actual_T:.0%})")
    
    ax.set_xlabel("Number of reviews seen")
    ax.set_ylabel("Tomatometer")
    ax.set_title(f"Nowcast Timeline: {example_mk}")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Not enough checkpoints for a timeline plot.")

# ═══ Cell 41 (id=histogram-example) ═══
# ── Histogram of final simulation at earliest checkpoint ──
if len(ex_checkpoints) > 0:
    early_cp = ex_checkpoints[0]
    as_of = ex_revs.iloc[early_cp - 1]["review_date"]
    nc = nowcast_movie(example_mk, as_of, df, critic_features, movie_features_all, n_sims=10000)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nc["simulations"], bins=50, density=True, alpha=0.7, color="C0",
            edgecolor="white")
    ax.axvline(x=example_actual_T, color="red", linestyle="--", linewidth=2,
               label=f"Actual ({example_actual_T:.0%})")
    ax.axvline(x=nc["point_estimate"], color="C0", linestyle="-", linewidth=2,
               label=f"Predicted ({nc['point_estimate']:.0%})")
    
    # Mark thresholds
    for t in THRESHOLDS:
        p = nc["threshold_probs"][t]
        ax.axvline(x=t, color="gray", linestyle=":", alpha=0.5)
        ax.text(t, ax.get_ylim()[1] * 0.9, f"{t:.0%}\nP={p:.0%}",
                ha="center", fontsize=8)
    
    ax.set_xlabel("Simulated Final Tomatometer")
    ax.set_ylabel("Density")
    ax.set_title(f"Monte Carlo Distribution after {early_cp} reviews — {example_mk}")
    ax.legend()
    plt.tight_layout()
    plt.show()

# ═══ Cell 43 (id=summary-table) ═══
# ── Summary table of backtest results ─────────────────────
summary_rows = []
for cp in BACKTEST_CHECKPOINTS:
    s = bt[bt["checkpoint"] == cp]
    if len(s) == 0:
        continue
    row = {
        "Reviews Seen": cp,
        "N Movies": len(s),
        "Model MAE": f"{mean_absolute_error(s['actual_T'], s['predicted_T']):.4f}",
        "Naive MAE": f"{mean_absolute_error(s['actual_T'], s['current_naive_T']):.4f}",
        "Model RMSE": f"{np.sqrt(mean_squared_error(s['actual_T'], s['predicted_T'])):.4f}",
        "Naive RMSE": f"{np.sqrt(mean_squared_error(s['actual_T'], s['current_naive_T'])):.4f}",
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print("BACKTEST SUMMARY")
print("=" * 80)
print(summary_df.to_string(index=False))
print("\nBrier scores by threshold:")
print(brier_df.to_string(index=False))
print("\n" + "=" * 80)
print("Model compares predicted final Tomatometer against the naive baseline")
print("(which simply uses the current average of observed reviews).")
print("Lower MAE/RMSE/Brier = better.")

# ═══ Cell 44 (id=final-note) ═══
print("Notebook complete.")
print(f"\nDataset: {CSV_PATH}")
print(f"Train movies: {len(train_movies):,} | Val movies: {len(val_movies):,}")
print(f"Critic universe: {n_critics:,}")
print(f"Cutoff date: {CUTOFF_DATE}")
print(f"\nTo nowcast a specific movie:")
print(f'  result = nowcast_movie("m/movie_slug", "2019-06-15", df, critic_features, movie_features_all)')
print(f'  print_nowcast(result)')

# ═══ Cell 46 (id=5x4mw2ucc1u) ═══
def register_new_movie(movie_key, reviews_list, release_month=None, genres=None):
    """
    Register a brand-new movie for nowcasting.
    
    Parameters
    ----------
    movie_key : str
        Unique identifier, e.g. "m/wuthering_heights_2025"
    reviews_list : list of dicts
        Each dict: {"critic_name": str, "is_fresh": 0 or 1, "review_date": str}
    release_month : int, optional
        Release month (1-12). If None, inferred from earliest review.
    genres : list of str, optional
        Genre labels, e.g. ["drama", "romance"]. Must match the genre names
        from the IMDB dataset (lowercase). If None, genres are set to 0.
    """
    global df, movie_features_all, movie_genre_all_df
    
    new_revs = pd.DataFrame(reviews_list)
    new_revs["critic_name"] = new_revs["critic_name"].str.strip().str.lower()
    new_revs["review_date"] = pd.to_datetime(new_revs["review_date"])
    new_revs["is_fresh"] = new_revs["is_fresh"].astype(int)
    new_revs["movie_key"] = movie_key
    new_revs["is_top_critic"] = 0
    if "outlet" not in new_revs.columns:
        new_revs["outlet"] = "unknown"
    
    # Set genre columns
    if HAS_GENRES and GENRE_COLS:
        for col in GENRE_COLS:
            new_revs[col] = 0
        if genres:
            for g in genres:
                g_lower = g.lower().replace(' ', '_').replace('-', '_')
                matching = [c for c in GENRE_COLS if g_lower in c]
                for mc in matching:
                    new_revs[mc] = 1
    
    for c in df.columns:
        if c not in new_revs.columns:
            new_revs[c] = np.nan
    df = pd.concat([df, new_revs[df.columns]], ignore_index=True)
    
    # ── NMF genre loadings ────────────────────────────────
    rev_critics = set(new_revs["critic_name"]) & set(critic_to_idx.keys())
    new_vec = np.zeros((1, n_critics))
    for c in rev_critics:
        new_vec[0, critic_to_idx[c]] = 1.0
    if new_vec.sum() > 0:
        W_new = nmf_model.transform(csr_matrix(new_vec))
        W_new_norm = W_new / (W_new.sum() + 1e-8)
    else:
        W_new_norm = np.full((1, N_LATENT_GENRES), 1.0 / N_LATENT_GENRES)
    
    genre_row = pd.DataFrame(W_new_norm, columns=LATENT_GENRE_COLS)
    genre_row["movie_key"] = movie_key
    movie_genre_all_df = movie_genre_all_df[movie_genre_all_df["movie_key"] != movie_key]
    movie_genre_all_df = pd.concat([movie_genre_all_df, genre_row], ignore_index=True)
    
    # ── Movie features ────────────────────────────────────
    first_date = new_revs["review_date"].min()
    r_month = release_month if release_month else first_date.month
    
    mf_row = {
        "movie_key": movie_key,
        "movie_n_reviews": len(new_revs),
        "movie_fresh_rate": new_revs["is_fresh"].mean(),
        "movie_pct_top_critic": 0.0,
        "movie_n_unique_outlets": new_revs["outlet"].nunique(),
        "movie_release_month": r_month,
        "movie_release_quarter": (r_month - 1) // 3 + 1,
        "movie_release_year": first_date.year,
        "movie_review_velocity": len(new_revs),  # approx
    }
    # Genre multi-hot
    if HAS_GENRES and GENRE_COLS:
        for col in GENRE_COLS:
            mf_row[col] = int(new_revs[col].max()) if col in new_revs.columns else 0
    # NMF
    for i, col in enumerate(LATENT_GENRE_COLS):
        mf_row[col] = float(W_new_norm[0, i])
    
    mf_df = pd.DataFrame([mf_row])
    movie_features_all = movie_features_all[movie_features_all["movie_key"] != movie_key]
    movie_features_all = pd.concat([movie_features_all, mf_df], ignore_index=True)
    
    genre_str = ", ".join(genres) if genres else "none specified"
    print(f"Registered '{movie_key}': {len(new_revs)} reviews, fresh rate {new_revs['is_fresh'].mean():.0%}")
    print(f"  Genres: {genre_str}")
    return new_revs

print("register_new_movie() defined.")

# ═══ Cell 47 (id=69k0n9uxdne) ═══
# ── Example: Wuthering Heights ────────────────────────────
# Fill in real early reviews here. Critic names should be lowercase
# and ideally match names in CRITIC_UNIVERSE for best results.
# (To see available critics: print(CRITIC_UNIVERSE[:50]))
#
# You can check if a critic is in the universe:
#   "peter bradshaw" in set(CRITIC_UNIVERSE)

wuthering_reviews = [
    # ---- Fill in actual reviews as they come in ----
    # Format: critic_name (lowercase), is_fresh (1=Fresh, 0=Rotten), review_date
    {"critic_name": "david gonzalez",    "is_fresh": 1, "review_date": "2025-02-10"},
    {"critic_name": "sophie ciminello",  "is_fresh": 1, "review_date": "2025-02-10"},
    {"critic_name": "rocco t thompson",    "is_fresh": 0, "review_date": "2025-02-11"},
    {"critic_name": "matt neglia",       "is_fresh": 1, "review_date": "2025-02-11"},
    {"critic_name": "liz shannon miller", "is_fresh": 1, "review_date": "2025-02-12"},
    {"critic_name": "joey magidson",      "is_fresh": 1, "review_date": "2025-02-12"},
    {"critic_name": "linda marric",        "is_fresh": 1, "review_date": "2025-02-13"},
    # ---- Add more as they come in ----
]

# Register the movie (creates features + appends reviews to df)
register_new_movie("m/wuthering_heights_2025", wuthering_reviews,
                   release_month=2, genres=["drama", "romance"])

# Check which of these critics are in our universe (model knows them)
wh_critics = [r["critic_name"].strip().lower() for r in wuthering_reviews]
in_universe = [c for c in wh_critics if c in set(CRITIC_UNIVERSE)]
print(f"\n{len(in_universe)}/{len(wh_critics)} reviewers are in the critic universe: {in_universe}")

# ═══ Cell 48 (id=cq5xbnowt9) ═══
# ── Run nowcast ───────────────────────────────────────────
# as_of_datetime controls how many reviews are "visible" to the model.
# Set it to the current date to use ALL reviews you entered above,
# or set it earlier to simulate "what if we only had reviews through date X?"

result = nowcast_movie(
    movie_key="m/wuthering_heights_2025",
    as_of_datetime="2025-02-15",       # <-- today's date; sees all 7 reviews above
    reviews_df=df,
    critic_feats=critic_features,
    movie_feats=movie_features_all,
    n_sims=10_000,
)
print_nowcast(result)

# ── You can also simulate "what if we only had the first 2 reviews?" ──
result_early = nowcast_movie(
    movie_key="m/wuthering_heights_2025",
    as_of_datetime="2025-02-10",        # only sees the Feb 10 reviews
    reviews_df=df,
    critic_feats=critic_features,
    movie_feats=movie_features_all,
    n_sims=10_000,
)
print_nowcast(result_early)

# ═══ Cell 49 (id=1t0218nlp25) ═══
# ── Histogram of the Monte Carlo distribution ────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(result["simulations"], bins=50, density=True, alpha=0.7, color="C0", edgecolor="white")
ax.axvline(x=result["point_estimate"], color="C0", linewidth=2, label=f"Predicted ({result['point_estimate']:.0%})")
for t in THRESHOLDS:
    p = result["threshold_probs"][t]
    ax.axvline(x=t, color="gray", linestyle=":", alpha=0.5)
    ax.text(t, ax.get_ylim()[1] * 0.85, f"{t:.0%}\nP={p:.0%}", ha="center", fontsize=8)
ax.set_xlabel("Simulated Final Tomatometer")
ax.set_ylabel("Density")
ax.set_title(f"Wuthering Heights (2025) — Monte Carlo after {result['observed']['N_obs']} reviews")
ax.legend()
plt.tight_layout()
plt.show()

# ═══ Cell 50 (id=induced-portugal) ═══


# ═══ Cell 51 (id=considered-module) ═══

