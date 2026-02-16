#!/usr/bin/env python3
# %%
# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "rotten_tomatoes_critic_reviews.csv"
IMDB_GENRES_PATH = "imdb_genres.csv"
EXTERNAL_METADATA_PATH = None  # Optional local CSV with movie-level metadata

COLUMN_MAP = {
    "movie_id": "rotten_tomatoes_link",
    "movie_title": "movie_title",
    "critic_name": "critic_name",
    "outlet": "publisher_name",
    "review_date": "review_date",
    "review_type": "review_type",
    "is_fresh": None,
    "review_score": "review_score",
    "is_top_critic": "top_critic",
    "movie_genres": None,
    "wide_release_date": None,
    "runtime": None,
    "mpaa_rating": None,
    "studio": None,
    "director": None,
    "cast": None,
}

NEGATIVE_RATIO = 8
MIN_REVIEWS_PER_MOVIE = 20
MIN_REVIEWS_PER_CRITIC = 5
MAX_CRITICS_UNIVERSE = 3000
CUTOFF_DATE = "2019-01-01"
MAX_REVIEW_DATE = None
RANDOM_SEED = 42

SELECTION_CHECKPOINTS = [5, 10, 25]
BACKTEST_CHECKPOINTS = [5, 10, 25]
RECENT_WINDOWS_DAYS = [7, 30, 90]

# Local runtime controls for laptop execution
MAX_SELECTION_TRAIN_MOVIES = 2500
MAX_SELECTION_VAL_MOVIES = 600
MAX_OUTCOME_TRAIN_MOVIES = 3500
MAX_OUTCOME_VAL_MOVIES = 700
PROGRESS_EVERY_MOVIES = 100
MAX_SELECTION_TRAIN_ROWS = 1500000
MAX_SELECTION_VAL_ROWS = 450000
MAX_OUTCOME_TRAIN_ROWS = 500000
MAX_OUTCOME_VAL_ROWS = 200000

N_SIMS = 10000
THRESHOLDS = [0.60, 0.70, 0.80, 0.90]
MAX_BACKTEST_MOVIES = 100

# %%
import warnings
warnings.filterwarnings("ignore")

import os
import re
from typing import Dict, List, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(RANDOM_SEED)

try:
    import lightgbm as lgb
    USE_LGBM = True
except (ImportError, OSError):
    USE_LGBM = False

print(f"LightGBM enabled: {USE_LGBM}")

# %%
def safe_rename(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    rename_dict = {}
    for std_name, csv_name in mapping.items():
        if csv_name is not None and csv_name in df.columns and csv_name != std_name:
            rename_dict[csv_name] = std_name
    return df.rename(columns=rename_dict)


def normalize_title(t: str) -> str:
    t = str(t).lower().strip()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    for article in ("the ", "a ", "an "):
        if t.startswith(article):
            t = t[len(article):]
    return t


def slug_to_title(slug: str) -> str:
    s = str(slug).replace("m/", "")
    s = re.sub(r"^\d+[-_]", "", s)
    return s.replace("_", " ").strip().lower()


def parse_runtime_to_minutes(v) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s.isdigit():
        return float(int(s))
    m = re.search(r"(\d+)\s*min", s)
    if m:
        return float(int(m.group(1)))
    m = re.search(r"(\d+)\s*h(?:our|rs?)?\s*(\d+)?", s)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2)) if m.group(2) else 0
        return float(60 * h + mm)
    return np.nan


def split_multi(v, sep="|") -> List[str]:
    if pd.isna(v):
        return []
    s = str(v).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [x.strip().strip("'").strip('"').lower() for x in s.split(sep)]
    return [p for p in parts if p]


def score_to_fresh(v) -> Optional[int]:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    # Percent score
    if s.endswith("%"):
        try:
            return int(float(s[:-1]) >= 60.0)
        except ValueError:
            return None
    # Fraction score like 3.5/5, 7/10, 2/4
    m = re.match(r"^\s*(\d+(\.\d+)?)\s*/\s*(\d+(\.\d+)?)\s*$", s)
    if m:
        num = float(m.group(1))
        den = float(m.group(3))
        if den > 0:
            return int((num / den) >= 0.6)
    # Letter grades (simple map)
    letter = s.replace("+", "").replace("-", "")
    letter_map = {"a": 1, "b": 1, "c": 0, "d": 0, "f": 0}
    if letter in letter_map:
        return letter_map[letter]
    # Plain numeric
    try:
        x = float(s)
        if x <= 10:
            return int(x >= 6)
        return int(x >= 60)
    except ValueError:
        return None


def ensure_movie_key(df: pd.DataFrame) -> pd.Series:
    if "movie_id" in df.columns and df["movie_id"].notna().any():
        return df["movie_id"].astype(str)
    if "movie_title" in df.columns:
        title = df["movie_title"].fillna("unknown_title").astype(str).str.strip().str.lower()
        if "wide_release_date" in df.columns:
            rel = pd.to_datetime(df["wide_release_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("na")
            return "title::" + title + "::" + rel
        return "title::" + title
    raise ValueError("Need movie_id or movie_title to create movie_key")


# %%
raw = pd.read_csv(CSV_PATH)
df = safe_rename(raw.copy(), COLUMN_MAP)

required = ["review_date", "critic_name"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
df = df[df["review_date"].notna()].copy()
df = df[df["review_date"] >= "1990-01-01"].copy()
if MAX_REVIEW_DATE is not None:
    df = df[df["review_date"] <= pd.Timestamp(MAX_REVIEW_DATE)].copy()

df["critic_name"] = df["critic_name"].astype(str).str.strip().str.lower()
df = df[(df["critic_name"] != "") & (df["critic_name"] != "nan")].copy()

if "is_fresh" in df.columns and df["is_fresh"].notna().any():
    df["is_fresh"] = pd.to_numeric(df["is_fresh"], errors="coerce")
else:
    df["is_fresh"] = np.nan

if df["is_fresh"].isna().all():
    if "review_type" in df.columns:
        rt = df["review_type"].astype(str).str.strip().str.lower()
        df["is_fresh"] = np.where(rt == "fresh", 1, np.where(rt == "rotten", 0, np.nan))

if df["is_fresh"].isna().any() and "review_score" in df.columns:
    parsed = df["review_score"].map(score_to_fresh)
    df["is_fresh"] = df["is_fresh"].fillna(parsed)

df = df[df["is_fresh"].notna()].copy()
df["is_fresh"] = df["is_fresh"].astype(int).clip(0, 1)

if "is_top_critic" in df.columns:
    df["is_top_critic"] = df["is_top_critic"].map(
        {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
    ).fillna(0).astype(int)
else:
    df["is_top_critic"] = 0

if "wide_release_date" in df.columns:
    df["wide_release_date"] = pd.to_datetime(df["wide_release_date"], errors="coerce")

df["movie_key"] = ensure_movie_key(df)

print(f"Reviews: {len(df):,}, movies: {df['movie_key'].nunique():,}, critics: {df['critic_name'].nunique():,}")

# %%
# ============================================================
# GENRE + OPTIONAL EXTERNAL METADATA
# ============================================================

movie_level = df.groupby("movie_key").agg(
    first_review=("review_date", "min"),
    n_reviews=("is_fresh", "size"),
).reset_index()

movie_level = movie_level[movie_level["n_reviews"] >= MIN_REVIEWS_PER_MOVIE].copy()
valid_movies = set(movie_level["movie_key"])
df = df[df["movie_key"].isin(valid_movies)].copy()

movie_level = df.groupby("movie_key").agg(
    first_review=("review_date", "min"),
    n_reviews=("is_fresh", "size"),
).reset_index()

movie_meta = df.groupby("movie_key").first().reset_index()[["movie_key"]]

optional_cols = ["runtime", "mpaa_rating", "studio", "director", "cast", "movie_genres", "wide_release_date", "movie_title"]
for c in optional_cols:
    if c in df.columns:
        movie_meta = movie_meta.merge(
            df.groupby("movie_key")[c].first().reset_index(),
            on="movie_key",
            how="left",
        )

if EXTERNAL_METADATA_PATH and os.path.exists(EXTERNAL_METADATA_PATH):
    ext = pd.read_csv(EXTERNAL_METADATA_PATH)
    ext = safe_rename(ext, COLUMN_MAP)
    if "movie_key" not in ext.columns:
        ext["movie_key"] = ensure_movie_key(ext)
    keep = ["movie_key"] + [c for c in optional_cols if c in ext.columns]
    ext = ext[keep].drop_duplicates("movie_key")
    movie_meta = movie_meta.merge(ext, on="movie_key", how="left", suffixes=("", "_ext"))
    for c in optional_cols:
        ce = f"{c}_ext"
        if ce in movie_meta.columns:
            movie_meta[c] = movie_meta[c].fillna(movie_meta[ce])
            movie_meta = movie_meta.drop(columns=[ce])

# genre multi-hot from in-file movie_genres or imdb match
GENRE_COLS: List[str] = []
if "movie_genres" in movie_meta.columns and movie_meta["movie_genres"].notna().any():
    genres = movie_meta["movie_genres"].apply(lambda x: split_multi(x, sep="|"))
    all_gen = pd.Series([g for xs in genres for g in xs])
    top_gen = all_gen.value_counts().head(16).index.tolist()
    for g in top_gen:
        col = f"genre_{g.replace(' ', '_').replace('-', '_')}"
        movie_meta[col] = genres.map(lambda xs: int(g in xs))
        GENRE_COLS.append(col)
elif os.path.exists(IMDB_GENRES_PATH):
    imdb = pd.read_csv(IMDB_GENRES_PATH)
    imdb["title_clean"] = imdb["movie title - year"].astype(str).str.rsplit(" - ", n=1).str[0].str.strip().str.lower()
    imdb["title_norm"] = imdb["title_clean"].map(normalize_title)
    imdb = imdb.drop_duplicates("title_norm", keep="first")
    mt = movie_meta[["movie_key"]].copy()
    mt["title_raw"] = mt["movie_key"].map(slug_to_title)
    mt["title_norm"] = mt["title_raw"].map(normalize_title)
    gm = mt.merge(imdb[["title_norm", "expanded-genres"]], on="title_norm", how="left")
    all_exp = gm["expanded-genres"].dropna().astype(str).str.split(", ").explode().str.strip().str.lower()
    top_gen = all_exp.value_counts().head(16).index.tolist()
    for g in top_gen:
        col = f"genre_{g.replace(' ', '_').replace('-', '_')}"
        gm[col] = gm["expanded-genres"].astype(str).str.lower().str.contains(g, na=False).astype(int)
        GENRE_COLS.append(col)
    movie_meta = movie_meta.merge(gm[["movie_key"] + GENRE_COLS], on="movie_key", how="left")
    for c in GENRE_COLS:
        movie_meta[c] = movie_meta[c].fillna(0).astype(int)
else:
    print("No genre source found; genre features disabled.")

if "runtime" in movie_meta.columns:
    movie_meta["runtime_min"] = movie_meta["runtime"].map(parse_runtime_to_minutes)
else:
    movie_meta["runtime_min"] = np.nan

if "wide_release_date" in movie_meta.columns:
    wrd = pd.to_datetime(movie_meta["wide_release_date"], errors="coerce")
else:
    wrd = pd.to_datetime(pd.Series([pd.NaT] * len(movie_meta)), errors="coerce")
movie_meta["release_month"] = wrd.dt.month
movie_meta["release_quarter"] = wrd.dt.quarter
movie_meta["release_year"] = wrd.dt.year

if movie_meta["release_month"].isna().all():
    movie_meta = movie_meta.merge(movie_level[["movie_key", "first_review"]], on="movie_key", how="left")
    movie_meta["release_month"] = movie_meta["first_review"].dt.month
    movie_meta["release_quarter"] = movie_meta["first_review"].dt.quarter
    movie_meta["release_year"] = movie_meta["first_review"].dt.year
    movie_meta = movie_meta.drop(columns=["first_review"])

for cat in ["mpaa_rating", "studio", "director"]:
    if cat in movie_meta.columns:
        vc = movie_meta[cat].fillna("unknown").astype(str).value_counts()
        top = set(vc.head(20).index)
        movie_meta[cat] = movie_meta[cat].fillna("unknown").astype(str).map(lambda x: x if x in top else "other")

if "cast" in movie_meta.columns:
    movie_meta["main_cast_1"] = movie_meta["cast"].fillna("").astype(str).map(
        lambda s: split_multi(s.replace(",", "|"), sep="|")[0] if split_multi(s.replace(",", "|"), sep="|") else "unknown"
    )
    vc = movie_meta["main_cast_1"].value_counts()
    top = set(vc.head(50).index)
    movie_meta["main_cast_1"] = movie_meta["main_cast_1"].map(lambda x: x if x in top else "other")

for c in ["runtime_min", "release_month", "release_quarter", "release_year"]:
    if c in movie_meta.columns:
        movie_meta[c] = pd.to_numeric(movie_meta[c], errors="coerce")

cat_cols = [c for c in ["mpaa_rating", "studio", "director", "main_cast_1"] if c in movie_meta.columns]
movie_meta = pd.get_dummies(movie_meta, columns=cat_cols, prefix=cat_cols, dummy_na=False)

movie_static_cols = [c for c in movie_meta.columns if c != "movie_key"]
movie_meta[movie_static_cols] = movie_meta[movie_static_cols].apply(pd.to_numeric, errors="ignore")
movie_meta[movie_static_cols] = movie_meta[movie_static_cols].fillna(0)

print(f"Movie static features: {len(movie_static_cols)} columns")

# %%
# ============================================================
# TIME SPLIT + TRAIN-ONLY CRITIC UNIVERSE
# ============================================================

movie_level = movie_level.merge(
    movie_meta[["movie_key"] + [c for c in ["release_year"] if c in movie_meta.columns]],
    on="movie_key",
    how="left",
)
cutoff = pd.Timestamp(CUTOFF_DATE)
train_movies = set(movie_level[movie_level["first_review"] < cutoff]["movie_key"])
val_movies = set(movie_level[movie_level["first_review"] >= cutoff]["movie_key"])

df_train = df[df["movie_key"].isin(train_movies)].copy()
df_val = df[df["movie_key"].isin(val_movies)].copy()

critic_counts = df_train.groupby("critic_name").size().reset_index(name="n_reviews")
critic_counts = critic_counts[critic_counts["n_reviews"] >= MIN_REVIEWS_PER_CRITIC]
if MAX_CRITICS_UNIVERSE and len(critic_counts) > MAX_CRITICS_UNIVERSE:
    critic_counts = critic_counts.nlargest(MAX_CRITICS_UNIVERSE, "n_reviews")

CRITIC_UNIVERSE = sorted(critic_counts["critic_name"].tolist())
critic_set = set(CRITIC_UNIVERSE)

df_train = df_train[df_train["critic_name"].isin(critic_set)].copy()
df_val = df_val[df_val["critic_name"].isin(critic_set)].copy()
df_all_univ = df[df["critic_name"].isin(critic_set)].copy()

print(f"Train movies: {len(train_movies):,}, Val movies: {len(val_movies):,}, Critic universe: {len(CRITIC_UNIVERSE):,}")

# %%
# ============================================================
# CRITIC PRIORS (TRAIN ONLY) + RECENCY INDEX
# ============================================================

critic_base = df_train.groupby("critic_name").agg(
    critic_n_reviews=("is_fresh", "size"),
    critic_fresh_rate=("is_fresh", "mean"),
    critic_top_rate=("is_top_critic", "mean"),
    critic_first_review=("review_date", "min"),
    critic_last_review=("review_date", "max"),
).reset_index()
career_days = (critic_base["critic_last_review"] - critic_base["critic_first_review"]).dt.days.clip(lower=30)
critic_base["critic_review_rate_per_year"] = critic_base["critic_n_reviews"] / (career_days / 365.25)
critic_base = critic_base.drop(columns=["critic_first_review", "critic_last_review"])

all_crit = pd.DataFrame({"critic_name": CRITIC_UNIVERSE})
critic_base = all_crit.merge(critic_base, on="critic_name", how="left")
global_fresh = float(df_train["is_fresh"].mean())
critic_base = critic_base.fillna({
    "critic_n_reviews": 0,
    "critic_fresh_rate": global_fresh,
    "critic_top_rate": 0.0,
    "critic_review_rate_per_year": 0.0,
})

critic_genre_fresh_cols = []
critic_genre_prop_cols = []
if GENRE_COLS:
    tmp = df_train[["critic_name", "movie_key", "is_fresh"]].drop_duplicates(["critic_name", "movie_key"])
    tmp = tmp.merge(movie_meta[["movie_key"] + GENRE_COLS], on="movie_key", how="left").fillna(0)
    for gc in GENRE_COLS:
        fresh_col = f"critic_fresh_{gc}"
        prop_col = f"critic_prop_{gc}"
        critic_genre_fresh_cols.append(fresh_col)
        critic_genre_prop_cols.append(prop_col)
        g = tmp[tmp[gc] == 1].groupby("critic_name")["is_fresh"].mean().reset_index()
        g.columns = ["critic_name", fresh_col]
        p = tmp.groupby("critic_name")[gc].mean().reset_index()
        p.columns = ["critic_name", prop_col]
        critic_base = critic_base.merge(g, on="critic_name", how="left")
        critic_base = critic_base.merge(p, on="critic_name", how="left")

for c in critic_genre_fresh_cols:
    critic_base[c] = critic_base[c].fillna(global_fresh)
for c in critic_genre_prop_cols:
    critic_base[c] = critic_base[c].fillna(0.0)

critic_feature_cols = [c for c in critic_base.columns if c != "critic_name"]

# critic review-date index for moving windows
def build_critic_dates_map(reviews_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    m = {}
    for c, g in reviews_df.groupby("critic_name"):
        arr = g["review_date"].sort_values().values.astype("datetime64[ns]")
        m[c] = arr
    return m


critic_dates_train = build_critic_dates_map(df_train)
critic_dates_all = build_critic_dates_map(df_all_univ)


def recent_count(arr: np.ndarray, as_of: np.datetime64, days: int) -> int:
    if arr is None or len(arr) == 0:
        return 0
    left = as_of - np.timedelta64(days, "D")
    i0 = np.searchsorted(arr, left, side="left")
    i1 = np.searchsorted(arr, as_of, side="right")
    return int(max(i1 - i0, 0))


def recent_counts_for_critics(
    critics: Iterable[str],
    as_of: pd.Timestamp,
    critic_dates_map: Dict[str, np.ndarray],
    windows_days: List[int],
) -> pd.DataFrame:
    as_of64 = np.datetime64(pd.Timestamp(as_of))
    rows = []
    for c in critics:
        arr = critic_dates_map.get(c)
        row = {"critic_name": c}
        for w in windows_days:
            row[f"critic_recent_{w}d_reviews"] = recent_count(arr, as_of64, w)
        rows.append(row)
    return pd.DataFrame(rows)

# %%
# ============================================================
# FEATURE BUILDER (STRICT AS-OF, NO LOOKAHEAD)
# ============================================================

movie_meta_small = movie_meta[["movie_key"] + movie_static_cols].copy()


def build_dynamic_movie_snapshot(movie_reviews_prefix: pd.DataFrame) -> Dict[str, float]:
    n_obs = len(movie_reviews_prefix)
    s_obs = int(movie_reviews_prefix["is_fresh"].sum()) if n_obs > 0 else 0
    current_t = s_obs / n_obs if n_obs > 0 else 0.0
    top_share = float(movie_reviews_prefix["is_top_critic"].mean()) if n_obs > 0 else 0.0
    if n_obs > 0:
        fr = movie_reviews_prefix["review_date"].min()
        lr = movie_reviews_prefix["review_date"].max()
        span = max((lr - fr).days, 1)
        velocity = n_obs / span
        days_since_first = span
    else:
        velocity = 0.0
        days_since_first = 0.0
    return {
        "n_obs_reviews": float(n_obs),
        "obs_fresh_rate": float(current_t),
        "obs_top_critic_share": float(top_share),
        "obs_review_velocity": float(velocity),
        "obs_days_since_first": float(days_since_first),
    }


def add_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not GENRE_COLS:
        frame["critic_genre_affinity_score"] = 0.0
        frame["critic_coverage_propensity_for_genre"] = 0.0
        return frame
    fresh_cols = [f"critic_fresh_{g}" for g in GENRE_COLS]
    prop_cols = [f"critic_prop_{g}" for g in GENRE_COLS]
    xg = frame[GENRE_COLS].values.astype(float)
    xf = frame[fresh_cols].values.astype(float)
    xp = frame[prop_cols].values.astype(float)
    frame["critic_genre_affinity_score"] = (xg * xf).sum(axis=1)
    frame["critic_coverage_propensity_for_genre"] = (xg * xp).sum(axis=1)
    return frame


def build_candidate_frame(
    movie_key: str,
    critics: List[str],
    as_of: pd.Timestamp,
    movie_prefix_reviews: pd.DataFrame,
    critic_dates_map: Dict[str, np.ndarray],
) -> pd.DataFrame:
    cf = pd.DataFrame({"critic_name": critics})
    cf = cf.merge(critic_base, on="critic_name", how="left")
    cf = cf.merge(recent_counts_for_critics(critics, as_of, critic_dates_map, RECENT_WINDOWS_DAYS), on="critic_name", how="left")
    mf = movie_meta_small[movie_meta_small["movie_key"] == movie_key].copy()
    if len(mf) == 0:
        mf = pd.DataFrame([{"movie_key": movie_key}])
        for c in movie_static_cols:
            mf[c] = 0.0
    mf = mf.drop(columns=["movie_key"])
    for c in mf.columns:
        cf[c] = float(mf.iloc[0][c])
    dyn = build_dynamic_movie_snapshot(movie_prefix_reviews)
    for k, v in dyn.items():
        cf[k] = v
    cf = add_interaction_features(cf)
    cf = cf.fillna(0)
    return cf

# %%
# ============================================================
# SELECTION DATASET: P(review after as_of | not yet reviewed)
# ============================================================

def dedup_movie_reviews(m: pd.DataFrame) -> pd.DataFrame:
    return m.sort_values(["review_date", "critic_name"]).drop_duplicates("critic_name", keep="first")


def limit_movie_keys(movie_keys: Iterable[str], max_n: Optional[int], seed: int = RANDOM_SEED) -> List[str]:
    mk = sorted(list(movie_keys))
    if max_n is None or max_n <= 0 or len(mk) <= max_n:
        return mk
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(mk), size=max_n, replace=False)
    return [mk[i] for i in sorted(idx)]


def downsample_binary_frame(df_in: pd.DataFrame, target_col: str, max_rows: Optional[int], seed: int = RANDOM_SEED) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df_in) <= max_rows:
        return df_in
    rng = np.random.RandomState(seed)
    pos = df_in[df_in[target_col] == 1]
    neg = df_in[df_in[target_col] == 0]
    pos_rate = len(pos) / max(len(df_in), 1)
    n_pos = int(round(max_rows * pos_rate))
    n_pos = min(max(n_pos, 1), len(pos))
    n_neg = max_rows - n_pos
    n_neg = min(max(n_neg, 1), len(neg))
    pos_idx = rng.choice(pos.index.values, size=n_pos, replace=False)
    neg_idx = rng.choice(neg.index.values, size=n_neg, replace=False)
    out = df_in.loc[np.concatenate([pos_idx, neg_idx])].copy()
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def build_selection_checkpoint_dataset(
    reviews_df: pd.DataFrame,
    movie_keys: Iterable[str],
    critic_universe: List[str],
    checkpoints: List[int],
    neg_ratio: int,
    critic_dates_map: Dict[str, np.ndarray],
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    u_set = set(critic_universe)
    rows = []
    movie_keys = list(movie_keys)
    total_rows = 0
    for mi, mk in enumerate(movie_keys, 1):
        if mi % PROGRESS_EVERY_MOVIES == 0:
            print(f"  selection progress: {mi:,}/{len(movie_keys):,} movies, rows={total_rows:,}")
        m = dedup_movie_reviews(reviews_df[reviews_df["movie_key"] == mk])
        if len(m) < 2:
            continue
        ordered = m[["critic_name", "review_date", "is_fresh", "is_top_critic"]].reset_index(drop=True)
        reviewer_list = ordered["critic_name"].tolist()
        reviewer_set = set(reviewer_list)
        for cp in checkpoints:
            if cp >= len(ordered):
                continue
            prefix = ordered.iloc[:cp].copy()
            as_of = pd.Timestamp(ordered.iloc[cp - 1]["review_date"])
            observed = set(prefix["critic_name"])
            future = reviewer_set - observed
            eligible = list((u_set - observed))
            positives = list(future & set(critic_universe))
            if not positives:
                continue
            pos_set = set(positives)
            neg_pool = [c for c in eligible if c not in pos_set]
            n_neg = min(len(neg_pool), neg_ratio * len(positives))
            if n_neg <= 0:
                continue
            negatives = rng.choice(neg_pool, size=n_neg, replace=False).tolist()
            candidates = positives + negatives
            labels = [1] * len(positives) + [0] * len(negatives)
            f = build_candidate_frame(mk, candidates, as_of, prefix, critic_dates_map)
            f["movie_key"] = mk
            f["as_of"] = as_of
            f["review_after_as_of"] = labels
            rows.append(f)
            total_rows += len(f)
            if max_rows is not None and total_rows >= max_rows:
                if mi % PROGRESS_EVERY_MOVIES != 0:
                    print(f"  selection capped at {total_rows:,} rows")
                out = pd.concat(rows, ignore_index=True)
                return out
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out


print("Building selection checkpoint train/val datasets...")
sel_train_movies = limit_movie_keys(train_movies, MAX_SELECTION_TRAIN_MOVIES, seed=RANDOM_SEED)
sel_val_movies = limit_movie_keys(val_movies, MAX_SELECTION_VAL_MOVIES, seed=RANDOM_SEED + 1)
print(f"  Using train movies: {len(sel_train_movies):,}/{len(train_movies):,}")
print(f"  Using val movies:   {len(sel_val_movies):,}/{len(val_movies):,}")
sel_train = build_selection_checkpoint_dataset(
    df_train,
    sel_train_movies,
    CRITIC_UNIVERSE,
    checkpoints=SELECTION_CHECKPOINTS,
    neg_ratio=NEGATIVE_RATIO,
    critic_dates_map=critic_dates_train,
    max_rows=MAX_SELECTION_TRAIN_ROWS,
    seed=RANDOM_SEED,
)
sel_val = build_selection_checkpoint_dataset(
    df_val,
    sel_val_movies,
    CRITIC_UNIVERSE,
    checkpoints=SELECTION_CHECKPOINTS,
    neg_ratio=NEGATIVE_RATIO,
    critic_dates_map=critic_dates_all,
    max_rows=MAX_SELECTION_VAL_ROWS,
    seed=RANDOM_SEED + 1,
)
print(f"Selection train rows: {len(sel_train):,}, val rows: {len(sel_val):,}")

# %%
# ============================================================
# OUTCOME DATASET (NO MOVIE-OUTCOME LEAKAGE)
# ============================================================

def build_outcome_dataset(
    reviews_df: pd.DataFrame,
    critic_dates_map: Dict[str, np.ndarray],
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    grouped = list(reviews_df.groupby("movie_key"))
    total_rows = 0
    for mi, (mk, g) in enumerate(grouped, 1):
        if mi % PROGRESS_EVERY_MOVIES == 0:
            print(f"  outcome progress: {mi:,}/{len(grouped):,} movies, rows={total_rows:,}")
        m = dedup_movie_reviews(g)
        if len(m) == 0:
            continue
        m = m.sort_values(["review_date", "critic_name"]).reset_index(drop=True)
        for i in range(len(m)):
            row = m.iloc[i]
            as_of = pd.Timestamp(row["review_date"])
            prefix = m.iloc[:i].copy()
            cf = build_candidate_frame(
                mk,
                [row["critic_name"]],
                as_of,
                prefix,
                critic_dates_map,
            )
            cf["is_fresh"] = int(row["is_fresh"])
            rows.append(cf)
            total_rows += 1
            if max_rows is not None and total_rows >= max_rows:
                if mi % PROGRESS_EVERY_MOVIES != 0:
                    print(f"  outcome capped at {total_rows:,} rows")
                return pd.concat(rows, ignore_index=True)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


print("Building outcome train/val datasets...")
out_train_movies = set(limit_movie_keys(train_movies, MAX_OUTCOME_TRAIN_MOVIES, seed=RANDOM_SEED + 2))
out_val_movies = set(limit_movie_keys(val_movies, MAX_OUTCOME_VAL_MOVIES, seed=RANDOM_SEED + 3))
print(f"  Using train movies: {len(out_train_movies):,}/{len(train_movies):,}")
print(f"  Using val movies:   {len(out_val_movies):,}/{len(val_movies):,}")
out_train = build_outcome_dataset(
    df_train[df_train["movie_key"].isin(out_train_movies)],
    critic_dates_train,
    max_rows=MAX_OUTCOME_TRAIN_ROWS,
)
out_val = build_outcome_dataset(
    df_val[df_val["movie_key"].isin(out_val_movies)],
    critic_dates_all,
    max_rows=MAX_OUTCOME_VAL_ROWS,
)
print(f"Outcome train rows: {len(out_train):,}, val rows: {len(out_val):,}")

sel_train = downsample_binary_frame(sel_train, "review_after_as_of", MAX_SELECTION_TRAIN_ROWS, seed=RANDOM_SEED + 10)
sel_val = downsample_binary_frame(sel_val, "review_after_as_of", MAX_SELECTION_VAL_ROWS, seed=RANDOM_SEED + 11)
out_train = downsample_binary_frame(out_train, "is_fresh", MAX_OUTCOME_TRAIN_ROWS, seed=RANDOM_SEED + 12)
out_val = downsample_binary_frame(out_val, "is_fresh", MAX_OUTCOME_VAL_ROWS, seed=RANDOM_SEED + 13)
print(f"After row caps — selection train/val: {len(sel_train):,}/{len(sel_val):,}, outcome train/val: {len(out_train):,}/{len(out_val):,}")

# %%
# ============================================================
# TRAIN MODELS
# ============================================================

SEL_META = ["critic_name", "movie_key", "as_of", "review_after_as_of"]
sel_features = [c for c in sel_train.columns if c not in SEL_META]

X_sel_train = sel_train[sel_features].values
y_sel_train = sel_train["review_after_as_of"].values
X_sel_val = sel_val[sel_features].values
y_sel_val = sel_val["review_after_as_of"].values

if USE_LGBM:
    sel_base = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=40,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    sel_model = CalibratedClassifierCV(sel_base, cv=2, method="sigmoid")
    sel_model.fit(X_sel_train, y_sel_train)
    scaler_sel = None
else:
    scaler_sel = StandardScaler()
    X_sel_train = scaler_sel.fit_transform(X_sel_train)
    X_sel_val = scaler_sel.transform(X_sel_val)
    sel_base = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=400,
        solver="saga",
        random_state=RANDOM_SEED,
    )
    sel_model = CalibratedClassifierCV(sel_base, cv=2, method="sigmoid")
    sel_model.fit(X_sel_train, y_sel_train)

sel_p_tr = sel_model.predict_proba(X_sel_train)[:, 1]
sel_p_va = sel_model.predict_proba(X_sel_val)[:, 1]
print("Selection model")
print(f"  train AUC={roc_auc_score(y_sel_train, sel_p_tr):.4f}, logloss={log_loss(y_sel_train, sel_p_tr):.4f}")
print(f"  val   AUC={roc_auc_score(y_sel_val, sel_p_va):.4f}, logloss={log_loss(y_sel_val, sel_p_va):.4f}")

OUT_META = ["critic_name", "is_fresh"]
out_features = [c for c in out_train.columns if c not in OUT_META]
X_out_train = out_train[out_features].values
y_out_train = out_train["is_fresh"].values
X_out_val = out_val[out_features].values
y_out_val = out_val["is_fresh"].values

if USE_LGBM:
    out_base = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=40,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    out_model = CalibratedClassifierCV(out_base, cv=2, method="sigmoid")
    out_model.fit(X_out_train, y_out_train)
    scaler_out = None
else:
    scaler_out = StandardScaler()
    X_out_train = scaler_out.fit_transform(X_out_train)
    X_out_val = scaler_out.transform(X_out_val)
    out_base = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=400,
        solver="saga",
        random_state=RANDOM_SEED,
    )
    out_model = CalibratedClassifierCV(out_base, cv=2, method="sigmoid")
    out_model.fit(X_out_train, y_out_train)

out_p_tr = out_model.predict_proba(X_out_train)[:, 1]
out_p_va = out_model.predict_proba(X_out_val)[:, 1]
print("Outcome model")
print(f"  train AUC={roc_auc_score(y_out_train, out_p_tr):.4f}, logloss={log_loss(y_out_train, out_p_tr):.4f}")
print(f"  val   AUC={roc_auc_score(y_out_val, out_p_va):.4f}, logloss={log_loss(y_out_val, out_p_va):.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
pt, pp = calibration_curve(y_sel_val, sel_p_va, n_bins=10)
axes[0].plot(pp, pt, "o-")
axes[0].plot([0, 1], [0, 1], "k--")
axes[0].set_title("Selection Calibration")
pt, pp = calibration_curve(y_out_val, out_p_va, n_bins=10)
axes[1].plot(pp, pt, "o-")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("Outcome Calibration")
plt.tight_layout()
plt.show()

# %%
# ============================================================
# NOWCAST (STRICT AS-OF)
# ============================================================

def predict_selection_remaining(
    movie_key: str,
    as_of: pd.Timestamp,
    observed_prefix: pd.DataFrame,
    remaining_critics: List[str],
) -> np.ndarray:
    if not remaining_critics:
        return np.array([])
    f = build_candidate_frame(
        movie_key=movie_key,
        critics=remaining_critics,
        as_of=as_of,
        movie_prefix_reviews=observed_prefix,
        critic_dates_map=critic_dates_all,
    )
    X = f[sel_features].values
    if scaler_sel is not None:
        X = scaler_sel.transform(X)
    return sel_model.predict_proba(X)[:, 1]


def predict_outcome_for_critics(
    movie_key: str,
    as_of: pd.Timestamp,
    observed_prefix: pd.DataFrame,
    critics: List[str],
) -> np.ndarray:
    if not critics:
        return np.array([])
    f = build_candidate_frame(
        movie_key=movie_key,
        critics=critics,
        as_of=as_of,
        movie_prefix_reviews=observed_prefix,
        critic_dates_map=critic_dates_all,
    )
    X = f[out_features].values
    if scaler_out is not None:
        X = scaler_out.transform(X)
    return out_model.predict_proba(X)[:, 1]


def nowcast_movie(
    movie_key: str,
    as_of_datetime,
    n_sims: int = N_SIMS,
    thresholds: List[float] = THRESHOLDS,
    chunk_size: int = 512,
    seed: int = RANDOM_SEED,
):
    as_of = pd.Timestamp(as_of_datetime)
    m = df_all_univ[df_all_univ["movie_key"] == movie_key].copy()
    if len(m) == 0:
        raise ValueError(f"Unknown movie_key: {movie_key}")
    m = dedup_movie_reviews(m)
    m = m.sort_values(["review_date", "critic_name"]).reset_index(drop=True)
    obs = m[m["review_date"] <= as_of].copy()

    S_obs = int(obs["is_fresh"].sum())
    N_obs = len(obs)
    current_t = S_obs / N_obs if N_obs else 0.0
    observed_critics = set(obs["critic_name"])
    remaining = [c for c in CRITIC_UNIVERSE if c not in observed_critics]

    if not remaining:
        sims = np.full(n_sims, current_t)
    else:
        p = predict_selection_remaining(movie_key, as_of, obs, remaining)
        q = predict_outcome_for_critics(movie_key, as_of, obs, remaining)
        p = np.clip(p, 1e-5, 1 - 1e-5)
        q = np.clip(q, 1e-5, 1 - 1e-5)
        rng = np.random.RandomState(seed)
        sims = np.empty(n_sims, dtype=float)
        n_rem = len(remaining)
        start = 0
        while start < n_sims:
            end = min(start + chunk_size, n_sims)
            bs = end - start
            U1 = rng.random((bs, n_rem))
            R = (U1 < p[np.newaxis, :]).astype(np.int16)
            U2 = rng.random((bs, n_rem))
            F = (U2 < q[np.newaxis, :]).astype(np.int16) * R
            new_n = R.sum(axis=1)
            new_s = F.sum(axis=1)
            tot_n = N_obs + new_n
            tot_s = S_obs + new_s
            sims[start:end] = np.where(tot_n > 0, tot_s / tot_n, 0.5)
            start = end

    return {
        "movie_key": movie_key,
        "as_of": as_of,
        "point_estimate": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "p5": float(np.percentile(sims, 5)),
        "p25": float(np.percentile(sims, 25)),
        "p75": float(np.percentile(sims, 75)),
        "p95": float(np.percentile(sims, 95)),
        "threshold_probs": {t: float((sims >= t).mean()) for t in thresholds},
        "observed": {
            "S_obs": S_obs,
            "N_obs": N_obs,
            "current_T": float(current_t),
            "n_remaining_critics": len(remaining),
        },
        "simulations": sims,
    }


def print_nowcast(r: dict):
    print(f"\nNowcast {r['movie_key']} @ {r['as_of']}")
    print(f"Observed: {r['observed']['S_obs']}/{r['observed']['N_obs']} ({r['observed']['current_T']:.1%})")
    print(f"Point: {r['point_estimate']:.1%}, median: {r['median']:.1%}, 90% CI: [{r['p5']:.1%}, {r['p95']:.1%}]")
    for t, p in r["threshold_probs"].items():
        print(f"P(T >= {t:.0%}) = {p:.1%}")

# %%
# ============================================================
# BACKTEST (EXACT CHECKPOINT PREFIX)
# ============================================================

movie_final = df_all_univ.groupby("movie_key").agg(
    n_reviews=("is_fresh", "size"),
    final_t=("is_fresh", "mean"),
).reset_index()


def run_backtest(
    movie_keys: Iterable[str],
    checkpoints: List[int],
    max_movies: int = MAX_BACKTEST_MOVIES,
    n_sims: int = 2000,
) -> pd.DataFrame:
    rows = []
    for i, mk in enumerate(list(movie_keys)[:max_movies]):
        m = dedup_movie_reviews(df_all_univ[df_all_univ["movie_key"] == mk])
        m = m.sort_values(["review_date", "critic_name"]).reset_index(drop=True)
        if len(m) == 0:
            continue
        actual = float(m["is_fresh"].mean())
        total_n = len(m)
        for cp in checkpoints:
            if cp > total_n:
                continue
            # Exact prefix checkpoint; no same-timestamp spillover leakage.
            as_of = pd.Timestamp(m.iloc[cp - 1]["review_date"])
            nc = nowcast_movie(mk, as_of, n_sims=n_sims, seed=RANDOM_SEED + i)
            row = {
                "movie_key": mk,
                "checkpoint": cp,
                "actual_T": actual,
                "predicted_T": nc["point_estimate"],
                "naive_T": nc["observed"]["current_T"],
                "error": nc["point_estimate"] - actual,
                "naive_error": nc["observed"]["current_T"] - actual,
                "pct_reviews_seen": cp / total_n,
            }
            for t in THRESHOLDS:
                t_int = int(t * 100)
                row[f"actual_ge_{t_int}"] = int(actual >= t)
                row[f"p_ge_{t_int}"] = nc["threshold_probs"][t]
            rows.append(row)
    return pd.DataFrame(rows)


print("Running backtest...")
bt = run_backtest(val_movies, BACKTEST_CHECKPOINTS, max_movies=MAX_BACKTEST_MOVIES, n_sims=1500)
print(f"Backtest rows: {len(bt):,}, movies: {bt['movie_key'].nunique() if len(bt) else 0}")

if len(bt):
    for cp in BACKTEST_CHECKPOINTS:
        s = bt[bt["checkpoint"] == cp]
        if len(s) == 0:
            continue
        mae_m = mean_absolute_error(s["actual_T"], s["predicted_T"])
        rmse_m = np.sqrt(mean_squared_error(s["actual_T"], s["predicted_T"]))
        mae_n = mean_absolute_error(s["actual_T"], s["naive_T"])
        rmse_n = np.sqrt(mean_squared_error(s["actual_T"], s["naive_T"]))
        print(f"cp={cp}: model MAE={mae_m:.4f}, RMSE={rmse_m:.4f} | naive MAE={mae_n:.4f}, RMSE={rmse_n:.4f}")

    print("Brier scores:")
    for cp in BACKTEST_CHECKPOINTS:
        s = bt[bt["checkpoint"] == cp]
        if len(s) == 0:
            continue
        vals = []
        for t in THRESHOLDS:
            t_int = int(t * 100)
            y = s[f"actual_ge_{t_int}"].values
            p = s[f"p_ge_{t_int}"].values
            if len(np.unique(y)) > 1:
                vals.append(f">={t_int}%:{brier_score_loss(y, p):.4f}")
            else:
                vals.append(f">={t_int}%:N/A")
        print(f"cp={cp}: " + ", ".join(vals))

# %%
# Worked example
if len(val_movies):
    ex = list(val_movies)[0]
    exm = dedup_movie_reviews(df_all_univ[df_all_univ["movie_key"] == ex]).sort_values(["review_date", "critic_name"]).reset_index(drop=True)
    print(f"Example movie: {ex}, total reviews={len(exm)}, final={exm['is_fresh'].mean():.1%}")
    for cp in [5, 10, 25]:
        if cp <= len(exm):
            r = nowcast_movie(ex, exm.iloc[cp - 1]["review_date"], n_sims=3000)
            print_nowcast(r)
