"""Pure Pandas data helpers for the six-question IMDb Streamlit practical.

The source CSV is left unchanged for Question 1, while analysis uses a cleaned copy.
"""
from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "primaryTitle", "averageRating", "numVotes", "genres",
    "countriesOfOrigin", "grossWorldwide",
}

ALIASES = {
    "primaryTitle": ("primaryTitle", "title", "Title", "movie_title"),
    "averageRating": ("averageRating", "rating", "imdb_rating", "IMDb Rating"),
    "numVotes": ("numVotes", "votes", "number_of_votes", "Votes"),
    "genres": ("genres", "genre", "Genre"),
    "countriesOfOrigin": ("countriesOfOrigin", "countries", "country", "Country"),
    "grossWorldwide": ("grossWorldwide", "worldwide_gross", "gross", "Gross"),
}


def parse_multi_value(value: object) -> list[str]:
    """Parse CSV cells such as "['Action', 'Drama']", JSON, or 'USA, UK'."""
    if isinstance(value, (list, tuple, set)):
        candidates: Iterable[object] = value
    elif value is None:
        return []
    elif isinstance(value, (float, int)) and pd.isna(value):
        return []
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    candidates = parsed
                else:
                    candidates = [parsed]
            except (ValueError, SyntaxError):
                try:
                    parsed = json.loads(text)
                    candidates = parsed if isinstance(parsed, list) else [parsed]
                except (ValueError, TypeError):
                    candidates = re.split(r"[,;|]", text.strip("[]"))
        else:
            candidates = re.split(r"[,;|]", text)
    seen: set[str] = set()
    values: list[str] = []
    for item in candidates:
        item_str = str(item).strip().strip("'\"")
        if item_str and item_str.lower() not in {"nan", "none", "null"}:
            if item_str not in seen:
                values.append(item_str)
                seen.add(item_str)
    return values


def parse_currency(value: object) -> float:
    """Normalize amounts such as 2920000000, '$2.92B', and '500,000'."""
    if value is None or pd.isna(value):
        return math.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raw = re.sub(r"[,$£€₹\s]", "", str(value)).strip().upper()
    if not raw or raw in {"NAN", "NONE", "NULL", "N/A"}:
        return math.nan
    multiplier = 1.0
    if raw.endswith("B"):
        raw, multiplier = raw[:-1], 1e9
    elif raw.endswith("M"):
        raw, multiplier = raw[:-1], 1e6
    elif raw.endswith("K"):
        raw, multiplier = raw[:-1], 1e3
    try:
        return float(raw) * multiplier
    except ValueError:
        return math.nan


def prepare_movies(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Make a normalized copy for analysis without touching raw_df."""
    df = raw_df.copy(deep=True)
    rename_map: dict[str, str] = {}
    for normalized_name, alternatives in ALIASES.items():
        if normalized_name in df.columns:
            continue
        for alias in alternatives:
            if alias in df.columns:
                rename_map[alias] = normalized_name
                break
    df = df.rename(columns=rename_map)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing))
        )
    df["primaryTitle"] = df["primaryTitle"].fillna("Unknown title").astype(str)
    df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce")
    df["numVotes"] = pd.to_numeric(
        df["numVotes"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0)
    df["grossWorldwide"] = df["grossWorldwide"].map(parse_currency)
    df["genreList"] = df["genres"].map(parse_multi_value)
    df["countryList"] = df["countriesOfOrigin"].map(parse_multi_value)
    df["genreDisplay"] = df["genreList"].map(lambda xs: ", ".join(xs) if xs else "Not listed")
    df["countryDisplay"] = df["countryList"].map(lambda xs: ", ".join(xs) if xs else "Not listed")
    return df


def unique_values(df: pd.DataFrame, list_col: str) -> list[str]:
    return sorted({item for items in df[list_col] for item in items}, key=str.casefold)


def genre_counts(df: pd.DataFrame) -> pd.Series:
    # One movie is counted in every genre it belongs to, as the paper requests.
    return df["genreList"].explode().dropna().value_counts()


def top_gross(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.dropna(subset=["grossWorldwide"])
        .sort_values("grossWorldwide", ascending=False, kind="stable")
        .head(n)
        .copy()
    )


def filter_movies(
    movies: pd.DataFrame,
    *,
    genre: str = "All Genres",
    countries: list[str] | None = None,
    rating_range: tuple[float, float] = (0.0, 10.0),
    min_votes: int = 0,
    title: str = "",
    sort_order: str = "Descending",
) -> pd.DataFrame:
    filtered = movies.copy()
    if genre != "All Genres":
        filtered = filtered[filtered["genreList"].map(lambda items: genre in items)]
    if countries:
        requested = set(countries)
        filtered = filtered[
            filtered["countryList"].map(lambda items: bool(requested.intersection(items)))
        ]
    min_rating, max_rating = rating_range
    filtered = filtered[filtered["averageRating"].between(min_rating, max_rating, inclusive="both")]
    filtered = filtered[filtered["numVotes"] >= min_votes]
    if title.strip():
        filtered = filtered[
            filtered["primaryTitle"].str.contains(title.strip(), case=False, na=False, regex=False)
        ]
    return filtered.sort_values(
        ["averageRating", "primaryTitle"],
        ascending=[sort_order == "Ascending", True],
        kind="stable",
    )


def find_movie(
    movies: pd.DataFrame,
    *,
    title: str,
    genre: str,
    minimum_rating: float,
) -> tuple[dict | None, int]:
    candidates = movies[movies["averageRating"] >= minimum_rating].copy()
    if genre != "All Genres":
        candidates = candidates[candidates["genreList"].map(lambda g: genre in g)]
    if title.strip():
        candidates = candidates[
            candidates["primaryTitle"].str.contains(title.strip(), case=False, na=False, regex=False)
        ]
    if candidates.empty:
        return None, 0
    # Exact matches take priority, then highest IMDb rating.
    candidates["_exact"] = candidates["primaryTitle"].str.casefold().eq(title.strip().casefold())
    candidates = candidates.sort_values(
        ["_exact", "averageRating", "numVotes"], ascending=[False, False, False]
    )
    return candidates.drop(columns="_exact").iloc[0].to_dict(), len(candidates)


def format_usd(amount: object) -> str:
    if amount is None or pd.isna(amount):
        return "N/A"
    amount = float(amount)
    if abs(amount) >= 1e9:
        return f"${amount / 1e9:,.2f}B"
    if abs(amount) >= 1e6:
        return f"${amount / 1e6:,.2f}M"
    return f"${amount:,.0f}"
