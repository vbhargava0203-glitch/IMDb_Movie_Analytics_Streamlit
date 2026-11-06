"""IMDb Movie Analytics Dashboard — all six Streamlit practical questions.

Run: streamlit run app.py
Dataset: https://www.kaggle.com/datasets/arjunsinghgangwar/top-250-movies-on-imdb-in-2026
"""
from __future__ import annotations

import base64
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    filter_movies,
    find_movie,
    format_usd,
    genre_counts,
    prepare_movies,
    top_gross,
    unique_values,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
KAGGLE_SLUG = "arjunsinghgangwar/top-250-movies-on-imdb-in-2026"
KAGGLE_URL = "https://www.kaggle.com/datasets/" + KAGGLE_SLUG

st.set_page_config(
    page_title="IMDb | Movie Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def img_data_uri(path: Path) -> str:
    suffix = "jpeg" if path.suffix.lower() in (".jpeg", ".jpg") else "png"
    return f"data:image/{suffix};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --blue:#2687f7; --green:#1da969; --violet:#8c4bd1; --orange:#ff8310; }
        .stApp { background: radial-gradient(circle at 78% 10%,#202739 0%,#111725 54%,#101522 100%); }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#161e2c,#20283b 64%,#171e2d); border-right:1px solid #3f4b64; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#f2f5ff; }
        .block-container { padding-top: 1.05rem !important; padding-bottom:2rem; max-width: 1900px; }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.75rem; }
        .imdb-logo { width:110px; border-radius:8px; display:block; margin:0 auto 11px 5px; }
        .side-name { color:white;font-size:1.6rem;font-weight:800;letter-spacing:-.6px;line-height:1.12;margin:0 0 14px 4px; }
        .side-section {border-top:1px solid #4b5873;margin-top:11px;padding-top:12px;}
        .hero {border-radius:6px;overflow:hidden;margin:-2px 0 13px;box-shadow:0 2px 8px #0007;}
        .hero img {display:block;width:100%;height:auto;}
        .mini-head {font-size:1.1rem;font-weight:800;margin:8px 0 6px; color:#f5f7ff;}
        .section-name {font-size:1.2rem;font-weight:850;color:#eef4ff;margin:.7rem 0 .3rem;}
        [data-testid="stMetric"] {padding:15px 16px; border-radius:12px; min-height:100px;
            background:linear-gradient(120deg,#0c61b6,#2485e9); border:1px solid #ffffff15;
            box-shadow:0 3px 15px #050b1740;}
        [data-testid="stMetric"] label {color:white !important;font-size:.91rem !important;font-weight:750 !important;}
        [data-testid="stMetric"] [data-testid="stMetricValue"] {color:white !important;font-weight:850 !important;font-size:1.94rem !important;}
        /* The four top-level Streamlit metric columns receive the screenshot's card colors. */
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {
            background:linear-gradient(120deg,#168d61,#24ab72);}
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {
            background:linear-gradient(120deg,#6831b1,#8e46cb);}
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] {
            background:linear-gradient(120deg,#f47a08,#ff951d);}
        [data-testid="stPlotlyChart"] {background:#f8faff;border-radius:10px;overflow:hidden;border:1px solid #dbe2f1;}
        [data-testid="stDataFrame"] {border-radius:8px;overflow:hidden;}
        [data-testid="stTabs"] button { font-weight:750; }
        [data-testid="stTabs"] button[aria-selected="true"] { background:#177ee8;color:white;border-radius:8px 8px 0 0; }
        [data-testid="stForm"] { background:linear-gradient(135deg,#242b48,#2b224e);border:1px solid #5e517e;border-radius:12px;padding:12px; }
        [data-testid="stExpander"] {background:#252e43;border-radius:9px;}
        .tiny-muted {color:#abb9d0;font-size:.82rem;}
        .result-card {background:linear-gradient(115deg,#17544b,#244c40);padding:14px;border-radius:10px;border:1px solid #34846e;}
        @media(max-width:800px){ .block-container{padding-left:.7rem;padding-right:.7rem;} }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_local_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_kaggle_csv() -> tuple[pd.DataFrame | None, str]:
    """Use the paper's exact Kaggle link, not an unrelated substituted dataset."""
    try:
        import kagglehub
        download_folder = Path(kagglehub.dataset_download(KAGGLE_SLUG))
        candidates = sorted(download_folder.rglob("*.csv"))
        if not candidates:
            return None, "Kaggle downloaded but the dataset did not contain a CSV file."
        csv_path = next((p for p in candidates if "250" in p.name), candidates[0])
        return pd.read_csv(csv_path), f"Kaggle: {csv_path.name}"
    except Exception as exc:
        return None, f"Automatic Kaggle download unavailable: {type(exc).__name__}: {exc}"


def load_dataset(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file), f"Uploaded: {uploaded_file.name}"
        except Exception as exc:
            st.error(f"Cannot read the uploaded CSV: {exc}")
            return None, "Invalid CSV upload"
    candidates = sorted(DATA_DIR.glob("*.csv"))
    if candidates:
        try:
            return load_local_csv(str(candidates[0])), f"Local CSV: {candidates[0].name}"
        except Exception as exc:
            st.warning(f"Local CSV could not be loaded: {exc}")
    return fetch_kaggle_csv()


def show_header() -> None:
    # Exact visual crop from the user's reference image; still responsive.
    st.markdown(
        f'<div class="hero"><img src="{img_data_uri(ROOT / "assets" / "cinema_header.jpg")}" '
        'alt="IMDb Movie Analytics Dashboard — Explore, Analyze, Discover"></div>',
        unsafe_allow_html=True,
    )


def style_chart(fig: go.Figure, height: int = 260) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=20, t=47, b=35),
        paper_bgcolor="#fbfcff",
        plot_bgcolor="#f5f8ff",
        font=dict(family="Arial, sans-serif", color="#0e1b37", size=12),
        title_font=dict(size=17, color="#101b33", family="Arial, sans-serif"),
        showlegend=False,
        hoverlabel=dict(bgcolor="#202e4a", font_color="white"),
    )
    fig.update_xaxes(gridcolor="#e5eaf4", zeroline=False, linecolor="#8b97b2")
    fig.update_yaxes(gridcolor="#e5eaf4", zeroline=False, linecolor="#8b97b2")
    return fig


def chart_genres(df: pd.DataFrame) -> go.Figure:
    counts = genre_counts(df).head(10)
    palette = ["#2384df", "#28ae70", "#ffb321", "#924fd4", "#2ca9e5", "#e89c15"]
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(), y=counts.values.tolist(),
        marker_color=[palette[i % len(palette)] for i in range(len(counts))],
        text=counts.values.tolist(), textposition="outside",
        hovertemplate="Genre: %{x}<br>Movies: %{y}<extra></extra>",
    ))
    fig.update_layout(title="📊 Movies by Genre", xaxis_title="Genre", yaxis_title="Count")
    return style_chart(fig)


def chart_ratings(df: pd.DataFrame) -> go.Figure:
    ratings = df["averageRating"].dropna()
    fig = go.Figure(go.Histogram(
        x=ratings, nbinsx=14, marker_color="#267fff",
        hovertemplate="Rating bin: %{x}<br>Movies: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="📊 IMDb Rating Distribution", xaxis_title="IMDb Rating",
        yaxis_title="Movie Count", bargap=0.05,
    )
    return style_chart(fig)


def chart_votes(df: pd.DataFrame) -> go.Figure:
    valid = df[df["numVotes"] > 0].dropna(subset=["averageRating"])
    fig = px.scatter(
        valid, x="averageRating", y="numVotes", hover_name="primaryTitle",
        labels={"averageRating": "IMDb Rating", "numVotes": "Number of Votes"},
        color_discrete_sequence=["#147ef5"],
        log_y=True,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.83))
    fig.update_layout(title="🔹 Rating vs Number of Votes")
    return style_chart(fig)


def chart_gross(df: pd.DataFrame) -> go.Figure:
    ranked = top_gross(df, 10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=ranked["grossWorldwide"].tolist(), y=ranked["primaryTitle"].tolist(),
        orientation="h", marker_color="#ff8714",
        text=[format_usd(x) for x in ranked["grossWorldwide"]],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Worldwide gross: $%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(title="🏆 Top 10 Movies by Worldwide Gross", xaxis_title="Worldwide Gross (USD)",
                      yaxis_title="")
    fig.update_xaxes(tickformat="~s")
    fig.update_yaxes(tickfont_size=10, automargin=True)
    fig.update_layout(margin=dict(l=12,r=64,t=47,b=35))
    return style_chart(fig, 285)


def draw_overview_charts(df: pd.DataFrame) -> None:
    top_left, top_right = st.columns(2, gap="small")
    with top_left:
        st.plotly_chart(chart_genres(df), use_container_width=True, key="overview_genre")
    with top_right:
        st.plotly_chart(chart_ratings(df), use_container_width=True, key="overview_rating")
    lower_left, lower_right = st.columns(2, gap="small")
    with lower_left:
        st.plotly_chart(chart_votes(df), use_container_width=True, key="overview_votes")
    with lower_right:
        st.plotly_chart(chart_gross(df), use_container_width=True, key="overview_gross")


def render_kpis(df: pd.DataFrame) -> None:
    # Question 2: st.columns(4) + st.metric() + Pandas aggregations.
    a, b, c, d = st.columns(4, gap="small")
    with a:
        st.metric("🎬  Total Movies", f"{len(df):,}")
    with b:
        avg = df["averageRating"].mean()
        st.metric("⭐  Average IMDb Rating", f"{avg:.1f}" if pd.notna(avg) else "N/A")
    with c:
        highest = df["averageRating"].max()
        st.metric("🏆  Highest Rating", f"{highest:.1f}" if pd.notna(highest) else "N/A")
    with d:
        st.metric("👥  Total Votes", f"{int(df['numVotes'].sum()):,}")


def render_chart_tabs(df: pd.DataFrame) -> None:
    # Question 4: actual Streamlit tabs for genre, rating, and gross.
    overview, genre_tab, rating_tab, gross_tab = st.tabs(
        ["🎬 Overview", "📊 Genre Analysis", "📈 Rating Analysis", "🪙 Gross Collection"]
    )
    with overview:
        draw_overview_charts(df)
    with genre_tab:
        st.plotly_chart(chart_genres(df), use_container_width=True, key="tab_genre")
        st.caption("Movies with multiple genres count once in each applicable genre.")
    with rating_tab:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(chart_ratings(df), use_container_width=True, key="tab_rating")
        with right:
            st.plotly_chart(chart_votes(df), use_container_width=True, key="tab_votes")
    with gross_tab:
        st.plotly_chart(chart_gross(df), use_container_width=True, key="tab_gross")
        gross_table = top_gross(df)[["primaryTitle", "grossWorldwide"]].copy()
        gross_table.columns = ["Movie Title", "Worldwide Gross (USD)"]
        st.dataframe(gross_table, hide_index=True, use_container_width=True)


def explorer_defaults(df: pd.DataFrame) -> dict:
    rating_min = math.floor(float(df["averageRating"].min()) * 10) / 10
    rating_max = math.ceil(float(df["averageRating"].max()) * 10) / 10
    return {
        "exp_genre": "All Genres",
        "exp_countries": [],
        "exp_rating": (rating_min, rating_max),
        "exp_votes": 0,
        "exp_title": "",
        "exp_sort": "Descending",
    }


def reset_explorer(df: pd.DataFrame) -> None:
    for key, val in explorer_defaults(df).items():
        st.session_state[key] = val


def init_explorer(df: pd.DataFrame) -> None:
    defaults = explorer_defaults(df)
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    valid_genres = ["All Genres", *unique_values(df, "genreList")]
    if st.session_state["exp_genre"] not in valid_genres:
        st.session_state["exp_genre"] = "All Genres"
    countries = set(unique_values(df, "countryList"))
    st.session_state["exp_countries"] = [
        c for c in st.session_state["exp_countries"] if c in countries
    ]
    lower, upper = defaults["exp_rating"]
    old_a, old_b = st.session_state["exp_rating"]
    if old_a < lower or old_b > upper or old_a > old_b:
        st.session_state["exp_rating"] = (lower, upper)


def render_movie_table(df: pd.DataFrame, *, key: str) -> None:
    shown = df[["primaryTitle", "genreDisplay", "countryDisplay", "averageRating", "numVotes", "grossWorldwide"]].copy()
    shown.columns = ["Title", "Genre", "Country", "IMDb Rating", "Votes", "Worldwide Gross (USD)"]
    st.dataframe(
        shown,
        hide_index=True,
        use_container_width=True,
        height=min(330, 80 + 36 * max(1, len(shown))),
        column_config={
            "IMDb Rating": st.column_config.NumberColumn(format="%.1f"),
            "Votes": st.column_config.NumberColumn(format="%d"),
            "Worldwide Gross (USD)": st.column_config.NumberColumn(format="$%.0f"),
        },
    )
    st.download_button(
        "⬇ Download Filtered Results (CSV)",
        data=shown.to_csv(index=False).encode("utf-8-sig"),
        file_name="imdb_filtered_movies.csv",
        mime="text/csv",
        key=f"download_{key}",
    )


def render_explorer(df: pd.DataFrame, *, page: str) -> None:
    # Question 5: selectbox, multiselect, slider, number_input, text_input,
    # radio, result count, dataframe and reset button.
    init_explorer(df)
    st.markdown('<div class="section-name">🔎 Interactive Movie Explorer</div>', unsafe_allow_html=True)
    genres = ["All Genres", *unique_values(df, "genreList")]
    countries = unique_values(df, "countryList")
    a, b, c = st.columns([1.05, 1.3, 1.55], gap="small")
    with a:
        st.selectbox("Genre", genres, key="exp_genre")
        st.multiselect("Countries of Origin", countries, key="exp_countries")
    with b:
        rating_min, rating_max = explorer_defaults(df)["exp_rating"]
        st.slider("IMDb Rating", min_value=rating_min,
                  max_value=max(rating_max, rating_min + 0.1),
                  step=0.1, key="exp_rating")
        st.number_input("Minimum Votes", min_value=0, max_value=2_000_000_000, step=10_000,
                        key="exp_votes")
    with c:
        st.text_input("Search Movie Title", placeholder="Enter movie title...", key="exp_title")
        st.radio("Sort IMDb Ratings", ["Ascending", "Descending"],
                 horizontal=True, key="exp_sort")
    filtered = filter_movies(
        df,
        genre=st.session_state["exp_genre"],
        countries=st.session_state["exp_countries"],
        rating_range=st.session_state["exp_rating"],
        min_votes=int(st.session_state["exp_votes"]),
        title=st.session_state["exp_title"],
        sort_order=st.session_state["exp_sort"],
    )
    count_col, reset_col = st.columns([2, 5], gap="small")
    with count_col:
        st.markdown(f"**🎯 Filtered Movies: {len(filtered):,}**")
    with reset_col:
        st.button("🔄 Reset Filters", key=f"reset_{page}", on_click=reset_explorer, args=(df,))
    if filtered.empty:
        st.info("No movies match these filters. Try resetting or broadening the search.")
    render_movie_table(filtered, key=page)


def render_search(df: pd.DataFrame, *, page: str) -> None:
    # Question 6: st.form, st.form_submit_button, persistent session state.
    st.markdown('<div class="section-name">🔍 Movie Search (Form + Session State)</div>', unsafe_allow_html=True)
    if "selected_movie" not in st.session_state:
        st.session_state["selected_movie"] = None
    if "search_feedback" not in st.session_state:
        st.session_state["search_feedback"] = None
    if "search_preference" not in st.session_state:
        st.session_state["search_preference"] = {"title": "", "genre": "All Genres", "rating": 7.0}

    with st.form(key=f"movie_search_form_{page}", clear_on_submit=False):
        pref = st.session_state["search_preference"]
        query_title = st.text_input("Movie Title", value=pref["title"],
                                     placeholder="e.g. Inception")
        genres = ["All Genres", *unique_values(df, "genreList")]
        genre_index = genres.index(pref["genre"]) if pref["genre"] in genres else 0
        preferred_genre = st.selectbox("Preferred Genre", genres, index=genre_index)
        min_rating = st.slider("Minimum IMDb Rating", 0.0, 10.0,
                               float(pref["rating"]), 0.1)
        submitted = st.form_submit_button("🔍 Search Movie", use_container_width=True,
                                          type="primary")
    if submitted:
        # Search is processed only on form submission; state survives other reruns.
        st.session_state["search_preference"] = {
            "title": query_title, "genre": preferred_genre, "rating": min_rating
        }
        movie, count = find_movie(
            df, title=query_title, genre=preferred_genre, minimum_rating=min_rating
        )
        st.session_state["selected_movie"] = movie
        st.session_state["search_feedback"] = ("found", count) if movie else ("empty", 0)

    feedback = st.session_state["search_feedback"]
    movie = st.session_state["selected_movie"]
    if feedback is None:
        st.info("Submit the form to find a movie. The selected result stays available across interactions.")
    elif feedback[0] == "empty":
        st.info("No matching movies found. Try a different title, genre, or minimum rating.")
    elif movie is not None:
        st.success(f"Movie found! {feedback[1]} matching result(s). Selection saved in session state.")
        st.markdown(f"### 🎬 {movie['primaryTitle']}")
        st.write(f"**Genre:** {movie['genreDisplay']}")
        st.write(f"**IMDb Rating:** ⭐ {movie['averageRating']:.1f} / 10")
        st.write(f"**Number of Votes:** {int(movie['numVotes']):,}")
        st.caption(f"Countries: {movie['countryDisplay']} · Gross: {format_usd(movie['grossWorldwide'])}")
        poster = movie.get("primaryImage", None)
        if isinstance(poster, str) and poster.startswith(("https://", "http://")):
            with st.expander("View movie poster"):
                st.image(poster, width=185)


def render_dataset_inspection(raw_df: pd.DataFrame, *, full_page: bool = False) -> None:
    # Question 1: first ten rows, shape, columns, dtypes, null counts, success.
    st.subheader("🗂️ Dataset Loading & Initial Exploration — Question 1")
    st.success("IMDb dataset loaded successfully!")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Rows", f"{raw_df.shape[0]:,}")
    with c2:
        st.metric("Total Columns", f"{raw_df.shape[1]:,}")
    st.markdown("**First 10 Records — `df.head(10)`**")
    st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)
    st.markdown("**Column Names**")
    st.write(list(raw_df.columns))
    left, right = st.columns(2)
    with left:
        st.markdown("**Data Types — `df.dtypes`**")
        st.dataframe(
            pd.DataFrame({"Column": raw_df.columns, "Data Type": raw_df.dtypes.astype(str).values}),
            hide_index=True, use_container_width=True,
        )
    with right:
        st.markdown("**Missing Values — `df.isnull().sum()`**")
        st.dataframe(
            raw_df.isnull().sum().rename_axis("Column").reset_index(name="Missing Values"),
            hide_index=True, use_container_width=True,
        )
    if full_page:
        st.info("The inspection uses the original raw CSV, before any cleaning or derived columns.")


def render_sidebar(raw_df: pd.DataFrame | None, source: str) -> str:
    with st.sidebar:
        nav = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🔎 Movie Explorer", "🔍 Movie Search (Form + Session State)",
             "🗂️ Dataset Inspection"],
            label_visibility="collapsed",
        )
        st.markdown('<div class="side-section"></div>', unsafe_allow_html=True)
        with st.expander("ℹ️ Dataset Information", expanded=True):
            st.markdown("**Top 250 Movies on IMDb (2026)**")
            if raw_df is not None:
                st.write(f"**Rows:** {raw_df.shape[0]:,}")
                st.write(f"**Columns:** {raw_df.shape[1]:,}")
                st.caption("Movie titles, ratings, votes, genres, countries, and worldwide gross.")
                st.caption(f"Source: {source}")
            else:
                st.caption("Upload the CSV below or enable Kaggle downloading.")
        st.markdown("[Open original Kaggle dataset](" + KAGGLE_URL + ")")
        st.caption("Upload a CSV in the uploader above to override the Kaggle data.")
        st.caption("Streamlit Practical • Questions 1–6 • 50 Marks")
        st.markdown(
            """
            <div class="student-footer">
                <div class="student-name">V. BHARGAVA</div>
                <div class="student-prn">PRN: 250200109</div>
            </div>
            <style>
                [data-testid="stSidebar"] .student-footer {
                    position: fixed;
                    bottom: 12px;
                    left: 12px;
                    width: calc(21rem - 24px);
                    max-width: calc(100vw - 24px);
                    padding: 11px 14px;
                    text-align: center;
                    background: #263449;
                    border-top: 2px solid #facc15;
                    border-radius: 9px;
                    box-shadow: 0 3px 14px #0005;
                    z-index: 100;
                }
                [data-testid="stSidebar"] .student-name {
                    color: white;
                    font-weight: 800;
                    font-size: 16px;
                }
                [data-testid="stSidebar"] .student-prn {
                    color: #cbd5e1;
                    font-size: 13px;
                    margin-top: 3px;
                }
                [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
                    padding-bottom: 110px;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    return nav


def main() -> None:
    apply_styles()
    with st.sidebar:
        st.markdown(
            f'<img class="imdb-logo" src="{img_data_uri(ROOT / "assets" / "imdb_logo.png")}" '
            'alt="IMDb logo" /><div class="side-name">Movie Analytics<br>Dashboard</div>',
            unsafe_allow_html=True,
        )
        uploaded_csv = st.file_uploader("📂 Upload IMDb CSV (optional)", type=["csv"],
                                         help="This overrides the auto-downloaded Kaggle CSV.")
    raw_df, source = load_dataset(uploaded_csv)
    nav = render_sidebar(raw_df, source)
    show_header()
    if raw_df is None:
        st.warning("The dataset has not been loaded yet, so no statistics have been invented.")
        st.error(source)
        st.markdown(
            "**To run:** download the CSV from the [exact Kaggle dataset](" + KAGGLE_URL +
            ") and upload it using the sidebar, or place it in the project's `data/` folder "
            "and reload. Automatic Kaggle downloading is also supported on machines with internet access."
        )
        st.stop()
    try:
        df = prepare_movies(raw_df)
    except ValueError as exc:
        st.error(str(exc))
        st.info("Please use the paper's exact Kaggle dataset, which contains the required columns.")
        st.stop()
    if df.empty or df["averageRating"].dropna().empty:
        st.error("The loaded CSV contains no usable movie ratings.")
        st.stop()

    if nav == "🗂️ Dataset Inspection":
        render_dataset_inspection(raw_df, full_page=True)
        return
    if nav == "🔎 Movie Explorer":
        render_explorer(df, page="standalone")
        return
    if nav == "🔍 Movie Search (Form + Session State)":
        render_search(df, page="standalone")
        return

    # Dashboard page: Q2–Q4, with Q5 and Q6 panels below, mirroring supplied image.
    render_kpis(df)
    st.divider()
    render_chart_tabs(df)
    st.success(f"Dashboard loaded successfully! Explore and analyze {len(df):,} movies on IMDb.")
    st.divider()
    left, right = st.columns([1.6, 1], gap="medium")
    with left:
        render_explorer(df, page="dashboard")
    with right:
        render_search(df, page="dashboard")
    with st.expander("🗂️ Question 1 — Preview raw data and dataset diagnostics"):
        render_dataset_inspection(raw_df)
    st.caption("Chart values and KPIs are calculated from the loaded CSV; the supplied screenshot is a design reference only.")


if __name__ == "__main__":
    main()
