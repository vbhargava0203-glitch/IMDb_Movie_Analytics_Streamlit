# IMDb Movie Analytics Dashboard — Streamlit Practical (All 6 Questions)

A complete Python + Streamlit solution for the **50-mark practical question paper**, designed using the supplied IMDb dashboard screenshot. It implements real CSV-driven analytics — no fake KPI totals are hard-coded.

## Quick start on Windows (PowerShell)

```powershell
cd IMDb_Movie_Analytics_6_Questions
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit opens in your browser, usually at `http://localhost:8501`.

## Dataset — IMPORTANT

The exact dataset specified in the examination paper is:

<https://www.kaggle.com/datasets/arjunsinghgangwar/top-250-movies-on-imdb-in-2026>

**How the application obtains it:**
1. If you upload a CSV in the left sidebar, it uses that file.
2. Otherwise, if you put the **original CSV** in the `data/` folder, it uses that.
3. Otherwise, it attempts to download the official Kaggle dataset automatically using `kagglehub` (internet access required).
4. If downloading fails, it displays directions for manual CSV download and upload.

The source CSV is **not included** in this ZIP because it was **not uploaded** with the paper and dashboard screenshot, and downloading it was not possible within the packaging environment. Do not substitute an unrelated or fabricated dataset: the assignment's actual dataset has **250 rows and 27 columns**, and the KPI and charts should reflect its values.

For manual setup, download the Kaggle dataset, unzip it, and put the CSV (e.g. `ToP 250 movies on imdb in 2026.csv`) inside `data/`.

## Question-by-question coverage

| Question | Marks | Exact features implemented | Where to view |
|---|---:|---|---|
| Q1 — Loading & Exploration | 5 | `pd.read_csv`, `head(10)`, `shape`, `columns`, `dtypes`, `isnull().sum()`, `st.success` | **Dataset Inspection** navigation and dashboard bottom expander |
| Q2 — KPI Metrics | 7 | 4 `st.metric` cards in `st.columns(4)`: total, mean rating, highest rating, sum of votes | Dashboard top |
| Q3 — Visualizations | 8 | exploded multi-genre counts, histogram of ratings, rating-votes scatter, top 10 worldwide grosses | Dashboard **Overview** and analysis tabs |
| Q4 — Layout | 3 | `st.tabs`, `st.expander`, `st.divider`, success/info messages | Dashboard and sidebar |
| Q5 — Interactive Explorer | 12 | genre selectbox, multi-country multiselect, rating range slider, minimum votes number input, title text input, ascending/descending radio, filtered count/table, reset button | **Movie Explorer** navigation and dashboard lower left |
| Q6 — Form & Session State | 10 | `st.form` with title/genre/minimum rating, submit button, persistent selection through `st.session_state`, movie title/genre/rating/votes, appropriate messages | **Movie Search** navigation and dashboard lower right |

## How to get your submission screenshots

Once the app is running with the actual Kaggle CSV:

1. On **Dashboard**, capture the four colored KPI cards and the four chart panels.
2. Scroll down and capture the **Interactive Movie Explorer** with filters applied.
3. Search for a movie such as *Inception* in the **Movie Search** form, then capture the result.
4. Navigate to **Dataset Inspection** and capture the first 10 rows, data types, and null counts.

The exam asks for actual screenshots **of the running application**. The original supplied screenshot is stored as `reference/design_reference.png` only as a *visual guide*, not as evidence that this app was executed.

## Files

- `app.py` — full dashboard, Streamlit widgets and charts
- `utils.py` — reusable, tested Pandas dataset preparation and filtering functions
- `.streamlit/config.toml` — dark UI theme
- `assets/` — cropped visuals from your uploaded screenshot
- `data/` — put the Kaggle CSV here if manually downloaded
- `tests/test_utils.py` — tests using a synthetic **test fixture** (not presented as IMDb data)
- `reference/` — your supplied exam PDF and reference screenshot
- `requirements.txt` — installation dependencies

## Checks

Run:

```powershell
python -m unittest discover -s tests -v
```

The app does not alter the original dataset while processing genre and country lists, so the raw-column / missing-value answer to Q1 stays faithful to the CSV.
