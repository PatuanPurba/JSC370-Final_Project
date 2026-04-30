from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go


DATA_DIR = Path("data/processed")

# If the CSV files are in the same folder as this script, use current directory.
if not (DATA_DIR / "movies_enrich.csv").exists():
    DATA_DIR = Path(".")

OUT_DIR = Path(".")


def safe_spearman(x, y, min_n=30):
    """Spearman correlation with pairwise missing-value handling."""
    d = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(d)

    if n < min_n or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return np.nan, n

    rho = d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())
    return rho, n


def line_fit(x, y, min_n=30):
    """Simple OLS visual trendline plus Spearman correlation."""
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(d) < min_n or d["x"].nunique() < 2:
        return None, None, None

    lo, hi = d["x"].quantile([0.01, 0.99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = d["x"].min(), d["x"].max()

    slope, intercept = np.polyfit(d["x"], d["y"], 1)
    xs = np.linspace(lo, hi, 80)
    ys = slope * xs + intercept

    rho, n = safe_spearman(d["x"], d["y"], min_n=min_n)
    return xs, ys, (rho, n, slope)


def main():
    movies = pd.read_csv(DATA_DIR / "movies_enrich.csv")

    # Loaded only to make the script clearly tied to your three sources.
    # These are not used directly in the movie-level plots because they can duplicate movies by person rows.
    _person_movie_history_check = pd.read_csv(DATA_DIR / "person_movie_history.csv", nrows=5)
    _person_check = pd.read_csv(DATA_DIR / "person.csv", nrows=5)

    df = movies.copy()

    df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
    if "Year" not in df.columns:
        df["Year"] = df["Release Date"].dt.year

    df = df[(df["Year"] >= 2000) & (df["Year"] <= 2025)].copy()

    # Log transforms for skewed variables.
    df["Revenue (log USD)"] = np.where(
        df["Revenue (USD)"] > 0,
        np.log10(df["Revenue (USD)"]),
        np.nan
    )

    df["Budget (log USD)"] = np.where(
        df["Budget (USD)"] > 0,
        np.log10(df["Budget (USD)"]),
        np.nan
    )

    df["Popularity (log)"] = np.log1p(df["Popularity"].clip(lower=0))

    df["Profit (signed log USD)"] = np.where(
        df["Profit (USD)"].notna(),
        np.sign(df["Profit (USD)"]) * np.log10(np.abs(df["Profit (USD)"]) + 1),
        np.nan
    )

    df["Blockbuster (0/1)"] = df["Blockbuster"].astype(float)

    df["Era"] = pd.cut(
        df["Year"],
        bins=[1999, 2009, 2019, 2025],
        labels=["2000-2009", "2010-2019", "2020-2025"],
        include_lowest=True,
    )

    genre_cols = [c for c in df.columns if c.startswith("genre_")]

    top_genre_cols = (
        df[genre_cols]
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .index
        .tolist()
    )

    selected_features = [
        "Length (min)",
        "Budget (log USD)",
        "Cast Size",
        "Crew Size",
        "top5_cast_pop",
        "top5_cast_wavg_mrating",
        "top5_cast_avg_mpop",
        "top5_cast_nblockbuster",
        "top5_cast_nprofit",
        "top5_cast_career_span",
        "n_experienced_cast",
        "top5_crew_wavg_mrating",
        "top5_crew_avg_mpop",
        "top5_crew_nblockbuster",
        "top5_crew_nprofit",
        "n_experienced_crew",
        "director_nmovies",
        "director_wavg_mrating",
        "director_avg_mpop",
        "director_nblockbuster",
        "director_nprofit",
        "director_career_span",
    ] + top_genre_cols

    feature_labels = {
        "Length (min)": "Length",
        "Budget (log USD)": "Budget (log10 USD)",
        "Cast Size": "Cast size",
        "Crew Size": "Crew size",
        "top5_cast_pop": "Top 5 cast: current popularity",
        "top5_cast_wavg_mrating": "Top 5 cast: past weighted rating",
        "top5_cast_avg_mpop": "Top 5 cast: past movie popularity",
        "top5_cast_nblockbuster": "Top 5 cast: past blockbusters",
        "top5_cast_nprofit": "Top 5 cast: past profitable movies",
        "top5_cast_career_span": "Top 5 cast: career span",
        "n_experienced_cast": "Experienced cast count",
        "top5_crew_wavg_mrating": "Top 5 crew: past weighted rating",
        "top5_crew_avg_mpop": "Top 5 crew: past movie popularity",
        "top5_crew_nblockbuster": "Top 5 crew: past blockbusters",
        "top5_crew_nprofit": "Top 5 crew: past profitable movies",
        "n_experienced_crew": "Experienced crew count",
        "director_nmovies": "Director: past movie count",
        "director_wavg_mrating": "Director: past weighted rating",
        "director_avg_mpop": "Director: past movie popularity",
        "director_nblockbuster": "Director: past blockbusters",
        "director_nprofit": "Director: past profitable movies",
        "director_career_span": "Director: career span",
    }

    for c in top_genre_cols:
        feature_labels[c] = "Genre: " + c.replace("genre_", "")

    target_cols = [
        "Weighted Rating",
        "Popularity (log)",
        "Revenue (log USD)",
        "Profit (signed log USD)",
        "Blockbuster (0/1)",
    ]

    target_labels = {
        "Weighted Rating": "Weighted rating",
        "Popularity (log)": "Popularity (log1p)",
        "Revenue (log USD)": "Revenue (log10 USD)",
        "Profit (signed log USD)": "Profit (signed log10 USD)",
        "Blockbuster (0/1)": "Blockbuster",
    }

    # =========================================================
    # Figure 2: Temporal relationship explorer
    # =========================================================

    relationship_specs = [
        (
            "Budget → Revenue",
            "Budget (log USD)",
            "Revenue (log USD)",
            "Budget-revenue relationship: tests whether higher production spending "
            "translates into larger revenue.",
        ),
        (
            "Cast size → Revenue",
            "Cast Size",
            "Revenue (log USD)",
            "Team size relationship: tests whether larger cast teams are associated "
            "with larger revenue.",
        ),
        (
            "Crew size → Revenue",
            "Crew Size",
            "Revenue (log USD)",
            "Team size relationship: tests whether larger crew teams are associated "
            "with larger revenue.",
        ),
        (
            "Top cast past popularity → Revenue",
            "top5_cast_avg_mpop",
            "Revenue (log USD)",
            "Cast history relationship: tests whether movies with previously popular "
            "cast members have higher revenue.",
        ),
        (
            "Top cast past rating → Weighted rating",
            "top5_cast_wavg_mrating",
            "Weighted Rating",
            "Cast quality relationship: tests whether cast members' past rating "
            "history aligns with current movie rating.",
        ),
        (
            "Director past popularity → Revenue",
            "director_avg_mpop",
            "Revenue (log USD)",
            "Director history relationship: tests whether directors with popular "
            "past movies are linked to higher revenue.",
        ),
        (
            "Director past rating → Weighted rating",
            "director_wavg_mrating",
            "Weighted Rating",
            "Director quality relationship: tests whether director past ratings "
            "align with current movie rating.",
        ),
        (
            "Experienced cast count → Revenue",
            "n_experienced_cast",
            "Revenue (log USD)",
            "Experience relationship: tests whether more experienced cast members "
            "are linked to higher revenue.",
        ),
        (
            "Length → Weighted rating",
            "Length (min)",
            "Weighted Rating",
            "Movie characteristic relationship: tests whether longer movies are "
            "associated with different audience ratings.",
        ),
    ]

    era_order = ["2000-2009", "2010-2019", "2020-2025"]

    fig2 = go.Figure()
    trace_groups = []
    trace_idx = 0

    for spec_i, (label, x_col, y_col, note) in enumerate(relationship_specs):
        group_indices = []

        for era in era_order:
            sub = df[df["Era"].astype(str) == era].copy()

            cols = list(dict.fromkeys([
                x_col,
                y_col,
                "Title",
                "Year",
                "Rating",
                "Revenue (USD)",
                "Weighted Rating",
                "Popularity",
            ]))

            d = (
                sub[cols]
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[x_col, y_col])
            )

            # Keeps the exported HTML smaller.
            # Remove this block if you want all points shown.
            if len(d) > 1000:
                d = d.sample(1000, random_state=42)

            visible = spec_i == 0

            revenue_strings = d["Revenue (USD)"].apply(
                lambda v: f"${v:,.0f}" if pd.notna(v) else "missing"
            )

            if len(d):
                custom = np.column_stack([
                    d["Title"].astype(str),
                    d["Year"].astype(int).astype(str),
                    d["Rating"].astype(float).round(3).astype(str),
                    revenue_strings.astype(str),
                ])
            else:
                custom = np.empty((0, 4), dtype=object)

            fig2.add_trace(
                go.Scattergl(
                    x=d[x_col].to_numpy(),
                    y=d[y_col].to_numpy(),
                    mode="markers",
                    name=f"{era} movies",
                    legendgroup=era,
                    showlegend=(spec_i == 0),
                    visible=visible,
                    marker=dict(size=5, opacity=0.38),
                    customdata=custom,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Year: %{customdata[1]}<br>"
                        f"{feature_labels.get(x_col, x_col)}: " + "%{x:.3f}<br>"
                        f"{target_labels.get(y_col, y_col)}: " + "%{y:.3f}<br>"
                        "Rating: %{customdata[2]}<br>"
                        "Revenue: %{customdata[3]}<extra></extra>"
                    ),
                )
            )

            group_indices.append(trace_idx)
            trace_idx += 1

            fit_x, fit_y, stat = line_fit(sub[x_col], sub[y_col])

            if fit_x is None:
                fit_x, fit_y, stat = [], [], (np.nan, 0, np.nan)

            rho, n, slope = stat

            fig2.add_trace(
                go.Scatter(
                    x=fit_x,
                    y=fit_y,
                    mode="lines",
                    name=f"{era} linear trend",
                    legendgroup=era,
                    showlegend=False,
                    visible=visible,
                    line=dict(width=3),
                    hovertemplate=(
                        f"{era}<br>"
                        f"Spearman ρ: {rho:.3f}<br>"
                        f"n = {n:,}<br>"
                        f"OLS slope: {slope:.3f}<extra></extra>"
                    ),
                )
            )

            group_indices.append(trace_idx)
            trace_idx += 1

        trace_groups.append(group_indices)

    buttons2 = []
    n_traces = len(fig2.data)

    for spec_i, (label, x_col, y_col, note) in enumerate(relationship_specs):
        visible = [False] * n_traces

        for idx in trace_groups[spec_i]:
            visible[idx] = True

        buttons2.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": f"Temporal relationship explorer: {label}",
                        "xaxis": {"title": feature_labels.get(x_col, x_col)},
                        "yaxis": {"title": target_labels.get(y_col, y_col)},
                        "annotations": [
                            dict(
                                text=(
                                    note
                                    + " Compare colored era trendlines to see whether "
                                    "the relationship changes over time."
                                ),
                                x=0,
                                y=1.08,
                                xref="paper",
                                yref="paper",
                                showarrow=False,
                                align="left",
                            )
                        ],
                    },
                ],
            )
        )

    first_label, first_x, first_y, first_note = relationship_specs[0]

    fig2.update_layout(
        title=f"Temporal relationship explorer: {first_label}",
        xaxis_title=feature_labels.get(first_x, first_x),
        yaxis_title=target_labels.get(first_y, first_y),
        width=1050,
        height=720,
        margin=dict(l=80, r=40, t=120, b=70),
        legend_title_text="Era",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=1.0,
                y=1.17,
                xanchor="right",
                yanchor="top",
                buttons=buttons2,
            )
        ],
        annotations=[
            dict(
                text=(
                    first_note + " Compare colored era trendlines to see whether the relationship changes over time."
                ),
                x=0,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
            )
        ],
    )

    fig2.write_html(
        OUT_DIR / "fig2_temporal_relationship_explorer.html",
        include_plotlyjs="cdn",
        full_html=True,
    )

    print("Saved fig1_movie_association_heatmap.html")
    print("Saved fig2_temporal_relationship_explorer.html")


if __name__ == "__main__":
    main()