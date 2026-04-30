import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go


# LOAD DATA
# ===========

# df_1 = "During Film Making"
# df_2 = "Current Career"
from pathlib import Path
import pandas as pd

files = sorted(Path("data").glob("full_movie_history_*.csv"))
df_1 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


df_2 = pd.read_csv("data/person.csv")
df_2["career_span"] = df_2["last_yr"] - df_2["first_yr"] + 1


movies_enrich = pd.read_csv("data/movies_enrich.csv")

features_1 = [
    "movie_id", "Title", "cast_id", "Name", "Gender", "Popularity_x", "Popularity_y", "Job", "prev_n_movies",
    "prev_wavg_rating", "prev_avg_pop", "prev_avg_mpop", "prev_recent_mpop", "prev_n_blockbuster", "prev_career_span"
]

features_2 = [
    "pid", "Name", "Gender", "main_job", "n_movies", "wavg_rating", "avg_mpop", "avg_pop", 
    "n_blockbuster", "career_span"
]


df_1 = df_1[features_1]
df_1 = df_1.rename(columns={"cast_id": "pid", "movie_id": "Movie ID"})
df_1["Movie ID"] = df_1["Movie ID"].astype(str)
df_1["pid"] = df_1["pid"].astype(str)
df_1["Gender"] = df_1["Gender"].replace({-1: "Male", 1: "Female", 0: "Missing"})
df_1["prev_career_span"] = df_1["prev_career_span"].apply(lambda x: np.where(pd.isna(x), "0 Years", str(x) + " Years"))

num_columns = ["Popularity_x", "Popularity_y", "prev_wavg_rating", "prev_avg_pop", "prev_avg_mpop", "prev_recent_mpop"]
for col in num_columns:
    df_1[col] = df_1[col].round(3)


df_1["prev_avg_pop"] = df_1["prev_avg_pop"].replace(np.nan, 0)
df_1 = df_1.replace(np.nan, "Missing")



df_2 = df_2[features_2]
df_2["pid"] = df_2["pid"].astype(str)
df_2["Gender"] = df_2["Gender"].replace({-1: "Male", 1: "Female", 0: "Missing"})
df_2["career_span"] = df_2["career_span"].apply(lambda x: np.where(pd.isna(x), "0 Years", str(x) + " Years"))

num_columns = ["wavg_rating", "avg_mpop", "avg_pop"]
for col in num_columns:
    df_2[col] = df_2[col].round(3)

df_2["avg_pop"] = df_2["avg_pop"].replace(np.nan, 0)
df_2 = df_2.replace(np.nan, "Missing")


# 2. X and Y for Scatter Plot
# =================================

DF1_X = "prev_avg_pop"
sp1_xlabel = "Average of Past Involvement Popularity"
DF1_Y = "Popularity_y"
sp1_ylabel = "Popularity in The Film"

DF2_X = "wavg_rating"
sp2_xlabel = "Average of Past Movie's Rating"
DF2_Y = "avg_pop"
sp2_ylabel = "Average of Past Movie's Popularity"


# Prevention to make sure each pid will only have 1 row per pid (This is current_statistics)
df_2 = (
    df_2.sort_values("avg_pop", ascending=False)
        .drop_duplicates(subset=["pid"])
        .copy()
)


# CUSTOMDATA Setup
# ===================


CUSTOM_FIELDS = [
    "movie_id", "pid", "Title", "Name", "Gender", "Popularity_x", "Popularity_y", "Job", "prev_n_movies",
    "prev_wavg_rating", "prev_avg_pop", "prev_recent_mpop", "prev_n_blockbuster", "prev_career_span",

    "main_job", "n_movies", "wavg_rating", "avg_mpop", "avg_pop", "n_blockbuster", "career_span"
]

CUSTOM_FIELDS = [col for col in CUSTOM_FIELDS if col in df_1.columns or col in df_2.columns]


def get_existing_fields(df, fields):
    return [col for col in fields if col in df.columns]


def make_customdata(df, fields):
    fields = get_existing_fields(df, fields)

    if not fields:
        return None

    temp = df[fields].copy()
    temp = temp.where(pd.notna(temp), None)

    return temp.to_numpy(), fields


def extract_pid_from_click(click_data, fields):
    """
    Plotly customdata is returned as a list, not a dict.
    This function converts the clicked point back into a pid.
    """
    if not click_data:
        return None

    try:
        customdata = click_data["points"][0]["customdata"]
        data_dict = dict(zip(fields, customdata))
        return str(data_dict.get("pid"))
    except Exception:
        return None



# Figure Function
# ========================================
rename_map = {
    "movie_id": "Movie ID",
    "Movie ID": "Movie ID",
    "Title": "Title",
    "pid": "pid",
    "Name": "Name",
    "Gender": "Gender",
    "Job": "Job",

    "cast_id": "pid",
    "Popularity_x": "Movie's Popularity",
    "Popularity_y": "Cast's Popularity",
    "prev_n_movies": "Number of Past Movies Involved",
    "prev_wavg_rating": "Past Movie's Rating (Average)",
    "prev_avg_pop": "Past Involvement's Popularity (Average)",
    "prev_avg_mpop": "Past Movie's Popularity (Average)", 
    "prev_recent_mpop": "Last Movie's Popularity",
    "prev_n_blockbuster": "Number of Past Blockbuster Movie",
    "prev_career_span": "Career Span",

    "main_job": "Main Job",
    "n_movies": "Number of Movies Involved",
    "wavg_rating": "Past Movie's Rating (Average)",
    "avg_mpop":  "Past Movie's Popularity (Average)", 
    "avg_pop": "Average Past Involvement Popularity",
    "n_blockbuster": "Number of Past Blockbuster Movie",
    "career_span": "Career Span"
}



def make_scatter(
    df,
    x_col,
    x_label, 
    y_col,
    y_label,
    title,
    selected_pid=None,
):
    df = df.copy()

    if selected_pid is not None:
        selected_pid = str(selected_pid)

    df["is_selected"] = df["pid"].astype(str).eq(selected_pid)

    normal_df = df[~df["is_selected"]]
    selected_df = df[df["is_selected"]]

    customdata_normal, fields_normal = make_customdata(normal_df, CUSTOM_FIELDS)
    customdata_selected, fields_selected = make_customdata(selected_df, CUSTOM_FIELDS)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=normal_df[x_col],
            y=normal_df[y_col],
            mode="markers",
            name="People",
            marker=dict(size=8, opacity=0.35),
            customdata=customdata_normal,
            hovertemplate=make_hovertemplate(fields_normal),
        )
    )

    if len(selected_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=selected_df[x_col],
                y=selected_df[y_col],
                mode="markers",
                name="Selected person",
                marker=dict(
                    size=16,
                    opacity=1,
                    line=dict(width=2),
                ),
                customdata=customdata_selected,
                hovertemplate=make_hovertemplate(fields_selected),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        clickmode="event+select",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.25)", gridwidth=0.5, griddash="dash")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.25)", gridwidth=0.5, griddash="dash")

    return fig


def make_hovertemplate(fields):
    if not fields:
        return None

    lines = []
    for i, field in enumerate(fields):
        lines.append(f"<b>  {rename_map[field]}</b>: \t %{{customdata[{i}]}}")

    return "<br>".join(lines) + "<extra></extra>"



# Initialize Dash and Layout
# ============================================================

app = Dash(__name__)

movie_options = (
    df_1[["Movie ID", "Title"]]
    .dropna()
    .drop_duplicates()
    .sort_values("Movie ID")
)

app.layout = html.Div(
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "fontFamily": "Arial, sans-serif",
        "padding": "20px",
    },
    children=[
        html.H2("Movie's Team Performance"),
        dcc.Dropdown(
            id="movie-id-dropdown",
            options=[{
                "label": f"{row['Title']} ({row['Movie ID']})",
                "value": str(row["Movie ID"])} for _, row in movie_options.iterrows()
            ],
            value=str(movie_options["Movie ID"].iloc[0]),
            searchable=True,
            clearable=False,
            placeholder="Search Movie Title...",
        ),

        dcc.Store(id="selected-pid-store"),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "20px",
                "marginTop": "25px",
            },
            children=[
                dcc.Graph(id="scatter-during"),
                dcc.Graph(id="scatter-current"),
            ],
        ),

        html.Div(
            id="selected-person-panel",
            style={
                "marginTop": "20px",
                "padding": "15px",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "backgroundColor": "#fafafa",
            },
        ),
    ],
)



# CALLBACK: STORE CLICKED PID
# ==================================

@app.callback(
    Output("selected-pid-store", "data"),
    Input("scatter-during", "clickData"),
    Input("scatter-current", "clickData"),
    State("selected-pid-store", "data"),
    prevent_initial_call=True,
)
def update_selected_pid(click_during, click_current, current_pid):
    triggered_id = ctx.triggered_id

    if triggered_id == "scatter-during":
        fields = get_existing_fields(df_1, CUSTOM_FIELDS)
        pid = extract_pid_from_click(click_during, fields)
    elif triggered_id == "scatter-current":
        fields = get_existing_fields(df_2, CUSTOM_FIELDS)
        pid = extract_pid_from_click(click_current, fields)
    else:
        return no_update

    if pid is None:
        return current_pid

    return pid


# CALLBACK: UPDATE BOTH FIGURES + DETAIL PANEL
# ==================================================

@app.callback(
    Output("scatter-during", "figure"),
    Output("scatter-current", "figure"),
    Output("selected-person-panel", "children"),
    Input("movie-id-dropdown", "value"),
    Input("selected-pid-store", "data"),
)
def update_figures(movie_id, selected_pid):
    movie_id = str(movie_id)

    # People involved in the selected movie
    df1_movie = df_1[df_1["Movie ID"] == movie_id].copy()

    # Same people in the current-career dataframe
    valid_pids = set(df1_movie["pid"])
    df2_movie_people = df_2[df_2["pid"].isin(valid_pids)].copy()

    # If selected pid is not part of the selected movie, ignore it.
    if selected_pid is not None and str(selected_pid) not in valid_pids:
        selected_pid = None

    fig1 = make_scatter(
        df=df1_movie,
        x_col=DF1_X,
        x_label=sp1_xlabel,
        y_col=DF1_Y,
        y_label=sp2_ylabel,
        title=f"During Film Making — Movie ID {movie_id}",
        selected_pid=selected_pid,
    )

    fig2 = make_scatter(
        df=df2_movie_people,
        x_col=DF2_X,
        x_label=sp2_xlabel,
        y_col=DF2_Y,
        y_label=sp2_ylabel,
        title="Current Career",
        selected_pid=selected_pid,
    )

    panel = make_detail_panel(movie_id)

    return fig1, fig2, panel


movies_dict = {
    "top5_cast_pop": "Top 5 Cast Popularity",
    "top5_cast_wavg_mrating": "Top 5 Cast Weighted Avg Movie Rating",
    "top5_cast_avg_mpop": "Top 5 Cast Avg Movie Popularity",
    "top5_cast_nblockbuster": "Top 5 Cast No. of Blockbusters",
    "top5_cast_nprofit": "Top 5 Cast No. of Profitable Movies",
    "top5_cast_career_span": "Top 5 Cast Avg Career Span ",
    "n_experienced_cast": "No. of Experienced Cast",

    "has_crew": "Has Crew",
    "top5_crew_wavg_mrating": "Top 5 Crew Weighted Avg Movie Rating",
    "top5_crew_avg_mpop": "Top 5 Crew Avg Movie Popularity",
    "top5_crew_nblockbuster": "Top 5 Crew No. of Blockbusters",
    "top5_crew_nprofit": "Top 5 Crew No. of Profitable Movies",
    "n_experienced_crew": "No. of Experienced Crew",

    "has_director": "Has Director",
    "director_nmovies": "Director No. of Movies",
    "director_pop": "Director Popularity",
    "director_wavg_mrating": "Director Weighted Avg Movie Rating",
    "director_avg_mpop": "Director Avg Movie Popularity",
    "director_recent_mpop": "Director Recent Movie Popularity",
    "director_nblockbuster": "Director No. of Blockbusters",
    "director_nprofit": "Director No. of Profitable Movies",
    "director_career_span": "Director Career Span",
}

["n_experienced_cast", "n_experienced_crew", "top5_cast_pop", "top5_cast_career_span"
 "director_nmovies", "director_avg_mpop", "director_nblockbuster", "director_career_span"]


def make_detail_panel(movie_id):
    if movie_id is None:
        return html.Div(
            [
                html.H4("Selected Movie"),
                html.P("Select a movie title to see Movie's Information"),
            ]
        )

    movie_row = movies_enrich[movies_enrich["movie_id"].astype(str).eq(str(movie_id))]

    if movie_row.empty:
        return html.Div(
            [
                html.H4("Selected Movie"),
                html.P("No detail found for the selected Movie"),
            ]
        )

    movie_info = movie_row.iloc[0].to_dict() if not movie_row.empty else {}
    title = movie_info.get("Title", "Unknown")

    return html.Div(
        [
            html.H4(f"{movie_id}: {title}"),

            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Movie's Metadata:", style={"marginBottom": "8px"}),
                            html.Ul(
                                [
                                    html.Li(
                                        f"{key}: {movie_info.get(key)}",
                                        style={"marginBottom": "6px"}
                                    )
                                    for key in [
                                        "Release Date", "Language", "Length (min)",
                                        "Budget (USD)", "Revenue (USD)",
                                        "Popularity", "Rating Count", "Rating"
                                    ]
                                    if key in movie_info
                                ],
                                style={
                                    "marginTop": "0",
                                    "paddingLeft": "20px"
                                }
                            )
                        ],
                        style={
                            "flex": "1"
                        }
                    ),

                    html.Div(
                        [
                            html.H4("Movie's Team Statistics:", style={"marginBottom": "8px"}),
                            html.Ul(
                                [
                                    html.Li(
                                        f"{movies_dict[key]}: {movie_info.get(key)}",
                                        style={"marginBottom": "6px"}
                                    )
                                    for key in [
                                        "top5_cast_pop", "top5_cast_career_span",
                                        "director_nmovies", "director_avg_mpop",
                                        "director_nblockbuster", "director_career_span"
                                    ]
                                    if key in movie_info
                                ],
                                style={
                                    "marginTop": "0",
                                    "paddingLeft": "20px"
                                }
                            )
                        ],
                        style={
                            "flex": "1"
                        }
                    )
                ],
                style={
                    "display": "flex",
                    "gap": "40px",
                    "alignItems": "flex-start",
                    "width": "100%"
                }
            ),
        ]
    )





if __name__ == "__main__":
    app.run(debug=True)
    server = app.server