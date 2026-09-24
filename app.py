"""
CampusPulse - Campus Placement Analytics & Prediction Dashboard
================================================================
Run with:  streamlit run app.py

Pages
-----
1. Overview          - headline KPIs and placement / package trends
2. Insights          - what drives placement (academics, experience, skills, employers)
3. Student Predictor - placement probability, expected package and "what-if" advice
4. SQL Explorer      - read-only SQL on the students table (SQLite)
5. Data & Models     - dataset preview, model metrics, feature importance
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "college_placement_analytics_5000.csv"
PLACEMENT_MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"
SALARY_MODEL_PATH = BASE_DIR / "models" / "salary_model.pkl"

# Must match the feature order used to train the models in the notebook.
FEATURES = [
    "CGPA",
    "Tenth_Percentage",
    "Twelfth_Percentage",
    "Backlogs",
    "Internship_flag",
    "Projects",
    "Certifications",
    "DSA_Score",
    "Technical_Score",
    "Aptitude_Score",
    "Communication_Score",
]

# Test-set metrics recorded in notebooks/CampusPulse.ipynb
PLACEMENT_ACCURACY = 0.797
SALARY_MAE, SALARY_RMSE, SALARY_R2 = 0.83, 1.04, 0.374

PAGES = [
    "Overview",
    "Insights",
    "Student Predictor",
    "SQL Explorer",
    "Data & Models",
]

st.set_page_config(
    page_title="CampusPulse",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Data, models, database
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    """Load the CSV and apply the same cleaning steps as the notebook."""
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    for col in ("Tenth_Percentage", "Twelfth_Percentage"):
        df[col] = df[col].fillna(df[col].median())
    df["Skills"] = df["Skills"].fillna("Unknown")
    df["Internship_flag"] = df["Internship"].map({"Yes": 1, "No": 0})
    df["Placed_flag"] = df["Placement_Status"].map({"Placed": 1, "Not Placed": 0})
    return df


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Return (placement_model, salary_model); a missing file gives None."""
    def _load(path: Path):
        return joblib.load(path) if path.exists() else None

    return _load(PLACEMENT_MODEL_PATH), _load(SALARY_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_connection() -> sqlite3.Connection:
    """In-memory, read-only SQLite copy of the cleaned data (table: students)."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    load_data().to_sql("students", conn, index=False, if_exists="replace")
    conn.execute("PRAGMA query_only = ON")
    return conn


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def placed_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Package_LPA"] > 0]


def rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Placement rate (%) and head-count per value of `col`."""
    out = (
        df.groupby(col)["Placed_flag"]
        .agg(Students="size", Placement_Rate="mean")
        .reset_index()
    )
    out["Placement_Rate"] = (out["Placement_Rate"] * 100).round(1)
    return out


def rate_bar(df: pd.DataFrame, col: str, title: str, horizontal: bool = False):
    data = rate_by(df, col)
    if horizontal:
        data = data.sort_values("Placement_Rate")
        fig = px.bar(
            data, x="Placement_Rate", y=col, orientation="h",
            text="Placement_Rate", hover_data=["Students"],
        )
    else:
        fig = px.bar(
            data, x=col, y="Placement_Rate", text="Placement_Rate",
            hover_data=["Students"],
        )
    fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
    fig.update_layout(title=title, yaxis_title=None, xaxis_title=None, margin=dict(t=50))
    if horizontal:
        fig.update_xaxes(range=[0, 105], title="Placement rate (%)")
    else:
        fig.update_yaxes(range=[0, 105], title="Placement rate (%)")
        fig.update_xaxes(type="category")
    return fig


def show(fig, height: int = 380) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch")


def skills_table(df: pd.DataFrame) -> pd.DataFrame:
    s = df.loc[df["Skills"] != "Unknown", ["Skills", "Placed_flag"]].copy()
    s["Skill"] = s["Skills"].str.split(", ")
    s = s.explode("Skill")
    t = s.groupby("Skill").agg(Students=("Placed_flag", "size"), Placement_Rate=("Placed_flag", "mean"))
    t["Placement_Rate"] = (t["Placement_Rate"] * 100).round(1)
    return t.reset_index().sort_values("Students", ascending=False)


def student_frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame([values], columns=FEATURES)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(df: pd.DataFrame):
    st.sidebar.title("🎓 CampusPulse")
    st.sidebar.caption("Placement analytics & prediction")
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

    filtered = df
    if page in ("Overview", "Insights"):
        st.sidebar.divider()
        st.sidebar.subheader("Filters")
        branches = st.sidebar.multiselect(
            "Branch", sorted(df["Branch"].unique()), default=sorted(df["Branch"].unique())
        )
        years = st.sidebar.multiselect(
            "Graduation year",
            sorted(df["Graduation_Year"].unique()),
            default=sorted(df["Graduation_Year"].unique()),
        )
        filtered = df[df["Branch"].isin(branches) & df["Graduation_Year"].isin(years)]
        st.sidebar.caption(f"{len(filtered):,} of {len(df):,} students selected")

    st.sidebar.divider()
    st.sidebar.caption("Built with Streamlit · scikit-learn · Plotly")
    return page, filtered


# --------------------------------------------------------------------------- #
# Page 1 - Overview
# --------------------------------------------------------------------------- #
def page_overview(df: pd.DataFrame) -> None:
    st.title("Placement Overview")
    placed = placed_only(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", f"{len(df):,}")
    c2.metric("Placement rate", f"{df['Placed_flag'].mean() * 100:.1f}%")
    c3.metric("Average package", f"₹{placed['Package_LPA'].mean():.2f} LPA")
    c4.metric("Median package", f"₹{placed['Package_LPA'].median():.2f} LPA")
    c5.metric("Highest package", f"₹{placed['Package_LPA'].max():.2f} LPA")

    left, right = st.columns(2)
    with left:
        show(rate_bar(df, "Branch", "Placement rate by branch", horizontal=True))
    with right:
        show(rate_bar(df, "Graduation_Year", "Placement rate by graduation year"))

    left, right = st.columns(2)
    with left:
        fig = px.histogram(placed, x="Package_LPA", nbins=25, title="Package distribution (placed students)")
        fig.update_layout(xaxis_title="Package (LPA)", yaxis_title="Students", margin=dict(t=50))
        show(fig)
    with right:
        avg = placed.groupby("Branch")["Package_LPA"].mean().round(2).sort_values().reset_index()
        fig = px.bar(avg, x="Package_LPA", y="Branch", orientation="h", text="Package_LPA",
                     title="Average package by branch")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="Average package (LPA)", yaxis_title=None, margin=dict(t=50))
        show(fig)


# --------------------------------------------------------------------------- #
# Page 2 - Insights
# --------------------------------------------------------------------------- #
def page_insights(df: pd.DataFrame) -> None:
    st.title("What Drives Placement?")
    tab_acad, tab_exp, tab_skill, tab_emp = st.tabs(
        ["📚 Academics & Scores", "🛠 Experience", "💡 Skills", "🏢 Employers"]
    )

    with tab_acad:
        left, right = st.columns(2)
        with left:
            fig = px.box(df, x="Placement_Status", y="CGPA", color="Placement_Status",
                         title="CGPA vs placement status")
            fig.update_layout(showlegend=False, xaxis_title=None, margin=dict(t=50))
            show(fig)
        with right:
            show(rate_bar(df, "Backlogs", "Placement rate by number of backlogs"))
        score_cols = ["DSA_Score", "Technical_Score", "Aptitude_Score", "Communication_Score"]
        means = df.groupby("Placement_Status")[score_cols].mean().round(1).T.reset_index()
        means = means.melt(id_vars="index", var_name="Status", value_name="Average score")
        fig = px.bar(means, x="index", y="Average score", color="Status", barmode="group",
                     title="Average test scores: placed vs not placed")
        fig.update_layout(xaxis_title=None, margin=dict(t=50))
        show(fig)

    with tab_exp:
        c1, c2, c3 = st.columns(3)
        with c1:
            show(rate_bar(df, "Internship", "Internship"), 340)
        with c2:
            show(rate_bar(df, "Projects", "Projects completed"), 340)
        with c3:
            show(rate_bar(df, "Certifications", "Certifications"), 340)

    with tab_skill:
        skills = skills_table(df)
        left, right = st.columns(2)
        with left:
            top = skills.head(15).sort_values("Students")
            fig = px.bar(top, x="Students", y="Skill", orientation="h", title="15 most common skills")
            fig.update_layout(yaxis_title=None, margin=dict(t=50))
            show(fig, 480)
        with right:
            rated = skills[skills["Students"] >= 50].sort_values("Placement_Rate").tail(15)
            fig = px.bar(rated, x="Placement_Rate", y="Skill", orientation="h",
                         text="Placement_Rate", hover_data=["Students"],
                         title="Placement rate of students who list each skill")
            fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
            fig.update_layout(yaxis_title=None, xaxis_title="Placement rate (%)", margin=dict(t=50))
            fig.update_xaxes(range=[0, 105])
            show(fig, 480)
        st.caption(
            "Skill placement rates mostly reflect branch: CSE/IT skills (Python, SQL, Java) "
            "appear on the branches with the highest placement rates. Use the Branch filter to compare fairly."
        )

    with tab_emp:
        placed = placed_only(df)
        left, right = st.columns(2)
        with left:
            top = placed["Company"].value_counts().head(12).sort_values().reset_index()
            top.columns = ["Company", "Students hired"]
            fig = px.bar(top, x="Students hired", y="Company", orientation="h", title="Top recruiters")
            fig.update_layout(yaxis_title=None, margin=dict(t=50))
            show(fig, 440)
        with right:
            roles = placed["Job_Role"].value_counts().head(12).sort_values().reset_index()
            roles.columns = ["Job role", "Students hired"]
            fig = px.bar(roles, x="Students hired", y="Job role", orientation="h", title="Most common job roles")
            fig.update_layout(yaxis_title=None, margin=dict(t=50))
            show(fig, 440)
        pay = (
            placed.groupby("Company")["Package_LPA"]
            .agg(Hires="size", Avg_Package_LPA="mean")
            .round(2)
            .sort_values("Avg_Package_LPA", ascending=False)
            .reset_index()
        )
        st.subheader("Average package by company")
        st.dataframe(pay, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Page 3 - Student predictor
# --------------------------------------------------------------------------- #
def gauge(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Placement probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f2937"},
                "steps": [
                    {"range": [0, 40], "color": "#f8b4b4"},
                    {"range": [40, 70], "color": "#fde68a"},
                    {"range": [70, 100], "color": "#a7f3d0"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def what_if(model, base: dict, base_prob: float) -> pd.DataFrame:
    """Change one factor at a time and report the shift in placement probability."""
    scenarios = {
        "Complete an internship": {"Internship_flag": 1},
        "Clear all backlogs": {"Backlogs": 0},
        "Complete 1 more project": {"Projects": base["Projects"] + 1},
        "Earn 1 more certification": {"Certifications": base["Certifications"] + 1},
        "Raise CGPA by 0.5": {"CGPA": min(10.0, base["CGPA"] + 0.5)},
        "Raise DSA score by 10": {"DSA_Score": min(100.0, base["DSA_Score"] + 10)},
        "Raise Technical score by 10": {"Technical_Score": min(100.0, base["Technical_Score"] + 10)},
        "Raise Aptitude score by 10": {"Aptitude_Score": min(100.0, base["Aptitude_Score"] + 10)},
        "Raise Communication score by 10": {"Communication_Score": min(100.0, base["Communication_Score"] + 10)},
    }
    rows = []
    for name, change in scenarios.items():
        if all(base[k] == v for k, v in change.items()):
            continue  # nothing would change
        new_prob = float(model.predict_proba(student_frame({**base, **change}))[0][1])
        rows.append({"Action": name, "New probability (%)": round(new_prob * 100, 1),
                     "Change (pp)": round((new_prob - base_prob) * 100, 1)})
    return pd.DataFrame(rows).sort_values("Change (pp)", ascending=False)


def page_predictor(df: pd.DataFrame) -> None:
    st.title("Student Predictor")
    st.caption("Enter a student profile to estimate placement chances and expected package.")

    placement_model, salary_model = load_models()
    if placement_model is None or salary_model is None:
        st.error(
            "Model files not found in `models/`. Run all cells of `notebooks/CampusPulse.ipynb` "
            "to create `placement_model.pkl` and `salary_model.pkl`."
        )
        return

    with st.form("student_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Academics**")
            cgpa = st.slider("CGPA", 5.0, 10.0, 8.0, 0.01)
            tenth = st.slider("10th percentage", 40.0, 100.0, 80.0, 0.1)
            twelfth = st.slider("12th percentage", 40.0, 100.0, 78.0, 0.1)
            backlogs = st.number_input("Active backlogs", 0, 10, 0)
        with c2:
            st.markdown("**Experience**")
            internship = st.radio("Internship completed?", ["Yes", "No"], horizontal=True, index=1)
            projects = st.number_input("Projects", 0, 10, 3)
            certs = st.number_input("Certifications", 0, 10, 1)
            branch = st.selectbox(
                "Branch (benchmark only, not a model input)", sorted(df["Branch"].unique())
            )
        with c3:
            st.markdown("**Test scores (0-100)**")
            dsa = st.slider("DSA score", 0.0, 100.0, 80.0, 0.1)
            tech = st.slider("Technical score", 0.0, 100.0, 82.0, 0.1)
            apt = st.slider("Aptitude score", 0.0, 100.0, 75.0, 0.1)
            comm = st.slider("Communication score", 0.0, 100.0, 78.0, 0.1)
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if not submitted:
        return

    base = {
        "CGPA": cgpa, "Tenth_Percentage": tenth, "Twelfth_Percentage": twelfth,
        "Backlogs": int(backlogs), "Internship_flag": 1 if internship == "Yes" else 0,
        "Projects": int(projects), "Certifications": int(certs),
        "DSA_Score": dsa, "Technical_Score": tech, "Aptitude_Score": apt, "Communication_Score": comm,
    }
    X = student_frame(base)
    prob = float(placement_model.predict_proba(X)[0][1])
    package = float(salary_model.predict(X)[0])

    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(gauge(prob), width="stretch")
        if prob >= 0.7:
            st.success("Strong profile: likely to be placed.")
        elif prob >= 0.4:
            st.warning("Borderline profile: targeted improvements could tip the balance.")
        else:
            st.error("At risk: focus on the highest-impact actions below.")
    with right:
        st.metric("Expected package if placed", f"₹{package:.2f} LPA")
        st.caption(f"Typical model error is about ±{SALARY_MAE} LPA (MAE). "
                   "Salary is hard to predict from these inputs, so treat it as a rough guide.")
        bdf = df[df["Branch"] == branch]
        st.metric(f"{branch} placement rate (all students)", f"{bdf['Placed_flag'].mean() * 100:.1f}%")
        st.metric(f"{branch} average package (placed)", f"₹{placed_only(bdf)['Package_LPA'].mean():.2f} LPA")

    st.subheader("What would improve this profile?")
    wi = what_if(placement_model, base, prob)
    if wi.empty:
        st.info("No single change would alter this profile further.")
    else:
        fig = px.bar(wi.sort_values("Change (pp)"), x="Change (pp)", y="Action", orientation="h",
                     text="Change (pp)", title="Change in placement probability (percentage points)")
        fig.update_traces(texttemplate="%{text:+.1f}", textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_title=None, margin=dict(t=50))
        show(fig, 380)
        st.dataframe(wi, width="stretch", hide_index=True)
    st.caption("Predictions come from a model trained on historical data. They show patterns, not guarantees.")


# --------------------------------------------------------------------------- #
# Page 4 - SQL explorer
# --------------------------------------------------------------------------- #
PRESET_QUERIES = {
    "Placement rate by branch": """SELECT Branch,
       COUNT(*) AS total_students,
       SUM(Placed_Flag) AS placed_students,
       ROUND(AVG(Placed_Flag) * 100, 2) AS placement_rate
FROM students
GROUP BY Branch
ORDER BY placement_rate DESC;""",
    "Average package by branch": """SELECT Branch,
       ROUND(AVG(CASE WHEN Package_LPA > 0 THEN Package_LPA END), 2) AS average_package
FROM students
GROUP BY Branch
ORDER BY average_package DESC;""",
    "Internship impact": """SELECT Internship,
       COUNT(*) AS total_students,
       SUM(Placed_Flag) AS placed_students,
       ROUND(AVG(Placed_Flag) * 100, 2) AS placement_rate
FROM students
GROUP BY Internship;""",
    "Top recruiters": """SELECT Company, COUNT(*) AS students_hired,
       ROUND(AVG(Package_LPA), 2) AS avg_package
FROM students
WHERE Placement_Status = 'Placed'
GROUP BY Company
ORDER BY students_hired DESC
LIMIT 15;""",
    "Most common job roles": """SELECT Job_Role, COUNT(*) AS students_hired
FROM students
WHERE Placement_Status = 'Placed'
GROUP BY Job_Role
ORDER BY students_hired DESC;""",
    "Top 20 packages": """SELECT Student_ID, Branch, CGPA, Company, Job_Role, Package_LPA
FROM students
ORDER BY Package_LPA DESC
LIMIT 20;""",
    "Unplaced students with strong scores": """SELECT Student_ID, Branch, CGPA, DSA_Score, Technical_Score, Backlogs
FROM students
WHERE Placement_Status = 'Not Placed' AND CGPA >= 8 AND Technical_Score >= 85
ORDER BY CGPA DESC
LIMIT 50;""",
}


def page_sql() -> None:
    st.title("SQL Explorer")
    st.caption("Run read-only SQL against the `students` table (SQLite).")

    conn = get_connection()
    preset = st.selectbox("Start from a preset query", list(PRESET_QUERIES))
    query = st.text_area("SQL query", PRESET_QUERIES[preset], height=190, key=f"sql_{preset}")

    with st.expander("Table schema"):
        schema = pd.read_sql_query("PRAGMA table_info(students)", conn)[["name", "type"]]
        st.dataframe(schema, width="stretch", hide_index=True)

    if st.button("Run query", type="primary"):
        stripped = query.strip().rstrip(";").strip()
        if not stripped.lower().startswith(("select", "with")):
            st.error("Only SELECT queries are allowed.")
            return
        try:
            result = pd.read_sql_query(stripped, conn)
        except Exception as exc:  # invalid SQL, unknown column, multiple statements...
            st.error(f"Query failed: {exc}")
            return
        st.success(f"{len(result):,} row(s) returned")
        st.dataframe(result.head(1000), width="stretch", hide_index=True)
        st.download_button("Download CSV", result.to_csv(index=False).encode(), "query_result.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Page 5 - Data & models
# --------------------------------------------------------------------------- #
def page_data_models(df: pd.DataFrame) -> None:
    st.title("Data & Models")

    st.subheader("Model performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Placement accuracy", f"{PLACEMENT_ACCURACY * 100:.1f}%", help="Random Forest classifier, 20% hold-out set")
    c2.metric("Salary MAE", f"{SALARY_MAE} LPA")
    c3.metric("Salary RMSE", f"{SALARY_RMSE} LPA")
    c4.metric("Salary R²", f"{SALARY_R2}")

    placement_model, _ = load_models()
    if placement_model is not None:
        imp = pd.Series(placement_model.feature_importances_, index=FEATURES).sort_values()
        fig = px.bar(imp, orientation="h", title="Placement model: feature importance")
        fig.update_layout(showlegend=False, xaxis_title="Importance", yaxis_title=None, margin=dict(t=50))
        show(fig, 420)

    with st.expander("Known limitations", expanded=True):
        st.markdown(
            "- The models do **not** use *Branch*, yet branch has the largest effect on placement rates "
            "(e.g. CSE ≈ 90% vs Civil ≈ 36%). Adding it is the most valuable next improvement.\n"
            "- The salary model explains only ~37% of package variance (R² = 0.37).\n"
            "- Results are based on one dataset and may not generalise to other colleges or years."
        )

    st.subheader("Dataset")
    st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns (after cleaning)")
    st.dataframe(df.head(200), width="stretch", hide_index=True)
    st.download_button("Download cleaned data (CSV)", df.to_csv(index=False).encode(),
                       "campuspulse_clean.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`. Place the CSV in the `data/` folder.")
        st.stop()

    df = load_data()
    page, filtered = sidebar(df)

    if page in ("Overview", "Insights") and filtered.empty:
        st.warning("No students match the current filters. Select at least one branch and year.")
        st.stop()

    if page == "Overview":
        page_overview(filtered)
    elif page == "Insights":
        page_insights(filtered)
    elif page == "Student Predictor":
        page_predictor(df)
    elif page == "SQL Explorer":
        page_sql()
    else:
        page_data_models(df)


main()
