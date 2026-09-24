# 🎓 CampusPulse

**Campus placement analytics and prediction dashboard.**
CampusPulse turns a 5,000-student placement dataset into an interactive Streamlit app. It shows what drives placement, predicts a student's chances and expected package, and lets you query the data with SQL.

## Features

| Page | What it does |
|------|--------------|
| **Overview** | KPIs (placement rate, average / median / highest package), placement by branch and year, package distribution. Filter by branch and graduation year. |
| **Insights** | Impact of CGPA, backlogs, test scores, internships, projects, certifications, skills, top recruiters and job roles. |
| **Student Predictor** | Enter a profile to get a placement probability, an expected package, and a *what-if* ranking of the actions that would help most. |
| **SQL Explorer** | Run read-only `SELECT` queries on the `students` table, with preset queries and CSV download. |
| **Data & Models** | Model metrics, feature importance, known limitations, dataset preview and download. |

## Project structure

```
CampusPulse/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml         # Theme and server settings
├── data/
│   └── college_placement_analytics_5000.csv
├── models/
│   ├── placement_model.pkl # Random Forest classifier
│   └── salary_model.pkl    # Random Forest regressor
└── notebooks/
    └── CampusPulse.ipynb   # EDA, SQL analysis and model training
```

## Quick start

Requires **Python 3.11+**.

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at http://localhost:8501.

## Dataset

`data/college_placement_analytics_5000.csv` has 5,000 students and 20 columns:

- **Profile:** `Student_ID`, `Gender`, `Graduation_Year` (2024-2027), `Branch` (CSE, IT, ECE, EEE, Mechanical, Civil)
- **Academics:** `CGPA`, `Tenth_Percentage`, `Twelfth_Percentage`, `Backlogs`
- **Experience:** `Internship`, `Projects`, `Certifications`, `Skills`
- **Test scores:** `DSA_Score`, `Technical_Score`, `Aptitude_Score`, `Communication_Score`
- **Outcome:** `Placement_Status`, `Company`, `Job_Role`, `Package_LPA`

Cleaning (same in the notebook and the app): duplicates dropped, missing 10th/12th percentages filled with the median, missing skills set to `Unknown`, and `Internship_flag` / `Placed_flag` created as 0/1 columns.

### Headline findings

- Overall placement rate is **70%**; the average package of placed students is **₹6.33 LPA** (median ₹6.35, highest ₹10.70).
- Branch matters most: CSE **89.6%** and IT **83.3%** versus Mechanical **46.1%** and Civil **35.7%**.
- Internships lift placement from **58.5% to 86.2%**.
- Placement rises steadily with projects (31% with none, 92% with four) and falls with backlogs (83% with none, 35% with four).

## Models

Both models use 11 features: CGPA, 10th %, 12th %, backlogs, internship, projects, certifications, and the four test scores.

| Model | Algorithm | Test result |
|-------|-----------|-------------|
| Placement (classification) | Random Forest, 300 trees, depth 10 | **79.7%** accuracy |
| Package (regression, placed students only) | Random Forest, 300 trees, depth 12 | MAE 0.83 LPA · RMSE 1.04 · R² 0.37 |

A Logistic Regression baseline scored 79.9%, so the models are about equally accurate.

### Retraining

Open `notebooks/CampusPulse.ipynb` and run all cells. It reads `../data/`, and writes the models to `../models/` and a SQLite copy of the data to `../data/campuspulse.db`. You will need the optional packages listed in `requirements.txt` (matplotlib, seaborn, jupyterlab). If you retrain with a different scikit-learn version, update the pin in `requirements.txt` to match.

## Limitations

- **Branch is not a model input**, even though it has the largest effect on placement rates. Adding it is the best next improvement.
- The salary model explains only about 37% of package variation; treat its output as a rough guide.
- Predictions describe patterns in this dataset, not guarantees for any individual student.

## Deployment (Streamlit Community Cloud)

1. Push the project to GitHub. The two model files total about 49 MB, which is under GitHub's 100 MB limit but large. For a smaller repo, retrain with fewer trees or lower `max_depth`.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing to `app.py`.
3. Choose Python 3.11 or newer in the advanced settings.

## Roadmap

- [ ] Add `Branch` (and one-hot encoded skills) to the models
- [ ] Try gradient boosting and report cross-validated scores
- [ ] Export a student's prediction report as PDF
- [ ] Add authentication for real placement-cell data
