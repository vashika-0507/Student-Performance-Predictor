"""
Student Performance Predictor
Predicts a student's math score from background factors and other test scores.
Dataset: "Students Performance in Exams" (Kaggle) -> save as StudentsPerformance.csv in this folder.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # lets charts save to files without a window
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

os.makedirs("charts", exist_ok=True)

# ---------- 1. Load and check the data ----------
df = pd.read_csv("StudentsPerformance.csv")
print("Rows, columns:", df.shape)
print("Missing values:", df.isnull().sum().sum(), "| Duplicate rows:", df.duplicated().sum())
print(df.describe())

# tidy column names: "race/ethnicity" -> "race_ethnicity"
df.columns = [c.replace("/", "_").replace(" ", "_") for c in df.columns]

# ---------- 2. Explore with charts ----------
sns.histplot(df["math_score"], bins=20, kde=True)
plt.title("Distribution of math scores"); plt.savefig("charts/math_distribution.png"); plt.clf()

sns.boxplot(data=df, x="test_preparation_course", y="math_score")
plt.title("Math score by test preparation course"); plt.savefig("charts/prep_course.png"); plt.clf()

sns.boxplot(data=df, x="lunch", y="math_score")
plt.title("Math score by lunch type"); plt.savefig("charts/lunch.png"); plt.clf()

sns.heatmap(df[["math_score", "reading_score", "writing_score"]].corr(), annot=True, cmap="Blues")
plt.title("Correlation between scores"); plt.savefig("charts/correlation.png"); plt.clf()

# ---------- 3. Train and compare models ----------
def run_experiment(features, label):
    X, y = df[features], df["math_score"]
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    # text columns -> 0/1 columns; numeric columns pass through unchanged
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                             remainder="passthrough")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    results, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("prep", prep), ("model", model)]).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            "Model": name,
            "R2": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "CV R2 (5-fold)": cross_val_score(pipe, X, y, cv=5, scoring="r2").mean(),
        })
        fitted[name] = pipe
    print(f"\n=== {label} ===")
    print(pd.DataFrame(results).round(3).to_string(index=False))
    return fitted

background = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
everything = background + ["reading_score", "writing_score"]

models_all = run_experiment(everything, "Experiment A: background + reading and writing scores")
models_bg = run_experiment(background, "Experiment B: background factors only")




lr = models_bg["Linear Regression"]
names = lr.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(lr.named_steps["model"].coef_, index=names).sort_values()
coefs.plot(kind="barh", figsize=(9, 7))
plt.axvline(0, color="black")
plt.title("Effect on math score in points (background factors only)")
plt.tight_layout(); plt.savefig("charts/linear_coefficients.png")
print(coefs.round(2))

# ---------- 5. Save the model for the web app (Project 3) ----------
joblib.dump(models_bg["Linear Regression"], "model_background.pkl")
joblib.dump(models_all["Linear Regression"], "model_full.pkl")
print("\nSaved both models")