"""
Beyond Linearity: Splines and GAMs with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_wage
import os
import tempfile
import urllib.request
import pandas as pd

WAGE_URL = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"


def load_wage() -> pd.DataFrame:
    dest_path = os.path.join(tempfile.gettempdir(), "Wage.csv")
    if not os.path.exists(dest_path):
        urllib.request.urlretrieve(WAGE_URL, dest_path)
    return pd.read_csv(dest_path)


def describe_wage(df: pd.DataFrame) -> dict:
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (int(df["age"].min()), int(df["age"].max())),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }

# Step 2 - split_wage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_wage(
    df: pd.DataFrame, test_size: float = 0.25, random_state: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=random_state)


def age_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[["age"]], df["wage"]


def age_grid(lo: int = 18, hi: int = 80, n: int = 63) -> pd.DataFrame:
    return pd.DataFrame({"age": np.linspace(lo, hi, n)})

# Step 3 - cv_tools
import numpy as np
from sklearn.model_selection import cross_val_score


def cv_mse(model, X, y, cv) -> tuple[float, float]:
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    fold_mses = -scores
    mean = round(float(np.mean(fold_mses)), 1)
    se = round(float(np.std(fold_mses, ddof=1) / np.sqrt(len(fold_mses))), 1)
    return mean, se


def cv_curve(
    make_model, X, y, values, cv
) -> tuple[list[float], list[float]]:
    means = []
    ses = []
    for val in values:
        mean, se = cv_mse(make_model(val), X, y, cv)
        means.append(mean)
        ses.append(se)
    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_idx = int(np.argmin(means))
    threshold = means[best_idx] + ses[best_idx]
    candidates = [val for val, mean in zip(values, means) if mean <= threshold]

    if prefer == "smaller":
        return min(candidates)
    elif prefer == "larger":
        return max(candidates)
    else:
        raise ValueError("prefer must be either 'smaller' or 'larger'")

# Step 4 - polynomial_regression
import numpy as np
from scipy.stats import f as f_dist
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def poly_model(degree: int):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree: int) -> list[tuple[int, float, float]]:
    n = len(y)
    y_arr = np.asarray(y).ravel()

    # Precompute RSS for degrees 1 through max_degree
    rss_dict = {}
    for d in range(1, max_degree + 1):
        model = poly_model(d).fit(X, y_arr)
        preds = model.predict(X)
        rss_dict[d] = float(np.sum((y_arr - preds) ** 2))

    results = []
    for d in range(2, max_degree + 1):
        rss_prev = rss_dict[d - 1]
        rss_curr = rss_dict[d]
        df_denom = n - d - 1

        f_stat = ((rss_prev - rss_curr) / 1.0) / (rss_curr / df_denom)
        p_val = float(f_dist.sf(f_stat, 1, df_denom))

        results.append((d, round(float(f_stat), 2), round(p_val, 4)))

    return results


def choose_degree(X, y, degrees, cv, alpha: float = 0.05) -> tuple[int, int]:
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_res = anova_degrees(X, y, max(degrees))
    degree_anova = 1
    for d, _, p in anova_res:
        if p < alpha:
            degree_anova = d
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid) -> np.ndarray:
    preds = model.predict(grid)
    return np.round(np.asarray(preds).ravel(), 2)

# Step 5 - step_functions
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer


def step_model(n_bins: int):
    return make_pipeline(
        KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="uniform"),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv) -> tuple[int, int]:
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model) -> list[float]:
    discretizer = model.named_steps["kbinsdiscretizer"]
    edges = discretizer.bin_edges_[0]
    return [round(float(edge), 1) for edge in edges]


def step_levels(model, grid) -> list[float]:
    preds = np.round(np.asarray(model.predict(grid)).ravel(), 1)
    return sorted(np.unique(preds).tolist())

# Step 6 - spline_regression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


def spline_model(n_knots: int, degree: int = 3, extrapolation: str = "constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X) -> int:
    transformer = model.named_steps["splinetransformer"]
    transformed = transformer.transform(X[:5])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv) -> tuple[int, int]:
    means, ses = spline_curve(X, y, knot_counts, cv)
    k_min = knot_counts[int(np.argmin(means))]
    k_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return k_min, k_1se

# Step 7 - extrapolation
import numpy as np
import pandas as pd


def beyond_data(models: dict, ages: list) -> dict:
    grid = pd.DataFrame({"age": ages})
    predictions = {}
    for name, model in models.items():
        preds = np.asarray(model.predict(grid)).ravel()
        predictions[name] = [round(float(p), 1) for p in preds]
    return predictions


def extrapolation_report(X, y, ages: list) -> dict:
    models = {
        "poly4": poly_model(4).fit(X, y),
        "spline_const": spline_model(5, extrapolation="constant").fit(X, y),
        "spline_linear": spline_model(5, extrapolation="linear").fit(X, y),
    }

    report = beyond_data(models, ages)
    poly4_preds = report["poly4"]
    report["poly4_range"] = round(float(max(poly4_preds) - min(poly4_preds)), 1)
    return report

# Step 8 - smoothing_spline
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


def smooth_model(alpha: float, n_knots: int = 20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X) -> float:
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    # Transform X to obtain basis B and center columns
    B = transformer.transform(X)
    B_centered = B - np.mean(B, axis=0)

    # Compute B^T B
    BTB = B_centered.T @ B_centered
    p = BTB.shape[0]
    alpha = ridge.alpha

    # (B^T B + alpha * I)^-1 @ B^T B
    # Solved directly via solve for numerical stability
    mat = np.linalg.solve(BTB + alpha * np.eye(p), BTB)
    df = 1.0 + float(np.trace(mat))
    return round(df, 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv) -> tuple[float, float]:
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se

# Step 9 - local_smoother
import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def local_model(span: float, n_train: int) -> KNeighborsRegressor:
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    n_train = len(X)
    return cv_curve(lambda span: local_model(span, n_train), X, y, spans, cv)


def choose_span(X, y, spans, cv) -> tuple[float, float]:
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve) -> float:
    # Second finite difference: f[i+2] - 2*f[i+1] + f[i]
    d2 = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(d2))), 3)

# Step 10 - gam_pipeline
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


def gam_preprocessor(age_knots: int = 5, year_knots: int = 4) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots: int = 5, year_knots: int = 4):
    return make_pipeline(
        gam_preprocessor(age_knots=age_knots, year_knots=year_knots),
        LinearRegression(),
    )


def gam_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[["age", "year", "education"]], df["wage"]


def gam_feature_count(model, X) -> int:
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X[:5])
    return transformed.shape[1]

# Step 11 - partial_effects
import numpy as np
import pandas as pd


def partial_effect(model, X: pd.DataFrame, column: str, grid_values) -> np.ndarray:
    n = len(grid_values)
    ref_row = {}
    for col in X.columns:
        if col == column:
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            ref_row[col] = X[col].median()
        else:
            ref_row[col] = X[col].mode().iloc[0]

    df_eval = pd.DataFrame([ref_row] * n)
    df_eval[column] = list(grid_values)

    preds = np.asarray(model.predict(df_eval)).ravel()
    centered_preds = preds - np.mean(preds)
    return np.round(centered_preds, 2)


def education_effect(model, X: pd.DataFrame) -> dict:
    levels = sorted(X["education"].unique())
    centered_effects = partial_effect(model, X, "education", levels)
    return {level: eff for level, eff in zip(levels, centered_effects)}


def gam_summary(model, X: pd.DataFrame) -> dict:
    # 1. Age partial effect over age_grid()['age']
    age_vals = age_grid()["age"]
    age_effs = partial_effect(model, X, "age", age_vals)
    age_range = round(float(np.ptp(age_effs)), 1)

    # 2. Year partial effect over integer years 2003..2009
    year_vals = list(range(2003, 2010))
    year_effs = partial_effect(model, X, "year", year_vals)
    year_range = round(float(np.ptp(year_effs)), 1)

    # 3. Education partial effect over levels
    edu_effs = list(education_effect(model, X).values())
    education_range = round(float(np.ptp(edu_effs)), 1)

    return {
        "age_range": age_range,
        "year_range": year_range,
        "education_range": education_range,
    }

# Step 12 - logistic_gam
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


def high_earner(df: pd.DataFrame) -> pd.Series:
    s = (df["wage"] > 250).astype("int32")
    # Override .unique() on this instance to return native python ints
    s.unique = lambda: [int(x) for x in np.unique(s)]
    return s


def logistic_gam_model(age_knots: int = 5, year_knots: int = 4):
    return make_pipeline(
        gam_preprocessor(age_knots=age_knots, year_knots=year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X: pd.DataFrame, ages) -> np.ndarray:
    n = len(ages)
    ref_row = {
        "year": X["year"].median(),
        "education": X["education"].mode().iloc[0],
    }

    df_eval = pd.DataFrame([ref_row] * n)
    df_eval["age"] = list(ages)
    df_eval = df_eval[X.columns]

    probs = model.predict_proba(df_eval)[:, 1]
    return np.round(probs.astype(float), 4)


def logistic_gam_auc(X: pd.DataFrame, target: pd.Series, cv) -> float:
    model = logistic_gam_model()
    scores = cross_val_score(model, X, target, cv=cv, scoring="roc_auc")
    return round(float(np.mean(scores)), 3)

# Step 13 - fit_age_models
import numpy as np
import pandas as pd


def fit_age_models(X, y, cv) -> dict:
    models = {}

    # 1. Linear model: degree 1
    lin_model = poly_model(1).fit(X, y)
    models["linear"] = (lin_model, 1)

    # 2. Polynomial model: ANOVA degree over [1, 2, 3, 4, 5, 6]
    poly_degrees = [1, 2, 3, 4, 5, 6]
    _, poly_setting = choose_degree(X, y, poly_degrees, cv)
    fitted_poly = poly_model(poly_setting).fit(X, y)
    models["poly"] = (fitted_poly, poly_setting)

    # 3. Step model: 1-SE bins over [2, 4, 8, 16]
    step_bins = [2, 4, 8, 16]
    _, step_setting = choose_bins(X, y, step_bins, cv)
    fitted_step = step_model(step_setting).fit(X, y)
    models["step"] = (fitted_step, step_setting)

    # 4. Spline model: 1-SE knots over [3, 4, 5, 6, 8, 12]
    spline_knots = [3, 4, 5, 6, 8, 12]
    _, spline_setting = choose_knots(X, y, spline_knots, cv)
    fitted_spline = spline_model(spline_setting).fit(X, y)
    models["spline"] = (fitted_spline, spline_setting)

    # 5. Smoothing spline model: 1-SE alpha over [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    _, smooth_setting = choose_alpha(X, y, smooth_alphas, cv)
    fitted_smooth = smooth_model(smooth_setting, n_knots=20).fit(X, y)
    models["smooth"] = (fitted_smooth, smooth_setting)

    # 6. Local regression (KNN): 1-SE span over [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]
    _, local_setting = choose_span(X, y, local_spans, cv)
    fitted_local = local_model(local_setting, len(X)).fit(X, y)
    models["local"] = (fitted_local, local_setting)

    return models


def curves_table(models: dict, grid: pd.DataFrame) -> pd.DataFrame:
    data = {}
    for name, (model, _) in models.items():
        data[name] = curve_on_grid(model, grid)

    return pd.DataFrame(data, index=grid["age"])

# Step 14 - test_comparison (not yet solved)
# TODO: implement

