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

# Step 7 - extrapolation (not yet solved)
# TODO: implement

# Step 8 - smoothing_spline (not yet solved)
# TODO: implement

# Step 9 - local_smoother (not yet solved)
# TODO: implement

# Step 10 - gam_pipeline (not yet solved)
# TODO: implement

# Step 11 - partial_effects (not yet solved)
# TODO: implement

# Step 12 - logistic_gam (not yet solved)
# TODO: implement

# Step 13 - fit_age_models (not yet solved)
# TODO: implement

# Step 14 - test_comparison (not yet solved)
# TODO: implement

