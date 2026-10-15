"""REML random-intercept variance summaries of reported treatment means."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from scipy.optimize import minimize
from models import mean

def mean(values):
    values=list(values)
    return sum(values)/len(values)

def design_matrix(rows: list[dict[str, str]], conditional: bool):
    y = []
    groups = []
    features = []
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    mean_initial = {
        source: mean(float(row["initial_weight_g"]) for row in source_rows)
        for source, source_rows in by_source.items()
    }
    for row in rows:
        source = row["source_id"]
        w0 = float(row["initial_weight_g"])
        wf = float(row["final_weight_g"])
        y.append((wf - w0) / mean_initial[source])
        groups.append(source)
        if conditional:
            features.append(
                [
                    math.log(w0),
                    math.log(float(row["duration_d"])),
                    float(row["diet_protein_DM_pct"]),
                    float(row["diet_lipid_DM_pct"]),
                ]
            )
    if conditional:
        z = np.asarray(features, dtype=float)
        center = z.mean(axis=0)
        scale = z.std(axis=0, ddof=0)
        if np.any(scale <= 0):
            raise ValueError("Conditional ICC covariates must have non-zero variance")
        z = (z - center) / scale
        x = np.column_stack([np.ones(len(rows)), z])
        metadata = {
            "fixed_effects": "intercept + z(log initial weight) + z(log duration) + z(CP) + z(CL)",
            "covariate_center": center.tolist(),
            "covariate_scale": scale.tolist(),
        }
    else:
        x = np.ones((len(rows), 1), dtype=float)
        metadata = {"fixed_effects": "intercept only"}
    return np.asarray(y, dtype=float), x, np.asarray(groups), metadata

def reml_random_intercept(y: np.ndarray, x: np.ndarray, groups: np.ndarray):
    """Fit y = Xb + study intercept + residual by direct REML optimization."""
    n, p = x.shape
    zz = (groups[:, None] == groups[None, :]).astype(float)

    def objective(log_variances):
        tau2, sigma2 = np.exp(log_variances)
        v = tau2 * zz + sigma2 * np.eye(n)
        sign_v, logdet_v = np.linalg.slogdet(v)
        if sign_v <= 0:
            return np.inf
        vinv_x = np.linalg.solve(v, x)
        xt_vinv_x = x.T @ vinv_x
        sign_x, logdet_x = np.linalg.slogdet(xt_vinv_x)
        if sign_x <= 0:
            return np.inf
        vinv_y = np.linalg.solve(v, y)
        beta = np.linalg.solve(xt_vinv_x, x.T @ vinv_y)
        resid = y - x @ beta
        quadratic = float(resid.T @ np.linalg.solve(v, resid))
        return 0.5 * ((n - p) * math.log(2 * math.pi) + logdet_v + logdet_x + quadratic)

    variance = float(np.var(y, ddof=1))
    starts = [
        np.log([max(variance * fraction, 1e-10), max(variance * (1 - fraction), 1e-10)])
        for fraction in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    fits = [
        minimize(objective, start, method="L-BFGS-B", bounds=[(-24, 16), (-24, 16)])
        for start in starts
    ]
    fit = min(fits, key=lambda item: item.fun)
    tau2, sigma2 = np.exp(fit.x)
    icc = tau2 / (tau2 + sigma2)
    boundary = bool(tau2 < 1e-8 * max(sigma2, 1e-30))
    return {
        "tau2_between_study": float(tau2),
        "sigma2_within_study": float(sigma2),
        "ICC": float(icc),
        "reml_negative_log_likelihood": float(fit.fun),
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "between_study_variance_boundary": boundary,
    }

def heterogeneity(rows: list[dict[str, str]]):
    outputs = []
    for label, conditional in [("unconditional", False), ("conditional", True)]:
        y, x, groups, metadata = design_matrix(rows, conditional)
        fit = reml_random_intercept(y, x, groups)
        outputs.append(
            {
                "analysis": label,
                "reported_treatment_means": len(rows),
                "studies": len(set(groups)),
                **metadata,
                **fit,
                "response": "u_si = (Wf_si - W0_si) / mean_s(W0)",
                "estimator": "Gaussian random-intercept meta-regression fitted by REML",
                "interpretation_boundary": (
                    "Descriptive proportion of residual variation among reported treatment means "
                    "attributable to between-study differences; not a fish- or tank-level causal partition"
                ),
            }
        )
    return outputs
