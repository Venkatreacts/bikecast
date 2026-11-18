"""Workflow diagrams for the report (one per chapter/unit), drawn with matplotlib."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).parent / "build" / "figs"
FILL = {"in": "#dbe9f4", "prep": "#eef4f8", "model": "#fdebd3", "eval": "#e3f1e0", "out": "#2a6f97"}


def flow(stages: list[tuple[str, str, str]], out: str, title: str) -> Path:
    """stages: (kind, heading, detail) drawn top to bottom with arrows."""
    n = len(stages)
    fig, ax = plt.subplots(figsize=(7.2, 1.05 * n + 0.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n * 1.05 + 0.3)
    ax.axis("off")
    for i, (kind, head, detail) in enumerate(stages):
        y = (n - 1 - i) * 1.05 + 0.25
        ax.add_patch(FancyBboxPatch((0.15, y), 9.7, 0.78, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=FILL[kind], ec="#2a6f97", lw=1.2))
        color = "white" if kind == "out" else "#1b1b1b"
        ax.text(0.35, y + 0.52, head, fontsize=10.5, weight="bold", va="center", color=color, family="serif")
        ax.text(0.35, y + 0.22, detail, fontsize=7.6, va="center", color=color, family="serif")
        if i < n - 1:
            ax.annotate("", xy=(5, y - 0.02), xytext=(5, y - 0.25),
                        arrowprops=dict(arrowstyle="<|-", color="#2a6f97", lw=1.4))
    del title  # the report caption names the figure
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / out
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def build_all() -> dict[str, Path]:
    return {
        "pbl1": flow([
            ("in", "Input data", "UCI Bike Sharing: hour.csv (17,379 h) and day.csv (731 days), 2011-2012"),
            ("prep", "Quality checks and leakage removal", "0 missing, 0 duplicates; drop casual, registered (sum to cnt), instant, dteday"),
            ("prep", "Chronological 80/20 split", "train 1 Jan 2011 - 7 Aug 2012 11:00 (13,903 h); test to 31 Dec 2012 (3,476 h)"),
            ("model", "Unit 1: probability and statistics", "quantiles, densities, Bayes rule, expectation, covariance, independence, polynomial fit"),
            ("prep", "Feature design", "RAW (12 numbers) and ENCODED (77 incl. hour x working-day), train-only scaling"),
            ("model", "Unit 2: regression and classification", "OLS, Huber, ridge, Bayesian linear; LDA, Naive Bayes, logistic, Bayesian logistic (Laplace), SVM"),
            ("eval", "Evaluation on the test period", "RMSE, MAE, R2, interval coverage; accuracy, precision, recall, F1, ROC-AUC"),
            ("out", "Outputs", "results.json, figures, baseline models for PBL 2 and PBL 3"),
        ], "flow_pbl1.png", "Workflow of PBL 1 (Units 1 and 2)"),
        "u3": flow([
            ("in", "Input", "hour.csv pivoted to a day x 24-hour matrix of rentals (731 days)"),
            ("prep", "Shape profiles", "divide each day by its total; drop 3 near-empty days; standardise per hour"),
            ("model", "Clustering", "K-means (K = 2..8), GMM by EM from scratch, hierarchical (3 measures x 4 linkages)"),
            ("eval", "Cluster evaluation", "silhouette, Davies-Bouldin, Calinski-Harabasz, BIC; ARI vs working day afterwards"),
            ("prep", "Weather-time features", "11 hourly features incl. sin/cos of hour and month; scaler fitted on training rows"),
            ("model", "Dimensionality reduction", "PCA by eigen-decomposition (scree, 90 %, Kaiser); factor analysis (held-out likelihood)"),
            ("out", "Outputs", "two daily demand shapes, PCA map, latent dimensions used in the app's Demand Patterns page"),
        ], "flow_u3.png", "Workflow of Unit 3 (clustering and dimensionality reduction)"),
        "u4": flow([
            ("in", "Input sequence", "z_t = log(1 + cnt_t), 17,379 hours in time order (165 missing hours skipped)"),
            ("prep", "Split and context", "train first 13,903 h; test decoded with a 200-hour warm-up from training"),
            ("model", "Hidden Markov model", "Gaussian emissions; scaled forward-backward; Baum-Welch; Viterbi (from scratch)"),
            ("prep", "Profile removal", "hour x working-day mean profile m from training; residual r_t = z_t - m"),
            ("model", "Linear dynamical system", "l_t = phi l_(t-1) + w_t,  r_t = l_t + v_t;  phi, Q, R by maximum likelihood"),
            ("eval", "Forecast evaluation", "1 h and 24 h ahead vs naive baselines; 95 % interval coverage"),
            ("out", "Outputs", "decoded demand states; Kalman 24-hour forecast used in the app"),
        ], "flow_u4.png", "Workflow of Unit 4 (sequential models)"),
        "u5": flow([
            ("in", "Input", "12 raw features of hour.csv; target high demand = cnt > 130 (training median)"),
            ("prep", "Time-ordered validation", "5-fold TimeSeriesSplit inside the training period for every hyper-parameter"),
            ("model", "Single tree", "CART with Gini splits; depth by CV; cost-complexity pruning"),
            ("model", "Ensembles", "bagging, random forest, AdaBoost, gradient boosting, stacking, BMA"),
            ("model", "Generalised additive model", "Poisson GAM: splines of temperature, humidity, wind, and hour per day type"),
            ("eval", "Evaluation on the test period", "accuracy, precision, recall, F1, ROC-AUC, log-loss; MAE, RMSE, R2"),
            ("out", "Outputs", "stacking classifier and GAM saved for the app's Predict page"),
        ], "flow_u5.png", "Workflow of Unit 5 (trees, ensembles and GAM)"),
        "system": flow([
            ("in", "Data", "UCI hour.csv / day.csv"),
            ("model", "Notebooks", "PBL 1 (Units 1-2) and PBL 2/3 (Units 3-5), chronological protocol"),
            ("prep", "Training script", "re-runs the Units 3-5 notebook, checks 8 headline numbers, saves models"),
            ("eval", "Tests", "27 automated tests: notebook parity to 1e-9, calendar rules, every page runs"),
            ("out", "Streamlit app", "Predict, Forecast 24h, Demand Patterns, Leaderboard (bikecast.streamlit.app)"),
        ], "flow_system.png", "From notebooks to the deployed demo"),
    }


if __name__ == "__main__":
    for k, p in build_all().items():
        print(k, p)
