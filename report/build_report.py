"""Builds the integrated ML project report (Units 1-5) on the faculty template.

Every number is read from reference/results.json (Units 1-2) and reference/results_units345.json (Units 3-5),
which the notebooks write; nothing is typed in by hand except values those notebooks print but do not save
(noted in comments). Run:  python report/build_report.py   then  report/word_export.ps1  for fields + PDF.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from docbuilder import Report  # noqa: E402
import diagrams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "pbl1_archive" / "figures"
BFIG = ROOT / "report" / "build" / "figs"
OUT = Path(os.environ.get("REPORT_OUT", ROOT / "report" / "Team8_ML_Project_Report.docx"))
REPO = "https://github.com/Venkatreacts/bikecast"
APP = "https://bikecast.streamlit.app"
NB1 = "Team8_Bike_Sharing_ML_PBL1_Final.ipynb"
NB2 = "Team8_Bike_Sharing_ML_PBL2_Units3-5.ipynb"
COLAB = "https://colab.research.google.com/github/Venkatreacts/bikecast/blob/main/notebooks/"

FACULTY = "Dr. Mohammad Ishtiyaq Qureshi"

MEMBERS = [
    ("Venkatanathan O R V", "RA2411026011163"),
    ("Ali Hasan Ansari", "RA2411026011160"),
    ("Fayas K N", "RA2411026011161"),
    ("Konanki Revanth Chowdary", "RA2411026011162"),
    ("Dhruva Mishra", "RA2411026011164"),
]

P1 = json.loads((ROOT / "reference" / "results.json").read_text(encoding="utf8"))
U = json.loads((ROOT / "reference" / "results_units345.json").read_text(encoding="utf8"))
R = P1["RES"]


def f(x, d=3):
    return f"{x:,.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def reg_row(model, target):
    return next(r for r in P1["regression"] if r["Model"] == model and r["Features"] == "encoded" and r["Target"] == target)


def cls_row(model, feats):
    return next(r for r in P1["classification"] if r["Model"] == model and r["Features"] == feats)


def u5(model_prefix):
    return next(r for r in U["cls"] if r["Model"].startswith(model_prefix))


def fc(model_prefix):
    return next(r for r in U["forecast"] if r["Model"].startswith(model_prefix))


rbf = cls_row("SVM RBF kernel", "encoded")
blog = cls_row("Bayesian logistic (Laplace)", "encoded")
stack, gb, rf = u5("Stacking"), u5("Gradient Boosting"), u5("Random Forest")
kal1, kal24 = fc("Kalman LDS, 1 h"), fc("Kalman LDS, 24 h")
pers, snw = fc("Persistence"), fc("Seasonal naive, same hour last week")
hmm1 = fc("HMM")
gam = U["gam"]["table"][0]
km, gmm, hmm, kal, pca = U["kmeans"], U["gmm"], U["hmm"], U["kalman"], U["pca"]
cm = P1["confusion"]


def abstract(rp: Report):
    rp.prelim("ABSTRACT")
    rp.p(
        "Bike-sharing operators must rebalance bikes between stations before docks run empty or full, which requires "
        "forecasting demand. This project studies hourly demand of the "
        "Capital Bikeshare system (Washington D.C.) using the public UCI Bike Sharing Dataset, 17,379 hourly records from "
        "2011 and 2012. The objectives were to understand demand with probability and statistics, to predict the hourly "
        "rental count and a high-demand label, to discover structure without labels, to model demand as a sequence, and to "
        "improve accuracy with tree ensembles and additive models. The same dataset served all five units; its hourly time "
        "series was the sequential data for Unit 4. Every model was trained on the first 80% of the timeline and tested on the "
        "last 20%. Probability analysis showed that working day and hour interact strongly. Bayesian linear regression on "
        f"encoded features with a log target reached R² {f(R['best_reg_r2'])} and an RBF support vector machine classified "
        f"{pct(R['rbf_enc_acc'])} of test hours correctly. K-means, Gaussian mixtures and hierarchical clustering all found "
        f"two daily demand shapes, commute and leisure, matching the working-day flag (adjusted Rand index "
        f"{f(km['ARI_workingday'], 2)}), and principal component analysis kept 90% of variance with {pca['M90']} components. "
        f"A Kalman filter forecast the next hour with mean absolute error {f(kal1['MAE'], 1)} rentals against "
        f"{f(pers['MAE'], 1)} for repeating the last hour. Stacked tree ensembles reached ROC-AUC {f(stack['ROC-AUC'])}, "
        f"and a Poisson generalised additive model predicted counts with R² {f(gam['R²'])} while remaining interpretable. "
        "A public web application demonstrates the models, showing that classical machine learning gives accurate, "
        "explainable demand forecasts for smart urban mobility.")


def front_matter(rp: Report):
    abstract(rp)
    rp.prelim("TABLE OF CONTENTS")
    rp.toc()
    rp.prelim("LIST OF FIGURES")
    rp.list_of("RptFigCaption")
    rp.prelim("LIST OF TABLES")
    rp.list_of("RptTabCaption")
    rp.prelim("ABBREVIATIONS")
    abbr = [
        ("ARI", "Adjusted Rand Index"), ("AUC", "Area Under the ROC Curve"), ("BIC", "Bayesian Information Criterion"),
        ("BMA", "Bayesian Model Averaging"), ("CART", "Classification and Regression Tree"),
        ("CV", "Cross-Validation"), ("DB", "Davies–Bouldin index"), ("EDA", "Exploratory Data Analysis"),
        ("EM", "Expectation–Maximisation"), ("FA", "Factor Analysis"), ("GAM", "Generalised Additive Model"),
        ("GB", "Gradient Boosting"), ("GMM", "Gaussian Mixture Model"), ("HMM", "Hidden Markov Model"),
        ("LDA", "Linear Discriminant Analysis"), ("LDS", "Linear Dynamical System"),
        ("MAE", "Mean Absolute Error"), ("MAP", "Maximum A Posteriori"), ("MLE", "Maximum Likelihood Estimation"),
        ("OLS", "Ordinary Least Squares"), ("OOB", "Out-Of-Bag"), ("OOF", "Out-Of-Fold"),
        ("PCA", "Principal Component Analysis"), ("R²", "Coefficient of determination"),
        ("RBF", "Radial Basis Function"), ("RF", "Random Forest"), ("RMSE", "Root Mean Squared Error"),
        ("ROC", "Receiver Operating Characteristic"), ("SVM", "Support Vector Machine"),
        ("UCI", "University of California, Irvine (Machine Learning Repository)"),
    ]
    rp.table(["Abbreviation", "Expansion"], abbr, None, widths=[1.4, 4.37], align_numbers=False)


# ------------------------------------------------------------------ chapter 1
def chapter1(rp: Report):
    rp.chapter("Introduction")
    rp.h2("1.1 Background, Motivation and Scope")
    rp.p("Public bike-sharing systems let a rider unlock a bike at one docking station and return it at any other. They "
         "reduce short car trips, congestion and emissions, and are part of most smart-city mobility plans. Their main "
         "operating problem is imbalance: in the morning commute bikes leave residential areas and fill the docks near "
         "offices, and in the evening the flow reverses. Operators rebalance the network with trucks, plan maintenance in "
         "quiet hours and decide where to add stations. All three tasks depend on knowing how many bikes will be rented "
         "in a given hour and why.")
    rp.p("The stakeholders are the operator (rebalancing crews, maintenance planners), the city transport department "
         "(station and lane planning) and riders (availability of bikes and free docks). The scope of this project is the "
         "hourly demand of one system, Capital Bikeshare in Washington D.C., for the two years 2011–2012, analysed with "
         "the algorithms allotted for Units 1 to 5 of 21CSC305P Machine Learning. The models are packaged in a public "
         f"web demonstration ({APP}).")
    rp.h2("1.2 Integrated Problem Statement")
    rp.p("Given the calendar and weather conditions of an hour, (i) estimate the number of bikes that will be rented "
         "(regression) and (ii) decide whether the hour will be a high-demand hour (classification, more than 130 rentals, "
         "the median of the training period). Around these two prediction tasks the units ask linked questions on the same "
         "data: which factors drive demand and how they interact (Unit 1); which linear, Bayesian and kernel models predict "
         "best (Unit 2); whether days fall into natural demand shapes and how many latent dimensions describe the weather and "
         "time features (Unit 3); how demand evolves hour by hour as a sequence and how far ahead it can be forecast "
         "(Unit 4); and how much tree ensembles and additive models add (Unit 5). Unit 4 uses the hourly time series of the "
         "same dataset as its sequential data, so no separate dataset was introduced.")
    rp.h2("1.3 Project Objectives")
    rp.bullets([
        "**Objective 1 (Unit 1):** describe hourly demand with summary statistics, densities, Bayes rule, expectation, "
        "covariance and (conditional) independence tests, and fit polynomial curves of daily rentals against temperature.",
        "**Objective 2 (Unit 2):** predict the count with least squares, robust, ridge and Bayesian linear regression, and "
        "the high-demand label with LDA, Naive Bayes, logistic and Bayesian logistic regression (Laplace) and SVMs.",
        "**Objective 3 (Unit 3):** cluster daily demand profiles with K-means, Gaussian mixtures (EM) and hierarchical "
        "clustering, and reduce the weather and time features with PCA and factor analysis.",
        "**Objective 4 (Unit 4):** model the hourly sequence with a hidden Markov model (forward–backward, Baum–Welch, "
        "Viterbi) and a linear dynamical system with a Kalman filter for 1-hour and 24-hour forecasts.",
        "**Objective 5 (Unit 5):** classify high demand with CART, bagging, random forest, AdaBoost, gradient boosting, "
        "stacking and Bayesian model averaging, and model the count with a generalised additive model.",
    ])
    rp.h2("1.4 Unit-wise Topic and Dataset Allocation")
    rp.table(["PBL", "Unit", "Allotted topic / algorithm", "Dataset"], [
        ("PBL 1", "1", "Load and view data; quantiles, densities, Bayes rule, expectation, covariance, independence, "
                       "polynomial curve fitting", "Original (UCI 275)"),
        ("PBL 1", "2", "Least squares, robust, ridge and Bayesian linear regression; discriminant functions, generative and "
                       "discriminative models, Laplace approximation, Bayesian logistic regression, kernel SVM",
         "Original (UCI 275)"),
        ("PBL 2", "3", "K-means, mixtures of Gaussians and EM, hierarchical clustering, dissimilarity measures, silhouette and "
                       "Davies–Bouldin index; PCA, choosing latent dimensions, factor analysis", "Original (UCI 275)"),
        ("PBL 2", "4", "Markov models, HMM, forward–backward, scaling, Baum–Welch, Viterbi; linear dynamical system and "
                       "Kalman filter", "Hourly sequence of the original dataset"),
        ("PBL 3", "5", "CART, bagging, random forest, boosting (AdaBoost, gradient boosting), stacking, Bayesian model "
                       "averaging, adaptive basis-function models, GAM", "Original (UCI 275)"),
    ], "Unit-wise allotted topics and datasets (Team 8)", widths=[0.6, 0.45, 3.2, 1.52], align_numbers=False)
    rp.p("All units share one evaluation protocol so their results can be compared: the hourly records are sorted by time, "
         "the first 80% (13,903 hours, 1 January 2011 to 7 August 2012 11:00) are used for training and model selection, and "
         "the last 20% (3,476 hours, to 31 December 2012) are used only once for testing. The high-demand threshold, scalers "
         "and every hyper-parameter are fitted on training data only, and the random seed is fixed at 42.")


# ------------------------------------------------------------------ chapter 2
def chapter2(rp: Report):
    rp.chapter("Data Analysis and Fundamental Machine Learning Techniques (Units 1 and 2)")
    rp.h2("2.1 Overview of Allotted Topics")
    rp.p("Unit 1 treats the hourly count as a random variable. Quantiles, mean, variance and a density estimate describe "
         "its distribution; Bayes rule links weather and demand; expectation, covariance and correlation summarise how "
         "season, year and weather move the average; chi-square tests check independence and conditional independence "
         "given the hour; and polynomial curve fitting shows model complexity on daily rentals against temperature. These "
         "tools [3], [4] fit the problem because demand is a noisy count whose distribution, not just its mean, matters to an operator.")
    rp.p("Unit 2 turns the analysis into predictors. Regression (least squares as maximum likelihood, Huber robust "
         "regression, ridge and Bayesian linear regression) estimates the count, and the Bayesian model also gives a "
         "predictive interval. Classification compares generative models (LDA, Gaussian Naive Bayes), discriminative "
         "models (logistic regression), a Bayesian logistic regression with the Laplace approximation, and maximum-margin "
         "SVMs with a linear and an RBF kernel. Their assumptions differ (Gaussian classes, linear log-odds, kernel "
         "similarity), so comparing them shows which assumption suits the data [3], [5].")
    rp.h2("2.2 Dataset Description")
    rp.p("The data is the UCI Machine Learning Repository Bike Sharing Dataset (id 275, H. Fanaee-T, 2013), built from Capital Bikeshare trip logs with weather "
         "and holiday information [1], [2]. hour.csv has 17,379 rows and 17 columns; day.csv has 731 rows and 16 columns. Together "
         "they record 3,292,679 rentals, 81% of them by registered members. There are no missing values and no duplicate "
         "rows or (date, hour) keys, and day.csv equals the sum of the hourly data on all 731 days. 165 clock-hours have no "
         "row at all (no rental logged) and are absent rather than zero. Table 2.1 lists the columns.")
    rp.table(["Column", "Type", "Meaning", "Use"], [
        ("instant", "integer", "Row number 1…N", "Dropped (time proxy)"),
        ("dteday", "date", "Calendar date", "Sorting and split only"),
        ("season", "category 1–4", "Winter, spring, summer, autumn", "Feature"),
        ("yr", "binary", "0 = 2011, 1 = 2012", "Feature"),
        ("mnth, hr, weekday", "integer", "Month, hour 0–23, day of week", "Features"),
        ("holiday, workingday", "binary", "Public holiday; neither weekend nor holiday", "Features"),
        ("weathersit", "category 1–4", "Clear … heavy rain or snow", "Feature"),
        ("temp, atemp", "float", "Temperature, feels-like temperature (normalised)", "Features"),
        ("hum, windspeed", "float", "Humidity, wind speed (normalised)", "Features"),
        ("casual, registered", "integer", "Rentals by casual and registered users", "Dropped (leakage)"),
        ("cnt", "integer", "Total rentals in the hour", "Regression target"),
        ("high_demand", "binary", "cnt > 130 (training median)", "Classification target"),
    ], "Columns of hour.csv and their use", widths=[1.35, 0.95, 2.17, 1.3], align_numbers=False)
    rp.p("Limitations: the data covers one city and two years, the second year is much busier than the first, and extreme "
         "weather (weathersit = 4) occurs in only three hours, so predictions for storms rest on very little evidence.")
    rp.h2("2.3 Data Preprocessing")
    rp.p("**Leakage removal.** casual + registered equals cnt on every row, so both columns are parts of the target and were "
         "dropped. instant is only the row number, a proxy for time that cannot be extrapolated, and was dropped. dteday was "
         "used to sort and split and then dropped, because yr, mnth, weekday and hr carry its content.")
    rp.p("**Chronological split.** Consecutive hours are near copies of each other and demand grows over time, so a random "
         "split would let the model see the future and overstate accuracy. The first 80% of rows form the training set and "
         "the last 20% the test set (Figure 2.2). The training median of cnt is 130, which defines the high-demand label; "
         "50.0% of training hours and 62.5% of test hours are high-demand hours.")
    rp.p(f"**Feature sets.** The RAW set keeps the {R['n_raw']} predictors as numbers. The ENCODED set has {R['n_enc']} "
         "columns: one-hot season, month, weekday and weather, an hour × working-day interaction (one column per hour for each "
         "day type), and the scaled weather variables. Standard scaling is fitted on the training rows only. The essential "
         "code is:")
    rp.code(
        "hour = hour.sort_values(['dteday', 'hr']).reset_index(drop=True)\n"
        "X = hour.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])\n"
        "n_train = int(0.8 * len(hour))                 # 13,903 hours\n"
        "X_tr, X_te = X.iloc[:n_train], X.iloc[n_train:]\n"
        "y_tr, y_te = hour.cnt.iloc[:n_train], hour.cnt.iloc[n_train:]\n"
        "threshold = y_tr.median()                      # 130 rentals\n"
        "scaler = StandardScaler().fit(X_tr[num_cols])  # fitted on training rows only")
    rp.h2("2.4 Exploratory Data Analysis (EDA)")
    rp.fig(FIG / "fig00_target_overview.png", "Distribution of hourly rentals (left) and daily rentals over time (right)")
    rp.p("Figure 2.1 shows a right-skewed count: most hours have few rentals and a long tail reaches the maximum of 977. "
         "Daily rentals rise every summer and are clearly higher in 2012 than in 2011, so the year is an important feature.")
    rp.fig(FIG / "fig01_split.png", "Chronological train/test split of daily rentals")
    rp.p("The test period (shaded, August to December 2012) lies entirely after the training period, so every reported test "
         "score is a genuine forecast of unseen future data.")
    rp.table(["Statistic", "Value", "Statistic", "Value"], [
        ("Mean", f(R["mean"], 1), "Median", f(R["median"], 0)),
        ("Variance", f(R["var"], 0), "Standard deviation", f(R["std"], 1)),
        ("Q1 / Q3", f"{R['Q1']:.0f} / {R['Q3']:.0f}", "99th percentile", f(R["p99"], 0)),
        ("Maximum", f(R["max"], 0), "Skewness", f(R["skew"], 2)),
        ("Variance / mean", f(R["vmr"], 0), "High-demand threshold", f(R["threshold"], 0)),
    ], "Summary statistics of the hourly count cnt (all 17,379 hours)", widths=[1.6, 1.28, 1.6, 1.29])
    rp.p("The variance is about 174 times the mean, whereas a Poisson count would have a ratio of 1, so the counts are "
         "strongly over-dispersed and their spread grows with their level. This motivates the log target used in Unit 2.")
    rp.fig(FIG / "fig02_cnt_hist.png", "Histogram of cnt (left) and of log(1 + cnt) (right)")
    rp.fig(FIG / "fig03_density.png", "Probability density of hourly rentals: histogram, kernel estimate, Gaussian and Poisson",
           width=4.6)
    rp.p("Figures 2.3 and 2.4 show that neither a Gaussian nor a Poisson density fits the data: a Gaussian with the same mean "
         "and standard deviation would put 14.8% of its mass below zero, and the Poisson is far too narrow. After the "
         "log(1 + x) transform the distribution is much less skewed.")
    rp.h3("2.4.1 Bayes rule, expectation and independence")
    rp.p(f"Let H be a high-demand hour and C clear weather. From counts, P(C) = {f(R['P_C'])}, P(H) = {f(R['P_H'])} and "
         f"P(C | H) = {f(R['P_C_H'])}. Bayes rule gives")
    rp.eq(f"P(H | C) = P(C | H) P(H) / P(C) = {f(R['P_C_H'])} × {f(R['P_H'])} / {f(R['P_C'])} = {f(R['P_H_C'])}")
    rp.p(f"which equals the direct empirical frequency. Under light rain or snow the probability falls to "
         f"{f(R['P_H_by_weather']['3'])}. The conditional expectation E[cnt | season] rises from {f(R['E_season']['1'], 1)} in "
         f"winter to {f(R['E_season']['3'], 1)} in summer, E[cnt | yr] is {f(R['E_yr']['0'], 1)} in 2011 and "
         f"{f(R['E_yr']['1'], 1)} in 2012, and the law of total expectation")
    rp.eq("E[cnt] = Σ_{s} E[cnt | S = s] P(S = s)")
    rp.p(f"returns E[cnt] = {f(R['mean'], 2)} for each grouping. A chi-square test rejects independence of the high-demand "
         f"label and the working-day flag (χ² = {f(R['chi2_wd'], 1)}, p < 10^{{-18}}), yet the pooled association is weak "
         f"(Cramér's V = {f(R['V_wd'])}).")
    rp.fig(FIG / "fig04_conditional_dependence.png",
           "Mean rentals by hour for working and non-working days, difference in P(high), and Cramér's V per hour")
    rp.p(f"Conditioning on the hour reveals the real structure (Figure 2.5). The working-day effect is significant in "
         f"{R['n_sig_hours']} of 24 hours (all {R['n_bonf_hours']} survive a Bonferroni correction) and changes sign: at "
         f"08:00 working days average {f(R['mean08_w'], 0)} rentals against {f(R['mean08_nw'], 0)} on non-working days "
         f"(V = {f(R['V08'], 2)}), while at 13:00 the order reverses ({f(R['mean13_w'], 0)} against {f(R['mean13_nw'], 0)}). "
         "The weak pooled association hides two strong opposite effects, so models need an hour × working-day interaction.")
    rp.fig(FIG / "fig05_cov_corr.png", "Covariance (left) and correlation (right) of cnt and the weather variables")
    rp.p(f"Temperature has the strongest linear relation with demand (correlation {f(R['corr_temp'], 2)}), humidity a "
         f"negative one ({f(R['corr_hum'], 2)}) and wind speed almost none ({f(R['corr_windspeed'], 2)}). temp and atemp are "
         f"almost the same variable (correlation {f(R['corr_temp_atemp'], 3)}), which matters for linear models and for PCA "
         "in Unit 3. Covariances depend on units, so only correlations are compared.")
    rp.h3("2.4.2 Polynomial curve fitting")
    rp.fig(FIG / "fig06_polynomial.png",
           "Polynomial fits of daily rentals against temperature: random split, chronological split and fitted curves")
    rp.p(f"Polynomials of degree 1 to 9 were fitted to daily rentals against temperature (in °C). On a random 70/30 split, "
         f"degree {R['poly_best']} gives the lowest held-out RMSE ({f(R['poly_rmse'], 0)} rentals per day, held-out "
         f"R² {f(R['poly_r2_test'])}) and places peak demand at {f(R['poly_peak'], 1)} °C; training error keeps falling with "
         "degree while held-out error rises again, the classic over-fitting curve. Over 200 random splits degree 3 has the "
         f"lowest median held-out RMSE, although degree 9 is best on {R['poly_wins9']} splits, so the choice is not stable. "
         "On a chronological split every degree has a negative held-out R² because 2012 is busier than 2011 at every "
         "temperature; temperature alone cannot explain the growth.")
    rp.h2("2.5 Proposed System Architecture / Workflow")
    rp.fig(BFIG / "flow_pbl1.png", "Workflow of PBL 1 (Units 1 and 2)", width=3.9)
    rp.p("Figure 2.8 shows the stages. The raw files are checked for quality and leakage, sorted and split by time. Unit 1 "
         "analyses the training-period distribution and dependences. The features are then built in two versions (RAW and "
         "ENCODED) with training-only scaling. Unit 2 fits the regression and classification models, selects "
         "hyper-parameters by time-series cross-validation inside the training period, and evaluates once on the test period. "
         "The outputs are the result tables and figures, and the baseline models against which Units 3–5 are compared.")
    rp.h2("2.6 Algorithm Theory and Mathematical Formulation")
    rp.p("Notation: x is the feature vector of an hour, X the training design matrix, y the target (cnt or log(1 + cnt)), "
         "w the weights, and t ∈ {0, 1} the high-demand label.")
    rp.h3("2.6.1 Regression")
    rp.p("**Least squares** is the maximum-likelihood estimate under Gaussian noise y = wᵀx + ε, ε ~ N(0, σ²):")
    rp.eq("ŵ_{OLS} = (XᵀX)^{-1} Xᵀy")
    rp.p("**Ridge regression** adds a penalty α‖w‖² that shrinks correlated weights; α is chosen by time-series CV:")
    rp.eq("ŵ_{ridge} = (XᵀX + αI)^{-1} Xᵀy")
    rp.p("**Huber regression** replaces the squared loss by a loss that is quadratic for residuals |r| ≤ ε and linear "
         "beyond, so single extreme hours have less influence (ε = 1.35):")
    rp.eq("L_{ε}(r) = ½ r² if |r| ≤ ε,   ε(|r| − ½ ε) otherwise")
    rp.p("**Bayesian linear regression** places a prior w ~ N(0, α^{-1}I) and noise precision β, both set by maximising "
         "the evidence. The posterior and the predictive distribution are Gaussian:")
    rp.eq("S_{N} = (αI + β XᵀX)^{-1},   m_{N} = β S_{N} Xᵀy")
    rp.eq("p(y′ | x′) = N(m_{N}ᵀx′, 1/β + x′ᵀ S_{N} x′)")
    rp.p("The predictive variance gives a 95% interval for every forecast hour, whose coverage is checked on the test set.")
    rp.h3("2.6.2 Classification")
    rp.p("**LDA** models each class as a Gaussian with a shared covariance Σ, giving the linear discriminant function")
    rp.eq("δ_{k}(x) = xᵀΣ^{-1}μ_{k} − ½ μ_{k}ᵀΣ^{-1}μ_{k} + ln π_{k}")
    rp.p("**Gaussian Naive Bayes** assumes the features are independent given the class. **Logistic regression** models "
         "the log-odds directly, p(t = 1 | x) = σ(wᵀx) with σ(a) = 1/(1 + e^{-a}), and is fitted by penalised maximum "
         "likelihood.")
    rp.p("**Bayesian logistic regression** (implemented from scratch in NumPy) uses a prior w ~ N(0, α^{-1}I). The "
         "posterior is not Gaussian, so the Laplace approximation replaces it by a Gaussian at the MAP estimate with the "
         "Hessian as precision, and the predictive probability is moderated by the uncertainty s² of the activation:")
    rp.eq("q(w) = N(w_{MAP}, H^{-1}),   H = αI + Σ_{n} σ_{n}(1 − σ_{n}) x_{n}x_{n}ᵀ")
    rp.eq("p(t = 1 | x) ≈ σ(κ(s²) μ_{a}),   κ(s²) = (1 + πs²/8)^{-1/2},   s² = xᵀH^{-1}x")
    rp.p("**Support vector machines** find the maximum-margin separating hyperplane; the kernel trick replaces inner "
         "products by a kernel so that the boundary can be non-linear. The RBF kernel used is")
    rp.eq("k(x, x′) = exp(−γ ‖x − x′‖²)")
    rp.p("The fixed settings (SVM C = 10, Laplace prior precision α = 1, Huber ε = 1.35) were chosen before seeing test data.")
    rp.h2("2.7 Results and Performance Evaluation")
    rp.h3("2.7.1 Regression")
    rows = []
    for r in P1["regression"]:
        rows.append((r["Model"].replace(" / least squares", ""), r["Features"], r["Target"], f(r["RMSE"], 1),
                     f(r["MAE"], 1), f(r["R2"])))
    rp.table(["Model", "Features", "Target", "RMSE", "MAE", "R²"], rows,
             "Regression results on the test period (errors in rentals per hour)", widths=[1.75, 0.8, 0.95, 0.75, 0.75, 0.77])
    best = reg_row("Bayesian linear (evidence)", "log1p(cnt)")
    rp.p(f"Table 2.3 shows that the inputs matter more than the algorithm. Least squares on the RAW features reaches only "
         f"R² {f(R['ols_raw_r2'])}; the ENCODED features with the hour × working-day interaction raise it to "
         f"{f(R['ols_enc_r2'])}, and the log target to {f(R['ols_enc_log_r2'])} while removing the "
         f"{f(R['ols_neg_cnt'], 1)}% negative forecasts. With 13,903 rows for {R['n_enc']} columns, ridge, Huber and Bayesian "
         f"regression are within a few rentals of each other. The best model is Bayesian linear regression with a log target "
         f"(RMSE {f(best['RMSE'], 1)}, MAE {f(best['MAE'], 2)}, R² {f(best['R2'])}). Huber is slightly worse because extreme "
         "hours are real demand, not outliers.")
    rp.fig(FIG / "fig07_ols_scatter.png", "Predicted against actual rentals: RAW, ENCODED, and ENCODED with log target")
    rp.fig(FIG / "fig08_ridge_cv.png", "Ridge regression: time-series CV error against the penalty α", width=4.3)
    rp.p(f"The CV curve (Figure 2.10) is flat over many orders of magnitude and selects α = {f(R['alpha_cnt'], 0)} for the "
         f"count and α = {f(R['alpha_log'], 0)} for the log target; regularisation matters little with this much data.")
    rp.fig(FIG / "fig09_blr_intervals.png", "Bayesian linear regression: 95% predictive intervals for one test week")
    rp.fig(FIG / "fig10_blr_coverage_by_hour.png", "Empirical coverage of the 95% interval by hour of day", width=4.6)
    cov = P1["coverage"]
    rp.table(["Target", "Coverage (all)", "Coverage (peak 07, 08, 17, 18 h)", "Coverage (off-peak)", "Mean width"], [
        (c["Target"], pct(c["coverage overall"]), pct(c["coverage peak hours (07,08,17,18)"]), pct(c["coverage off-peak"]),
         f(c["mean interval width (rentals)"], 0)) for c in cov],
        "Coverage of the nominal 95% predictive interval on the test period", widths=[1.1, 1.0, 1.47, 1.1, 1.1])
    rp.p("With the count target the noise variance is the same for every hour, so the interval is too wide at night and "
         f"far too narrow at the commute peaks (coverage {pct(R['cov_cnt_peak'])}). The log target makes the noise "
         f"multiplicative and brings coverage close to nominal everywhere ({pct(R['cov_log_all'])} overall, "
         f"{pct(R['cov_log_peak'])} at peaks), at the cost of wider intervals at busy hours (Figures 2.11 and 2.12).")
    rp.fig(FIG / "fig11_regression_comparison.png", "One test week with forecasts (left) and test R² of all regression models")
    rp.h3("2.7.2 Classification")
    rows = [(r["Model"], r["Features"], f(r["Accuracy"]), f(r["Precision"]), f(r["Recall"]), f(r["F1"]), f(r["ROC-AUC"]))
            for r in P1["classification"]]
    rp.table(["Model", "Features", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], rows,
             "Classification of high-demand hours on the test period",
             widths=[1.65, 0.72, 0.68, 0.7, 0.62, 0.62, 0.78])
    rp.p(f"Every model gains from the ENCODED features. The RBF SVM on encoded features is best on accuracy "
         f"({f(rbf['Accuracy'])}), F1 ({f(rbf['F1'])}) and ROC-AUC ({f(rbf['ROC-AUC'])}); logistic regression, its Bayesian "
         f"version and the linear SVM follow within 0.01. Gaussian Naive Bayes is weakest on encoded features (AUC "
         f"{f(R['nb_auc'])}) because the one-hot columns are strongly dependent, which breaks its independence assumption.")
    rp.fig(FIG / "fig13_confusion.png", "Confusion matrix of the RBF SVM (encoded): counts and row-normalised")
    rp.p(f"Of the 1,302 low-demand test hours, {cm[0][0]:,} are correctly called low and {cm[0][1]} wrongly called high; of the "
         f"2,174 high-demand hours, {cm[1][1]:,} are correctly called high and {cm[1][0]} missed. Errors concentrate near the "
         f"threshold: accuracy is {pct(R['acc_far'])} when demand is more than 40 rentals from 130 and "
         f"{pct(R['acc_near'])} within ±40.")
    rp.fig(FIG / "fig14_roc.png", "ROC curves of all classifiers: encoded features (left) and raw features (right)")
    rp.fig(FIG / "fig12_laplace.png", "Bayesian logistic regression: Laplace predictive probabilities and moderation")
    rp.p(f"The Laplace predictive probability is pulled slightly toward 0.5 relative to the MAP plug-in (mean change "
         f"{f(R['laplace_dp_mean'], 4)}, maximum {f(R['laplace_dp_max'], 3)}), so accuracy is unchanged "
         f"({f(R['laplace_acc'])}), but the probabilities are useful: {pct(R['conf_share'])} of test hours are predicted with "
         f"p < 0.1 or p > 0.9 and only {pct(R['conf_err'])} of them are wrong, while the {pct(R['unsure_share'])} of hours "
         f"with 0.4 ≤ p ≤ 0.6 have an error rate of {pct(R['unsure_err'], 0)}.")
    rp.h2("2.8 Discussion and Findings")
    rp.bullets([
        "Demand is driven by the hour, the working-day flag, season, year and temperature, and the hour and working-day "
        "effects interact; a model without that interaction loses most of its accuracy.",
        "Feature design (encoding and the log target) improved R² from 0.31 to 0.89, far more than the choice between "
        "least squares, ridge, Huber and Bayesian regression.",
        f"The largest errors occur in the evening peak (MAE {f(R['mae_evening'], 0)} rentals at 16–19 h against "
        f"{f(R['mae_night'], 1)} at night), exactly where operators need buffer stock.",
        "Gaussian-noise models ignore the discrete, over-dispersed nature of counts, and no lagged demand is used, so "
        "sudden weather changes within a day are not captured. Unit 4 addresses the time dependence and Unit 5 the "
        "count distribution (Poisson GAM).",
        "The test period (August–December 2012) has more high-demand hours (62.5%) than training (50%), so test accuracy "
        "partly reflects this seasonal slice.",
    ])
    rp.h2("2.9 Chapter Conclusion")
    rp.p(f"PBL 1 established a leakage-free, chronological protocol and two baselines used by every later unit: Bayesian "
         f"linear regression with a log target for the count (MAE {f(best['MAE'], 2)}, R² {f(best['R2'])}) and the RBF SVM "
         f"for the high-demand label (accuracy {f(rbf['Accuracy'])}, ROC-AUC {f(rbf['ROC-AUC'])}). The analysis also "
         "identified the two questions left open: whether days fall into distinct demand shapes, and how much the "
         "time order of the data can improve forecasts.")


# ------------------------------------------------------------------ chapter 3
def chapter3(rp: Report):
    rp.chapter("Unsupervised Learning and Sequential Data Modelling (Units 3 and 4)")
    rp.h2("3.1 Overview of Allotted Unit 3 and Unit 4 Topics")
    rp.p("Unit 3 is unsupervised: K-means, mixtures of Gaussians fitted by EM, hierarchical clustering with different "
         "dissimilarity measures, and cluster validation with the silhouette and Davies–Bouldin indices (Practice 1); and "
         "principal component analysis, the choice of the number of latent dimensions, and factor analysis (Practice 2). "
         "Unit 4 is sequential: Markov models and hidden Markov models with the forward–backward algorithm, scaling factors, "
         "Baum–Welch and Viterbi (Practice 1), and a linear dynamical system with a Kalman filter used for forecasting the "
         "hourly count.")
    rp.h2("3.2 Unit 3 – Dataset Description and Preprocessing")
    rp.p("Unit 3 uses the dataset and split of Chapter 2; only the additional preprocessing is described here.")
    rp.p(f"**Daily demand profiles (clustering).** hour.csv is pivoted into a day × 24-hour matrix. Each day is divided by "
         f"its total, so clustering compares the shape of the day and not its size, and each hour column is standardised. "
         f"Three near-empty days (fewer than 500 rentals) were excluded because their shape is sampling noise: 27 January "
         f"2011 ({U['excluded_days']['2011-01-27']} rentals), 29 October 2012, Hurricane Sandy "
         f"({U['excluded_days']['2012-10-29']}), and 26 December 2012 ({U['excluded_days']['2012-12-26']}). In an earlier run "
         "they formed a meaningless one-day cluster. 728 days remain. The working-day, season and weekday labels are not "
         "used for clustering, only afterwards to interpret clusters.")
    rp.p("**Weather–time features (PCA and factor analysis).** Eleven hourly features are used: temp, atemp, hum, "
         "windspeed, sine and cosine of the hour and of the month (so 23:00 is close to 00:00), workingday, holiday and "
         "weathersit. The scaler is fitted on training rows only.")
    rp.h2("3.3 Unit 3 – Theory, EDA, Implementation and Results")
    rp.fig(FIG / "u3_fig01_profiles.png",
           "Mean normalised daily profile for working and non-working days (left) and a sample of daily profiles (right)")
    rp.p("Figure 3.1 is the additional EDA for Unit 3: the individual profiles visibly fall into a double-peaked commute "
         "shape and a single afternoon hump, which motivates clustering on shapes. The implementation details follow in "
         "Section 3.4 (theory), Section 3.5 (workflow) and Section 3.6 (results).")
    rp.h2("3.4 Unit 3 – Algorithm Theory and Mathematical Formulation")
    rp.p("**K-means** [3] partitions N points x_{n} into K clusters with centres μ_{k} by minimising the distortion")
    rp.eq("J = Σ_{n} Σ_{k} r_{nk} ‖x_{n} − μ_{k}‖²")
    rp.p("alternating assignment to the nearest centre and recomputing centres as cluster means (20 restarts, K = 2 to 8).")
    rp.p("**Gaussian mixture model** (EM [7] implemented from scratch) models the density as")
    rp.eq("p(x) = Σ_{k} π_{k} N(x | μ_{k}, Σ_{k})")
    rp.p("The E-step computes responsibilities γ_{nk} = π_{k} N(x_{n} | μ_{k}, Σ_{k}) / Σ_{j} π_{j} N(x_{n} | μ_{j}, Σ_{j}); "
         "the M-step re-estimates π, μ and Σ from them. The log-likelihood never decreases. Models are compared with")
    rp.eq("BIC = −2 ln L + p ln N")
    rp.p("where p is the number of parameters. The GMM is fitted on the first 7 principal components of the profiles "
         "(90% of their variance) with full covariances, because a 24-dimensional full covariance per cluster would be "
         "poorly estimated from 728 days.")
    rp.p("**Hierarchical clustering** merges the closest clusters bottom-up. Three dissimilarities (Euclidean, cosine and "
         "correlation distance) were combined with four linkages (single, complete, average, Ward). The cophenetic "
         "correlation measures how faithfully the tree preserves the original distances.")
    rp.p("**Validation.** The silhouette [8] of point i compares its mean distance to its own cluster a(i) with that to the "
         "nearest other cluster b(i); the Davies–Bouldin index [9] averages the worst ratio of within-cluster spread to "
         "between-centre distance (lower is better):")
    rp.eq("s(i) = (b(i) − a(i)) / max(a(i), b(i))")
    rp.eq("DB = (1/K) Σ_{k} max_{j≠k} (s_{k} + s_{j}) / d(μ_{k}, μ_{j})")
    rp.p("The adjusted Rand index (ARI) compares a clustering with an external label (1 = identical, 0 = chance).")
    rp.p("**PCA** [10] (implemented by eigen-decomposition) diagonalises the training covariance S = (1/N) XᵀX of the centred, "
         "scaled features: S u_{i} = λ_{i} u_{i}. The first M components keep the fraction Σ_{i≤M} λ_{i} / Σ_{i} λ_{i} of the "
         "variance; M is chosen by the 90% rule, the Kaiser rule (λ > 1) and the scree plot.")
    rp.p("**Factor analysis** is a latent-variable model with a diagonal noise covariance Ψ, so each feature has its own "
         "unique variance; the number of factors q is chosen by the log-likelihood on held-out test rows:")
    rp.eq("x = Wz + μ + ε,   z ~ N(0, I_{q}),   ε ~ N(0, Ψ)")
    rp.h2("3.5 Unit 3 Proposed System Architecture / Workflow")
    rp.fig(BFIG / "flow_u3.png", "Workflow of Unit 3 (clustering and dimensionality reduction)", width=3.9)
    rp.p("The day × hour matrix is turned into standardised shape profiles; three clustering families are run and "
         "validated with internal indices; external labels are used only after clustering. In parallel, the hourly "
         "weather–time features are scaled on training rows and reduced with PCA and factor analysis. The outputs (two "
         "daily demand shapes and a PCA map of days) are shown in the web application's Demand Patterns page.")
    rp.h2("3.6 Unit 3 – Results and Performance Evaluation")
    rp.h3("3.6.1 K-means")
    rp.table(["K", "Distortion J", "Silhouette", "Davies–Bouldin", "Calinski–Harabasz", "ARI vs working day"],
             [(r["K"], f(r["inertia J"], 0), f(r["silhouette"]), f(r["Davies-Bouldin"]), f(r["Calinski-Harabasz"], 0),
               f(r["ARI vs workingday"])) for r in U["kmeans_table"]],
             "K-means model selection on daily profiles (728 days)", widths=[0.4, 1.0, 0.85, 1.05, 1.27, 1.2])
    rp.fig(FIG / "u3_fig02_kmeans_selection.png", "K-means: distortion (elbow), silhouette and Davies–Bouldin against K")
    rp.p(f"Silhouette and Davies–Bouldin both choose K = 2 (silhouette {f(km['sil'])}, DB {f(km['DB'])}); at K = 3 the "
         "silhouette halves to 0.239, and the distortion curve has no other elbow. Calinski–Harabasz also peaks at K = 2.")
    rp.fig(FIG / "u3_fig03_kmeans_profiles.png", "Mean demand profile of the two K-means clusters", width=4.6)
    rp.p(f"The clusters ({km['sizes'][0]} and {km['sizes'][1]} days) are the leisure shape, with one broad peak at "
         f"{km['peak_hours'][0]}:00, and the commute shape, with peaks at 08:00 and {km['peak_hours'][1]}:00. They agree "
         f"with the working-day flag with ARI {f(km['ARI_workingday'])}; only {U['kmeans_mismatch']['n']} days disagree, "
         f"{U['kmeans_mismatch']['holidays']} of them official holidays whose shape differs from their calendar type.")
    rp.fig(FIG / "u3_fig07_silhouette.png", "Silhouette of every day for K-means with K = 2", width=4.3)
    rp.p("Almost every day has a positive silhouette (Figure 3.5); the commute cluster is tighter than the leisure cluster, "
         "whose days vary more with weather.")
    rp.h3("3.6.2 Gaussian mixture and EM")
    rp.table(["K", "Log-likelihood (own EM)", "Parameters", "BIC", "EM iterations"],
             [(r["K"], f(r["logL (own EM)"], 1), r["params"], f(r["BIC"], 1), r["EM iterations"])
              for r in gmm["BIC_table"]],
             "Gaussian mixture models fitted by our EM (first 7 principal components)", widths=[0.5, 1.6, 1.05, 1.3, 1.32])
    rp.fig(FIG / "u3_fig04_gmm.png", "EM log-likelihood gain per iteration (left) and BIC against K (right)")
    rp.p(f"The EM log-likelihood never decreases, and at K = 2 it matches scikit-learn's GaussianMixture "
         f"({f(gmm['own_ll_K'], 1)} against {f(gmm['sk_ll_K'], 1)}). BIC alone prefers K = {gmm['K_BIC']}: with full "
         "covariances, extra components describe within-shape variation. Because the distance-based indices clearly prefer "
         f"K = 2, we report both and use K = 2 for interpretation. At K = 2 the GMM agrees with K-means (ARI "
         f"{f(gmm['ARI_vs_kmeans'])}), and only {gmm['uncertain_days']} day has a maximum responsibility below 0.9.")
    rp.h3("3.6.3 Hierarchical clustering and dissimilarity measures")
    rp.table(["Dissimilarity", "Linkage", "Cophenetic corr.", "Silhouette", "ARI vs working day", "Cluster sizes"],
             [(r["dissimilarity"], r["linkage"], f(r["cophenetic corr"]), f(r["silhouette (cut at K)"]),
               f(r["ARI vs workingday"]), r["cluster sizes"]) for r in U["hier_table"]],
             "Hierarchical clustering: 3 dissimilarities × 4 linkages, cut at K = 2",
             widths=[1.05, 0.8, 0.95, 0.8, 1.05, 1.12])
    rp.p("Single linkage, and Euclidean complete or average linkage, chain the data and isolate one to three unusual days, "
         "which gives a high silhouette but useless clusters; the silhouette alone can therefore mislead. Ward linkage and "
         "the cosine and correlation dissimilarities, which compare shapes, recover the two demand shapes (ARI 0.85–0.95).")
    rp.fig(FIG / "u3_fig05_dendrogram.png", "Ward dendrogram (last 30 merges); a horizontal cut gives K = 2")
    rp.fig(FIG / "u3_fig06_methods_profiles.png", "Mean cluster profiles found by K-means, GMM and Ward clustering")
    rp.h3("3.6.4 Cluster evaluation")
    rp.table(["Method", "K", "Silhouette", "DB", "ARI working day", "ARI season", "ARI weekday"],
             [(r["Method"], r["K"], f(r["Silhouette"]), f(r["Davies-Bouldin"]), f(r["ARI vs workingday"]),
               f(r["ARI vs season"]), f(r["ARI vs weekday"])) for r in U["cluster_eval"]],
             "Internal and external evaluation of the three clustering methods",
             widths=[1.7, 0.35, 0.75, 0.6, 0.85, 0.72, 0.8])
    rp.p("The three methods agree with each other (pairwise ARI 0.93–0.97) and with the working-day flag, but not with "
         "the season (ARI ≈ 0): the season changes the height of the daily curve, not its shape.")
    rp.h3("3.6.5 PCA and factor analysis")
    ev = pca["eigenvalues"]
    rp.table(["Component"] + [str(i) for i in range(1, 12)],
             [["Eigenvalue"] + [f(x, 2) for x in ev], ["Cumulative"] + [f(x, 2) for x in pca["cum"]]],
             "PCA of the 11 scaled weather–time features: eigenvalues and cumulative variance",
             widths=[0.95] + [0.438] * 11)
    rp.fig(FIG / "u3_fig08_pca.png", "PCA: scree plot, cumulative explained variance and loadings of the first three components")
    rp.p(f"Our eigen-decomposition reproduces scikit-learn's PCA [6]. Eight components keep 90% of the variance and the Kaiser "
         f"rule keeps {pca['Kaiser']}; there is no sharp elbow, so the 90% rule (M = {pca['M90']}) is used. PC1 is "
         "temperature and season, PC2 humidity and time of day, and PC3 working day against holiday. A linear model of "
         f"log(1 + cnt) on the {pca['M90']} components reaches test R² {f(pca['r2_pc'])} against {f(pca['r2_all'])} on all "
         f"11 features, so almost nothing predictive is lost; two components alone give only {f(pca['r2_2pc'])}.")
    rp.fig(FIG / "u3_fig09_pca_map.png", "Hours projected on PC1 and PC2, coloured by demand (left) and by hour of day (right)")
    rp.table(["Latent dimension q", "Training avg. log-likelihood", "Test avg. log-likelihood"],
             [(r["latent dim q"], f(r["train avg logL"], 4), f(r["test avg logL"], 4)) for r in U["fa"]["table"]],
             "Factor analysis: choosing the number of factors by held-out likelihood", widths=[1.6, 2.1, 2.07])
    rp.p(f"The held-out log-likelihood is highest at q = {U['fa']['q_best']} factors and falls slightly beyond, so six "
         "factors are kept; temp and atemp load on the same factor, as their correlation of 0.99 suggests.")
    rp.h2("3.7 Unit 4 – Sequential Dataset Description")
    rp.p("Unit 4 uses the hourly sequence of the same UCI dataset. The timestamp is dteday + hr, the sampling interval is "
         "one hour, and there is a single sequence (one city system) of 17,379 observations from 1 January 2011 00:00 to "
         "31 December 2012 23:00. The observed variable is z_{t} = ln(1 + cnt_{t}); the hidden variables are a discrete "
         "demand state (HMM) or a continuous demand level (LDS). 165 clock-hours are missing (no rentals logged, often "
         "overnight or during storms); they are skipped, which is a small approximation because the models then treat "
         "the hours on either side of a gap as adjacent. The calendar and weather columns are not used by the sequential "
         "models except the hour and working-day profile of the Kalman model.")
    rp.h2("3.8 Unit 4 – Sequential Preprocessing and EDA")
    rp.p("Temporal order is preserved throughout. The first 13,903 hours train the models; the test hours are decoded or "
         "filtered with a 200-hour warm-up taken from the end of the training period, so the test sequence starts from a "
         "realistic state. For the Kalman model, a profile m(hour, working day) of mean z_{t} is computed on training hours "
         f"only; it explains {pct(kal['profile_var_explained'])} of the training variance, and the residual "
         "r_{t} = z_{t} − m is modelled as a slowly varying level. Figure 3.12 (first ten days) and the daily curves in "
         "Chapter 2 show the strong 24-hour cycle that dominates the series.")
    rp.h2("3.9 Unit 4 – Mathematical Formulation")
    rp.p("**Hidden Markov model** [11]. A hidden state s_{t} ∈ {1, …, K} follows a Markov chain with initial probabilities π and "
         "transition matrix A_{ij} = P(s_{t} = j | s_{t−1} = i); each state emits z_{t} ~ N(μ_{j}, σ_{j}²). The forward "
         "recursion computes the likelihood, with scaling factors c_{t} that keep the numbers in range:")
    rp.eq("α_{t}(j) = [Σ_{i} α_{t−1}(i) A_{ij}] N(z_{t} | μ_{j}, σ_{j}²) / c_{t},   ln p(z_{1:T}) = Σ_{t} ln c_{t}")
    rp.p("The backward recursion gives β_{t}(i); together they give the state posteriors γ_{t}(i) and pair posteriors "
         "ξ_{t}(i, j). Baum–Welch (EM) re-estimates the parameters, for example")
    rp.eq("A_{ij} = Σ_{t} ξ_{t}(i, j) / Σ_{t} γ_{t}(i),   μ_{j} = Σ_{t} γ_{t}(j) z_{t} / Σ_{t} γ_{t}(j)")
    rp.p("and the Viterbi algorithm finds the most probable state path in log space:")
    rp.eq("δ_{t}(j) = max_{i} [δ_{t−1}(i) + ln A_{ij}] + ln N(z_{t} | μ_{j}, σ_{j}²)")
    rp.p("**Linear dynamical system and Kalman filter** [12]. The profile residual is modelled as a latent level l_{t} that "
         "decays towards zero and an observation with noise:")
    rp.eq("l_{t} = φ l_{t−1} + w_{t},  w_{t} ~ N(0, Q);    r_{t} = l_{t} + v_{t},  v_{t} ~ N(0, R)")
    rp.p("The Kalman filter alternates prediction and update with gain K_{t}:")
    rp.eq("l̂_{t|t−1} = φ l̂_{t−1},  P_{t|t−1} = φ² P_{t−1} + Q,  K_{t} = P_{t|t−1} / (P_{t|t−1} + R)")
    rp.eq("l̂_{t} = l̂_{t|t−1} + K_{t}(r_{t} − l̂_{t|t−1}),  P_{t} = (1 − K_{t}) P_{t|t−1}")
    rp.p("φ, Q and R are estimated by maximising the innovation log-likelihood on training hours. An h-step forecast is "
         "ẑ_{t+h} = m(t + h) + φ^{h} l̂_{t} with variance φ^{2h} P_{t} + Q Σ_{k<h} φ^{2k} + R, which gives the 95% interval; "
         "rentals are recovered as exp(ẑ) − 1. Assumptions: Markov dynamics, Gaussian emissions on the log scale, and "
         "parameters that do not change over the two years.")
    rp.h2("3.10 Unit 4 Proposed System Architecture / Workflow")
    rp.fig(BFIG / "flow_u4.png", "Workflow of Unit 4 (sequential models)", width=3.9)
    rp.p("The log-count sequence is split in time order. The HMM branch learns discrete demand states by Baum–Welch and "
         "decodes them with Viterbi. The LDS branch removes the training profile and filters the residual level with the "
         "Kalman filter. Both are evaluated on the test hours, against naive baselines, at 1 hour and 24 hours ahead. The "
         "decoded states and the 24-hour Kalman forecast are shown in the web application.")
    rp.h2("3.11 Results and Performance Evaluation")
    rp.h3("3.11.1 Hidden Markov model")
    rp.p("Correctness was checked on a 6-hour toy series by enumerating all 2⁶ = 64 state paths: our forward "
         "log-likelihood agrees with the brute-force value and Viterbi returns the brute-force best path. Baum–Welch "
         "never decreased the log-likelihood for any K.")
    rp.table(["K", "Train log-likelihood", "Test avg. log-lik. per hour", "Parameters", "BIC (train)", "EM iterations"],
             [(r["K"], f(r["train logL"], 1), f(r["test logL (avg per hour)"]), r["params"], f(r["BIC (train)"], 1),
               r["EM iters"]) for r in U["hmm_table"]],
             "HMM model selection on the hourly log-count sequence", widths=[0.4, 1.2, 1.3, 0.85, 1.12, 0.9])
    rp.p("BIC and the held-out likelihood keep improving up to K = 6 on this long series, so they do not identify a "
         "single best K. K = 4 was kept for interpretability: four demand levels that an operator can name.")
    st = [round(2.718281828 ** m - 1) for m in hmm["mu"]]
    rp.table(["State", "Mean of ln(1 + cnt)", "≈ rentals", "Std. dev.", "Mean dwell (h)", "Share of training hours"],
             [(f"S{i}", f(hmm["mu"][i], 2), st[i], f(hmm["std"][i], 2), f(hmm["dwell"][i], 1),
               pct(hmm["state_share"][i])) for i in range(4)],
             "Learned HMM states (K = 4)", widths=[0.55, 1.15, 0.85, 0.8, 1.05, 1.37])
    A = hmm["A"]
    rp.table(["From \\ to", "S0", "S1", "S2", "S3"],
             [(f"S{i}",) + tuple(f(x, 3) for x in A[i]) for i in range(4)],
             "Transition matrix learned by Baum–Welch", widths=[1.17, 1.15, 1.15, 1.15, 1.15])
    rp.p("The diagonal entries are large, so states persist for several hours (6.1 h for the quietest and 5.4 h for the "
         "busiest). The chain never jumps directly between the quietest and busiest "
         f"states. Viterbi and the posterior-mode state agree on {pct(hmm['viterbi_vs_gamma_agree'])} of test hours.")
    rp.fig(FIG / "u4_fig01_hmm_states.png",
           "Viterbi state against hour of day (training, left) and the first ten days coloured by decoded state (right)")
    rp.fig(FIG / "u4_fig02_hmm_test_week.png", "Test week 15–21 October 2012: log-count and mean of the Viterbi state")
    rp.p("Figures 3.12 and 3.13 show that the states follow the daily rhythm: night hours sit in S0 and the commute peaks in "
         "S3. The HMM therefore describes demand levels well but learns little beyond the time of day.")
    rp.h3("3.11.2 Kalman filter and forecasting")
    rp.p(f"Maximum likelihood gives φ = {f(kal['phi'])}, Q = {f(kal['Q'], 4)} and R = {f(kal['R'], 4)}. A level shock "
         f"halves in about {f(kal['half_life_h'], 1)} hours. Our filter's likelihood equals the exact multivariate-Gaussian "
         "likelihood of the same model, which verifies the recursion.")
    rp.table(["Model", "MAE", "RMSE", "R²", "MAE (log scale)"],
             [(r["Model"], f(r["MAE"], 1), f(r["RMSE"], 1), f(r["R²"]), f(r["MAE (log scale)"])) for r in U["forecast"]],
             "Forecasting the hourly count on the test period (errors in rentals)", widths=[2.6, 0.7, 0.75, 0.7, 1.02])
    rp.fig(FIG / "u4_fig03_kalman_forecast.png",
           "Kalman forecasts for the test week 15–21 October 2012 (top) and the filtered demand level (bottom)")
    rp.p(f"One hour ahead the Kalman filter has MAE {f(kal1['MAE'], 1)} rentals and R² {f(kal1['R²'])}, about 2.5 times "
         f"better than persistence ({f(pers['MAE'], 1)}) and the seasonal naive forecasts ({f(snw['MAE'], 1)}–80.5). "
         f"Twenty-four hours ahead the level has decayed (φ²⁴ ≈ 0.20), so the forecast is close to the profile and the "
         f"MAE grows to {f(kal24['MAE'], 1)}. The 95% intervals are well calibrated: they contain the true value "
         f"{pct(kal['cov95_1h'])} of the time at 1 hour and {pct(kal['cov95_24h'])} at 24 hours.")
    rp.fig(FIG / "u4_fig04_kalman_diag.png", "MAE by hour of day (left) and weekly mean of the filtered level (right)")
    rp.p("The filter helps most at the commute peaks, where the profile alone is worst (Figure 3.15). The weekly level "
         "stays above zero during the test period (mean anomaly "
         f"{f(kal['mean_level_anomaly_test'], 2)} on the log scale) because 2012 was busier than the training average, and "
         "falls sharply in late December. The HMM is a poor forecaster: its 1-hour MAE "
         f"({f(hmm1['MAE'], 1)}) is worse than persistence, and its next-state accuracy ({pct(hmm['next_state_acc'])}) is below "
         f"simply keeping the current state ({pct(hmm['persist_state_acc'])}).")
    rp.h2("3.12 Discussion, Findings and Limitations")
    rp.bullets([
        "Two demand shapes, commute and leisure, describe the days; three independent clustering families agree, which "
        "makes the finding robust. More than two clusters are weakly supported by distances, although BIC prefers four "
        "Gaussian components.",
        "Shape-based dissimilarities (cosine, correlation) and Ward linkage suit this problem; single linkage and the "
        "silhouette alone can be misleading when a few outlying days exist.",
        "The weather–time features are redundant: 8 of 11 components keep 90% of the variance and the same predictive power.",
        "The HMM states are interpretable demand levels but largely restate the hour of day, so the state definition has "
        "limited value for forecasting. The Kalman filter, which adds a persistent level to the known daily profile, is "
        "the useful sequential model.",
        "Validation respected time order throughout. Limitations: missing hours are skipped, parameters are assumed "
        "constant over two years, and the sequential models use no weather inputs, so a sudden storm is only seen after "
        "it lowers demand.",
    ])
    rp.h2("3.13 Chapter Conclusion")
    rp.p(f"Unit 3 showed that daily demand has two shapes that match the working-day flag (ARI {f(km['ARI_workingday'], 2)}) "
         f"and that {pca['M90']} principal components summarise the weather–time features without losing predictive "
         f"information. Unit 4 showed that demand has memory with a half-life of about {f(kal['half_life_h'], 0)} hours; the "
         f"Kalman filter turns this into the best short-term forecast of the project (1-hour MAE {f(kal1['MAE'], 1)} "
         "rentals), while the HMM gives an interpretable description of demand states.")


# ------------------------------------------------------------------ chapter 4
def chapter4(rp: Report):
    rp.chapter("Advanced Predictive Modelling and Ensemble Learning (Unit 5)")
    rp.h2("4.1 Overview of Allotted Unit 5 Topics")
    rp.p("Unit 5 covers classification and regression trees (CART) for categorisation (Practice 1); ensemble learning by "
         "bagging, random forests, boosting (AdaBoost and gradient boosting as adaptive basis-function models), stacking "
         "and Bayesian model averaging (Practice 2); and a generalised additive model (GAM) for the count against "
         "temperature and hour. The classifiers predict the same high-demand label as Unit 2, so the results are directly "
         "comparable.")
    rp.h2("4.2 Dataset Description and Preprocessing")
    rp.p("Unit 5 uses the dataset, split and high-demand threshold of Chapter 2. The trees use the 12 RAW features without "
         "scaling or encoding, because trees split on thresholds and capture interactions such as hour × working day "
         "themselves. Every hyper-parameter is chosen by 5-fold TimeSeriesSplit cross-validation inside the training "
         "period (each fold trains on earlier hours and validates on later ones). The GAM uses temp (converted to °C), hum, "
         "windspeed, the hour separately for working and non-working days, and one-hot season, year and weather.")
    rp.h2("4.3 EDA and Feature Analysis")
    rp.fig(FIG / "u5_fig05_importance.png",
           "Feature importance: random-forest Gini importance, gradient-boosting importance and permutation importance (AUC)")
    imp = U["importance"]
    rp.p(f"Figure 4.1 is the feature analysis specific to Unit 5. All three measures rank the hour first by far (random-forest "
         f"Gini importance {imp['rf_gini_top']['hr']:.2f}, permutation importance {imp['perm_top']['hr']:.2f} in test AUC). "
         f"Gini importance then lists atemp, temp and humidity, but permutation importance shows that removing any of them "
         f"costs at most {imp['perm_top']['temp']:.3f} AUC, because they are correlated and can replace each other; "
         "Gini importance is inflated for continuous features with many split points. Working day ranks second by "
         "permutation importance, confirming the hour × working-day interaction found in Units 1 and 3.")
    rp.h2("4.4 Proposed Workflow / Architecture")
    rp.fig(BFIG / "flow_u5.png", "Workflow of Unit 5 (trees, ensembles and GAM)", width=3.9)
    rp.p("The raw features and the training-median label feed a time-ordered cross-validation loop used for every "
         "hyper-parameter. A single tree is grown and pruned; bagging, random forest, AdaBoost and gradient boosting build "
         "ensembles of trees; stacking and BMA combine them. In parallel, a Poisson GAM models the count. All models are "
         "evaluated once on the test period. The stacking classifier and the GAM are saved for the web application's "
         "Predict page.")
    rp.h2("4.5 Algorithm Theory and Mathematical Formulation")
    rp.p("**CART** [13] splits the feature space recursively; at each node it chooses the feature and threshold that most "
         "reduce the Gini impurity of the children, where p_{k} is the class share at the node:")
    rp.eq("G = 1 − Σ_{k} p_{k}²")
    rp.p("The depth was chosen by CV, and alternatively the full tree was pruned by cost-complexity pruning, which "
         "minimises the training error R(T) plus a penalty on the number of leaves |T|, with α chosen by CV:")
    rp.eq("R_{α}(T) = R(T) + α |T|")
    rp.p("**Bagging** averages B trees grown on bootstrap samples, which reduces variance; the rows left out of each sample "
         "give an out-of-bag (OOB) estimate. **Random forest** [14] also samples a random subset of features at each split, "
         "which de-correlates the trees.")
    rp.p("**AdaBoost** [15] fits weak learners h_{m} in sequence, raising the weight of misclassified hours:")
    rp.eq("α_{m} = ln((1 − err_{m}) / err_{m}),   F(x) = Σ_{m} α_{m} h_{m}(x)")
    rp.eq("w_{i} ← w_{i} exp(α_{m} I(y_{i} ≠ h_{m}(x_{i})))")
    rp.p("**Gradient boosting** [16] is an adaptive basis-function model: each new tree h_{m} fits the negative gradient of the "
         "log-loss at the current model, and is added with a learning rate ν:")
    rp.eq("F_{m}(x) = F_{m−1}(x) + ν h_{m}(x)")
    rp.p("The learning rate, tree depth and number of trees M were tuned by CV (Table 4.2).")
    rp.p("**Stacking** [17] trains a logistic-regression meta-model on the probabilities of CART, random forest, gradient "
         "boosting and AdaBoost. The meta-features are out-of-fold predictions from expanding windows, so the meta-model "
         "never sees a prediction made with knowledge of the future. **Bayesian model averaging** [18] weights each model k by its "
         "posterior probability, approximated by its held-out predictive likelihood L_{k}:")
    rp.eq("p(y | x) = Σ_{k} w_{k} p_{k}(y | x),   w_{k} ∝ exp(ln L_{k})")
    rp.p("**Generalised additive model** [19]. The count is modelled as Poisson with a log link and smooth functions built from "
         "cubic B-splines, fitted with an L2 penalty α (chosen by CV):")
    rp.eq("ln E[cnt] = β_{0} + f_{1}(temp) + f_{2}(hum) + f_{3}(wind) + f_{wd}(hr) + βᵀc")
    rp.p("where c holds the one-hot season, year and weather indicators, f_{wd} is a separate smooth curve of the hour for working and non-working days. The Poisson likelihood "
         "suits counts (never negative, variance growing with the mean), and each f can be plotted and read.")
    rp.h2("4.6 Results and Performance Evaluation")
    rp.h3("4.6.1 CART")
    sel = [r for r in U["cart_depth_table"] if r["max_depth"] in (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 15)]
    rp.table(["Max depth", "CV accuracy (TimeSeriesSplit)", "Training accuracy", "Test accuracy"],
             [(r["max_depth"], f(r["CV accuracy (TimeSeriesSplit)"]), f(r["train accuracy"]), f(r["test accuracy"]))
              for r in sel],
             "CART: accuracy against tree depth (test accuracy shown only for reference)", widths=[1.1, 2.0, 1.3, 1.37])
    rp.fig(FIG / "u5_fig01_cart_depth.png", "CART complexity against accuracy: training, CV and test", width=4.6)
    rp.p(f"Training accuracy keeps rising with depth while CV accuracy levels off at depth 7–11; CV selects depth "
         f"{U['cart_depth']}. Test accuracy was not used for selection. Depth 1 has a high test accuracy (0.863) only by "
         "accident of the test period, where most hours are high-demand, which shows why selection must use CV. "
         "Cost-complexity pruning (α = 0.00151) gives a tree with only 26 leaves and similar test accuracy (Table 4.3).")
    rp.fig(FIG / "u5_fig02_cart_tree.png", "Top three levels of the CART tree")
    rp.p("The first split is on the hour (night against day) and the next on hour and working day, so the tree rediscovers "
         "the daily profile found in Units 1 and 3 and is easy to explain.")
    rp.h3("4.6.2 Ensembles")
    rp.p(f"Bagging with 200 trees has an OOB accuracy of {f(U['bag_oob'])} while its individual trees average "
         f"{f(U['bag_tree_acc_mean'])} (s.d. {f(U['bag_tree_acc_sd'], 3)}), so averaging removes much of the variance. The "
         f"random forest with 500 trees has OOB accuracy {f(U['rf_oob'])}.")
    rp.table(["Learning rate", "Max depth", "Best number of trees", "CV accuracy"],
             [(r["learning_rate"], r["max_depth"], r["best #trees"], f(r["CV accuracy"], 4)) for r in U["gb_grid"]],
             "Gradient boosting: time-ordered CV over learning rate, depth and number of trees",
             widths=[1.3, 1.1, 1.8, 1.57])
    rp.fig(FIG / "u5_fig03_boosting.png", "Test accuracy against boosting rounds (left) and gradient-boosting training loss (right)")
    rp.p("A smaller learning rate needs more trees but gives the best CV accuracy (ν = 0.05, depth 3, 381 trees). "
         "AdaBoost with stumps learns slowly; with depth-3 trees it is close to gradient boosting (Figure 4.5).")
    rows = [(r["Model"], f(r["Accuracy"]), f(r["Precision"]), f(r["Recall"]), f(r["F1"]), f(r["ROC-AUC"]),
             f(r["Log-loss"])) for r in U["cls"]]
    rp.table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Log-loss"], rows,
             "Unit 5 classifiers on the test period (high-demand label)", widths=[1.95, 0.68, 0.7, 0.62, 0.6, 0.66, 0.56])
    rp.fig(FIG / "u5_fig04_classification.png",
           "Stacking confusion matrix (left), ROC curves (centre) and test accuracy of all Unit 5 models (right)")
    rp.p(f"Every ensemble beats the single tree on ranking quality: CART reaches ROC-AUC 0.94, the ensembles "
         f"0.976–{f(stack['ROC-AUC'])}. Stacking has the highest accuracy ({f(stack['Accuracy'])}), F1 ({f(stack['F1'])}) "
         f"and AUC ({f(stack['ROC-AUC'])}); gradient boosting has the lowest log-loss ({f(gb['Log-loss'])}), i.e. the best "
         f"calibrated probabilities. The random forest has the highest precision ({f(rf['Precision'])}) and stacking the "
         f"highest recall ({f(stack['Recall'])}). The differences between ensembles are small (accuracy 0.92–0.93). BMA "
         "put all its weight on gradient boosting (weight 1.0, the others below 10⁻²⁵) because, with 13,903 hours, small "
         "likelihood differences become overwhelming; it therefore reproduces gradient boosting.")
    rp.h3("4.6.3 Generalised additive model")
    rp.table(["Model", "RMSE", "MAE", "R²"],
             [(r["Model"], f(r["RMSE"], 1), f(r["MAE"], 2), f(r["R²"])) for r in U["gam"]["table"]],
             "Count models on the test period (errors in rentals per hour)", widths=[2.9, 0.95, 0.95, 0.97])
    rp.fig(FIG / "u5_fig06_gam.png",
           "GAM effects of temperature and hour by day type, and GAM forecasts for the test week 15–21 October 2012")
    rp.p(f"The Poisson GAM (α = {U['gam']['alpha']}) is the most accurate count model of this unit (RMSE {f(gam['RMSE'], 1)}, "
         f"MAE {f(gam['MAE'], 2)}, R² {f(gam['R²'])}), slightly ahead of gradient-boosting and random-forest regressors, while "
         f"each effect can be read from Figure 4.7: expected rentals rise with temperature up to about "
         f"{f(U['gam']['temp_peak_C'], 0)} °C and decline beyond, and the hour curve has a double commute peak on working "
         "days and a broad afternoon hump otherwise.")
    rp.h2("4.7 Comparison with Previous Models")
    p1lr = cls_row("Logistic regression (L2)", "raw")
    rp.table(["Task", "Unit 2 model", "Unit 2 result", "Unit 5 model", "Unit 5 result"], [
        ("High demand, RAW features", "Logistic regression", f"Acc {f(p1lr['Accuracy'])}, AUC {f(p1lr['ROC-AUC'])}",
         "Stacking", f"Acc {f(stack['Accuracy'])}, AUC {f(stack['ROC-AUC'])}"),
        ("High demand, best overall", "RBF SVM (encoded)", f"Acc {f(rbf['Accuracy'])}, AUC {f(rbf['ROC-AUC'])}",
         "Stacking (raw)", f"Acc {f(stack['Accuracy'])}, AUC {f(stack['ROC-AUC'])}"),
        ("Count, best overall", "Bayesian linear (log)", f"MAE {f(R['best_reg_mae'], 2)}, R² {f(R['best_reg_r2'])}",
         "Poisson GAM", f"MAE {f(gam['MAE'], 2)}, R² {f(gam['R²'])}"),
    ], "Unit 5 against Unit 2 on the same test period and metrics", widths=[1.35, 1.2, 1.15, 0.95, 1.12],
        align_numbers=False)
    rp.p("On the same 3,476 test hours and the same label, the tree ensembles improve the RAW-feature baseline sharply "
         f"(accuracy from {f(p1lr['Accuracy'])} to {f(stack['Accuracy'])}, AUC from {f(p1lr['ROC-AUC'])} to "
         f"{f(stack['ROC-AUC'])}) without any manual feature engineering, because trees learn the hour × working-day "
         "interaction themselves. Against the best Unit 2 model, which uses the hand-built interaction, there is no "
         f"improvement: the RBF SVM is slightly more accurate ({f(rbf['Accuracy'])} against {f(stack['Accuracy'])}) with "
         "the same AUC. For counts, the GAM matches Bayesian linear regression on MAE and has a lower RMSE "
         f"({f(gam['RMSE'], 1)} against {f(R['best_reg_rmse'], 1)}), and it gives readable effect curves.")
    rp.h2("4.8 Discussion, Findings and Limitations")
    rp.bullets([
        "Ensembles generalise better than a single tree (AUC 0.98 against 0.94) and need no feature engineering, but they "
        "do not beat a well-engineered kernel model on this data.",
        "The single tree and the GAM are interpretable; random forest and boosting need importance measures, which "
        "disagree for correlated features (Gini against permutation importance).",
        "Training costs are small on this dataset (all ensembles fit in under 3 seconds); stacking costs five model fits "
        "because of its expanding-window meta-features.",
        "BMA collapsed to a single model, a known behaviour with large data; stacking's gain over gradient boosting is small.",
        "Limitations: no weather forecast or lagged demand is used, and the high-demand label hides the size of the error "
        "near the threshold of 130 rentals.",
    ])
    rp.h2("4.9 Chapter Conclusion")
    rp.p(f"Unit 5 contributed the strongest feature-free classifier of the project (stacking, ROC-AUC "
         f"{f(stack['ROC-AUC'])}) and an interpretable count model (Poisson GAM, R² {f(gam['R²'])}) that explains how "
         "temperature and hour drive demand. These two models are the ones deployed in the web application's Predict page.")


# ------------------------------------------------------------------ chapters 5-6
def chapter5(rp: Report):
    rp.chapter("Integrated Discussion and Findings")
    rp.p("The five units answer one question from different angles: how many bikes will be rented in an hour, and why. "
         "All results below use the same data, the same chronological split and the same test period, so models of the "
         "same task can be compared directly.")
    rp.table(["Unit", "Task", "Best model", "Test result"], [
        ("1", "Understanding demand", "Probability analysis", f"Working-day effect significant in {R['n_sig_hours']}/24 hours; "
                                                               "sign reverses between 08:00 and 13:00"),
        ("2", "Count", "Bayesian linear regression (log target)", f"MAE {f(R['best_reg_mae'], 2)}, R² {f(R['best_reg_r2'])}"),
        ("2", "High-demand label", "RBF SVM (encoded)", f"Accuracy {f(rbf['Accuracy'])}, AUC {f(rbf['ROC-AUC'])}"),
        ("3", "Day shapes", "K-means / GMM / Ward, K = 2", f"Silhouette {f(km['sil'])}, ARI {f(km['ARI_workingday'])}"),
        ("3", "Feature reduction", "PCA", f"{pca['M90']} of 11 components keep 90%; R² {f(pca['r2_pc'])} vs {f(pca['r2_all'])}"),
        ("4", "1-hour forecast", "Kalman filter (LDS)", f"MAE {f(kal1['MAE'], 1)}, R² {f(kal1['R²'])}"),
        ("4", "24-hour forecast", "Kalman filter (LDS)", f"MAE {f(kal24['MAE'], 1)}, R² {f(kal24['R²'])}"),
        ("5", "High-demand label", "Stacking ensemble", f"Accuracy {f(stack['Accuracy'])}, AUC {f(stack['ROC-AUC'])}"),
        ("5", "Count", "Poisson GAM", f"MAE {f(gam['MAE'], 2)}, R² {f(gam['R²'])}"),
    ], "Best model of each unit on the common test period", widths=[0.45, 1.25, 1.95, 2.12], align_numbers=False)
    rp.h2("5.1 Critical Findings")
    rp.bullets([
        "**The hour and the working-day flag explain most of the demand.** Unit 1 found the interaction, Unit 2 needed it "
        "in the features, Unit 3 found it again as the two daily shapes, Unit 4 used it as the forecasting profile and "
        "Unit 5's trees and importance measures rediscovered it.",
        "**Representation beats algorithm.** Encoding and the log target raised R² from 0.31 to 0.89; ensembles on raw "
        "features reached the same classification quality as a kernel model on engineered features.",
        f"**Time order matters.** Using the previous hours (Kalman filter) cuts the 1-hour forecast error from "
        f"{f(R['best_reg_mae'], 1)} rentals (Bayesian regression, no history) to {f(kal1['MAE'], 1)}. For a 24-hour horizon "
        "the memory has faded, and the calendar–weather models, which use the hour's observed weather, are more accurate.",
        "**Uncertainty must be modelled on the right scale.** Intervals on the log scale are calibrated (93–98% for a 95% "
        "target) at peaks and at night; on the count scale they fail at peaks.",
        "**Interpretable models are competitive.** The GAM and the HMM states explain demand as clearly as the ensembles "
        "predict it.",
    ])
    rp.h2("5.2 Deployed Demonstration")
    rp.fig(BFIG / "flow_system.png", "From notebooks to the deployed demonstration", width=3.9)
    rp.p("The Units 3–5 notebook is re-run by a training script that checks eight headline numbers against the saved "
         "results before it exports the models, and 27 automated tests check the notebook parity, the calendar rules and "
         f"every page. The Streamlit application is public at {APP} and its source at {REPO}. It has four pages: Predict "
         "(GAM count and stacking probability for a chosen hour and weather), Forecast 24h (Kalman forecast with interval), "
         "Demand Patterns (K-means shapes, PCA map and HMM states) and Model Leaderboard.")
    rp.fig(BFIG / "app_predict.png", "Predict page of the web application", width=5.2)
    rp.p("For the default inputs (Wednesday 15 August 2012, 08:00, clear weather, 24 °C, 60% humidity) the GAM predicts 747 rentals and the "
         "stacking classifier gives a high-demand probability of 0.99.")
    rp.fig(BFIG / "app_forecast.png", "Forecast 24h page: Kalman forecast from 17 October 2012 06:00 with 95% interval",
           width=5.2)
    rp.p("For this start time the 24-hour forecast has an MAE of 99.8 rentals and under-forecasts the evening peak, "
         f"consistent with the 24-hour test MAE of {f(kal24['MAE'], 1)}; short horizons are much more accurate.")
    rp.h2("5.3 Study Limitations")
    rp.bullets([
        "One city and two years of data; the strong growth from 2011 to 2012 means models need retraining before use in a "
        "later year or another city.",
        "The test period (August–December 2012) is a seasonal slice with more high-demand hours than training.",
        "Extreme weather is rare in the data (3 hours with weathersit = 4), so predictions for storms are unreliable.",
        "Unit 4 used the hourly sequence of the same dataset rather than a separate sequential dataset; 165 missing hours "
        "were skipped.",
        "Results for clusters (K = 2 against BIC's K = 4) and HMM states (K = 4) involve modelling choices that we report "
        "rather than hide.",
    ])


def chapter6(rp: Report):
    rp.chapter("Overall Conclusion and Future Scope")
    rp.h2("6.1 Conclusion")
    rp.p("The project applied every allotted technique of Units 1–5 to one real problem under a single, leakage-free, "
         "chronological protocol. The evidence supports clear conclusions: hourly bike demand is governed by the hour and "
         "working-day pattern, modulated by season, year and weather; days fall into a commute and a leisure shape; demand "
         f"has a memory of about {f(kal['half_life_h'], 0)} hours that a Kalman filter exploits for a 1-hour forecast error "
         f"of {f(kal1['MAE'], 1)} rentals; and both kernel models and tree ensembles classify high-demand hours with "
         f"ROC-AUC about {f(stack['ROC-AUC'], 2)}. Several algorithms (EM, PCA, HMM, Kalman filter, Bayesian logistic "
         "regression) were implemented from scratch and verified against libraries or brute force. The final models are "
         "deployed in a public web application with automated tests.")
    rp.h2("6.2 Future Scope")
    rp.bullets([
        "**Better data:** station-level trips to forecast demand per station and plan rebalancing routes; weather "
        "forecasts as inputs; more years and other cities to test transfer.",
        "**Better models:** negative-binomial or Poisson regression for over-dispersed counts; Kalman filtering with "
        "weather inputs; lagged-demand features for the ensembles; gradient-boosted count models.",
        "**Tuning and validation:** rolling-origin evaluation over several test periods instead of one seasonal slice; "
        "probability calibration of the classifiers.",
        "**Deployment:** connect the application to live data, retrain automatically, and alert operators when a station "
        "is forecast to run empty or full.",
        "**Interpretability:** partial-dependence and SHAP explanations of the ensembles to compare with the GAM curves.",
    ])


REFS = [
    'H. Fanaee-T and J. Gama, "Event labeling combining ensemble detectors and background knowledge," *Progress in '
    'Artificial Intelligence*, vol. 2, no. 2–3, pp. 113–127, 2014.',
    'H. Fanaee-T, "Bike Sharing Dataset," UCI Machine Learning Repository, 2013. [Online]. Available: '
    'https://doi.org/10.24432/C5W894',
    'C. M. Bishop, *Pattern Recognition and Machine Learning*. New York, NY, USA: Springer, 2006.',
    'K. P. Murphy, *Machine Learning: A Probabilistic Perspective*. Cambridge, MA, USA: MIT Press, 2012.',
    'T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. New York, NY, USA: '
    'Springer, 2009.',
    'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, '
    'pp. 2825–2830, 2011.',
    'A. P. Dempster, N. M. Laird, and D. B. Rubin, "Maximum likelihood from incomplete data via the EM algorithm," '
    '*Journal of the Royal Statistical Society: Series B*, vol. 39, no. 1, pp. 1–38, 1977.',
    'P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," '
    '*Journal of Computational and Applied Mathematics*, vol. 20, pp. 53–65, 1987.',
    'D. L. Davies and D. W. Bouldin, "A cluster separation measure," *IEEE Transactions on Pattern Analysis and Machine '
    'Intelligence*, vol. PAMI-1, no. 2, pp. 224–227, 1979.',
    'I. T. Jolliffe, *Principal Component Analysis*, 2nd ed. New York, NY, USA: Springer, 2002.',
    'L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," '
    '*Proceedings of the IEEE*, vol. 77, no. 2, pp. 257–286, 1989.',
    'R. E. Kalman, "A new approach to linear filtering and prediction problems," *Journal of Basic Engineering*, '
    'vol. 82, no. 1, pp. 35–45, 1960.',
    'L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, *Classification and Regression Trees*. Belmont, CA, USA: '
    'Wadsworth, 1984.',
    'L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.',
    'Y. Freund and R. E. Schapire, "A decision-theoretic generalization of on-line learning and an application to '
    'boosting," *Journal of Computer and System Sciences*, vol. 55, no. 1, pp. 119–139, 1997.',
    'J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, '
    'no. 5, pp. 1189–1232, 2001.',
    'D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992.',
    'J. A. Hoeting, D. Madigan, A. E. Raftery, and C. T. Volinsky, "Bayesian model averaging: A tutorial," '
    '*Statistical Science*, vol. 14, no. 4, pp. 382–417, 1999.',
    'T. Hastie and R. Tibshirani, *Generalized Additive Models*. London, U.K.: Chapman & Hall, 1990.',
]


def references(rp: Report):
    rp.unnumbered_chapter("References")
    for i, r in enumerate(REFS, 1):
        rp._runs(rp.doc.add_paragraph(style="RptRef"), f"[{i}]\t{r}")


def notebook_code(nb_path: Path):
    """Yield (markdown heading or None, code) for each code cell, in order."""
    nb = json.loads(nb_path.read_text(encoding="utf8"))
    head = None
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" and src.lstrip().startswith("#"):
            head = src.strip().splitlines()[0].lstrip("# ").strip()
        elif c["cell_type"] == "code" and src.strip():
            yield head, src
            head = None


def appendix(rp: Report):
    rp.unnumbered_chapter("Appendix A")
    rp.doc.add_paragraph("CODING", style="RptChapterNo")
    rp.p(f"The complete Python implementation is given below, cell by cell, exactly as in the two executed notebooks. Both "
         f"notebooks, the data and the web application are in the project repository {REPO}. To run a notebook in Google "
         "Colab, open its link and upload hour.csv and day.csv (from the repository's data folder) to the session before "
         "running all cells.")
    nb2 = list(notebook_code(ROOT / "notebooks" / NB2))
    i4 = next(i for i, (h, _) in enumerate(nb2) if h and h.startswith("UNIT 4") or h and h.startswith("10."))
    i5 = next(i for i, (h, _) in enumerate(nb2) if h and (h.startswith("UNIT 5") or h.startswith("14.")))
    parts = [
        ("Chapter 2 – Implementation Methodology and Algorithm Configuration (Units 1 and 2)", NB1,
         list(notebook_code(ROOT / "notebooks" / NB1))),
        ("Chapter 3 – Unit 3 Proposed Workflow and Implementation Methodology", NB2, nb2[:i4]),
        ("Chapter 3 – Unit 4 Proposed Workflow and Implementation Methodology", NB2, nb2[i4:i5]),
        ("Chapter 4 – Implementation Methodology and Hyperparameter Configuration (Unit 5)", NB2, nb2[i5:]),
    ]
    for title, nb, cells in parts:
        rp.h2(title)
        rp.p(f"Notebook: {nb}. Google Colab link: {COLAB}{nb}")
        for head, src in cells:
            if head:  # notebook section title in bold body text, kept out of the table of contents
                clean = head.replace("*", "").replace("`", "")
                rp.doc.add_paragraph(style="RptBody").add_run(clean).bold = True
            rp.code(src)
            rp.doc.add_paragraph(style="RptCode")


def main():
    diagrams.build_all()
    rp = Report(ROOT / "report" / "template.docx", MEMBERS, "Bike-Sharing Demand Forecasting for Smart Urban Mobility",
                FACULTY)
    front_matter(rp)
    for ch in (chapter1, chapter2, chapter3, chapter4, chapter5, chapter6):
        ch(rp)
    references(rp)
    appendix(rp)
    rp.finish(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
