# Team 8 – Units 3, 4, 5 (PBL II): Audit and Viva Notes
Notebook: `Team8_Bike_Sharing_ML_PBL2_Units3-5.ipynb` (executed from a clean kernel, 0 errors, ~5 min). Numbers: `results_units345.json`. Figures: `figures/u3_*, u4_*, u5_*`. Needs `hour.csv`, `day.csv` beside it. Same chronological 80/20 split and threshold (130) as PBL I.

## Excel requirement → where implemented
| Unit | Requirement (Team 8 row) | Section |
|---|---|---|
| 3 | K-means | §3 (K by elbow/silhouette/DB; assignment=nearest centre verified) |
| 3 | Mixtures of Gaussians, EM | §4 (EM written from scratch, checked vs sklearn: logL within ~1 unit; log-lik never decreases) |
| 3 | Hierarchical clustering, dissimilarity measures | §5 (euclid/cosine/correlation × single/complete/average/Ward, cophenetic corr, dendrogram) |
| 3 | Silhouette, DB index | §3, §6 |
| 3 | PCA, choosing latent dimensions | §7 (eigen-decomposition from scratch = sklearn; scree, 90 %, Kaiser, held-out recon error) |
| 3 | Factor analysis | §8 (q by held-out likelihood; communalities) |
| 4 | Markov model, HMM, forward–backward, scaling, Baum–Welch, Viterbi | §10 (from scratch; matches brute-force enumeration exactly) |
| 4 | LDS / Kalman → forecasting hourly count | §11 (MLE of φ,Q,R; matches exact Gaussian likelihood; 1 h and 24 h forecasts + intervals) |
| 4 | Practice: HMM to predict sequential data | §10, §12 |
| 5 | CART, pruning | §14 |
| 5 | Bagging / Random Forest / OOB | §15 |
| 5 | AdaBoost / Gradient Boosting / adaptive basis functions | §16 |
| 5 | Stacking, Bayesian model averaging | §17 |
| 5 | GAM for count vs temperature & hour | §19 (Poisson spline GAM) |

## Key results
- **Clustering:** silhouette and DB both choose K=2 (working-day vs non-working-day shape; ARI vs workingday 0.94). K=3 is weak (silhouette 0.24). GMM BIC prefers K=4 – disagreement is disclosed. 3 near-empty days (incl. Hurricane Sandy, 29 Oct 2012, 22 rentals) are excluded from shape clustering.
- **PCA:** 8 PCs keep 90 % of weather–time variance (Kaiser: 4); a linear model on the 8 PCs has test R² 0.484 vs 0.485 on all features.
- **HMM (K=4):** decoding works and is verified, but forecasting is weak (1 h MAE 105 vs persistence 85). Honest role: decoding/interpretation.
- **Kalman LDS:** φ=0.935 (half-life 10 h). 1 h MAE 31.2 vs persistence 85.2, same-hour-yesterday 80.5, profile-only 117.2. 95 % interval coverage 0.943 (1 h), 0.977 (24 h).
- **Classification (test):** Stacking AUC 0.978 / acc 0.927; GB 0.978 / 0.921; RF 0.977 / 0.919; single CART 0.941 / 0.909; Unit-2 logistic reference 0.886 / 0.824. Top inputs: hour, workingday, temperature.
- **GAM:** test MAE 44.9 vs gradient-boosting regressor 45.5; temperature effect peaks ≈ 30 °C.

## Disclosed limitations / decisions
1. K=2 chosen (not 3): initial run with Sandy day included gave a misleading singleton cluster – fixed by excluding days with < 500 rentals.
2. HMM is a coarse level model; it is *not* better than naive baselines for forecasting.
3. BMA uses held-out predictive likelihood as proxy for marginal likelihood and collapses onto gradient boosting (weight 1.0).
4. sklearn's StackingClassifier cannot use time-ordered folds, so stacking is built manually with expanding-window out-of-fold predictions.
5. RF OOB accuracy (0.928) is optimistic vs chronological test (0.919).
6. Kalman forecast = expm1 of the log-mean (a median-type estimate).
7. Not yet done: Unit 3–5 slides / PDF report (PPT and PDF still cover Units 1–2 only).

## Viva questions
1. **Why K=2 when GMM BIC says 4?** Silhouette/DB (distance-based) both pick 2; BIC rewards extra flexible components. Two shapes (commute vs leisure) are what the data support; K=3 silhouette is 0.24.
2. **K-means vs GMM?** K-means = hard assignment, equal spherical clusters (the Σ=εI, ε→0 limit); GMM = soft responsibilities, full covariances, EM maximises a lower bound on log-likelihood.
3. **Why standardise before PCA?** Features have different units; PCA maximises variance so unscaled features would dominate. Scaler fitted on train only.
4. **PCA vs FA?** PCA explains total variance; FA models shared variance plus per-feature noise Ψ.
5. **Why scaling factors in forward–backward?** α_t shrinks exponentially; dividing by c_t=p(z_t|z_1:t-1) keeps values O(1) and ln p(Z)=Σ ln c_t.
6. **How did you verify Viterbi/forward?** Enumerated all 2^6 state paths on a 6-hour toy series: same log-likelihood to 10 decimals and same best path.
7. **Why a Kalman filter on top of an hour×workday profile?** Profile removes the daily cycle; the residual level is slow (AR(1), half-life 10 h), which a scalar LDS tracks. Result: MAE 31 vs 80–85 for naive baselines.
8. **Why is HMM worse?** Four Gaussian level buckets, no hour input; next-hour state accuracy 0.725 vs 0.777 for "state unchanged".
9. **CART vs Random Forest vs boosting?** CART: one greedy tree, high variance. RF: bagging + random feature subsets decorrelate trees (variance↓). Boosting: stage-wise adaptive basis functions fit to errors/gradients (bias↓).
10. **Why is OOB not enough?** Random resampling ignores time order; chronological test is the honest estimate.
11. **What is a GAM and why Poisson?** Sum of smooth 1-D functions on a link scale; Poisson/log link suits counts and gives multiplicative effects; keeps interpretability (temperature peak ≈ 30 °C, two hourly shapes).
12. **Bayesian model averaging vs stacking?** BMA weights ∝ posterior model probability (→ best single model: GB weight 1.0); stacking learns a combiner (Ada 1.8, GB 0.43) and can beat every base model.

## Presentation files (added)
- `Team8_Bike_Sharing_ML_PBL2_Units3-5.pptx` – 12-slide deck (same dark style as the PBL 1 deck), all numbers read from `results_units345.json`.
- `Team8_Bike_Sharing_ML_PBL2_Units3-5_Report.pdf` – report sections 5–9 (Units 3–5), 4 pages.
- `Team8_Full_Report_Units1-5.pdf` – PBL 1 report followed by the Units 3–5 report (simple concatenation; the PBL 1 pages are unchanged and still carry the "Units 1 & 2 only" scope sentence).
