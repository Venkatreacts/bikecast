# Team 8 – Viva Preparation (answers based on the final verified notebook)

**1. Why a chronological split?** Demand is a time series. We train on 1 Jan 2011 – 7 Aug 2012 (13,903 h) and test on 7 Aug – 31 Dec 2012 (3,476 h), so each score is a real forecast of an unseen future.

**2. Why not a random split?** Consecutive hours are almost copies of each other, so a random split puts neighbours of every test hour in training and the score becomes interpolation. It also lets the model "see the future" and ignores growth: test mean demand is 249 rentals/h vs 175 in training.

**3. Why remove casual and registered?** casual + registered = cnt on all 17,379 rows (checked in code). They are the target split in two parts and are unknown before the hour happens – target leakage.

**4. Why remove `instant`?** It is just the row number 1…N (checked). It is a monotone proxy for time, which a model could use to extrapolate a trend, and it has no causal meaning.

**5. Why handle `dteday` separately?** It is a string date. We parse it to sort and split chronologically, then drop it; yr, mnth, weekday and hr already carry its information (verified consistent).

**6. Why log1p(cnt)?** Counts are right-skewed (skewness 1.28) and over-dispersed (variance/mean ≈ 174); spread grows with level. On log1p the noise is closer to constant. OLS R² goes 0.774 → 0.891, MAE 78.1 → 44.8, and negative forecasts (0.8%) disappear. Predictions are back-transformed with expm1 and scored on the count scale.

**7. What does variance tell us?** Spread around the mean. cnt has mean 189.5 and s.d. 181.4; variance/mean ≈ 174 versus 1 for a Poisson count → strong over-dispersion.

**8. What does covariance tell us?** The direction (and scale-dependent size) of joint variation: Cov(cnt,temp) = 14.14 > 0 (warmer → more rentals), Cov(cnt,hum) = -11.30 < 0.

**9. Covariance vs correlation?** Correlation = covariance / (σ_X σ_Y): unit-free, in [−1, 1]. Changing temp to °C multiplies the covariance by 41 (579.6) but leaves ρ = 0.405 unchanged. We report both; correlation does not replace covariance.

**10. What is Bayes' theorem here?** P(H|C) = P(C|H)P(H)/P(C) = 0.704 × 0.525 / 0.657 = 0.562, where H = high demand (cnt > 130) and C = clear weather. It equals the direct empirical proportion.

**11. What is independence?** A and B are independent if P(A,B) = P(A)P(B). Pooled, working day vs high demand is significant (p ≈ 3e−19) but weak (Cramér's V = 0.068); significance ≠ strength.

**12. What is conditional dependence?** Dependence that appears (or disappears) once we condition on a third variable. Given the hour, working day and demand are dependent in 18 of 24 hours, with opposite signs (commute hours vs midday).

**13. Why hour × workingday?** Because of #12: working days peak at 08:00 and 17:00 (477 vs 106 at 08:00), weekends peak around 13:00 (373 vs 198). A model without the interaction cannot represent both shapes.

**14. Why polynomial degree 3?** On the seed-42 random split it has the lowest held-out RMSE (1519), and the median held-out RMSE over 200 splits is lowest at degree 3. Honest caveat: degree 3 and 9 each win ~40% of individual splits; 3 is the stable, parsimonious choice. Peak ≈ 27.4 °C.

**15. What is overfitting?** Fitting noise: training error keeps falling with degree (1480 → 1359) while held-out error stops improving after degree 3 and then rises.

**16. Why OLS?** It is the maximum-likelihood estimator under Gaussian noise: maximising the log-likelihood equals minimising squared error, w = (ΦᵀΦ)⁻¹Φᵀy. It is our baseline (encoded R² = 0.774).

**17. Why ridge?** L2 penalty = MAP under a Gaussian prior; it stabilises collinear features (temp, atemp ρ = 0.988). With 13,903 rows and 77 features it barely changes accuracy (RMSE 105.7 vs 104.9; α chosen by TimeSeriesSplit).

**18. Why Bayesian linear regression?** It gives a full predictive distribution, not just a point forecast. Its posterior mean equals ridge (λ = α/β), so point accuracy is the same; the value is the 95% interval: coverage 93.1% (log target) but only 51.7% at rush hour for the raw target.

**19. Why Laplace approximation?** The logistic posterior has no closed form. We approximate it by a Gaussian at the MAP weights with covariance H⁻¹ (H = ΦᵀRΦ + αI) and predict with σ(κμ_a), κ = (1+πσ_a²/8)^(−1/2). The posterior is narrow here (mean |Δp| = 0.006); its value is flagging uncertain hours (2.4% of hours, 35% wrong vs 3.5% for confident ones).

**20. Why SVM, and why RBF?** SVM finds the maximum-margin separator; the RBF kernel gives a non-linear boundary without explicit features. On raw features it lifts AUC from 0.883 to 0.961; on well-encoded features linear ≈ RBF (0.975 vs 0.978).

**21. Why does Naive Bayes perform worse?** AUC 0.840 vs 0.976 for logistic regression: the hour and hour×workingday dummies overlap, so independence is badly violated and the evidence is double-counted; binary columns are not Gaussian.

**22. Why is ROC-AUC useful?** Threshold-free: the probability that a random high-demand hour scores above a random low-demand hour. Test positives are 62.5%, so accuracy alone is misleading (always predicting "high" gives 0.625). Best model: SVM RBF kernel (encoded), AUC 0.978.

**23. What does R² = 0.891 actually mean?** On the test period (Aug–Dec 2012) the model's squared errors are 10.9% of those of a constant-mean predictor, i.e. it explains 89.1% of the variance of hourly counts. It says nothing about any single hour (MAE ≈ 45 bikes; MAE ≈ 85 in the 16–19 h peak).

**24. What are the limitations?** Gaussian-noise models for counts; log forecasts are median-type; one chronological split with a distribution shift (test 62.5% high-demand vs 50% in training); no confidence intervals on test metrics; fixed hyper-parameters; no lagged demand; weather assumed known; ridge α fragile; probit approximation rough in tails; the polynomial "best degree" depends on the split.

**25. What was corrected from your earlier report?** Ridge α for the cnt target (0.001 → 100) and its RMSE/MAE (105.7/78.7), polynomial training RMSE at M = 9 (1359), linear-SVM raw accuracy/AUC (0.834/0.883); covariance was added; PPT bars were rescaled. 105 of 111 old claims reproduced.

**Bonus – likely follow-ups.** *Why is the 130 threshold legitimate?* It is the training median (verified), computed without test data. *Why is the cut mid-day on 7 Aug?* It is exactly the first 80% of rows. *Is the Excel member split given?* No – the contribution table is a proposal. *Why not Units 3–5?* They are PBL II in the Excel.
