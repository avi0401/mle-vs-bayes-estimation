# Maximum Likelihood vs Bayesian Parameter Estimation — and What It Does to a Classifier

This project compares **maximum likelihood estimation (MLE)** and **Bayesian parameter estimation** on
**simulated data**, where the true parameters are known. It then analyses how the choice of estimator changes the
behaviour of the **Gaussian generative classifier (QDA)** built on top of it: accuracy, log-loss, calibration,
decision boundaries, and sensitivity to the prior.

## Quick start

**Easiest:** open `MLE_vs_Bayesian_Estimation.ipynb` in Google Colab (File → Upload notebook, or open it from GitHub), then Runtime → Run all.

**Locally:**

```bash
pip install -r requirements.txt
python main.py --quick     # ~15 s smoke test
python main.py             # full run, ~2 min; writes everything to results/
```

Outputs in `results/`:

| File | What it shows |
|---|---|
| `REPORT.md` | Auto-generated tables of all key numbers |
| `01_parameter_estimation_mse.png` | MSE vs n for a Bernoulli p, a Gaussian mean, and a 2-D covariance matrix |
| `02_posterior_evolution.png` | How the Beta posterior over p sharpens as coin flips arrive |
| `03_learning_curves_2d.png` | Accuracy, log-loss, Brier score and ECE vs training size (2-D problem) |
| `04_learning_curves_highdim.png` | Same in 10-D, where n ≈ d breaks MLE |
| `05_decision_boundaries.png` | Learned P(class 1 \| x) maps at n = 4, 15, 200, vs the Bayes-optimal boundary |
| `06_calibration_small_n.png` | Reliability diagram + confidence histogram at n = 5 per class |
| `07_prior_sensitivity.png` | Effect of prior strength with a correct vs a wrong prior |
| `*.csv`, `all_results.json` | Raw numbers |

## Project structure

```
MLE_vs_Bayesian_Estimation.ipynb   self-contained Colab notebook (same code + explanations)
main.py              runs all experiments, writes results/
src/data.py          simulated Gaussian class problems (2-D and 10-D) + Bayes-optimal posterior
src/estimators.py    MLE & Bayesian (NIW) estimators, the three classifiers, Gaussian / Student-t densities
src/metrics.py       accuracy, log-loss, Brier, expected calibration error, reliability curve
src/experiments.py   the five experiments and their plots
```

## Theory

### Maximum likelihood
MLE picks the single parameter value that makes the observed data most probable:
θ̂_MLE = argmax_θ p(D | θ). For a Gaussian class this gives

* μ̂ = x̄  (sample mean)
* Σ̂ = (1/n) Σᵢ (xᵢ − x̄)(xᵢ − x̄)ᵀ  (biased sample covariance)
* class prior π̂ₖ = nₖ / n

The **plug-in classifier** then uses p(x | y=k) = N(x; μ̂ₖ, Σ̂ₖ) as if those estimates were exact.

### Bayesian estimation
Bayesian estimation treats θ as a random variable with a prior p(θ). It computes the full posterior
p(θ | D) ∝ p(D | θ) p(θ) instead of a single point. This project uses the **conjugate Normal-Inverse-Wishart prior**:

Σ ~ IW(Ψ₀, ν₀),  μ | Σ ~ N(μ₀, Σ/κ₀)

The posterior is again NIW, with these closed-form updates:

* κₙ = κ₀ + n,  νₙ = ν₀ + n
* μₙ = (κ₀μ₀ + n x̄) / κₙ  ← a weighted average of the prior mean and the data mean (*shrinkage*)
* Ψₙ = Ψ₀ + S + (κ₀n/κₙ)(x̄ − μ₀)(x̄ − μ₀)ᵀ

Class priors use a Dirichlet(α=1) prior, which gives Laplace smoothing: (nₖ+1)/(n+K).

To classify, the Bayesian model **integrates over parameter uncertainty** instead of plugging in one estimate.
It uses the posterior predictive p(x | D) = ∫ p(x | θ) p(θ | D) dθ, which here is a **multivariate Student-t**:

t_{νₙ−d+1}( μₙ, Ψₙ(κₙ+1) / (κₙ(νₙ−d+1)) )

Its heavier tails with little data are exactly what keeps the classifier from being over-confident.

Three classifiers are compared:
1. **MLE plug-in QDA**
2. **Bayesian posterior predictive** (Student-t), the fully Bayesian model
3. **Bayesian posterior-mean plug-in**, an ablation that uses the Bayesian estimates but a Gaussian, not a Student-t.
   It separates the benefit of the *prior* (shrinkage) from the benefit of *averaging over uncertainty*.

The default prior is weak: κ₀ = 0.01, ν₀ = d+2, E[Σ] = I. Its job is to regularise, not to inject knowledge.

### Simulated data
* **2-D problem:** class 0 ~ N((0,0), [[1, .5], [.5, 1]]), class 1 ~ N((1.5,1), [[1.5, −.4], [−.4, .8]]).
  The classes have different covariances, so the Bayes-optimal boundary is quadratic and the best achievable
  accuracy is about 0.82.
* **10-D problem:** random correlated covariances, class means √d·1.6/d apart.
* Each setting is averaged over 100–200 independent training sets, evaluated on 20 000 fresh test points.

## Key findings (full run)

1. **Small samples: Bayes wins clearly.** With 3 points per class (2-D), MLE reaches 63.0 % accuracy and Bayes
   reaches 74.7 %. At n = 5, the MLE of a Bernoulli p has 2× the MSE of the Bayesian posterior mean (0.044 vs 0.021).
2. **MLE is over-confident.** At n = 5 per class, 47 % of MLE's predictions claim more than 99 % confidence,
   against 16 % for Bayes. MLE's expected calibration error is about 2× Bayes's, and its median log-loss is 1.22 vs 0.53.
   With one sample estimate of Σ, MLE underestimates the spread (it uses 1/n, and it ignores uncertainty in μ).
   The Student-t predictive corrects both.
3. **Where the benefit comes from.** The posterior-mean plug-in recovers most of the *accuracy* gain, so the
   prior/shrinkage fixes the decision boundary. The full predictive gives the best *log-loss and calibration*,
   so integrating out uncertainty fixes the probabilities.
4. **Dimensionality amplifies the gap.** In 10-D with n = 11 per class (barely more than d), MLE's covariance
   is nearly singular: accuracy 53 %, median log-loss 15. Bayes gets 61 % and 1.9.
5. **Asymptotically they agree.** From about 35 points per class (2-D) or about 300 (10-D), both match each other
   and approach the Bayes-optimal classifier. The prior washes out, as the Bernstein–von Mises theorem predicts.
6. **The prior is a double-edged sword.** A strong *correct* prior (κ₀ = 50) gives about 80 % accuracy with only 3
   points per class. A strong *wrong* prior (class means swapped) drops accuracy to about 31 %, worse than chance,
   until enough data overrides it. Weak priors (κ₀ ≤ 1) are safe in both cases.

## Takeaways

| | MLE | Bayesian |
|---|---|---|
| Output | single point estimate | full posterior distribution |
| Small n | high variance, over-confident, can be singular | regularised, calibrated |
| Large n | consistent, efficient | converges to MLE |
| Needs | nothing extra | a prior (which can hurt if strong and wrong) |
| Cost | closed form | closed form here (conjugate); generally MCMC/VI |

## Possible extensions
* Unequal class sizes / imbalanced priors (`sample` instead of `sample_per_class`)
* Non-Gaussian true data (model misspecification)
* Empirical Bayes: set μ₀ and Ψ₀ from pooled data
* Compare with a discriminative baseline (logistic regression)
