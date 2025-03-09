### R code explanation
```R
library(TrialEmulation)

trial_pp  <- trial_sequence(estimand = "PP")  # Per-protocol

trial_itt <- trial_sequence(estimand = "ITT") # Intention-to-treat
```

Purpose: Initializes two target trial emulation frameworks:
- PP (Per-Protocol): Analyzes subjects who adhered to the protocol
- ITT (Intention-to-Treat): Analyzes all subjects as originally allocated

```R
trial_pp_dir  <- file.path(tempdir(), "trial_pp")
dir.create(trial_pp_dir)

trial_itt_dir <- file.path(tempdir(), "trial_itt")
dir.create(trial_itt_dir)
```

Purpose: Creates temporary directories to store:
- Weight models
- Intermediate results
- Diagnostic outputs

```R
data("data_censored") # dummy data in the package
head(data_censored)
```
Purpose: Loads a synthetic dataset with:
- Longitudinal patient records
- Columns like id, period, treatment, outcome, censored, and covariates (age, x1, etc.)

```R
# Per-protocol
trial_pp <- trial_pp |>
  set_data(
    data      = data_censored,
    id        = "id",
    period    = "period",
    treatment = "treatment",
    outcome   = "outcome",
    eligible  = "eligible"
  )

# ITT
# Function style without pipes
trial_itt <- set_data( 
  trial_itt,
  data      = data_censored,
  id        = "id",
  period    = "period",
  treatment = "treatment",
  outcome   = "outcome",
  eligible  = "eligible"
)
```
Key Arguments:
- id: Unique patient identifier
- period: Time period indicator
- treatment: Treatment status (binary)
- outcome: Outcome of interest
- eligible: Eligibility criteria indicator
Purpose: Formalizes the data structure for emulation.

```R
trial_pp <- trial_pp |>
  set_switch_weight_model(
    numerator    = ~ age,
    denominator  = ~ age + x1 + x3,
    model_fitter = stats_glm_logit(save_path = file.path(trial_pp_dir, "switch_models"))
  )
  trial_pp@switch_weights
```
Purpose: Models treatment switching using inverse probability weights (IPWs):
- Numerator: Probability of treatment given baseline covariates (age)
- Denominator: Probability of treatment given baseline + time-varying covariates (x1, x3)

```R
trial_pp <- trial_pp |>
  set_censor_weight_model(
    censor_event = "censored",
    numerator    = ~ x2,
    denominator  = ~ x2 + x1,
    pool_models  = "none",
    model_fitter = stats_glm_logit(save_path = file.path(trial_pp_dir, "switch_models"))
  )
trial_pp@censor_weights

trial_itt <- set_censor_weight_model(
  trial_itt,
  censor_event = "censored",
  numerator    = ~x2,
  denominator  = ~ x2 + x1,
  pool_models  = "numerator",
  model_fitter = stats_glm_logit(save_path = file.path(trial_itt_dir, "switch_models"))
)
trial_itt@censor_weights
```
Purpose: Models censoring using IPWs:
- Numerator/Denominator: Similar logic to switch weights but for censoring events
- pool_models: Determines whether models are pooled across time periods

```R
trial_pp  <- trial_pp |> calculate_weights()
trial_itt <- calculate_weights(trial_itt)
```
Purpose: Combines:
- Switch weights (for treatment adherence)
- Censor weights (for loss to follow-up)
Final weights = Switch Weights × Censor Weights

```R
trial_pp  <- set_outcome_model(trial_pp)
trial_itt <- set_outcome_model(trial_itt, adjustment_terms = ~x2)
```
Purpose: Specifies the outcome model (typically a survival model):
- Adjusted for covariates (x2 in ITT)
- Uses inverse probability weights to account for time-varying confounding

```R
trial_pp <- set_expansion_options(
  trial_pp,
  output     = save_to_datatable(),
  chunk_size = 500 # the number of patients to include in each expansion iteration
)
trial_pp  <- expand_trials(trial_pp)
```
Purpose:
- Clones patient records to emulate sequential trials
- Processes data in chunks of 500 patients for memory efficiency

```R
trial_itt <- fit_msm(
  trial_itt,
  weight_cols    = c("weight", "sample_weight"),
  modify_weights = function(w) { # winsorization of extreme weights
    q99 <- quantile(w, probs = 0.99)
    pmin(w, q99)
  }
)
```
Purpose: Fits a Marginal Structural Model (MSM):
- Uses Winsorization to handle extreme weights (trimming at 99th percentile)
- Estimates causal treatment effects

```R
preds <- predict(
  trial_itt,
  newdata       = outcome_data(trial_itt)[trial_period == 1, ],
  predict_times = 0:10,
  type          = "survival",
)

plot(preds$difference$followup_time, preds$difference$survival_diff,
     type = "l", xlab = "Follow up", ylab = "Survival difference")
lines(preds$difference$followup_time, preds$difference$`2.5%`, type = "l", col = "red", lty = 2)
lines(preds$difference$followup_time, preds$difference$`97.5%`, type = "l", col = "red", lty = 2)
```
Purpose:
- Predicts survival curves for treated vs. untreated
- Plots survival differences with 95% confidence intervals
Outputs results like:
- Survival probabilities
- Risk differences
- Hazard ratios
