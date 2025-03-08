import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

# ---------------------------
# 1. Set-up Directories
# ---------------------------
trial_pp_dir = os.path.join(os.getenv("TMPDIR", "/tmp"), "trial_pp")
os.makedirs(trial_pp_dir, exist_ok=True)
trial_itt_dir = os.path.join(os.getenv("TMPDIR", "/tmp"), "trial_itt")
os.makedirs(trial_itt_dir, exist_ok=True)

# ---------------------------
# 2. Create Dummy Data (data_censored)
# ---------------------------
# Assume columns: id, period, treatment, outcome, eligible, age, x1, x3, x2, censored
np.random.seed(42)
n_total = 1000
n_patients = n_total // 10  # assume 10 periods per patient

data_censored = pd.DataFrame({
    "id": np.repeat(np.arange(1, n_patients + 1), 10),
    "period": np.tile(np.arange(1, 11), n_patients),
    "treatment": np.random.binomial(1, 0.5, n_total),
    "outcome": np.random.binomial(1, 0.1, n_total),
    "eligible": np.random.binomial(1, 0.8, n_total),
    "age": np.random.normal(50, 10, n_total),
    "x1": np.random.normal(0, 1, n_total),
    "x3": np.random.normal(0, 1, n_total),
    "x2": np.random.normal(0, 1, n_total),
    "censored": np.random.binomial(1, 0.05, n_total)
})
print(data_censored.head())

# ---------------------------
# 3. Define “Trial” Objects
# ---------------------------
# For demonstration, we use dictionaries to hold trial configurations.
trial_pp = {"estimand": "PP", "data": data_censored.copy()}
trial_itt = {"estimand": "ITT", "data": data_censored.copy()}

# ---------------------------
# 4. Set Weight Models (Placeholders)
# ---------------------------
def set_switch_weight_model(trial, numerator_formula, denominator_formula, save_path):
    # Placeholder: store formulas and save path
    trial['switch_weight_model'] = {
        "numerator_formula": numerator_formula,
        "denominator_formula": denominator_formula,
        "save_path": save_path
    }
    return trial

def set_censor_weight_model(trial, censor_event, numerator_formula, denominator_formula, pool_models, save_path):
    trial['censor_weight_model'] = {
        "censor_event": censor_event,
        "numerator_formula": numerator_formula,
        "denominator_formula": denominator_formula,
        "pool_models": pool_models,
        "save_path": save_path
    }
    return trial

# For Per-protocol trial (trial_pp)
trial_pp = set_switch_weight_model(
    trial_pp,
    numerator_formula="age",
    denominator_formula="age + x1 + x3",
    save_path=os.path.join(trial_pp_dir, "switch_models")
)
print("trial_pp switch weight model:", trial_pp.get("switch_weight_model"))

trial_pp = set_censor_weight_model(
    trial_pp,
    censor_event="censored",
    numerator_formula="x2",
    denominator_formula="x2 + x1",
    pool_models="none",
    save_path=os.path.join(trial_pp_dir, "switch_models")
)
print("trial_pp censor weight model:", trial_pp.get("censor_weight_model"))

# For ITT trial (trial_itt)
trial_itt = set_censor_weight_model(
    trial_itt,
    censor_event="censored",
    numerator_formula="x2",
    denominator_formula="x2 + x1",
    pool_models="numerator",
    save_path=os.path.join(trial_itt_dir, "switch_models")
)
print("trial_itt censor weight model:", trial_itt.get("censor_weight_model"))

# ---------------------------
# 5. Calculate Weights (Simulation)
# ---------------------------
def calculate_weights(trial):
    # In practice, weight estimation involves fitting the specified models.
    # Here we simply add dummy weight columns.
    trial['data']['weight'] = np.random.uniform(0.8, 1.2, len(trial['data']))
    trial['data']['sample_weight'] = np.random.uniform(0.9, 1.1, len(trial['data']))
    return trial

trial_pp = calculate_weights(trial_pp)
trial_itt = calculate_weights(trial_itt)

print("trial_pp weights added:", trial_pp['data'][['weight', 'sample_weight']].head())
print("trial_itt weights added:", trial_itt['data'][['weight', 'sample_weight']].head())

# ---------------------------
# 6. Set Outcome Model
# ---------------------------
def set_outcome_model(trial, adjustment_terms=None):
    # For demonstration, fit a Cox proportional hazards model using lifelines.
    # We assume 'period' is the time variable and 'outcome' is the event indicator.
    cph = CoxPHFitter()
    df = trial['data']
    
    # Use 'treatment' as the main predictor; add adjustments if provided.
    predictors = ['treatment']
    if adjustment_terms:
        # This is a simplified parsing of the adjustment formula (e.g. "~ x2")
        adjustments = adjustment_terms.replace("~", "").strip().split(" + ")
        predictors += adjustments
    
    # Use only the necessary columns and drop missing values.
    cols = ['period', 'outcome'] + predictors
    df_model = df[cols].dropna()
    
    try:
        cph.fit(df_model, duration_col='period', event_col='outcome')
    except Exception as e:
        print("Outcome model fitting error:", e)
    
    trial['outcome_model'] = cph
    return trial

trial_pp = set_outcome_model(trial_pp)
trial_itt = set_outcome_model(trial_itt, adjustment_terms="~ x2")

# ---------------------------
# 7. Set Expansion Options & Expand Trials
# ---------------------------
def set_expansion_options(trial, output, chunk_size):
    trial['expansion_options'] = {"output": output, "chunk_size": chunk_size}
    return trial

def save_to_datatable():
    # Dummy function: in practice, this might store data in a specific format.
    return lambda df: df  # identity function

trial_pp = set_expansion_options(trial_pp, output=save_to_datatable(), chunk_size=500)
trial_itt = set_expansion_options(trial_itt, output=save_to_datatable(), chunk_size=500)

def expand_trials(trial):
    # Placeholder: here, expansion might mean replicating data or restructuring it.
    # For this example, we simply store the original data as "expanded."
    trial['expansion'] = trial['data']
    return trial

trial_pp = expand_trials(trial_pp)
trial_itt = expand_trials(trial_itt)
print("Expanded trial_pp data shape:", trial_pp['expansion'].shape)

def load_expanded_data(trial, seed=1234, p_control=0.5):
    np.random.seed(seed)
    # For demonstration, add a control flag to the expanded data.
    df = trial['expansion'].copy()
    df['control_flag'] = np.random.binomial(1, p_control, len(df))
    trial['expansion'] = df
    return trial

trial_itt = load_expanded_data(trial_itt, seed=1234, p_control=0.5)

# ---------------------------
# 8. Fit Marginal Structural Model (MSM)
# ---------------------------
def fit_msm(trial, weight_cols, modify_weights):
    # Apply the provided modification to weights (e.g., winsorization)
    weights = trial['data']['weight']
    trial['data']['modified_weight'] = modify_weights(weights)
    
    # Fit a weighted Cox model as a placeholder for MSM fitting.
    cph = CoxPHFitter()
    try:
        cph.fit(trial['data'], duration_col='period', event_col='outcome', weights_col='modified_weight')
    except Exception as e:
        print("MSM fitting error:", e)
    trial['msm'] = cph
    return trial

def winsorize_weights(w):
    q99 = np.quantile(w, 0.99)
    return np.minimum(w, q99)

trial_itt = fit_msm(trial_itt, weight_cols=["weight", "sample_weight"], modify_weights=winsorize_weights)

print("MSM outcome model summary for trial_itt:")
print(trial_itt['msm'].summary)

# ---------------------------
# 9. Predict Survival & Plot Differences
# ---------------------------
def predict_survival(trial, newdata, predict_times):
    # Use the fitted Cox model to predict survival functions.
    cph = trial['msm']
    surv_funcs = cph.predict_survival_function(newdata)
    
    # For demonstration, compare survival curves by treatment group.
    # Here we compute the average survival function for treatment=1 vs treatment=0.
    treatment_1_idx = newdata[newdata['treatment'] == 1].index
    treatment_0_idx = newdata[newdata['treatment'] == 0].index
    
    surv_t1 = surv_funcs[treatment_1_idx].mean(axis=1)
    surv_t0 = surv_funcs[treatment_0_idx].mean(axis=1)
    
    survival_diff = surv_t1 - surv_t0
    # Dummy confidence intervals (for illustration only)
    lower_ci = survival_diff * 0.95
    upper_ci = survival_diff * 1.05
    
    return pd.DataFrame({
        "followup_time": surv_funcs.index,
        "survival_diff": survival_diff,
        "lower": lower_ci,
        "upper": upper_ci
    })

# Select newdata from the expanded ITT data where period == 1
newdata = trial_itt['expansion'][trial_itt['expansion']['period'] == 1]

preds = predict_survival(trial_itt, newdata, predict_times=np.arange(0, 11))
print(preds.head())

# Plot survival difference and confidence bounds
plt.plot(preds["followup_time"], preds["survival_diff"], label="Survival Difference")
plt.plot(preds["followup_time"], preds["lower"], linestyle="--", color="red", label="2.5% CI")
plt.plot(preds["followup_time"], preds["upper"], linestyle="--", color="red", label="97.5% CI")
plt.xlabel("Follow up")
plt.ylabel("Survival difference")
plt.legend()
plt.show()
