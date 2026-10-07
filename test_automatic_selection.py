import sys, os, pandas as pd

PROJECT_ROOT = r'C:\Users\Sukesh\.gemini\antigravity\scratch\credit-card-fraud-detection'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'app'))

import utils
import predictor

print("Loading cached reference stats and results...")
ref13, ref23 = utils.get_cached_reference_stats()
comp_df, cv_df, dist_df = utils.load_results()
m13 = utils.load_model("2013", "Random Forest")
m23 = utils.load_model("2023", "Random Forest")
sample_models = {"2013": m13, "2023": m23}

# Test 1: Upload 2013 CSV Sample
df13_sample = utils.load_dataset_sample("2013", sample_size=100)
dec13 = predictor.evaluate_automatic_selection(df13_sample, ref13, ref23, comp_df, sample_models)

print("\n=== TEST 1: 2013 CSV Upload ===")
print("Selected Dataset:", dec13["selected_dataset"])
print("Selected Model:", dec13["selected_model"])
print("Overall Compatibility (2013 vs 2023):", f"{dec13['overall_13']}% vs {dec13['overall_23']}%")
print("Dataset Reason:", dec13["dataset_explanation"])
print("Model Reason:", dec13["model_explanation"])
assert dec13["selected_dataset"] == "2013", "Expected 2013 selection!"

# Test 2: Upload 2023 CSV Sample
df23_sample = utils.load_dataset_sample("2023", sample_size=100)
dec23 = predictor.evaluate_automatic_selection(df23_sample, ref13, ref23, comp_df, sample_models)

print("\n=== TEST 2: 2023 CSV Upload ===")
print("Selected Dataset:", dec23["selected_dataset"])
print("Selected Model:", dec23["selected_model"])
print("Overall Compatibility (2013 vs 2023):", f"{dec23['overall_13']}% vs {dec23['overall_23']}%")
print("Dataset Reason:", dec23["dataset_explanation"])
print("Model Reason:", dec23["model_explanation"])
assert dec23["selected_dataset"] == "2023", "Expected 2023 selection!"

# Test 3: Upload 2015 CSV Sample (Schema/distribution matches 2013 with timestamp shift)
df15_sample = df13_sample.copy()
if "Time" in df15_sample.columns:
    df15_sample["Time"] = df15_sample["Time"] + 63072000 # 2 years shift
dec15 = predictor.evaluate_automatic_selection(df15_sample, ref13, ref23, comp_df, sample_models)

print("\n=== TEST 3: 2015 CSV Upload ===")
print("Selected Dataset:", dec15["selected_dataset"])
print("Selected Model:", dec15["selected_model"])
print("Overall Compatibility (2013 vs 2023):", f"{dec15['overall_13']}% vs {dec15['overall_23']}%")
print("Dataset Reason:", dec15["dataset_explanation"])
print("Model Reason:", dec15["model_explanation"])

print("\nAll automatic selection tests passed successfully!")
