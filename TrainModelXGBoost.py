import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# 1. Load the recorded data
df_0 = pd.read_csv("dataset_label_0.csv", header=None)
df_1 = pd.read_csv("dataset_label_1.csv", header=None)
df_2 = pd.read_csv("dataset_label_2.csv", header=None)

# Merge datasets
df_all = pd.concat([df_0, df_1, df_2], ignore_index=True)

# 2. Split into features (X) and labels (y)
# The label is the last column
X = df_all.iloc[:, :-1].values
y = df_all.iloc[:, -1].values

# 3. Split into training, validation, and testing sets
# 70% training
# 15% validation  <- used during training
# 15% testing    <- final evaluation
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=145
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=5
)

# 4. Initialize and train XGBoost
print("Training XGBoost...")

model = xgb.XGBClassifier(

    # =========================
    # GENERAL PARAMETERS
    # =========================

    objective="multi:softprob",      # The task the model is solving.
                                      # "binary:logistic" = binary classification (0/1).
                                      # Outputs a probability between 0 and 1.

    num_class=3,

    booster="gbtree",                 # Type of model.
                                      # "gbtree"  -> decision trees (almost always used)
                                      # "dart"    -> trees with dropout
                                      # "gblinear"-> linear model instead of trees

    device="cuda",                     # Hardware used for training.
                                      # "cpu" for processor
                                      # "cuda" for NVIDIA GPU

    tree_method="hist",               # Algorithm used to build trees.
                                      # "auto" lets XGBoost choose.
                                      # Other options:
                                      # "hist", "approx", "exact"

    n_estimators=5000,                # Number of trees to build.
                                      # More trees usually increase accuracy
                                      # but also training time.

    learning_rate=0.005,                # How much each tree corrects the previous ones.
                                      # Lower values learn slower but usually generalize better.
                                      # Common values:
                                      # 0.3, 0.1, 0.05, 0.01

    random_state=1415,                # Random seed.
                                      # Set to an integer (e.g. 42)
                                      # for reproducible results.

    verbosity=1,                      # Amount of information printed.
                                      # 0 = silent
                                      # 1 = warnings
                                      # 2 = info
                                      # 3 = debug

    n_jobs=None,                      # Number of CPU threads.
                                      # None = default
                                      # -1 = use every CPU core

    # =========================
    # TREE COMPLEXITY
    # =========================

    max_depth=3,                      # Maximum depth of every tree.
                                      # Higher values allow more complex trees
                                      # but increase overfitting.

    min_child_weight=10,               # Minimum "weight" required before creating
                                      # another split.
                                      # Larger values make the tree simpler.

    gamma=5,                          # Minimum improvement required before
                                      # making a split.
                                      # Larger values create fewer branches.

    max_delta_step=0,                 # Maximum change allowed for leaf values.
                                      # Mostly useful for highly imbalanced datasets.

    max_leaves=0,                     # Maximum number of leaves.
                                      # 0 means unlimited.

    grow_policy="depthwise",          # How trees grow.
                                      # "depthwise" grows evenly.
                                      # "lossguide" grows where improvement is greatest.

    # =========================
    # DATA SAMPLING
    # =========================

    subsample=0.8,                    # Fraction of training samples used
                                      # for each tree.
                                      # 0.8 means every tree sees 80% of the data.

    sampling_method="uniform",        # How rows are sampled.
                                      # Usually leave as "uniform".

    colsample_bytree=0.8,             # Fraction of features used
                                      # when creating each tree.

    colsample_bylevel=1.0,            # Fraction of features used
                                      # at every tree level.

    colsample_bynode=1.0,             # Fraction of features used
                                      # for every individual split.

    # =========================
    # REGULARIZATION
    # =========================

    reg_alpha=1.0,                    # L1 regularization.
                                      # Encourages simpler models.
                                      # Higher values reduce overfitting.

    reg_lambda=10,                   # L2 regularization.
                                      # Penalizes very large weights.
                                      # Also helps reduce overfitting.

    # =========================
    # HISTOGRAM SETTINGS
    # =========================

    max_bin=512,                     # Number of bins used by histogram-based trees.
                                      # Larger values increase precision
                                      # but require more memory.

    # =========================
    # MISSING VALUES
    # =========================

    missing=np.nan,                   # Value treated as missing.
                                      # Usually leave as np.nan.

    # =========================
    # IMBALANCED DATASETS
    # =========================

    scale_pos_weight=1.0,             # Gives more importance to the positive class.
                                      # Useful when positives are rare.
                                      # Example:
                                      # 900 negatives / 100 positives
                                      # -> scale_pos_weight ≈ 9

    # =========================
    # CATEGORICAL FEATURES
    # =========================

    enable_categorical=False,         # Enables native categorical support.
                                      # Requires categorical columns.

    max_cat_to_onehot=None,           # Threshold deciding when one-hot encoding
                                      # is used for categorical variables.

    max_cat_threshold=None,           # Controls splitting strategy
                                      # for categorical features.

    # =========================
    # CONSTRAINTS
    # =========================

    monotone_constraints=None,        # Forces predictions to always increase
                                      # or decrease with selected features.

    interaction_constraints=None,     # Restricts which features
                                      # are allowed to interact together.

    # =========================
    # DART BOOSTER ONLY
    # =========================

    rate_drop=0.0,                    # Percentage of trees randomly dropped
                                      # during training.

    one_drop=False,                   # Forces at least one tree
                                      # to be dropped every iteration.

    skip_drop=0.0,                    # Probability of skipping dropout.

    sample_type="uniform",            # Tree selection method for dropout.

    normalize_type="tree",            # How dropped trees are normalized.

    # =========================
    # GBLINEAR BOOSTER ONLY
    # =========================

    updater=None,                     # Optimization algorithm
                                      # for linear booster.

    feature_selector=None,            # Strategy for selecting features
                                      # in the linear booster.

    top_k=0,                          # Number of top features considered
                                      # by some feature selectors.

    # =========================
    # EVALUATION
    # =========================

    eval_metric="aucpr",                 # Metric used during training.
                                      # Examples:
                                      # "logloss"
                                      # "auc"
                                      # "error"
                                      # "rmse"

    importance_type="gain",           # How feature importance is computed.
                                      #
                                      # "weight"    -> number of times used
                                      # "gain"      -> average improvement
                                      # "cover"     -> samples affected
                                      # "total_gain"
                                      # "total_cover"

    validate_parameters=True,         # Checks that all parameters are valid.
                                      # Recommended to keep True.

    # =========================
    # EXTRA PARAMETERS
    # =========================

    # **kwargs                          # Any additional parameters supported by
                                      # the underlying XGBoost library.
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test,y_test)],
    verbose=True
)

# 5. Test metrics
# model.predict automatically picks the class with the highest probability
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100}%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# For multi-class, you must specify an 'average' method
# 'macro' calculates metrics for each label, and finds their unweighted mean.
precision = precision_score(y_test, y_pred, average='macro')
print("Macro Precision:", precision)

recall = recall_score(y_test, y_pred, average='macro')
print("Macro Recall:", recall)



# 6. Save the model to a file so we can load it in real-time later
model.save_model("fitness_coach_xgboost.json")
print("Model saved to fitness_coach_xgboost.json")