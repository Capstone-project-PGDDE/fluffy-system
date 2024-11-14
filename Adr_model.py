import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv("data/hotel_bookings.csv")

# Categorical Encoding and Data Preparation for Cancellation Prediction
features = ['lead_time', 'hotel', 'market_segment', 'previous_cancellations', 
            'booking_changes', 'total_of_special_requests', 'arrival_date_month']
df_cancellation = df[features + ['is_canceled']].dropna()

label_encoders = {}
for col in ['hotel', 'market_segment', 'arrival_date_month']:
    le = LabelEncoder()
    df_cancellation[col] = le.fit_transform(df_cancellation[col])
    label_encoders[col] = le
df_cancellation['is_canceled'] = df_cancellation['is_canceled'].apply(lambda x: 1 if x == 'yes' else 0)

# Split data into training and testing sets for Cancellation Prediction
X_cancel = df_cancellation[features]
y_cancel = df_cancellation['is_canceled']
X_train_cancel, X_test_cancel, y_train_cancel, y_test_cancel = train_test_split(X_cancel, y_cancel, test_size=0.3, random_state=42)

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train_cancel, X_test_cancel, y_train_cancel, y_test_cancel):
    model.fit(X_train_cancel, y_train_cancel)
    y_pred_cancel = model.predict(X_test_cancel)
    accuracy = accuracy_score(y_test_cancel, y_pred_cancel)
    report = classification_report(y_test_cancel, y_pred_cancel)
    return accuracy, report

# Function to define objective for Optuna hyperparameter tuning
def objective(trial, model_class, X_train, y_train):
    if model_class == RandomForestClassifier:
        n_estimators = trial.suggest_int('n_estimators', 100, 300)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       random_state=42)
    elif model_class == GradientBoostingClassifier:
        n_estimators = trial.suggest_int('n_estimators', 100, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                           max_depth=max_depth, random_state=42)
    elif model_class == LogisticRegression:
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    elif model_class == DecisionTreeClassifier:
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    
    model.fit(X_train, y_train)
    return model.score(X_train, y_train)

# Models to evaluate
models = {
    'Random Forest': RandomForestClassifier,
    'Logistic Regression': LogisticRegression,
    'Decision Tree': DecisionTreeClassifier,
    'Gradient Boosting': GradientBoostingClassifier
}

# Hyperparameter tuning with Optuna for Cancellation Prediction
best_models = {}
best_accuracy = 0
best_model_name = None
best_model = None
for model_name, model_class in models.items():
    print(f"Tuning hyperparameters for {model_name} using Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_class, X_train_cancel, y_train_cancel), n_trials=20)
    best_params = study.best_params
    print(f"Best hyperparameters for {model_name}: {best_params}")
    
    # Train model with best hyperparameters
    if model_class == RandomForestClassifier:
        model = RandomForestClassifier(**best_params, random_state=42)
    elif model_class == GradientBoostingClassifier:
        model = GradientBoostingClassifier(**best_params, random_state=42)
    elif model_class == LogisticRegression:
        model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
    elif model_class == DecisionTreeClassifier:
        model = DecisionTreeClassifier(**best_params, random_state=42)
    
    model.fit(X_train_cancel, y_train_cancel)
    accuracy, report = train_and_evaluate_model(model, X_train_cancel, X_test_cancel, y_train_cancel, y_test_cancel)
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)
    print("-" * 60)
    
    # Save the model with the highest accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model

# Save the best model for Cancellation Prediction
if best_model is not None:
    file_name = f'models/{best_model_name.lower().replace(" ", "_")}_best_model.pkl'
    joblib.dump(best_model, file_name)
    print(f"Best model '{best_model_name}' saved to {file_name} with accuracy: {best_accuracy * 100:.2f}%")

# Categorical Encoding and Data Preparation for ADR Prediction
adr_features = ['hotel', 'lead_time', 'market_segment', 'arrival_date_month',
                'previous_cancellations', 'booking_changes', 'total_of_special_requests', 'is_repeated_guest']
adr_df = df[adr_features + ['adr']].dropna()
for col in ['hotel', 'market_segment', 'arrival_date_month']:
    adr_df[col] = label_encoders[col].transform(adr_df[col])

# Split data into training and testing sets for ADR Prediction
X_adr = adr_df[adr_features]
y_adr = adr_df['adr']
X_train_adr, X_test_adr, y_train_adr, y_test_adr = train_test_split(X_adr, y_adr, test_size=0.3, random_state=42)

# Function to train and evaluate regression models
def train_and_evaluate_regression(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

# Regression models to evaluate
regression_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}

# Train and evaluate each regression model for ADR Prediction
print("\nAverage Daily Rate (ADR) Prediction Results:\n")
best_regression_model = None
best_r2 = -np.inf
best_regression_model_name = None
for model_name, model in regression_models.items():
    mae, rmse, r2 = train_and_evaluate_regression(model, X_train_adr, X_test_adr, y_train_adr, y_test_adr)
    print(f"Model: {model_name}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")
    print("-" * 60)
    
    # Save the model with the best R² score
    if r2 > best_r2:
        best_r2 = r2
        best_regression_model = model
        best_regression_model_name = model_name

# Save the best regression model for ADR Prediction
if best_regression_model is not None:
    file_name = f'models/{best_regression_model_name.lower().replace(" ", "_")}_best_model.pkl'
    joblib.dump(best_regression_model, file_name)
    print(f"Best regression model '{best_regression_model_name}' saved to {file_name} with R²: {best_r2:.2f}")
