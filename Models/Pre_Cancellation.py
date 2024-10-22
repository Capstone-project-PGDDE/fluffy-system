
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/hotel_bookings.csv")
features = ['lead_time', 'hotel', 'market_segment', 'previous_cancellations', 
            'booking_changes', 'total_of_special_requests', 'arrival_date_month']
df_clean = df[features + ['is_canceled']].dropna()

label_encoders = {}
for col in ['hotel', 'market_segment', 'arrival_date_month']:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le
df_clean['is_canceled'] = df_clean['is_canceled'].apply(lambda x: 1 if x == 'yes' else 0)
X = df_clean[features]
y = df_clean['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_rep)
