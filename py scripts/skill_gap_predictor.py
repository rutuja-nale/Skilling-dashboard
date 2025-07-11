
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_excel("employee_skills_sample.xlsx")

# Feature engineering
df['Needs_Upskilling'] = df['Skill_Status'].apply(lambda x: 1 if x == 'To Be Skilled' else 0)

# Encode categorical features
df_encoded = pd.get_dummies(df[['Department', 'Current_Technology', 'Target_Technology']])
X = pd.concat([df[['Experience_Years']], df_encoded], axis=1)
y = df['Needs_Upskilling']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on full data
df['Prediction'] = model.predict(X)
df['Prediction_Label'] = df['Prediction'].apply(lambda x: "Likely Upskilling" if x == 1 else "Not Required")

# Save output
df.to_excel("employee_skills_predictions.xlsx", index=False)
print("Prediction saved to 'employee_skills_predictions.xlsx'")
