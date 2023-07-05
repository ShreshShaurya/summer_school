import pandas as pd

# Load the data
data = pd.read_csv('loan_dataset.csv')

# Perform preprocessing steps

# Remove missing values
data = data.dropna()

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)

# Normalize numeric features
numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
data[numeric_features] = (data[numeric_features] - data[numeric_features].mean()) / data[numeric_features].std()

# Convert target variable to numerical
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

# Save preprocessed data to a new file
data.to_csv('preprocessed_data.csv', index=False)
