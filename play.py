import pandas as pd

# Sample dataset
data = {
    'age': [25, 30, 22, 28, 35],
    'gender': ['M', 'F', 'M', 'M', 'F'],
    'income': [50000, 60000, 45000, 55000, 70000],
    'is_student': [False, False, True, False, True],
    'class': ['A', 'B', 'A', 'A', 'B']
}

# Convert the data dictionary into a DataFrame
df = pd.DataFrame(data)

# Determine feature types
feature_types = {}
for column in df.columns:
    if df[column].dtype == 'int64' or df[column].dtype == 'float64':
        feature_types[column] = 'Numeric'
    elif df[column].dtype == 'object':
        unique_values = df[column].nunique()
        if unique_values <= 5:
            feature_types[column] = 'Categorical (Low Cardinality)'
        else:
            feature_types[column] = 'Categorical (High Cardinality)'
    elif df[column].dtype == 'bool':
        feature_types[column] = 'Boolean'

print("Feature Types:")
for feature, feature_type in feature_types.items():
    print(f"{feature}: {feature_type}")
