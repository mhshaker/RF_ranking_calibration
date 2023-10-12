import Data.data_provider as dp 
import pandas as pd

X, y = dp.load_data("kc2", ".")
print("shape X", y.shape)


# Sample DataFrame with categorical features
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small']
}

df = pd.DataFrame(X)
print(df.head())

# # Perform one-hot encoding on categorical columns
# categorical_columns = ['Color', 'Size']
# df_encoded = pd.get_dummies(df, columns=categorical_columns)

# # The resulting DataFrame df_encoded will have one-hot encoded columns
# print(df_encoded)
