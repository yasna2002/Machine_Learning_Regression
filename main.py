import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('CreditPrediction.csv')

# all the data in 'Unnamed: 19' column is zero, so we delete this column
df.drop('Unnamed: 19', axis=1, inplace=True)
df.drop('CLIENTNUM', axis=1, inplace=True)

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Marital_Status'] = df['Marital_Status'].fillna(df['Marital_Status'].mode()[0])
df['Card_Category'] = df['Card_Category'].fillna(df['Card_Category'].mode()[0])
df['Months_on_book'] = df['Months_on_book'].fillna(df['Months_on_book'].mean().round())
df['Total_Relationship_Count'] = df['Total_Relationship_Count'].fillna(df['Total_Relationship_Count'].mean().round())

df = df.drop_duplicates()

# Encoding Sequential Features Using Mapping
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

df['Education_Level'] = df['Education_Level'].replace("Unknown", df['Education_Level'].mode()[0])
df_encoded = pd.get_dummies(df['Education_Level'], prefix='Education_Level')
df_encoded = df_encoded.astype(int)
df = pd.concat([df, df_encoded], axis=1)
df.drop('Education_Level', axis=1, inplace=True)

df['Income_Category'] = df['Income_Category'].replace("Unknown", df['Income_Category'].mode()[0])
df_encoded = pd.get_dummies(df['Income_Category'], prefix='Income_Category')
df_encoded = df_encoded.astype(int)
df = pd.concat([df, df_encoded], axis=1)
df.drop('Income_Category', axis=1, inplace=True)

df['Marital_Status'] = df['Marital_Status'].replace("Unknown", df['Marital_Status'].mode()[0])
df_encoded = pd.get_dummies(df['Marital_Status'], prefix='Marital_Status')
df_encoded = df_encoded.astype(int)
df = pd.concat([df, df_encoded], axis=1)
df.drop('Marital_Status', axis=1, inplace=True)

df_encoded = pd.get_dummies(df['Card_Category'], prefix='Card_Category')
df_encoded = df_encoded.astype(int)
df = pd.concat([df, df_encoded], axis=1)
df.drop('Card_Category', axis=1, inplace=True)


# Defining a function to replace outliers with the mode of each column using IQR
def replace_outliers_with_mode_iqr(data):
    # Initialize an empty DataFrame to store the results
    result = pd.DataFrame(index=data.index, columns=data.columns)

    # Loop through each column in the DataFrame
    for col in data.columns:
        Q1 = data[col].quantile(0.25)  # 25th percentile (Q1)
        Q3 = data[col].quantile(0.75)  # 75th percentile (Q3)
        IQR = Q3 - Q1  # Interquartile Range (IQR)
        lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
        upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers

        # Replace outliers with mode for the current column
        mode_val = data.loc[(data[col] >= lower_bound) & (data[col] <= upper_bound), col].mean()
        result[col] = data[col].apply(lambda x: mode_val if x < lower_bound or x > upper_bound else x)

    return result

# Apply the function to replace outliers with mode for each column separately
df = replace_outliers_with_mode_iqr(df)



Y = df['Credit_Limit']
df.drop('Credit_Limit', axis=1, inplace=True)

X = df

linear_regression_y = []

poly_y = []

GB_y = []

RF_y = []

for _ in range(10):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    scaler = StandardScaler()

    # Apply Min-Max scaling to the DataFrame
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Linear Regression

    linear_regression = LinearRegression()

    linear_regression.fit(X_train, Y_train)

    Y_pred_LR = linear_regression.predict(X_test)

    mse_LR = mean_squared_error(Y_test, Y_pred_LR)

    print("MSE : ", mse_LR)

    # Polynomial Regression with Ridge

    poly = PolynomialFeatures(degree=2)

    X_train_poly = poly.fit_transform(X_train)

    X_test_poly = poly.transform(X_test)

    ridge = Ridge()
    ridge.fit(X_train_poly, Y_train)

    Y_pred_poly = ridge.predict(X_test_poly)

    mse_poly = mean_squared_error(Y_test, Y_pred_poly)

    print("\nMSE Poly : ", mse_poly)

    # Gradient Boosting

    gb = GradientBoostingRegressor()

    gb.fit(X_train, Y_train)

    Y_pred_gb = gb.predict(X_test)

    mse_gb = mean_squared_error(Y_test, Y_pred_gb)

    print("\nMSE GB : ", mse_gb)

    # Random Forest

    random_forest = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=10)

    random_forest.fit(X_train, Y_train)

    Y_pred_rf = random_forest.predict(X_test)

    mse_rf = mean_squared_error(Y_test, Y_pred_rf)

    print("\nMSE RF : ", mse_rf)

    linear_regression_y.append(round(mse_LR, 0))

    poly_y.append(round(mse_poly, 0))

    GB_y.append(round(mse_gb, 0))

    RF_y.append(round(mse_rf, 0))

linear_regression_y = np.array(linear_regression_y)
mean_LR = linear_regression_y.mean()

poly_y = np.array(poly_y)
mean_poly = poly_y.mean()

GB_y = np.array(GB_y)
mean_GB = GB_y.mean()

RF_y = np.array(RF_y)
mean_RF = RF_y.mean()

print("\nLinear Regression mean : ", mean_LR)
print("\nPolynomial Regression mean : ", mean_poly)
print("\nGradient Boosting mean : ", mean_GB)
print("\nRandom Forest mean : ", mean_RF)

plt.plot(linear_regression_y, marker='o', color='r', label='Linear Regression - '+str(mean_LR))

plt.plot(poly_y, marker='*', color='blue', label='Polynomial Regression - '+str(mean_poly))

plt.plot(GB_y, marker='H', color='green', label='Gradient Boosting - '+str(mean_GB))

plt.plot(RF_y, marker='D', color='hotpink', label='Random Forest - '+str(mean_RF))

plt.title("4 MODELS MSE")
plt.xlabel("Data Sample")
plt.ylabel("MSE")
plt.grid()
plt.legend()
plt.show()
