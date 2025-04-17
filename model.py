import numpy as np  # Numerical Python library for linear algebra and computations
import pandas as pd  # Python library for data analysis and data frame

# Set option to display all columns
pd.set_option('display.max_columns', None)

# Libraries for handling categorical column and scaling numeric columns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Libraries for clustering and evaluation
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

import warnings
warnings.filterwarnings("ignore")  # To prevent kernel from showing any warning

# Example dataset loading and preprocessing
# Load dataset
df = pd.read_csv('./marketing_campaign.csv', sep='\t')

# Data Cleaning
df.rename(columns = {'MntGoldProds':'MntGoldProducts'}, inplace = True)
# converting columns to DateTime format
df['Year_Birth'] = pd.to_datetime(df['Year_Birth'], format ='%Y')
df['Year_Birth'] = pd.to_datetime(df['Year_Birth'], format ='%Y')

df['Income'].skew()
df['Income'].fillna(df['Income'].median(), inplace = True)

# Convert Dt_Customer to datetime with error handling
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')

# Drop rows with NaT (Not a Time)
df = df.dropna(subset=['Dt_Customer'])

# Extract the year and find min and max
min_year = df["Dt_Customer"].dt.year.min()
max_year = df["Dt_Customer"].dt.year.max()

min_year, max_year

# Creating Age and Years_Customer (Amount of years a person has been a customer) columns.
df['Age'] = (df["Dt_Customer"].dt.year.max()) - (df['Year_Birth'].dt.year)
df['Years_Customer'] = (df["Dt_Customer"].dt.year.max()) - (df['Dt_Customer'].dt.year)
df['Days_Customer'] = (df["Dt_Customer"].max()) - (df["Dt_Customer"])

# Total amount spent on products
df['TotalMntSpent'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProducts']

# Total number of purchases made
df['TotalNumPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']

# Total number of accepted campaigns
df['Total_Acc_Cmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']

# adding columns about the day, month, and year customer joined
df['Year_Joined'] = df['Dt_Customer'].dt.year
df['Month_Joined'] = df['Dt_Customer'].dt.strftime("%B")
df['Day_Joined'] = df['Dt_Customer'].dt.day_name()

# dividing age into groups
df['Age_Group'] = pd.cut(x = df['Age'], bins = [17, 24, 44, 64, 150],
                         labels = ['Young adult','Adult','Middle Aged','Senior Citizen'])
# Total children living in the household
df["Children"] = df["Kidhome"] +  df["Teenhome"]

# Deriving living situation by marital status
df["Partner"]=df["Marital_Status"].replace({"Married":"Yes", "Together":"Yes", "Absurd":"No", "Widow":"No", "YOLO":"No", "Divorced":"No", "Single":"No","Alone":"No"})

# Segmenting education levels into three groups
df["Education_Level"]=df["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

# Dropping useless columns
df.drop(['ID','Z_CostContact','Z_Revenue','Year_Birth','Dt_Customer'], axis=1, inplace=True)

# Converting Days_Joined to int format
df['Days_Customer'] = df['Days_Customer'].dt.days.astype('int16')

df1 = df.copy()  # make a copy
df1.drop(['Education','Marital_Status','Years_Customer','Year_Joined','Month_Joined','Day_Joined'], axis=1, inplace=True)

num_col = df1.select_dtypes(include = np.number).columns

# Handle outliers
for col in num_col:
    q1 = df1[col].quantile(0.25)
    q3 = df1[col].quantile(0.75)
    iqr = q3 - q1
    ll = q1 - (1.5 * iqr)
    ul = q3 + (1.5 * iqr)
    for ind in df1[col].index:
        if df1.loc[ind, col] > ul:
            df1.loc[ind, col] = ul
        elif df1.loc[ind, col] < ll:
            df1.loc[ind, col] = ll
        else:
            pass
print("Outliers have been taken care of")

# Selecting the columns to use for clustering
subset = df1[['Income','Kidhome','Teenhome','Age','Partner','Education_Level']]
print('This is the data we will use for clustering:')
subset.head()

# Importing essential libraries for building pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# Skipping scaling for kidhome, teenhome cols, as their range is between 0 & 2
num_cols = ['Income', 'Age']
numeric_pipeline = make_pipeline(StandardScaler())

ord_cols = ['Education_Level']
ordinal_pipeline = make_pipeline(OrdinalEncoder(categories=[['Undergraduate','Graduate','Postgraduate']]))

nom_cols = ['Partner']
nominal_pipeline = make_pipeline(OneHotEncoder())

# Stack the pipelines in a column transformer
transformer = ColumnTransformer(transformers=[('num', numeric_pipeline, num_cols),
                                              ('ordinal', ordinal_pipeline, ord_cols),
                                              ('nominal', nominal_pipeline, nom_cols)])

# Fit and transform the data
transformed = transformer.fit_transform(subset)
print('Data has been Transformed')

# K-Means Clustering & Cluster's Analysis

# Using k-means to form clusters
kmeans = KMeans(n_clusters=4, random_state=42)
subset['Clusters'] = kmeans.fit_predict(transformed)  # Fit the data and adding clusters back to the data in the 'Clusters' column

# Import required libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Separate features and target column
x = subset.drop('Clusters', axis=1)
y = subset['Clusters']

# Create train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Adding GradientBoostingClassifier to the transformer pipeline
final_pipeline = make_pipeline(transformer, GradientBoostingClassifier())

# Fit the data to the new pipeline & model
final_pipeline.fit(x_train, y_train)

# Check the accuracy of the model
final_pipeline.score(x_test, y_test)

# Save the model using pickle
import pickle
filename = 'classifier.pkl'  # Create a variable with the name you want to give to the file
pickle.dump(final_pipeline, open(filename, 'wb'))
