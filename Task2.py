#Titanic EDA Task
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Optional, for interactive plots

#Load the Dataset
df = pd.read_csv("Titanic.csv")
print("First 5 Rows:\n", df.head())

#Generate Summary Statistics
print("\nSummary Statistics:\n", df.describe())

#Histogram and Boxplot for Numeric Features

sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.xlabel('Fare')
plt.show()

#Correlation Matrix
corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", corr_matrix)

#Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#Identifying patterns,trends,or anomalies in the data

#Survival rate by gender
survival_by_gender = df.groupby('Sex')['Survived'].mean()
print("\nSurvival Rate by Gender:\n", survival_by_gender)

#Barplot for gender vs survival
sns.barplot(x=survival_by_gender.index, y=survival_by_gender.values)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.show()