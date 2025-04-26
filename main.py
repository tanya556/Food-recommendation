# main_pipeline.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Paths to your CSV files
DATASET_1 = 'FOOD-DATA-GROUP1.csv'
DATASET_2 = 'FOOD-DATA-GROUP2.csv'
MERGED_DATASET = 'merged_nutrition.csv'
LABELED_DATASET = 'labeled_nutrition.csv'
MODEL_FILE = 'nutrition_recommendation_model.pkl'

def merge_datasets():
    print("Merging datasets...")
    df1 = pd.read_csv(DATASET_1)
    df2 = pd.read_csv(DATASET_2)
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset='food')
    merged_df.to_csv(MERGED_DATASET, index=False)
    print(f"Merged dataset saved as {MERGED_DATASET} with {merged_df.shape[0]} foods.")
    return merged_df

def visualize_data(df):
    print("Visualizing dataset...")

    # Plot Calories Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df['Caloric Value'].dropna(), kde=True, color='skyblue')
    plt.title('Calories Distribution')
    plt.xlabel('Calories')
    plt.ylabel('Frequency')
    plt.show()

    # Plot Correlation Heatmap
    plt.figure(figsize=(10,8))
    features_to_correlate = ['Caloric Value', 'Fat', 'Carbohydrates', 'Sugars', 'Protein', 'Sodium']
    numeric_df = df[features_to_correlate].dropna()  # Only numeric columns
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Nutrition Feature Correlation')
    plt.show()


def label_data(df):
    print("Labeling data for health conditions...")
    df['good_for_diabetes'] = (df['Carbohydrates'] < 20) & (df['Sugars'] < 5)
    df['good_for_hypertension'] = (df['Sodium'] < 300) & (df['Cholesterol'] < 50)
    df['good_for_obesity'] = (df['Caloric Value'] < 400) & (df['Protein'] > 10)
    df.to_csv(LABELED_DATASET, index=False)
    print(f"Labeled dataset saved as {LABELED_DATASET}")
    return df

def train_model(df):
    print("Training the model...")
    features = ['Caloric Value', 'Fat', 'Carbohydrates', 'Sugars', 'Protein', 'Sodium']
    X = df[features]
    y = df[['good_for_diabetes', 'good_for_hypertension', 'good_for_obesity']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)

    print(f"Model saved as {MODEL_FILE}")
    print("Training Score:", model.score(X_train, y_train))
    print("Testing Score:", model.score(X_test, y_test))

def main():
    merged_df = merge_datasets()
    visualize_data(merged_df)
    labeled_df = label_data(merged_df)
    train_model(labeled_df)
    print("\n✅ All steps completed successfully!")

if __name__ == "__main__":
    main()
