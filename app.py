from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

@app.route('/')
def home():
    # Load dataset
    data = pd.read_csv('static/titanic.csv')

    # Ensure visualization directory exists
    vis_dir = "static/visualizations"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # Plot 1: Survival Count
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Survived', data=data)
    plt.title("Survival Overview")
    plt.savefig(f"{vis_dir}/survival_count.png")
    plt.clf()

    # Plot 2: Age Distribution
    sns.histplot(data['Age'], bins=30, kde=True)
    plt.title("Passenger Age Distribution")
    plt.savefig(f"{vis_dir}/age_distribution.png")
    plt.clf()

    # Plot 3: Class vs Survival
    sns.countplot(x='Pclass', hue='Survived', data=data)
    plt.title("Passenger Class vs Survival")
    plt.savefig(f"{vis_dir}/class_vs_survival.png")
    plt.clf()

    # Plot 4: Gender Count
    sns.countplot(x='Sex', data=data)
    plt.title("Gender Distribution")
    plt.savefig(f"{vis_dir}/gender_distribution.png")
    plt.clf()

    # Plot 5: Gender vs Survival
    sns.countplot(x='Sex', hue='Survived', data=data)
    plt.title("Gender vs Survival")
    plt.savefig(f"{vis_dir}/gender_vs_survival.png")
    plt.clf()

    # Plot 6: Fare Distribution
    sns.histplot(data['Fare'], bins=40, kde=True)
    plt.title("Fare Distribution")
    plt.savefig(f"{vis_dir}/fare_distribution.png")
    plt.clf()

    # Plot 7: Embarked Port Count
    sns.countplot(x='Embarked', data=data)
    plt.title("Embarkation Port Distribution")
    plt.savefig(f"{vis_dir}/embarked_distribution.png")
    plt.clf()

    # Plot 8: Embarked vs Survival
    sns.countplot(x='Embarked', hue='Survived', data=data)
    plt.title("Survival by Embarkation Port")
    plt.savefig(f"{vis_dir}/embarked_vs_survival.png")
    plt.clf()

    # Plot 9: Age vs Fare (scatter)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='Age', y='Fare', hue='Survived', data=data)
    plt.title("Age vs Fare (Colored by Survival)")
    plt.savefig(f"{vis_dir}/age_vs_fare.png")
    plt.clf()

    # Plot 10: Heatmap of Correlation
    plt.figure(figsize=(6, 4))
    corr = data[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{vis_dir}/correlation_heatmap.png")
    plt.clf()

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
