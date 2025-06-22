from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

@app.route('/')
def index():
    df = pd.read_csv('static/titanic.csv')

    if not os.path.exists("static/visualizations"):
        os.makedirs("static/visualizations")

    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Survived')
    plt.title("Survival Count")
    plt.savefig('static/visualizations/survival_count.png')
    plt.clf()

    sns.histplot(data=df, x='Age', bins=30, kde=True)
    plt.title("Age Distribution")
    plt.savefig('static/visualizations/age_distribution.png')
    plt.clf()

    sns.countplot(data=df, x='Pclass', hue='Survived')
    plt.title("Class vs Survival")
    plt.savefig('static/visualizations/class_vs_survival.png')
    plt.clf()

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
