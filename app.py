from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

@app.route('/')
def home():
    # Load the Titanic dataset
    data = pd.read_csv('static/titanic.csv')

    # Create directory for storing plots if not already existing
    output_dir = "static/visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot 1: Survival count
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Survived', data=data)
    plt.title("Survival Overview")
    plt.savefig(os.path.join(output_dir, 'survival_count.png'))
    plt.clf()

    # Plot 2: Age distribution
    sns.histplot(data['Age'], bins=30, kde=True)
    plt.title("Distribution of Passenger Ages")
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
    plt.clf()

    # Plot 3: Passenger class vs survival
    sns.countplot(x='Pclass', hue='Survived', data=data)
    plt.title("Survival Rate by Class")
    plt.savefig(os.path.join(output_dir, 'class_vs_survival.png'))
    plt.clf()

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
