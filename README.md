# Telecom Customer Churn Analysis

This project analyzes telecom customer churn data to identify patterns and predict which customers are likely to churn. It uses machine learning models allowed for Project 1: Linear Regression, Logistic Regression, SVM, Decision Trees, and KNN.

## Setup and Running the Analysis

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analysis script:
   ```bash
   python telco_churn_analysis.py
   ```

3. The script will:
   - Load and explore the data
   - Preprocess the data
   - Perform exploratory data analysis
   - Build and evaluate multiple models
   - Optimize the best-performing model
   - Generate business insights

## Generated Visualizations

After running the script, you'll find several visualizations saved to the project directory:
- `churn_by_categories.png`: Churn by categorical variables (Contract, Internet Type, etc.)
- `churn_by_numerics.png`: Churn by numeric variables (Monthly Charge, Tenure, etc.)
- `correlation_matrix.png`: Correlation matrix of numeric features
- `roc_curves.png`: ROC curves for different models
- Model feature importance charts
- `churn_by_tenure.png`: Churn rate by tenure group
- `top_churn_reasons.png`: Top 10 churn reasons
- `churn_by_charge.png`: Churn rate by monthly charge group

## Data Story

The analysis tells a story about customer churn in the telecom industry:
1. Identifies which customers are most likely to churn
2. Determines the key factors influencing churn
3. Shows when in the customer lifecycle churn is most likely
4. Identifies the most common reasons for churn
5. Provides a basis for targeted retention strategies

## Project Structure

Data + Pre-processing/exploration that we began
