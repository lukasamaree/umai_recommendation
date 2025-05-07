# Recipe Recommendation System

A collaborative filtering-based recommendation system for recipes using Funk SVD matrix factorization.

## Features

- Collaborative filtering using Funk SVD matrix factorization
- User and item bias terms for better accuracy
- Learning rate scheduling and early stopping
- Cross-validation for model evaluation
- Hyperparameter tuning
- Web interface for easy interaction
- REST API for integration with other systems

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Supabase credentials:
   ```
   SUPABASE_DEMO_URL=your_supabase_url
   SUPABASE_DEMO_API=your_supabase_api_key
   ```

## Usage

### Command Line Interface

Run the recommendation system from the command line:

```
python run_recommender.py
```

This will:
1. Load data from Supabase
2. Train the recommendation model
3. Evaluate the model on a test set
4. Generate recommendations for a few users
5. Optionally tune hyperparameters


## Model Details

The recommendation system uses Funk SVD matrix factorization with the following features:

- User and item bias terms to capture systematic rating patterns
- Learning rate scheduling to improve convergence
- Early stopping to prevent overfitting
- Regularization to prevent overfitting
- Cross-validation for robust evaluation

## Hyperparameter Tuning

The system supports hyperparameter tuning for:

- Number of latent factors (k)
- Learning rate (alpha)
- Regularization strength (beta)

You can tune hyperparameters using the command line interface or by calling the `tune_hyperparameters` method directly.

# Workflow 

- This script is triggered once a day on Google Cloud to our Supabase Instance where recommendations are read via SQL for each user.