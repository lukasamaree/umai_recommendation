import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from postgrest.exceptions import APIError
import pprint


class RecipeRecommender:
    def __init__(self, k=30, epochs=100, alpha=0.001, beta=0.01, verbose=True,supabase=None):
        self.k = k
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        self.P = None
        self.Q = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.best_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0
        self.user_map = None
        self.item_map = None
        self.supabase = supabase
        
    def preprocess_data(self, df):
        """Preprocess the ratings dataframe to create user and item mappings and rating matrix"""
        # Create user and item mappings
        user_ids = df['user_id'].unique()
        item_ids = df['recipe_id'].unique()
        self.user_map = {uid: i for i, uid in enumerate(user_ids)}
        self.item_map = {iid: i for i, iid in enumerate(item_ids)}
        
        # Create rating matrix
        R = np.zeros((len(user_ids), len(item_ids)))
        for _, row in df.iterrows():
            R[self.user_map[row['user_id']], self.item_map[row['recipe_id']]] = row['rating']
        
        return R
    
    def fit(self, ratings_df):
        """Train the recommendation model"""
        # Preprocess data
        R = self.preprocess_data(ratings_df)
        self.users, self.items = R.shape
        
        # Initialize global mean
        known_ratings = R[R > 0]
        self.global_mean = np.mean(known_ratings) if len(known_ratings) > 0 else 0
        
        # Initialize biases
        self.user_bias = np.zeros(self.users)
        self.item_bias = np.zeros(self.items)
        
        # Calculate initial biases
        for i in range(self.users):
            user_ratings = R[i, R[i] > 0]
            if len(user_ratings) > 0:
                self.user_bias[i] = np.mean(user_ratings) - self.global_mean
        
        for j in range(self.items):
            item_ratings = R[R[:, j] > 0, j]
            if len(item_ratings) > 0:
                self.item_bias[j] = np.mean(item_ratings) - self.global_mean
        
        # Initialize latent factors with better scaling
        self.P = np.random.normal(scale=0.1, size=(self.users, self.k))
        self.Q = np.random.normal(scale=0.1, size=(self.items, self.k))
        
        known_indices = np.argwhere(R > 0)
        n_samples = len(known_indices)
        
        # Learning rate schedule
        initial_lr = self.alpha
        min_lr = initial_lr * 0.01
        
        for epoch in range(self.epochs):
            np.random.shuffle(known_indices)
            epoch_loss = 0
            
            # Decay learning rate
            current_lr = max(min_lr, initial_lr * (1.0 / (1.0 + epoch * 0.1)))
            
            for i, j in known_indices:
                # Calculate prediction with bias terms
                pred = (self.global_mean + 
                       self.user_bias[i] + 
                       self.item_bias[j] + 
                       np.dot(self.P[i, :], self.Q[j, :]))
                
                error = R[i, j] - pred
                epoch_loss += error ** 2
                
                # Update latent factors
                self.P[i, :] += current_lr * (error * self.Q[j, :] - self.beta * self.P[i, :])
                self.Q[j, :] += current_lr * (error * self.P[i, :] - self.beta * self.Q[j, :])
                
                # Update bias terms
                self.user_bias[i] += current_lr * (error - self.beta * self.user_bias[i])
                self.item_bias[j] += current_lr * (error - self.beta * self.item_bias[j])
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / n_samples
            
            # Early stopping
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            if self.verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}: MSE = {avg_loss:.4f}")
    
    def predict(self, user_id=None, recipe_id=None):
        """Predict ratings for all user-item pairs or a specific user-item pair"""
        if user_id is not None and recipe_id is not None:
            # Predict for a specific user-item pair
            if user_id in self.user_map and recipe_id in self.item_map:
                u = self.user_map[user_id]
                i = self.item_map[recipe_id]
                return (self.global_mean + 
                        self.user_bias[u] + 
                        self.item_bias[i] + 
                        np.dot(self.P[u, :], self.Q[i, :]))
            else:
                return None
        else:
            # Predict for all user-item pairs
            return (self.global_mean + 
                    self.user_bias[:, np.newaxis] + 
                    self.item_bias[np.newaxis, :] + 
                    np.dot(self.P, self.Q.T))
    
    def get_top_recommendations(self, user_id, n=5):
        """Get top N recipe recommendations for a user"""
        if user_id not in self.user_map:
            return []
        
        # Get all predictions for this user
        user_idx = self.user_map[user_id]
        predictions = self.predict()[user_idx]
        
        # Get indices of top N items
        top_indices = np.argsort(predictions)[-n:][::-1]
        
        # Map back to recipe IDs
        reverse_item_map = {v: k for k, v in self.item_map.items()}
        recommendations = []
        
        for idx in top_indices:
            recipe_id = reverse_item_map[idx]
            predicted_rating = predictions[idx]
            recommendations.append((recipe_id, predicted_rating))
        
        return recommendations
    
    def evaluate(self, test_df):
        """Evaluate the model on a test set"""
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            if row['user_id'] in self.user_map and row['recipe_id'] in self.item_map:
                pred = self.predict(row['user_id'], row['recipe_id'])
                predictions.append(pred)
                actuals.append(row['rating'])
        
        if len(predictions) == 0:
            return None, None, None
        
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        
        return mse, rmse, mae
    
    def cross_validate(self, df, k=5):
        """Perform k-fold cross-validation"""
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        mse_scores = []
        rmse_scores = []
        mae_scores = []
        
        for train_idx, val_idx in kf.split(df):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            # Train model
            self.fit(train_df)
            
            # Evaluate
            mse, rmse, mae = self.evaluate(val_df)
            if mse is not None:
                mse_scores.append(mse)
                rmse_scores.append(rmse)
                mae_scores.append(mae)
        
        return {
            'mse': np.mean(mse_scores),
            'rmse': np.mean(rmse_scores),
            'mae': np.mean(mae_scores),
            'mse_std': np.std(mse_scores),
            'rmse_std': np.std(rmse_scores),
            'mae_std': np.std(mae_scores)
        }
    
    def tune_hyperparameters(self, df, k_vals=[10, 20, 30, 50], 
                            alpha_vals=[0.0001, 0.0005, 0.001, 0.005], 
                            beta_vals=[0.001, 0.005, 0.01, 0.02]):
        """Tune hyperparameters using grid search with cross-validation"""
        best_score = float('inf')
        best_params = None
        results = []
        
        for k in k_vals:
            for alpha in alpha_vals:
                for beta in beta_vals:
                    # Create model with current parameters
                    model = RecipeRecommender(k=k, alpha=alpha, beta=beta, epochs=50, verbose=False)
                    
                    # Perform cross-validation
                    cv_results = model.cross_validate(df, k=3)  # Use 3-fold CV for speed
                    
                    if cv_results['rmse'] < best_score:
                        best_score = cv_results['rmse']
                        best_params = (k, alpha, beta)
                    
                    results.append({
                        'k': k,
                        'alpha': alpha,
                        'beta': beta,
                        'rmse': cv_results['rmse'],
                        'mae': cv_results['mae']
                    })
        
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Create heatmap visualizations for better interpretability
        
        # 1. Heatmap for k vs alpha (averaging over beta)
        plt.figure(figsize=(10, 8))
        pivot_k_alpha = results_df.pivot_table(
            values='rmse', 
            index='alpha', 
            columns='k', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_k_alpha, annot=True, fmt='.4f', cmap='YlGnBu_r', 
                    cbar_kws={'label': 'RMSE'})
        plt.title('RMSE for Different k and α Values (Averaged over β)')
        plt.xlabel('Number of Latent Factors (k)')
        plt.ylabel('Learning Rate (α)')
        
        # 2. Heatmap for k vs beta (averaging over alpha)
        plt.figure(figsize=(10, 8))
        pivot_k_beta = results_df.pivot_table(
            values='rmse', 
            index='beta', 
            columns='k', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_k_beta, annot=True, fmt='.4f', cmap='YlGnBu_r', 
                    cbar_kws={'label': 'RMSE'})
        plt.title('RMSE for Different k and β Values (Averaged over α)')
        plt.xlabel('Number of Latent Factors (k)')
        plt.ylabel('Regularization (β)')
        
        # 3. Heatmap for alpha vs beta (averaging over k)
        plt.figure(figsize=(10, 8))
        pivot_alpha_beta = results_df.pivot_table(
            values='rmse', 
            index='alpha', 
            columns='beta', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_alpha_beta, annot=True, fmt='.4f', cmap='YlGnBu_r', 
                    cbar_kws={'label': 'RMSE'})
        plt.title('RMSE for Different α and β Values (Averaged over k)')
        plt.xlabel('Regularization (β)')
        plt.ylabel('Learning Rate (α)')
        
        # 4. 3D scatter plot for all parameters
        plt.figure(figsize=(12, 10))
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(
            results_df['k'], 
            results_df['alpha'], 
            results_df['beta'],
            c=results_df['rmse'],
            cmap='YlGnBu_r',
            s=100
        )
        plt.colorbar(scatter, label='RMSE')
        ax.set_xlabel('Number of Latent Factors (k)')
        ax.set_ylabel('Learning Rate (α)')
        ax.set_zlabel('Regularization (β)')
        ax.set_title('3D Visualization of Hyperparameter Impact on RMSE')
        
        return best_params, results_df
    
    def get_all_recommendations(self, recipes_df):
        """Get recommendations for all users and return as a DataFrame
        
        Returns:
            pandas.DataFrame: DataFrame containing all recommendations with columns:
                - id: Unique identifier for the user-recipe pair (format: user_id_recipe_id)
                - user_id: The user ID
                - recipe_id: The recipe ID
                - recipe_title: The title of the recipe
                - predicted_rating: The predicted rating
        """
        # Get all predictions
        predictions = self.predict()
        
        # Create reverse mappings
        reverse_user_map = {v: k for k, v in self.user_map.items()}
        reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        # Create a list to store all recommendations
        all_recommendations = []
        
        # For each user
        for user_idx in range(len(self.user_map)):
            user_id = reverse_user_map[user_idx]
            user_predictions = predictions[user_idx]
            
            # Get indices of all items sorted by rating (descending)
            sorted_indices = np.argsort(user_predictions)[::-1]
            
            # Add each recipe to the recommendations
            for idx in sorted_indices:
                recipe_id = reverse_item_map[idx]
                predicted_rating = user_predictions[idx]
                
                # Create unique ID for the user-recipe pair
                recommendation_id = f"{user_id}_{recipe_id}"
                
                # Get recipe title
                recipe_title = recipes_df[recipes_df['id'] == recipe_id]['title'].iloc[0] if not recipes_df[recipes_df['id'] == recipe_id].empty else "Unknown Recipe"
                
                all_recommendations.append({
                    'id': recommendation_id,
                    'user_id': user_id,
                    'recipe_id': recipe_id,
                    'recipe_title': recipe_title,
                    'predicted_rating': predicted_rating
                })
        
        # Convert to DataFrame
        recommendations_df = pd.DataFrame(all_recommendations)
        
        # Group by user_id and sort by predicted_rating within each group
        recommendations_df = recommendations_df.groupby('user_id', group_keys=False).apply(
            lambda x: x.sort_values('predicted_rating', ascending=False)
        ).reset_index(drop=True)
        
        return recommendations_df

    def upsert_recommendations(self, recipes_df):
        """Upsert recommendations to the database
        
        Args:
            recipes_df (pandas.DataFrame): DataFrame containing recipe information
            
        Returns:
            dict: Response from the Supabase upsert operation
        """
        recommendations_df = self.get_all_recommendations(recipes_df)
        recommendations = recommendations_df.to_dict('records')

        try:
            response = self.supabase.table("recommendations_test").upsert(
                recommendations,
                on_conflict="id"
            ).execute()
            return response
        except APIError as e:
            print("Supabase APIError:", e.message)  # or just `print(e)` to dump everything
            raise

    def convert_hyperparameters_to_csv(self, hyperparameters):
        """Convert hyperparameters JSON to a CSV format for upserting
        
        Args:
            hyperparameters (dict): Dictionary containing hyperparameters and metrics
            
        Returns:
            pandas.DataFrame: DataFrame with flattened hyperparameters and metrics
        """
        # Extract metrics
        val_metrics = hyperparameters['metrics']['validation']
        test_metrics = hyperparameters['metrics']['test']
        
        # Create a single row DataFrame
        hyperparams_df = pd.DataFrame([{
            'k': hyperparameters['k'],
            'alpha': hyperparameters['alpha'],
            'beta': hyperparameters['beta'],
            'epochs': hyperparameters['epochs'],
            'val_mse': val_metrics['mse'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'test_mse': test_metrics['mse'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae']
        }])
        
        return hyperparams_df

    def upsert_hyperparameters(self, hyperparameters):
        """Upsert hyperparameters to the database
        
        Args:
            hyperparameters (dict): Dictionary containing hyperparameters and metrics
            
        Returns:
            dict: Response from the Supabase upsert operation
        """
        # Convert to DataFrame
        hyperparams_df = self.convert_hyperparameters_to_csv(hyperparameters)
        
        # Convert to list of dictionaries
        hyperparams_data = hyperparams_df.to_dict('records')
        
        # Upsert to Supabase
        
        response = self.supabase.table("hyperparameters").upsert(
            hyperparams_data).execute()
        
        return response

def main():
    # Create hyperparameters directory if it doesn't exist
    hyperparams_dir = 'hyperparameters'
    if not os.path.exists(hyperparams_dir):
        os.makedirs(hyperparams_dir)
    
    # Load data
    load_dotenv("supabase.env")
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_api = os.getenv("SUPABASE_API")
    supabase = create_client(supabase_url, supabase_api)
    
    # Get recipes and ratings
    recipes = supabase.table("recipes_test").select("*").execute().data
    ratings = supabase.table("ratings_test").select("*").execute().data
    
    # Convert to DataFrames
    recipes_df = pd.DataFrame(recipes)
    ratings_df = pd.DataFrame(ratings)

    print(f' this is length of recipe {len(recipes_df)}')
    
    print(f"Loaded {len(ratings_df)} ratings from {len(ratings_df['user_id'].unique())} users for {len(ratings_df['recipe_id'].unique())} recipes")
    
    # Split data into train, validation, and test sets
    # First split: 80% train+val, 20% test
    train_val_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    # Second split: 80% train, 20% validation (from the 80% train+val)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
    
    print(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    
    # Tune hyperparameters using only training data
    print("\n=== HYPERPARAMETER TUNING ===")
    print("Tuning hyperparameters using training data...")
    recommender = RecipeRecommender(verbose=True,supabase=supabase)
    best_params, results_df = recommender.tune_hyperparameters(train_df)
    
    k, alpha, beta = best_params
    print(f"Best parameters: k={k}, alpha={alpha}, beta={beta}")
    
    # Train final model with best parameters on training data
    print("\n=== MODEL TRAINING ===")
    print("Training final model with best parameters on training data...")
    final_model = RecipeRecommender(k=k, alpha=alpha, beta=beta, epochs=100, verbose=True, supabase=supabase)
    final_model.fit(train_df)
    
    # Evaluate model on validation set
    print("\n=== MODEL EVALUATION ===")
    print("Evaluating model on validation set...")
    val_mse, val_rmse, val_mae = final_model.evaluate(val_df)
    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    
    # Evaluate model on test set (unseen data)
    print("\nEvaluating model on test set (unseen data)...")
    test_mse, test_rmse, test_mae = final_model.evaluate(test_df)
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Create hyperparameters dictionary
    hyperparameters = {
        'k': k,
        'alpha': alpha,
        'beta': beta,
        'epochs': 100,
        'metrics': {
            'validation': {
                'mse': float(val_mse),
                'rmse': float(val_rmse),
                'mae': float(val_mae)
            },
            'test': {
                'mse': float(test_mse),
                'rmse': float(test_rmse),
                'mae': float(test_mae)
            }
        }
    }
    
    # Save current hyperparameters (overwriting previous file)
    hyperparams_path = os.path.join(hyperparams_dir, 'recipe_recommender_hyperparameters.json')
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"\nHyperparameters saved to {hyperparams_path}")
    
    # Convert hyperparameters to CSV format and upsert to database
    print("\n=== UPSERTING HYPERPARAMETERS ===")
    hyperparams_df = final_model.convert_hyperparameters_to_csv(hyperparameters)
    hyperparams_csv_path = os.path.join(hyperparams_dir, 'recipe_recommender_hyperparameters.csv')
    hyperparams_df.to_csv(hyperparams_csv_path, index=False)
    print(f"Hyperparameters saved to CSV at {hyperparams_csv_path}")
    
    # Upsert hyperparameters to database
    response = final_model.upsert_hyperparameters(hyperparameters)
    print("Hyperparameters upserted to database")
    
    # Save hyperparameter tuning results (overwriting previous file)
    results_path = os.path.join(hyperparams_dir, 'hyperparameter_tuning_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Hyperparameter tuning results saved to {results_path}")
    
    # Save tuning plots (overwriting previous files)
    k_alpha_plot_path = os.path.join(hyperparams_dir, 'hyperparameter_tuning_k_alpha.png')
    plt.figure(1)
    plt.savefig(k_alpha_plot_path)
    print(f"k vs α hyperparameter tuning plot saved to {k_alpha_plot_path}")
    
    k_beta_plot_path = os.path.join(hyperparams_dir, 'hyperparameter_tuning_k_beta.png')
    plt.figure(2)
    plt.savefig(k_beta_plot_path)
    print(f"k vs β hyperparameter tuning plot saved to {k_beta_plot_path}")
    
    alpha_beta_plot_path = os.path.join(hyperparams_dir, 'hyperparameter_tuning_alpha_beta.png')
    plt.figure(3)
    plt.savefig(alpha_beta_plot_path)
    print(f"α vs β hyperparameter tuning plot saved to {alpha_beta_plot_path}")
    
    scatter_3d_plot_path = os.path.join(hyperparams_dir, 'hyperparameter_tuning_3d.png')
    plt.figure(4)
    plt.savefig(scatter_3d_plot_path)
    print(f"3D hyperparameter tuning plot saved to {scatter_3d_plot_path}")
    
    # Generate recommendations for all users
    print("\n=== GENERATING RECOMMENDATIONS ===")
    print("Generating recommendations for all users...")
    recommendations_df = final_model.get_all_recommendations(recipes_df)
    
    # Convert DataFrame to list of dictionaries for Supabase upsert
    recommendations_data = recommendations_df.to_dict(orient='records')
    
    # Upsert recommendations to Supabase
    print("Upserting recommendations to Supabase...")
    response = final_model.upsert_recommendations(recipes_df)
    
    print(f"Successfully upserted {len(recommendations_data)} recommendations to Supabase")
    
    # Display sample of recommendations
    print("\n=== SAMPLE RECOMMENDATIONS ===")
    sample_users = recommendations_df['user_id'].unique()[:3]  # First 3 users
    for user_id in sample_users:
        user_recs = recommendations_df[recommendations_df['user_id'] == user_id].head(5)
        print(f"\nTop 5 recommendations for user {user_id}:")
        for _, rec in user_recs.iterrows():
            print(f"- {rec['recipe_title']}: {rec['predicted_rating']:.2f}")
    
    print("\n=== PROCESS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main() 