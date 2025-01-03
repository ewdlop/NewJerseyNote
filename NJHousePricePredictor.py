import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

class NJHousePricePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_importance = None
        
    def create_preprocessor(self, categorical_features, numeric_features):
        """Create an advanced preprocessor for housing data"""
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ])
        
        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

    def build_model(self, X, y, model_type='rf'):
        """Build and train model with choice of algorithms"""
        # Define features
        categorical_features = ['town', 'property_type', 'school_district', 'condition']
        numeric_features = ['sqft', 'bedrooms', 'bathrooms', 'year_built', 
                          'lot_size', 'taxes', 'crime_rate', 'school_rating']
        
        # Create preprocessor
        self.preprocessor = self.create_preprocessor(categorical_features, numeric_features)
        
        # Choose model based on type
        if model_type == 'rf':
            regressor = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'gb':
            regressor = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
        elif model_type == 'xgb':
            regressor = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                random_state=42
            )
            
        # Create pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', regressor)
        ])
        
        # Train model
        self.model.fit(X, y)
        
        # Store feature importance
        self.calculate_feature_importance(X.columns)
        
    def hyperparameter_tuning(self, X, y):
        """Perform grid search for hyperparameter tuning"""
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [10, 15, 20],
            'regressor__min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_params_
        
    def calculate_feature_importance(self, feature_names):
        """Calculate and store feature importance"""
        if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
            importances = self.model.named_steps['regressor'].feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.feature_importance is not None:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='importance', y='feature', data=self.feature_importance[:10])
            plt.title('Top 10 Most Important Features')
            plt.show()
            
    def predict(self, X):
        """Make predictions with confidence intervals"""
        predictions = self.model.predict(X)
        
        if isinstance(self.model.named_steps['regressor'], RandomForestRegressor):
            predictions_all = np.array([tree.predict(self.preprocessor.transform(X)) 
                                     for tree in self.model.named_steps['regressor'].estimators_])
            conf_int = np.percentile(predictions_all, [5, 95], axis=0).T
            return predictions, conf_int
        
        return predictions, None
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        predictions = self.predict(X_test)[0]
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'MAE': mean_absolute_error(y_test, predictions),
            'R2': r2_score(y_test, predictions)
        }
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        metrics['CV_Mean'] = cv_scores.mean()
        metrics['CV_Std'] = cv_scores.std()
        
        return metrics
    
    def plot_predictions(self, X_test, y_test):
        """Plot actual vs predicted values"""
        predictions = self.predict(X_test)[0]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted House Prices')
        plt.tight_layout()
        plt.show()

def main():
    # Example usage with synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic NJ housing data
    data = pd.DataFrame({
        'town': np.random.choice(['Newark', 'Jersey City', 'Princeton', 'Hoboken'], n_samples),
        'property_type': np.random.choice(['Single', 'Condo', 'Multi-Family'], n_samples),
        'sqft': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'year_built': np.random.randint(1950, 2024, n_samples),
        'lot_size': np.random.normal(5000, 1000, n_samples),
        'school_district': np.random.choice(['A', 'B', 'C'], n_samples),
        'condition': np.random.choice(['Excellent', 'Good', 'Fair'], n_samples),
        'taxes': np.random.normal(8000, 2000, n_samples),
        'crime_rate': np.random.normal(5, 2, n_samples),
        'school_rating': np.random.randint(1, 11, n_samples)
    })
    
    # Generate synthetic prices
    base_price = 300000
    data['price'] = base_price + \
                   data['sqft'] * 100 + \
                   data['bedrooms'] * 50000 + \
                   data['bathrooms'] * 25000 + \
                   np.random.normal(0, 50000, n_samples)
    
    # Split features and target
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    predictor = NJHousePricePredictor()
    predictor.build_model(X_train, y_train, model_type='rf')
    
    # Evaluate model
    metrics = predictor.evaluate(X_test, y_test)
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot feature importance
    predictor.plot_feature_importance()
    
    # Plot predictions
    predictor.plot_predictions(X_test, y_test)

if __name__ == "__main__":
    main()
