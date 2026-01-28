#!/usr/bin/env python3
"""
NEXUS AI v3.0 - Data Preparation and Model Training
Processes Football-Data.co.uk data and trains cutting-edge models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class FootballDataProcessor:
    """Process and prepare football data for ML training"""
    
    def __init__(self, data_dir: str = "data/raw/football_data"):
        self.data_dir = Path(data_dir)
        self.processed_data = None
        self.features = None
        self.targets = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        
    def load_excel_files(self) -> pd.DataFrame:
        """Load all Excel files and combine them"""
        all_data = []
        
        excel_files = sorted(self.data_dir.glob("seasons-*.xlsx"))
        logger.info(f"Found {len(excel_files)} Excel files")
        
        for file in excel_files:
            try:
                logger.info(f"Loading {file.name}...")
                # Load all sheets (each sheet is a different league)
                xls = pd.ExcelFile(file)
                
                for sheet_name in xls.sheet_names:
                    try:
                        df = pd.read_excel(file, sheet_name=sheet_name)
                        if len(df) > 0:
                            df['league'] = sheet_name
                            df['season'] = file.stem.replace('seasons-', '')
                            all_data.append(df)
                            logger.info(f"  - {sheet_name}: {len(df)} matches")
                    except Exception as e:
                        logger.warning(f"  - Error loading {sheet_name}: {e}")
                        
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                
        if not all_data:
            raise ValueError("No data loaded!")
            
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total matches loaded: {len(combined)}")
        return combined
    
    def clean_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the data"""
        logger.info("Cleaning data...")
        
        # Keep only finished matches (with results)
        df = df[df['FTR'].notna()].copy()
        logger.info(f"Matches with results: {len(df)}")
        
        # Convert date
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    def calculate_team_form(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Calculate team form features"""
        logger.info(f"Calculating team form (window={window})...")
        
        df = df.copy()
        df['home_form_points'] = 0.0
        df['away_form_points'] = 0.0
        df['home_form_goals'] = 0.0
        df['away_form_goals'] = 0.0
        df['home_form_conceded'] = 0.0
        df['away_form_conceded'] = 0.0
        
        # Calculate form for each team
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        for team in teams:
            if pd.isna(team):
                continue
                
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
            team_matches = team_matches.sort_values('Date')
            
            for idx, row in team_matches.iterrows():
                # Get previous matches for this team
                prev_matches = team_matches[team_matches['Date'] < row['Date']].tail(window)
                
                if len(prev_matches) == 0:
                    continue
                    
                points = 0
                goals_scored = 0
                goals_conceded = 0
                
                for _, prev in prev_matches.iterrows():
                    if prev['HomeTeam'] == team:
                        goals_scored += prev['FTHG'] if not pd.isna(prev['FTHG']) else 0
                        goals_conceded += prev['FTAG'] if not pd.isna(prev['FTAG']) else 0
                        if prev['FTR'] == 'H':
                            points += 3
                        elif prev['FTR'] == 'D':
                            points += 1
                    else:
                        goals_scored += prev['FTAG'] if not pd.isna(prev['FTAG']) else 0
                        goals_conceded += prev['FTHG'] if not pd.isna(prev['FTHG']) else 0
                        if prev['FTR'] == 'A':
                            points += 3
                        elif prev['FTR'] == 'D':
                            points += 1
                
                n_matches = len(prev_matches)
                if row['HomeTeam'] == team:
                    df.loc[idx, 'home_form_points'] = points / (n_matches * 3)
                    df.loc[idx, 'home_form_goals'] = goals_scored / n_matches
                    df.loc[idx, 'home_form_conceded'] = goals_conceded / n_matches
                else:
                    df.loc[idx, 'away_form_points'] = points / (n_matches * 3)
                    df.loc[idx, 'away_form_goals'] = goals_scored / n_matches
                    df.loc[idx, 'away_form_conceded'] = goals_conceded / n_matches
        
        return df
    
    def calculate_h2h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate head-to-head statistics"""
        logger.info("Calculating head-to-head stats...")
        
        df = df.copy()
        df['h2h_home_wins'] = 0
        df['h2h_draws'] = 0
        df['h2h_away_wins'] = 0
        
        for idx, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match_date = row['Date']
            
            if pd.isna(home_team) or pd.isna(away_team):
                continue
            
            # Find previous matches between these teams
            h2h_matches = df[
                ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
            ]
            h2h_matches = h2h_matches[h2h_matches['Date'] < match_date].tail(5)
            
            if len(h2h_matches) > 0:
                home_wins = 0
                draws = 0
                away_wins = 0
                
                for _, h2h in h2h_matches.iterrows():
                    if h2h['HomeTeam'] == home_team:
                        if h2h['FTR'] == 'H':
                            home_wins += 1
                        elif h2h['FTR'] == 'D':
                            draws += 1
                        else:
                            away_wins += 1
                    else:
                        if h2h['FTR'] == 'A':
                            home_wins += 1
                        elif h2h['FTR'] == 'D':
                            draws += 1
                        else:
                            away_wins += 1
                
                total = len(h2h_matches)
                df.loc[idx, 'h2h_home_wins'] = home_wins / total
                df.loc[idx, 'h2h_draws'] = draws / total
                df.loc[idx, 'h2h_away_wins'] = away_wins / total
        
        return df
    
    def extract_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract betting odds features"""
        logger.info("Extracting odds features...")
        
        # Use average market odds if available
        if 'AvgH' in df.columns and 'AvgD' in df.columns and 'AvgA' in df.columns:
            df['odds_home'] = df['AvgH']
            df['odds_draw'] = df['AvgD']
            df['odds_away'] = df['AvgA']
        elif 'B365H' in df.columns and 'B365D' in df.columns and 'B365A' in df.columns:
            df['odds_home'] = df['B365H']
            df['odds_draw'] = df['B365D']
            df['odds_away'] = df['B365A']
        else:
            # Fallback - set default odds
            df['odds_home'] = 2.0
            df['odds_draw'] = 3.4
            df['odds_away'] = 3.6
        
        # Calculate implied probabilities
        df['prob_home'] = 1 / df['odds_home']
        df['prob_draw'] = 1 / df['odds_draw']
        df['prob_away'] = 1 / df['odds_away']
        
        # Normalize probabilities
        total_prob = df['prob_home'] + df['prob_draw'] + df['prob_away']
        df['prob_home'] = df['prob_home'] / total_prob
        df['prob_draw'] = df['prob_draw'] / total_prob
        df['prob_away'] = df['prob_away'] / total_prob
        
        return df
    
    def extract_match_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract match statistics features"""
        logger.info("Extracting match statistics...")
        
        # Calculate rolling averages for teams
        df = df.copy()
        
        # Initialize columns with NaN
        stat_cols = ['avg_shots_home', 'avg_shots_away', 'avg_corners_home', 'avg_corners_away',
                     'avg_cards_home', 'avg_cards_away']
        for col in stat_cols:
            df[col] = 0.0
        
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        for team in teams:
            if pd.isna(team):
                continue
            
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
            team_matches = team_matches.sort_values('Date')
            
            for idx, row in team_matches.iterrows():
                prev_matches = team_matches[team_matches['Date'] < row['Date']].tail(5)
                
                if len(prev_matches) == 0:
                    continue
                
                shots = []
                corners = []
                cards = []
                
                for _, prev in prev_matches.iterrows():
                    if prev['HomeTeam'] == team:
                        if 'HS' in prev and not pd.isna(prev['HS']):
                            shots.append(prev['HS'])
                        if 'HC' in prev and not pd.isna(prev['HC']):
                            corners.append(prev['HC'])
                        if 'HY' in prev and 'HR' in prev:
                            cards.append(prev['HY'] + prev['HR'])
                    else:
                        if 'AS' in prev and not pd.isna(prev['AS']):
                            shots.append(prev['AS'])
                        if 'AC' in prev and not pd.isna(prev['AC']):
                            corners.append(prev['AC'])
                        if 'AY' in prev and 'AR' in prev:
                            cards.append(prev['AY'] + prev['AR'])
                
                if row['HomeTeam'] == team:
                    df.loc[idx, 'avg_shots_home'] = np.mean(shots) if shots else 0
                    df.loc[idx, 'avg_corners_home'] = np.mean(corners) if corners else 0
                    df.loc[idx, 'avg_cards_home'] = np.mean(cards) if cards else 0
                else:
                    df.loc[idx, 'avg_shots_away'] = np.mean(shots) if shots else 0
                    df.loc[idx, 'avg_corners_away'] = np.mean(corners) if corners else 0
                    df.loc[idx, 'avg_cards_away'] = np.mean(cards) if cards else 0
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create feature matrix and target vector"""
        logger.info("Creating feature matrix...")
        
        # Apply all feature engineering
        df = self.clean_and_prepare(df)
        df = self.calculate_team_form(df, window=5)
        df = self.calculate_h2h(df)
        df = self.extract_odds_features(df)
        df = self.extract_match_stats(df)
        
        # Select features for model
        feature_cols = [
            'home_form_points', 'away_form_points',
            'home_form_goals', 'away_form_goals',
            'home_form_conceded', 'away_form_conceded',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
            'odds_home', 'odds_draw', 'odds_away',
            'prob_home', 'prob_draw', 'prob_away',
            'avg_shots_home', 'avg_shots_away',
            'avg_corners_home', 'avg_corners_away',
            'avg_cards_home', 'avg_cards_away',
        ]
        
        # Filter to only rows with valid targets and features
        df = df[df['FTR'].isin(['H', 'D', 'A'])].copy()
        
        # Fill NaN values
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = 0
        
        X = df[feature_cols].copy()
        y = df['FTR'].copy()
        
        # Encode target
        y_encoded = self.le.fit_transform(y)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: H={sum(y=='H')}, D={sum(y=='D')}, A={sum(y=='A')}")
        
        return X, y_encoded, df


class ModelTrainer:
    """Train and save ML models"""
    
    def __init__(self, output_dir: str = "models/trained"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.scaler = StandardScaler()
        self.pca = None
        
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train Random Forest with hyperparameter tuning"""
        logger.info("Training Random Forest...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
        
        # Use smaller grid for faster training
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluate
        train_acc = best_model.score(X_train, y_train)
        val_acc = best_model.score(X_val, y_val)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
        
        logger.info(f"RF Best params: {grid_search.best_params_}")
        logger.info(f"RF Train accuracy: {train_acc:.4f}")
        logger.info(f"RF Val accuracy: {val_acc:.4f}")
        logger.info(f"RF CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Feature importance
        feature_importance = dict(zip(
            range(X_train.shape[1]),
            best_model.feature_importances_
        ))
        
        self.models['random_forest'] = {
            'model': best_model,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'cv_acc': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_
        }
        
        return self.models['random_forest']
    
    def train_mlp(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train MLP Neural Network with PCA"""
        logger.info("Training MLP Neural Network...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Apply PCA
        n_components = min(15, X_train.shape[1])
        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        
        logger.info(f"PCA explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Train MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=True
        )
        
        mlp.fit(X_train_pca, y_train)
        
        # Evaluate
        train_acc = mlp.score(X_train_pca, y_train)
        val_acc = mlp.score(X_val_pca, y_val)
        
        # Predictions for log loss
        y_pred_proba = mlp.predict_proba(X_val_pca)
        val_loss = log_loss(y_val, y_pred_proba)
        
        logger.info(f"MLP Train accuracy: {train_acc:.4f}")
        logger.info(f"MLP Val accuracy: {val_acc:.4f}")
        logger.info(f"MLP Val loss: {val_loss:.4f}")
        logger.info(f"MLP Iterations: {mlp.n_iter_}")
        
        self.models['mlp'] = {
            'model': mlp,
            'scaler': self.scaler,
            'pca': self.pca,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'n_iterations': mlp.n_iter_
        }
        
        return self.models['mlp']
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train Gradient Boosting classifier"""
        logger.info("Training Gradient Boosting...")
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=1
        )
        
        gb.fit(X_train, y_train)
        
        # Evaluate
        train_acc = gb.score(X_train, y_train)
        val_acc = gb.score(X_val, y_val)
        
        logger.info(f"GB Train accuracy: {train_acc:.4f}")
        logger.info(f"GB Val accuracy: {val_acc:.4f}")
        
        self.models['gradient_boosting'] = {
            'model': gb,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        
        return self.models['gradient_boosting']
    
    def save_models(self):
        """Save all trained models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model_data in self.models.items():
            filepath = self.output_dir / f"{name}_{timestamp}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Saved {name} to {filepath}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'models': list(self.models.keys()),
            'performance': {
                name: {
                    'train_acc': data['train_acc'],
                    'val_acc': data['val_acc'],
                }
                for name, data in self.models.items()
            }
        }
        
        metadata_path = self.output_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
        
        return metadata


def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("NEXUS AI v3.0 - Model Training Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load and process data
    logger.info("\n[STEP 1] Loading data...")
    processor = FootballDataProcessor()
    raw_data = processor.load_excel_files()
    
    # Step 2: Create features
    logger.info("\n[STEP 2] Creating features...")
    X, y, processed_df = processor.create_features(raw_data)
    
    # Step 3: Split data
    logger.info("\n[STEP 3] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Val set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Step 4: Train models
    logger.info("\n[STEP 4] Training models...")
    trainer = ModelTrainer()
    
    # Train Random Forest
    rf_results = trainer.train_random_forest(X_train.values, y_train, X_val.values, y_val)
    
    # Train MLP
    mlp_results = trainer.train_mlp(X_train.values, y_train, X_val.values, y_val)
    
    # Train Gradient Boosting
    gb_results = trainer.train_gradient_boosting(X_train.values, y_train, X_val.values, y_val)
    
    # Step 5: Save models
    logger.info("\n[STEP 5] Saving models...")
    metadata = trainer.save_models()
    
    # Final evaluation on test set
    logger.info("\n[STEP 6] Final evaluation on test set...")
    
    # RF Test
    rf_model = trainer.models['random_forest']['model']
    rf_test_acc = rf_model.score(X_test.values, y_test)
    logger.info(f"RF Test accuracy: {rf_test_acc:.4f}")
    
    # MLP Test
    mlp_data = trainer.models['mlp']
    X_test_scaled = mlp_data['scaler'].transform(X_test.values)
    X_test_pca = mlp_data['pca'].transform(X_test_scaled)
    mlp_test_acc = mlp_data['model'].score(X_test_pca, y_test)
    logger.info(f"MLP Test accuracy: {mlp_test_acc:.4f}")
    
    # GB Test
    gb_model = trainer.models['gradient_boosting']['model']
    gb_test_acc = gb_model.score(X_test.values, y_test)
    logger.info(f"GB Test accuracy: {gb_test_acc:.4f}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Random Forest: Train={rf_results['train_acc']:.4f}, Val={rf_results['val_acc']:.4f}, Test={rf_test_acc:.4f}")
    logger.info(f"MLP Neural Net: Train={mlp_results['train_acc']:.4f}, Val={mlp_results['val_acc']:.4f}, Test={mlp_test_acc:.4f}")
    logger.info(f"Gradient Boosting: Train={gb_results['train_acc']:.4f}, Val={gb_results['val_acc']:.4f}, Test={gb_test_acc:.4f}")
    
    return trainer, processor


if __name__ == "__main__":
    trainer, processor = main()
