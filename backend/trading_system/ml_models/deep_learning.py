"""
Advanced Deep Learning Models for Price Prediction and Market Analysis
LSTM, Transformer, and Ensemble Models for Full ML Potential
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
from transformers import AutoModel, AutoConfig
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import joblib
import json
import os
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for deep learning models"""
    sequence_length: int = 60
    prediction_horizon: int = 1
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    validation_split: float = 0.2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMNetwork(nn.Module):
    """Advanced LSTM network for price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2, bidirectional: bool = True):
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * (2 if bidirectional else 1),
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Dense layers
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.fc2 = nn.Linear(lstm_output_size // 2, lstm_output_size // 4)
        self.fc3 = nn.Linear(lstm_output_size // 4, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last time step
        last_hidden = attn_out[:, -1, :]
        
        # Dense layers with residual connections
        x1 = F.relu(self.fc1(self.dropout(last_hidden)))
        x2 = F.relu(self.fc2(self.dropout(x1)))
        output = self.fc3(self.dropout(x2))
        
        return output


class TransformerNetwork(nn.Module):
    """Transformer network for market analysis"""
    
    def __init__(self, input_size: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, output_size: int = 1, dropout: float = 0.1):
        super(TransformerNetwork, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_size)
        )
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Global pooling
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        
        # Classification/regression
        output = self.classifier(pooled)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class LSTMPricePredictor:
    """
    Advanced LSTM-based price prediction system
    Supports multi-step prediction and uncertainty quantification
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_scalers = {}
        self.is_trained = False
        self.training_history = []
        self.device = torch.device(self.config.device)
        
        logger.info("LSTM Price Predictor initialized")
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for training"""
        try:
            # Feature engineering
            features = self._engineer_features(data)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Create sequences
            sequences, targets = self._create_sequences(
                scaled_features, 
                target_column=features.columns.get_loc(target_column),
                sequence_length=self.config.sequence_length,
                prediction_horizon=self.config.prediction_horizon
            )
            
            logger.info(f"Prepared {len(sequences)} sequences with {scaled_features.shape[1]} features")
            
            return sequences, targets
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for price prediction"""
        try:
            features = data.copy()
            
            # Price-based features
            if 'close' in features.columns:
                features['returns'] = features['close'].pct_change()
                features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
                features['price_momentum_5'] = features['close'] / features['close'].shift(5) - 1
                features['price_momentum_10'] = features['close'] / features['close'].shift(10) - 1
                features['price_momentum_20'] = features['close'] / features['close'].shift(20) - 1
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                if 'close' in features.columns:
                    ma_col = f'ma_{window}'
                    features[ma_col] = features['close'].rolling(window).mean()
                    features[f'ma_ratio_{window}'] = features['close'] / features[ma_col] - 1
            
            # Volatility features
            if 'close' in features.columns:
                features['volatility_5'] = features['returns'].rolling(5).std()
                features['volatility_10'] = features['returns'].rolling(10).std()
                features['volatility_20'] = features['returns'].rolling(20).std()
            
            # Volume features
            if 'volume' in features.columns:
                features['volume_ma_5'] = features['volume'].rolling(5).mean()
                features['volume_ratio'] = features['volume'] / features['volume_ma_5']
                features['volume_momentum'] = features['volume'] / features['volume'].shift(5) - 1
            
            # Technical indicators
            if all(col in features.columns for col in ['high', 'low', 'close']):
                # RSI
                features['rsi'] = self._calculate_rsi(features['close'])
                
                # Bollinger Bands
                bb_upper, bb_lower = self._calculate_bollinger_bands(features['close'])
                features['bb_upper'] = bb_upper
                features['bb_lower'] = bb_lower
                features['bb_position'] = (features['close'] - bb_lower) / (bb_upper - bb_lower)
                
                # MACD
                macd_line, macd_signal = self._calculate_macd(features['close'])
                features['macd'] = macd_line
                features['macd_signal'] = macd_signal
                features['macd_histogram'] = macd_line - macd_signal
            
            # Time-based features
            if 'timestamp' in features.columns:
                features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
                features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
                features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
                features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
                features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
                features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            
            # Remove NaN values
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            # Select numeric columns only
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            features = features[numeric_columns]
            
            return features
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        return macd_line, macd_signal
    
    def _create_sequences(self, data: np.ndarray, target_column: int, 
                         sequence_length: int, prediction_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            # Input sequence
            seq = data[i:i + sequence_length]
            sequences.append(seq)
            
            # Target (future price)
            target = data[i + sequence_length + prediction_horizon - 1, target_column]
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train(self, data: pd.DataFrame, target_column: str = 'close', validation_data: Optional[pd.DataFrame] = None) -> Dict:
        """Train the LSTM model"""
        try:
            logger.info("Starting LSTM training...")
            
            # Prepare training data
            sequences, targets = self.prepare_data(data, target_column)
            
            # Split data
            split_idx = int(len(sequences) * (1 - self.config.validation_split))
            train_sequences = sequences[:split_idx]
            train_targets = targets[:split_idx]
            
            if validation_data is not None:
                val_sequences, val_targets = self.prepare_data(validation_data, target_column)
            else:
                val_sequences = sequences[split_idx:]
                val_targets = targets[split_idx:]
            
            # Create datasets and dataloaders
            train_dataset = TimeSeriesDataset(train_sequences, train_targets)
            val_dataset = TimeSeriesDataset(val_sequences, val_targets)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Initialize model
            input_size = sequences.shape[2]
            self.model = LSTMNetwork(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=1,
                dropout=self.config.dropout
            ).to(self.device)
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(self.config.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_sequences, batch_targets in train_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_sequences)
                    loss = criterion(outputs.squeeze(), batch_targets)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_sequences, batch_targets in val_loader:
                        batch_sequences = batch_sequences.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        outputs = self.model(batch_sequences)
                        loss = criterion(outputs.squeeze(), batch_targets)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), './models/best_lstm_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Load best model
            self.model.load_state_dict(torch.load('./models/best_lstm_model.pth'))
            self.is_trained = True
            
            # Training history
            self.training_history = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'epochs_trained': len(train_losses)
            }
            
            logger.info(f"LSTM training completed. Best validation loss: {best_val_loss:.6f}")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            raise
    
    def predict(self, data: pd.DataFrame, return_confidence: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with the trained model"""
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before making predictions")
            
            # Prepare data
            features = self._engineer_features(data)
            scaled_features = self.scaler.transform(features)
            
            # Create sequences for prediction
            sequence_length = self.config.sequence_length
            if len(scaled_features) < sequence_length:
                raise ValueError(f"Not enough data points. Need at least {sequence_length}, got {len(scaled_features)}")
            
            # Take the last sequence
            last_sequence = scaled_features[-sequence_length:].reshape(1, sequence_length, -1)
            
            self.model.eval()
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(last_sequence).to(self.device)
                prediction = self.model(sequence_tensor).cpu().numpy()
            
            # Denormalize prediction
            # This is simplified - in practice, you'd need to properly denormalize based on the target column scaling
            prediction_scaled = prediction.flatten()
            
            if return_confidence:
                # Monte Carlo dropout for uncertainty estimation
                confidence_predictions = []
                self.model.train()  # Enable dropout
                
                with torch.no_grad():
                    for _ in range(50):  # 50 Monte Carlo samples
                        mc_pred = self.model(sequence_tensor).cpu().numpy().flatten()
                        confidence_predictions.append(mc_pred)
                
                confidence_predictions = np.array(confidence_predictions)
                confidence_interval = np.std(confidence_predictions, axis=0)
                
                return prediction_scaled, confidence_interval
            
            return prediction_scaled
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def evaluate(self, test_data: pd.DataFrame, target_column: str = 'close') -> Dict:
        """Evaluate model performance"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Prepare test data
            sequences, targets = self.prepare_data(test_data, target_column)
            
            # Make predictions
            test_dataset = TimeSeriesDataset(sequences, targets)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            self.model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_sequences, batch_targets in test_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    outputs = self.model(batch_sequences)
                    
                    predictions.extend(outputs.cpu().numpy().flatten())
                    actuals.extend(batch_targets.numpy().flatten())
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate metrics
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(actuals, predictions)
            
            # Direction accuracy
            actual_directions = np.sign(np.diff(actuals))
            pred_directions = np.sign(np.diff(predictions))
            direction_accuracy = np.mean(actual_directions == pred_directions)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'direction_accuracy': direction_accuracy,
                'num_predictions': len(predictions)
            }
            
            logger.info(f"Model evaluation completed: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def save_model(self, path: str):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'config': self.config.__dict__,
                'training_history': self.training_history,
                'is_trained': self.is_trained
            }, path)
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Restore configuration
            config_dict = checkpoint['config']
            self.config = ModelConfig(**config_dict)
            
            # Create model with loaded config
            # Note: This assumes input_size is stored in config or can be inferred
            input_size = checkpoint.get('input_size', 50)  # Default fallback
            
            self.model = LSTMNetwork(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=1,
                dropout=self.config.dropout
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore other attributes
            self.scaler = checkpoint['scaler']
            self.training_history = checkpoint['training_history']
            self.is_trained = checkpoint['is_trained']
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'LSTM',
            'is_trained': self.is_trained,
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }


class TransformerMarketAnalyzer:
    """
    Advanced Transformer-based market analysis system
    For pattern recognition and market regime detection
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = []
        self.device = torch.device(self.config.device)
        
        logger.info("Transformer Market Analyzer initialized")
    
    def train(self, data: pd.DataFrame, target_type: str = 'trend') -> Dict:
        """Train transformer for market analysis"""
        try:
            logger.info("Starting Transformer training for market analysis...")
            
            # Prepare data based on target type
            features = self._engineer_features(data)
            if target_type == 'trend':
                targets = self._create_trend_labels(data)
            elif target_type == 'volatility':
                targets = self._create_volatility_labels(data)
            elif target_type == 'regime':
                targets = self._create_regime_labels(data)
            else:
                raise ValueError(f"Unknown target type: {target_type}")
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Create sequences
            sequences, sequence_targets = self._create_sequences_for_classification(
                scaled_features, targets, self.config.sequence_length
            )
            
            # Split data
            split_idx = int(len(sequences) * (1 - self.config.validation_split))
            train_sequences = sequences[:split_idx]
            train_targets = sequence_targets[:split_idx]
            val_sequences = sequences[split_idx:]
            val_targets = sequence_targets[split_idx:]
            
            # Create datasets
            train_dataset = TimeSeriesDataset(train_sequences, train_targets)
            val_dataset = TimeSeriesDataset(val_sequences, val_targets)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Initialize model
            input_size = sequences.shape[2]
            output_size = len(np.unique(targets))
            
            self.model = TransformerNetwork(
                input_size=input_size,
                d_model=self.config.hidden_size,
                nhead=8,
                num_layers=self.config.num_layers,
                output_size=output_size,
                dropout=self.config.dropout
            ).to(self.device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            
            # Training loop
            best_val_acc = 0.0
            patience_counter = 0
            train_losses = []
            val_accuracies = []
            
            for epoch in range(self.config.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_sequences, batch_targets in train_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_targets = batch_targets.long().to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_sequences)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_sequences, batch_targets in val_loader:
                        batch_sequences = batch_sequences.to(self.device)
                        batch_targets = batch_targets.long().to(self.device)
                        
                        outputs = self.model(batch_sequences)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_targets.size(0)
                        val_correct += (predicted == batch_targets).sum().item()
                
                train_loss /= len(train_loader)
                val_accuracy = 100 * val_correct / val_total
                
                train_losses.append(train_loss)
                val_accuracies.append(val_accuracy)
                
                scheduler.step()
                
                # Early stopping based on validation accuracy
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    patience_counter = 0
                    torch.save(self.model.state_dict(), './models/best_transformer_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Acc = {val_accuracy:.2f}%")
            
            # Load best model
            self.model.load_state_dict(torch.load('./models/best_transformer_model.pth'))
            self.is_trained = True
            
            self.training_history = {
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'best_val_accuracy': best_val_acc,
                'epochs_trained': len(train_losses)
            }
            
            logger.info(f"Transformer training completed. Best validation accuracy: {best_val_acc:.2f}%")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Error training Transformer: {e}")
            raise
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for transformer analysis"""
        # Reuse the feature engineering from LSTM
        predictor = LSTMPricePredictor()
        return predictor._engineer_features(data)
    
    def _create_trend_labels(self, data: pd.DataFrame, window: int = 20) -> np.ndarray:
        """Create trend labels for classification"""
        try:
            prices = data['close'].values
            trends = []
            
            for i in range(len(prices)):
                if i < window:
                    trends.append(1)  # Neutral
                else:
                    recent_trend = (prices[i] - prices[i-window]) / prices[i-window]
                    if recent_trend > 0.02:  # Bullish threshold
                        trends.append(2)
                    elif recent_trend < -0.02:  # Bearish threshold
                        trends.append(0)
                    else:
                        trends.append(1)  # Neutral
            
            return np.array(trends)
            
        except Exception as e:
            logger.error(f"Error creating trend labels: {e}")
            return np.ones(len(data))
    
    def _create_volatility_labels(self, data: pd.DataFrame, window: int = 20) -> np.ndarray:
        """Create volatility regime labels"""
        try:
            returns = data['close'].pct_change()
            volatilities = returns.rolling(window).std()
            
            # Define volatility regimes based on quantiles
            low_vol = volatilities.quantile(0.33)
            high_vol = volatilities.quantile(0.67)
            
            labels = []
            for vol in volatilities:
                if pd.isna(vol):
                    labels.append(1)  # Medium
                elif vol < low_vol:
                    labels.append(0)  # Low
                elif vol > high_vol:
                    labels.append(2)  # High
                else:
                    labels.append(1)  # Medium
            
            return np.array(labels)
            
        except Exception as e:
            logger.error(f"Error creating volatility labels: {e}")
            return np.ones(len(data))
    
    def _create_regime_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Create market regime labels using multiple indicators"""
        try:
            # Combine trend and volatility for regime detection
            trend_labels = self._create_trend_labels(data)
            vol_labels = self._create_volatility_labels(data)
            
            # Create composite regime labels
            # 0: Bear Market (Low Vol), 1: Bear Market (High Vol)
            # 2: Sideways (Low Vol), 3: Sideways (High Vol)  
            # 4: Bull Market (Low Vol), 5: Bull Market (High Vol)
            
            regime_labels = trend_labels * 2 + vol_labels
            
            # Simplify to 3 main regimes for easier learning
            simplified_regimes = []
            for regime in regime_labels:
                if regime in [0, 1]:  # Bear markets
                    simplified_regimes.append(0)
                elif regime in [2, 3]:  # Sideways markets
                    simplified_regimes.append(1)
                else:  # Bull markets [4, 5]
                    simplified_regimes.append(2)
            
            return np.array(simplified_regimes)
            
        except Exception as e:
            logger.error(f"Error creating regime labels: {e}")
            return np.ones(len(data))
    
    def _create_sequences_for_classification(self, features: np.ndarray, targets: np.ndarray, 
                                          sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for classification tasks"""
        sequences = []
        sequence_targets = []
        
        for i in range(len(features) - sequence_length):
            seq = features[i:i + sequence_length]
            target = targets[i + sequence_length]
            
            sequences.append(seq)
            sequence_targets.append(target)
        
        return np.array(sequences), np.array(sequence_targets)
    
    def predict_market_state(self, data: pd.DataFrame) -> Dict:
        """Predict current market state/regime"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")
            
            # Prepare data
            features = self._engineer_features(data)
            scaled_features = self.scaler.transform(features)
            
            # Create sequence
            sequence_length = self.config.sequence_length
            if len(scaled_features) < sequence_length:
                raise ValueError(f"Not enough data points. Need at least {sequence_length}")
            
            last_sequence = scaled_features[-sequence_length:].reshape(1, sequence_length, -1)
            
            self.model.eval()
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(last_sequence).to(self.device)
                outputs = self.model(sequence_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # Map prediction to market state
            state_mapping = {
                0: 'Bearish',
                1: 'Neutral/Sideways',
                2: 'Bullish'
            }
            
            return {
                'predicted_state': state_mapping.get(predicted_class, 'Unknown'),
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy().flatten(),
                'class_index': predicted_class
            }
            
        except Exception as e:
            logger.error(f"Error predicting market state: {e}")
            raise


class EnsembleMLPredictor:
    """
    Ensemble ML predictor combining multiple algorithms
    Includes LSTM, Transformer, XGBoost, LightGBM, and traditional ML models
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.scalers = {}
        self.weights = {}
        self.is_trained = False
        self.performance_history = {}
        
        # Initialize component models
        self.lstm_predictor = LSTMPricePredictor(config)
        self.transformer_analyzer = TransformerMarketAnalyzer(config)
        
        # Traditional ML models
        self.ml_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'linear_regression': LinearRegression()
        }
        
        logger.info("Ensemble ML Predictor initialized with multiple algorithms")
    
    def train(self, data: pd.DataFrame, target_column: str = 'close') -> Dict:
        """Train all models in the ensemble"""
        try:
            logger.info("Starting ensemble training...")
            
            # Prepare features
            features_df = self._prepare_ensemble_features(data)
            target = data[target_column].shift(-1).dropna()  # Predict next period
            
            # Align features and target
            min_length = min(len(features_df), len(target))
            features_df = features_df.iloc[:min_length]
            target = target.iloc[:min_length]
            
            # Split data
            split_idx = int(len(features_df) * 0.8)
            
            train_features = features_df.iloc[:split_idx]
            train_target = target.iloc[:split_idx]
            val_features = features_df.iloc[split_idx:]
            val_target = target.iloc[split_idx:]
            
            # Train deep learning models
            logger.info("Training deep learning models...")
            
            # Train LSTM
            try:
                lstm_history = self.lstm_predictor.train(data.iloc[:split_idx], target_column)
                self.performance_history['lstm'] = lstm_history
                logger.info("LSTM training completed")
            except Exception as e:
                logger.warning(f"LSTM training failed: {e}")
                self.performance_history['lstm'] = {'error': str(e)}
            
            # Train traditional ML models
            logger.info("Training traditional ML models...")
            model_performances = {}
            
            for name, model in self.ml_models.items():
                try:
                    # Scale features for this model
                    scaler = StandardScaler()
                    scaled_train_features = scaler.fit_transform(train_features)
                    scaled_val_features = scaler.transform(val_features)
                    
                    # Train model
                    model.fit(scaled_train_features, train_target)
                    
                    # Evaluate
                    train_pred = model.predict(scaled_train_features)
                    val_pred = model.predict(scaled_val_features)
                    
                    train_mse = mean_squared_error(train_target, train_pred)
                    val_mse = mean_squared_error(val_target, val_pred)
                    val_r2 = r2_score(val_target, val_pred)
                    
                    model_performances[name] = {
                        'train_mse': train_mse,
                        'val_mse': val_mse,
                        'val_r2': val_r2,
                        'model': model,
                        'scaler': scaler
                    }
                    
                    logger.info(f"{name}: Val MSE = {val_mse:.6f}, Val R² = {val_r2:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Training {name} failed: {e}")
                    model_performances[name] = {'error': str(e)}
            
            # Calculate ensemble weights based on performance
            self._calculate_ensemble_weights(model_performances)
            
            # Store models and scalers
            for name, perf in model_performances.items():
                if 'error' not in perf:
                    self.models[name] = perf['model']
                    self.scalers[name] = perf['scaler']
            
            self.performance_history['traditional_ml'] = model_performances
            self.is_trained = True
            
            logger.info("Ensemble training completed successfully")
            
            return {
                'lstm_history': self.performance_history.get('lstm', {}),
                'ml_performances': model_performances,
                'ensemble_weights': self.weights
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble training: {e}")
            raise
    
    def _prepare_ensemble_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for ensemble models"""
        try:
            # Start with engineered features from LSTM predictor
            predictor = LSTMPricePredictor()
            features = predictor._engineer_features(data)
            
            # Add additional features for traditional ML models
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features[f'close_lag_{lag}'] = features['close'].shift(lag)
                if 'volume' in features.columns:
                    features[f'volume_lag_{lag}'] = features['volume'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features[f'close_std_{window}'] = features['close'].rolling(window).std()
                features[f'close_min_{window}'] = features['close'].rolling(window).min()
                features[f'close_max_{window}'] = features['close'].rolling(window).max()
                features[f'close_median_{window}'] = features['close'].rolling(window).median()
            
            # Interaction features
            if 'volume' in features.columns:
                features['price_volume_interaction'] = features['close'] * features['volume']
                features['volume_price_ratio'] = features['volume'] / features['close']
            
            # Statistical features
            features['close_skew_20'] = features['close'].rolling(20).skew()
            features['close_kurtosis_20'] = features['close'].rolling(20).kurt()
            
            # Remove rows with NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing ensemble features: {e}")
            raise
    
    def _calculate_ensemble_weights(self, performances: Dict):
        """Calculate weights for ensemble based on model performance"""
        try:
            valid_models = {name: perf for name, perf in performances.items() 
                          if 'error' not in perf and 'val_r2' in perf}
            
            if not valid_models:
                logger.warning("No valid models for ensemble weighting")
                return
            
            # Calculate weights based on validation R²
            r2_scores = {name: max(0, perf['val_r2']) for name, perf in valid_models.items()}
            total_r2 = sum(r2_scores.values())
            
            if total_r2 > 0:
                # Normalize weights
                self.weights = {name: score / total_r2 for name, score in r2_scores.items()}
            else:
                # Equal weights if all models perform poorly
                self.weights = {name: 1.0 / len(valid_models) for name in valid_models.keys()}
            
            # Add weight for LSTM if trained successfully
            if hasattr(self.lstm_predictor, 'is_trained') and self.lstm_predictor.is_trained:
                # Give LSTM a base weight
                lstm_weight = 0.3
                # Adjust other weights
                adjustment_factor = (1 - lstm_weight) / sum(self.weights.values())
                self.weights = {name: weight * adjustment_factor for name, weight in self.weights.items()}
                self.weights['lstm'] = lstm_weight
            
            logger.info(f"Ensemble weights calculated: {self.weights}")
            
        except Exception as e:
            logger.error(f"Error calculating ensemble weights: {e}")
            # Fallback to equal weights
            model_names = list(self.models.keys())
            if model_names:
                self.weights = {name: 1.0 / len(model_names) for name in model_names}
    
    def predict(self, data: pd.DataFrame, return_individual: bool = False) -> Union[float, Dict]:
        """Make ensemble prediction"""
        try:
            if not self.is_trained:
                raise ValueError("Ensemble must be trained before making predictions")
            
            predictions = {}
            
            # Get LSTM prediction
            if hasattr(self.lstm_predictor, 'is_trained') and self.lstm_predictor.is_trained:
                try:
                    lstm_pred = self.lstm_predictor.predict(data, return_confidence=False)
                    predictions['lstm'] = float(lstm_pred[0]) if isinstance(lstm_pred, np.ndarray) else float(lstm_pred)
                except Exception as e:
                    logger.warning(f"LSTM prediction failed: {e}")
            
            # Get traditional ML predictions
            features_df = self._prepare_ensemble_features(data)
            if len(features_df) > 0:
                latest_features = features_df.iloc[-1:].values
                
                for name, model in self.models.items():
                    try:
                        scaler = self.scalers[name]
                        scaled_features = scaler.transform(latest_features)
                        pred = model.predict(scaled_features)[0]
                        predictions[name] = float(pred)
                    except Exception as e:
                        logger.warning(f"Prediction failed for {name}: {e}")
            
            if not predictions:
                raise ValueError("No models could make predictions")
            
            # Calculate weighted ensemble prediction
            ensemble_pred = 0.0
            total_weight = 0.0
            
            for name, pred in predictions.items():
                weight = self.weights.get(name, 0.0)
                ensemble_pred += weight * pred
                total_weight += weight
            
            if total_weight > 0:
                ensemble_pred /= total_weight
            else:
                # Fallback to simple average
                ensemble_pred = np.mean(list(predictions.values()))
            
            if return_individual:
                return {
                    'ensemble_prediction': ensemble_pred,
                    'individual_predictions': predictions,
                    'weights_used': {name: self.weights.get(name, 0.0) for name in predictions.keys()}
                }
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            raise
    
    def get_model_importance(self) -> Dict:
        """Get feature importance from tree-based models"""
        try:
            importance_dict = {}
            
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_dict[name] = model.feature_importances_.tolist()
                elif hasattr(model, 'coef_'):
                    importance_dict[name] = np.abs(model.coef_).tolist()
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error getting model importance: {e}")
            return {}
    
    def get_ensemble_status(self) -> Dict:
        """Get comprehensive ensemble status"""
        return {
            'is_trained': self.is_trained,
            'models_available': list(self.models.keys()),
            'ensemble_weights': self.weights,
            'lstm_status': self.lstm_predictor.get_model_info() if hasattr(self.lstm_predictor, 'get_model_info') else {},
            'performance_history': self.performance_history
        }