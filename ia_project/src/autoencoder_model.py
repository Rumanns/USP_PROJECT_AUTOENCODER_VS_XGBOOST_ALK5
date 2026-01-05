import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class AutoencoderAnomalyDetector:
    def __init__(self, input_dim, encoding_dim=8, hidden_layers=None, dropout_rate=0.2):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers if hidden_layers else []
        self.dropout_rate = dropout_rate
        self.model = None
        self.encoder = None
        self.history = None
        self.threshold = None
        
        self._build_model()
    
    def _build_model(self):
        """Constrói a arquitetura do autoencoder"""
        # Input
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = input_layer
        for units in self.hidden_layers:
            encoded = Dense(units, activation='relu')(encoded)
            encoded = Dropout(self.dropout_rate)(encoded)
        
        # Bottleneck
        encoded = Dense(self.encoding_dim, activation='relu', name='bottleneck')(encoded)
        
        # Decoder
        decoded = encoded
        for units in reversed(self.hidden_layers):
            decoded = Dense(units, activation='relu')(decoded)
            decoded = Dropout(self.dropout_rate)(decoded)
        
        # Output
        output_layer = Dense(self.input_dim, activation='linear')(decoded)
        
        # Modelos
        self.model = Model(input_layer, output_layer)
        self.encoder = Model(input_layer, encoded)
        
        # Compilar
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        print("Autoencoder construído com sucesso!")
        print(self.model.summary())
    
    def train(self, X_train, X_val, epochs=100, batch_size=32):
        """Treina o autoencoder"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calcular threshold baseado nos dados de treino
        train_reconstructions = self.model.predict(X_train)
        train_mse = np.mean(np.power(X_train - train_reconstructions, 2), axis=1)
        self.threshold = np.percentile(train_mse, 95)
        
        print(f"Threshold definido: {self.threshold:.4f}")
        
        return self.history
    
    def predict(self, X):
        """Faz previsões e detecta anomalias"""
        # Reconstruir dados
        reconstructions = self.model.predict(X)
        
        # Calcular erro de reconstrução
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        # Classificar como anomalia se o erro for maior que o threshold
        predictions = (mse > self.threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """Retorna probabilidades de ser anomalia [0, 1]"""
        # Reconstruir dados
        reconstructions = self.model.predict(X)
        
        # Calcular erro de reconstrução
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        # Normalizar o MSE para probabilidades entre 0 e 1
        # Quanto maior o erro, maior a probabilidade de ser anomalia
        if self.threshold is None:
            # Se não foi treinado, usar threshold temporário
            temp_threshold = np.percentile(mse, 95) if len(mse) > 0 else 1.0
        else:
            temp_threshold = self.threshold
        
        # Converter MSE em probabilidades usando função sigmoid
        # Probabilidade aumenta conforme se aproxima do threshold
        probabilities = 1 / (1 + np.exp(-(mse - temp_threshold/2) / (temp_threshold/10)))
        
        # Garantir que temos probabilidades para a classe 1 (anomalia)
        # Para curva ROC, precisamos da probabilidade da classe positiva
        return probabilities
    
    def get_reconstruction_error(self, X):
        """Retorna o erro de reconstrução"""
        reconstructions = self.model.predict(X)
        return np.mean(np.power(X - reconstructions, 2), axis=1)