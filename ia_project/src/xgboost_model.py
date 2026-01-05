import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd  # Adicionei esta importação que estava faltando

class XGBoostClassifier:
    def __init__(self):
        self.model = None
        self.best_params = None
    
    def train(self, X_train, y_train, cv_folds=5):
        """Treina o modelo XGBoost com validação cruzada"""
        
        # Definir parâmetros para busca
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        # Buscar melhores parâmetros
        xgb_model = xgb.XGBClassifier(random_state=42)
        grid_search = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=cv_folds, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Melhores parâmetros encontrados: {self.best_params}")
        
        return self.model
    
    def predict(self, X):
        """Faz previsões"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Retorna probabilidades da classe positiva (anomalia)"""
        # XGBoost retorna probabilidades para todas as classes
        # Para problemas binários: [prob_classe_0, prob_classe_1]
        probas = self.model.predict_proba(X)
        
        # Se for problema binário, retornar apenas probabilidade da classe 1
        if probas.shape[1] == 2:
            return probas[:, 1]  # Probabilidade da classe positiva
        else:
            # Para multi-classes, retornar a probabilidade da classe com maior score
            return np.max(probas, axis=1)
    
    def get_feature_importance(self, feature_names):
        """Retorna a importância das features"""
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df