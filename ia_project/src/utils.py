import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def plot_results(X_test, y_test, ae_predictions, xgb_predictions, history):
    """Plota os resultados comparativos"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Histórico de treinamento do Autoencoder
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Autoencoder - Loss durante Treinamento')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # 2. Matriz de confusão - Autoencoder
    cm_ae = confusion_matrix(y_test, ae_predictions)
    sns.heatmap(cm_ae, annot=True, fmt='d', ax=axes[0, 1])
    axes[0, 1].set_title('Matriz de Confusão - Autoencoder')
    
    # 3. Matriz de confusão - XGBoost
    cm_xgb = confusion_matrix(y_test, xgb_predictions)
    sns.heatmap(cm_xgb, annot=True, fmt='d', ax=axes[0, 2])
    axes[0, 2].set_title('Matriz de Confusão - XGBoost')
    
    # 4. Comparação de previsões
    comparison = pd.DataFrame({
        'Real': y_test.values,
        'Autoencoder': ae_predictions,
        'XGBoost': xgb_predictions
    })
    
    axes[1, 0].scatter(range(len(y_test)), y_test, alpha=0.5, label='Real')
    axes[1, 0].scatter(range(len(y_test)), ae_predictions, alpha=0.5, label='Autoencoder')
    axes[1, 0].set_title('Previsões vs Real - Autoencoder')
    axes[1, 0].legend()
    
    axes[1, 1].scatter(range(len(y_test)), y_test, alpha=0.5, label='Real')
    axes[1, 1].scatter(range(len(y_test)), xgb_predictions, alpha=0.5, label='XGBoost')
    axes[1, 1].set_title('Previsões vs Real - XGBoost')
    axes[1, 1].legend()
    
    # 5. Acertos e erros
    ae_correct = (ae_predictions == y_test).sum()
    xgb_correct = (xgb_predictions == y_test).sum()
    total = len(y_test)
    
    models = ['Autoencoder', 'XGBoost']
    correct = [ae_correct, xgb_correct]
    incorrect = [total - ae_correct, total - xgb_correct]
    
    x = np.arange(len(models))
    axes[1, 2].bar(x - 0.2, correct, 0.4, label='Corretos')
    axes[1, 2].bar(x + 0.2, incorrect, 0.4, label='Incorretos')
    axes[1, 2].set_title('Comparação de Acertos')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(models)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(y_true, y_pred, model_name):
    """Avalia o modelo e retorna métricas"""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n--- {model_name} ---")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred))
    
    return {
        'Acurácia': accuracy,
        'F1-Score': f1
    }