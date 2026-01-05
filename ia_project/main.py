import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans

from src.autoencoder_model import AutoencoderAnomalyDetector
from src.xgboost_model import XGBoostClassifier
from src.utils import plot_results, evaluate_model

class IAProject:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = StandardScaler()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, ax=None):
        """Plota a curva ROC e calcula AUC"""
        # Calcular ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plotar
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Classificador Aleatório')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.set_title(f'Curva ROC - {model_name}')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        return roc_auc
    
    def load_your_data(self, x_file_path, y_file_path, problem_type='regression'):
        """Carrega seus próprios dados de arquivos separados X e Y"""
        print(f"Carregando features de: {x_file_path}")
        print(f"Carregando target de: {y_file_path}")
        
        # Carregar features (X)
        x_data = pd.read_csv(x_file_path)
        
        # Carregar target (Y)
        y_data = pd.read_csv(y_file_path)
        
        # Verificar e corrigir problemas nos dados
        x_data = self._clean_data(x_data)
        
        # Juntar os dados
        self.data = x_data.copy()
        
        # Encontrar a coluna de target no y_data (deve ser a segunda coluna)
        target_column = y_data.columns[1] if len(y_data.columns) > 1 else y_data.columns[0]
        self.data['target'] = y_data[target_column].values
        
        print(f"Shape dos dados carregados: {self.data.shape}")
        print(f"Colunas: {self.data.columns.tolist()}")
        print(f"Estatísticas do target:")
        print(f"  Min: {self.data['target'].min():.3f}")
        print(f"  Max: {self.data['target'].max():.3f}")
        print(f"  Mean: {self.data['target'].mean():.3f}")
        print(f"  Std: {self.data['target'].std():.3f}")
        
        # Converter para problema de classificação se necessário
        if problem_type == 'classification':
            self._convert_to_classification()
        elif problem_type == 'anomaly':
            self._create_anomaly_target()
        
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        return X, y
    
    def _clean_data(self, df):
        """Limpa e corrige problemas nos dados"""
        print("Limpando dados...")
        
        # Fazer uma cópia
        df_clean = df.copy()
        
        # Remover coluna ID se existir
        id_columns = [col for col in df_clean.columns if 'ID' in col or 'id' in col]
        if id_columns:
            print(f"Removendo colunas ID: {id_columns}")
            df_clean = df_clean.drop(columns=id_columns)
        
        # Corrigir problemas de parsing (vírgulas extras)
        for col in df_clean.columns:
            # Converter para string, tratar problemas e converter de volta
            df_clean[col] = df_clean[col].astype(str)
            # Remover múltiplos pontos consecutivos
            df_clean[col] = df_clean[col].str.replace('\.\.', '.', regex=True)
            # Corrigir padrões como "0.0.1" para "0.001"
            df_clean[col] = df_clean[col].str.replace(r'(\d+)\.(\d+)\.(\d+)', 
                                                    lambda x: f"{x.group(1)}.{x.group(2)}{x.group(3)}", 
                                                    regex=True)
            # Converter de volta para numérico
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Preencher valores NaN
        if df_clean.isnull().sum().sum() > 0:
            print(f"Preenchendo {df_clean.isnull().sum().sum()} valores NaN...")
            df_clean = df_clean.fillna(df_clean.mean())
        
        print(f"Shape após limpeza: {df_clean.shape}")
        return df_clean
    
    def _convert_to_classification(self):
        """Converte problema de regressão para classificação"""
        print("Convertendo para problema de classificação...")
        
        # Usar quantis para criar classes balanceadas
        n_classes = 3  # Podemos ajustar isso
        self.data['target_class'] = pd.qcut(self.data['target'], n_classes, labels=False)
        
        # Substituir o target original
        original_target = self.data['target'].copy()
        self.data['target'] = self.data['target_class']
        self.data = self.data.drop('target_class', axis=1)
        
        print(f"Distribuição das classes: {self.data['target'].value_counts().sort_index().to_dict()}")
        print(f"Intervalos originais:")
        for i in range(n_classes):
            mask = self.data['target'] == i
            min_val = original_target[mask].min()
            max_val = original_target[mask].max()
            print(f"  Classe {i}: [{min_val:.3f}, {max_val:.3f}]")
    
    def _create_anomaly_target(self):
        """Cria um target binário para detecção de anomalias"""
        print("Criando problema de detecção de anomalias...")
        
        from sklearn.ensemble import IsolationForest
        
        # Usar apenas as features para detectar anomalias
        features = self.data.drop('target', axis=1)
        contamination = 0.1  # 10% dos dados como anomalias
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(features)
        
        # Salvar o target original
        original_target = self.data['target'].copy()
        self.data['original_target'] = original_target
        
        # Converter para 0 (normal) e 1 (anomalia)
        self.data['target'] = (anomalies == -1).astype(int)
        
        print(f"Distribuição criada: {self.data['target'].value_counts().to_dict()}")
    
    def preprocess_data(self, test_size=0.2, use_stratify=True):
        """Preprocessa os dados"""
        print("Preprocessando dados...")
        
        # Verificar se os dados foram carregados
        if self.data is None:
            raise ValueError("Dados não carregados. Chame load_your_data() primeiro.")
        
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # Verificar se podemos usar stratify (apenas para classificação)
        if y.dtype == 'object' or len(y.unique()) > 10:
            use_stratify = False
        
        if use_stratify:
            min_samples_per_class = y.value_counts().min()
            can_stratify = min_samples_per_class >= 2
            
            print(f"Menor classe tem {min_samples_per_class} amostras")
            print(f"Pode usar stratify: {can_stratify}")
        else:
            can_stratify = False
        
        if can_stratify:
            # Dividir os dados com stratify
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # Dividir sem stratify
            print("Dividindo dados sem stratify...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=None
            )
        
        # Normalizar os dados
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Shape dos dados de treino: {self.X_train_scaled.shape}")
        print(f"Shape dos dados de teste: {self.X_test_scaled.shape}")
        print(f"Tipo do problema: {'Classificação' if len(np.unique(y)) < 10 else 'Regressão'}")
        
        if len(np.unique(y)) < 10:
            print(f"Distribuição no treino: {pd.Series(self.y_train).value_counts().to_dict()}")
            print(f"Distribuição no teste: {pd.Series(self.y_test).value_counts().to_dict()}")
    
    def run_autoencoder_experiment(self):
        """Executa o experimento com Autoencoder"""
        print("\n" + "="*50)
        print("EXPERIMENTO COM AUTOENCODER")
        print("="*50)
        
        # Para autoencoder, vamos usar detecção de anomalias não supervisionada
        autoencoder = AutoencoderAnomalyDetector(
            input_dim=self.X_train_scaled.shape[1],
            encoding_dim=8,
            hidden_layers=[16, 12]
        )
        
        # Treinar apenas com features (sem usar labels)
        history = autoencoder.train(
            self.X_train_scaled, 
            self.X_test_scaled,
            epochs=100,
            batch_size=32
        )
        
        # Detectar anomalias - obter probabilidades também
        autoencoder_predictions = autoencoder.predict(self.X_test_scaled)
        autoencoder_proba = autoencoder.predict_proba(self.X_test_scaled)
        
        # Plotar ROC para Autoencoder
        fig, ax = plt.subplots(figsize=(8, 6))
        
        auc_autoencoder = self.plot_roc_curve(
            self.y_test, autoencoder_proba, "Autoencoder", ax
        )
        
        plt.tight_layout()
        plt.show()
        
        # Avaliar
        autoencoder_metrics = evaluate_model(
            self.y_test, 
            autoencoder_predictions, 
            "Autoencoder"
        )
        
        # Adicionar AUC nas métricas
        autoencoder_metrics['AUC'] = auc_autoencoder
        
        return autoencoder, history, autoencoder_predictions, autoencoder_metrics, autoencoder_proba
    
    def run_xgboost_experiment(self):
        """Executa o experimento com XGBoost"""
        print("\n" + "="*50)
        print("EXPERIMENTO COM XGBOOST")
        print("="*50)
        
        # Criar e treinar o XGBoost
        xgb_model = XGBoostClassifier()
        xgb_model.train(self.X_train_scaled, self.y_train)
        
        # Fazer previsões - obter probabilidades também
        xgb_predictions = xgb_model.predict(self.X_test_scaled)
        xgb_proba = xgb_model.predict_proba(self.X_test_scaled)
        
        # Plotar ROC para XGBoost
        fig, ax = plt.subplots(figsize=(8, 6))
        
        auc_xgb = self.plot_roc_curve(
            self.y_test, xgb_proba, "XGBoost", ax
        )
        
        plt.tight_layout()
        plt.show()
        
        # Avaliar
        xgb_metrics = evaluate_model(
            self.y_test, 
            xgb_predictions, 
            "XGBoost"
        )
        
        # Adicionar AUC nas métricas
        xgb_metrics['AUC'] = auc_xgb
        
        return xgb_model, xgb_predictions, xgb_metrics, xgb_proba
    
    def compare_models(self, autoencoder_metrics, xgb_metrics):
        """Compara os resultados dos dois modelos"""
        print("\n" + "="*50)
        print("COMPARAÇÃO DOS MODELOS")
        print("="*50)
        
        comparison_df = pd.DataFrame({
            'Autoencoder': autoencoder_metrics,
            'XGBoost': xgb_metrics
        })
        
        print(comparison_df)
        
        # Plotar comparação
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # Agora com 3 gráficos
        
        # Acurácia
        models = ['Autoencoder', 'XGBoost']
        accuracies = [autoencoder_metrics['Acurácia'], xgb_metrics['Acurácia']]
        
        axes[0].bar(models, accuracies, color=['blue', 'orange'])
        axes[0].set_title('Comparação de Acurácia')
        axes[0].set_ylabel('Acurácia')
        
        # F1-Score
        f1_scores = [autoencoder_metrics['F1-Score'], xgb_metrics['F1-Score']]
        
        axes[1].bar(models, f1_scores, color=['blue', 'orange'])
        axes[1].set_title('Comparação de F1-Score')
        axes[1].set_ylabel('F1-Score')
        
        # AUC
        auc_scores = [autoencoder_metrics['AUC'], xgb_metrics['AUC']]
        
        axes[2].bar(models, auc_scores, color=['blue', 'orange'])
        axes[2].set_title('Comparação de AUC')
        axes[2].set_ylabel('AUC Score')
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df

def main():
    # Inicializar o projeto
    project = IAProject()
    
    # Caminhos para SEUS dados
    x_file_path = "C:\\Users\\Rumanns\\Desktop\\_USP IA\\ia_project\\data\\x.csv"
    y_file_path = "C:\\Users\\Rumanns\\Desktop\\_USP IA\\ia_project\\data\\y.csv"
    
    # ESCOLHA UMA DAS OPÇÕES:
    
    # Opção 1: Detecção de Anomalias (Recomendado para Autoencoder)
    X, y = project.load_your_data(x_file_path, y_file_path, problem_type='anomaly')
    
    # Opção 2: Classificação (Converte regressão em classificação)
    # X, y = project.load_your_data(x_file_path, y_file_path, problem_type='classification')
    
    # Preprocessar dados
    project.preprocess_data()
    
    # Executar experimento com Autoencoder
    autoencoder, history, ae_predictions, ae_metrics, ae_proba = project.run_autoencoder_experiment()
    
    # Executar experimento com XGBoost
    xgb_model, xgb_predictions, xgb_metrics, xgb_proba = project.run_xgboost_experiment()
    
    # Plotar ROC comparativa
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Curva ROC para Autoencoder
    fpr_ae, tpr_ae, _ = roc_curve(project.y_test, ae_proba)
    auc_ae = auc(fpr_ae, tpr_ae)
    
    # Curva ROC para XGBoost
    fpr_xgb, tpr_xgb, _ = roc_curve(project.y_test, xgb_proba)
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    
    ax.plot(fpr_ae, tpr_ae, color='blue', lw=2, 
            label=f'Autoencoder (AUC = {auc_ae:.3f})')
    ax.plot(fpr_xgb, tpr_xgb, color='red', lw=2, 
            label=f'XGBoost (AUC = {auc_xgb:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
            label='Classificador Aleatório')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Comparação das Curvas ROC')
    ax.legend(loc="lower right")
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Comparar modelos
    comparison = project.compare_models(ae_metrics, xgb_metrics)
    
    # Plotar resultados detalhados
    plot_results(
        project.X_test_scaled,
        project.y_test,
        ae_predictions,
        xgb_predictions,
        history
    )

if __name__ == "__main__":
    main()