import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
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
        
        # Detectar anomalias
        autoencoder_predictions = autoencoder.predict(self.X_test_scaled)
        
        # Avaliar
        autoencoder_metrics = evaluate_model(
            self.y_test, 
            autoencoder_predictions, 
            "Autoencoder"
        )
        
        return autoencoder, history, autoencoder_predictions, autoencoder_metrics
    
    def run_xgboost_experiment(self):
        """Executa o experimento com XGBoost"""
        print("\n" + "="*50)
        print("EXPERIMENTO COM XGBOOST")
        print("="*50)
        
        # Criar e treinar o XGBoost
        xgb_model = XGBoostClassifier()
        xgb_model.train(self.X_train_scaled, self.y_train)
        
        # Fazer previsões
        xgb_predictions = xgb_model.predict(self.X_test_scaled)
        
        # Avaliar
        xgb_metrics = evaluate_model(
            self.y_test, 
            xgb_predictions, 
            "XGBoost"
        )
        
        return xgb_model, xgb_predictions, xgb_metrics
    
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
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
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
    autoencoder, history, ae_predictions, ae_metrics = project.run_autoencoder_experiment()
    
    # Executar experimento com XGBoost
    xgb_model, xgb_predictions, xgb_metrics = project.run_xgboost_experiment()
    
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