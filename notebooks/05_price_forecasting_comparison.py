"""
Script para comparação dos modelos de previsão de preços ARIMA, VAR e LSTM.

Este script carrega dados históricos de preços de metais, treina os três modelos,
e compara seu desempenho preditivo para análise de viabilidade econômica de 
nódulos polimetálicos.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime, timedelta

# Adicionar diretório raiz ao path de forma mais robusta
script_dir = os.path.dirname(os.path.abspath(__file__))  # Diretório do script atual
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))  # Diretório pai (raiz do projeto)
sys.path.insert(0, project_root)  # Adiciona a raiz do projeto ao path

# Agora importe o módulo
from src.models.price_forecasting import MetalPriceForecaster

def compare_models(self, metal, periods=12, train_size=0.8, arima_forecast=None, var_forecast=None, lstm_forecast=None):
    """
    Compara o desempenho de diferentes modelos de previsão.
    """
    print(f"Comparando modelos para {metal}...")
    
    # Resultados da comparação
    comparison = {
        'metal': metal,
        'metrics': {}
    }
    
    # Verificar se temos dados suficientes para testar
    if not hasattr(self, 'data') or len(self.data) < 2*periods:
        print("Aviso: Dados insuficientes para avaliação completa do modelo.")
        return comparison
    
    # Dividir dados em treino e teste para avaliação
    train_size_idx = int(len(self.data) * train_size)
    train_data = self.data.iloc[:train_size_idx]
    test_data = self.data.iloc[train_size_idx:train_size_idx+periods]
    
    if len(test_data) == 0:
        print("Aviso: Não há dados de teste disponíveis para avaliação.")
        return comparison
    
    # Obter previsões se não fornecidas
    if arima_forecast is None and f'ARIMA_{metal}' in self.models:
        try:
            arima_forecast = self.forecast_arima(metal, periods=len(test_data))
        except Exception as e:
            print(f"Erro ao obter previsão ARIMA: {e}")
    
    if var_forecast is None and 'VAR' in self.models:
        try:
            var_forecast = self.forecast_var(periods=len(test_data))
        except Exception as e:
            print(f"Erro ao obter previsão VAR: {e}")
    
    if lstm_forecast is None and f'LSTM_{metal}' in self.models:
        try:
            lstm_forecast = self.forecast_lstm(metal, periods=len(test_data))
        except Exception as e:
            print(f"Erro ao obter previsão LSTM: {e}")
    
    # Valores reais para comparação
    try:
        actual_values = test_data[metal].values
    except:
        print(f"Erro ao obter valores reais para {metal}")
        return comparison
    
    # Calcular métricas para cada modelo
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # ARIMA
    if arima_forecast is not None and 'forecast' in arima_forecast:
        # Ajustar índices para corresponder ao período de teste
        arima_values = arima_forecast['forecast'].values[:len(actual_values)]
        if len(arima_values) > 0:
            mse = mean_squared_error(actual_values, arima_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_values, arima_values)
            
            comparison['metrics']['ARIMA'] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae
            }
    
    # VAR
    if var_forecast is not None and metal in var_forecast.columns:
        var_values = var_forecast[metal].values[:len(actual_values)]
        if len(var_values) > 0:
            mse = mean_squared_error(actual_values, var_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_values, var_values)
            
            comparison['metrics']['VAR'] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae
            }
    
    # LSTM
    if lstm_forecast is not None:
        lstm_values = lstm_forecast.values[:len(actual_values)]
        if len(lstm_values) > 0:
            mse = mean_squared_error(actual_values, lstm_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_values, lstm_values)
            
            comparison['metrics']['LSTM'] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae
            }
    
    # Determinar o melhor modelo
    if comparison['metrics']:
        best_model = min(comparison['metrics'].items(), 
                        key=lambda x: x[1]['RMSE'])[0]
        comparison['best_model'] = best_model
        print(f"Melhor modelo para {metal}: {best_model}")
    
    # Visualizar comparação (versão simplificada)
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(actual_values)), actual_values, 'k-', label='Valores Reais')
    
    if 'ARIMA' in comparison['metrics']:
        arima_values = arima_forecast['forecast'].values[:len(actual_values)]
        plt.plot(range(len(arima_values)), arima_values, 'b-', label='ARIMA')
    
    if 'VAR' in comparison['metrics']:
        var_values = var_forecast[metal].values[:len(actual_values)]
        plt.plot(range(len(var_values)), var_values, 'g-', label='VAR')
    
    if 'LSTM' in comparison['metrics']:
        lstm_values = lstm_forecast.values[:len(actual_values)]
        plt.plot(range(len(lstm_values)), lstm_values, 'r-', label='LSTM')
    
    plt.title(f'Comparação de Modelos de Previsão para {metal}')
    plt.xlabel('Período de Previsão')
    plt.ylabel('Preço (USD/ton)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(self.results_dir, f'model_comparison_{metal}.png'))
    plt.close()
    
    return comparison

# Adicionar o método à classe MetalPriceForecaster
MetalPriceForecaster.compare_models = compare_models

# Adicionar método split_data se não existir
if not hasattr(MetalPriceForecaster, 'split_data'):
    def split_data(self, data, train_size=0.8):
        """Divide os dados em conjuntos de treino e teste."""
        split_index = int(len(data) * train_size)
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
        return train_data, test_data
    
    MetalPriceForecaster.split_data = split_data

# Configurações
plt.style.use('ggplot')
results_dir = '../results/forecasting'
os.makedirs(results_dir, exist_ok=True)

# Verificar caminhos possíveis do banco de dados
possible_paths = [
    '../data/nodules.db',           # Relativo ao diretório notebooks
    'data/nodules.db',              # Relativo ao diretório raiz
    './data/nodules.db',            # Explicitamente no diretório data do diretório atual
    '../data/raw/nodules.db',       # Talvez esteja no diretório raw
    'data/raw/nodules.db'           # Outra possibilidade
]

# Primeiro caminho que existir
db_path = None
for path in possible_paths:
    if os.path.exists(os.path.abspath(path)):
        db_path = path
        print(f"Banco de dados encontrado em: {os.path.abspath(path)}")
        break

if db_path is None:
    print("ERRO: Banco de dados não encontrado em nenhum dos caminhos possíveis!")
    # Verificar se há algum arquivo .db em algum lugar
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.db'):
                print(f"Arquivo .db encontrado: {os.path.join(root, file)}")
    sys.exit(1)

def main():
    """Função principal para executar a comparação de modelos."""
    print("Iniciando comparação de modelos de previsão de preços...")
    print("=" * 80)
    
    # Inicializar forecaster
    forecaster = MetalPriceForecaster(database_path=db_path)
    
    # Carregar dados históricos
    print("\n1. Carregando dados de preços...")
    data = forecaster.load_data()
    
    # Dividir dados em treino e teste
    train_size = 0.8
    train_data, test_data = forecaster.split_data(data, train_size)
    print(f"Dados divididos em treino ({len(train_data)} registros) e teste ({len(test_data)} registros)")
    
    # Realizar análise exploratória
    print("\n2. Realizando análise exploratória...")
    eda_results = forecaster.exploratory_analysis()
    print("Análise exploratória concluída. Figuras salvas em:", results_dir)
    
    # Período de previsão
    forecast_periods = 12
    
    # Resultados da comparação
    comparison_results = {}
    
    # Para cada metal, treinar e avaliar os três modelos
    metals = data.columns.tolist()
    
    for metal in metals:
        print(f"\n3. Analisando {metal}...")
        print("-" * 50)
        
        # 3.1 Treinar modelo ARIMA
        print(f"3.1 Treinando modelo ARIMA para {metal}...")
        arima_results = forecaster.train_arima(metal, auto=True)
        
        # 3.2 Realizar previsão ARIMA
        print(f"3.2 Realizando previsão ARIMA para {metal}...")
        arima_forecast = forecaster.forecast_arima(metal, periods=forecast_periods)
        
        # 3.3 Treinar modelo VAR (apenas uma vez)
        if metal == metals[0]:
            print("\n3.3 Treinando modelo VAR para todos os metais...")
            try:
                var_results = forecaster.train_var()
                
                print("3.4 Realizando previsão VAR...")
                var_forecast = forecaster.forecast_var(periods=forecast_periods)
                has_var = True
            except Exception as e:
                print(f"  AVISO: Não foi possível treinar o modelo VAR: {e}")
                var_forecast = None
                has_var = False
        
        # 3.5 Treinar modelo LSTM
        print(f"\n3.5 Treinando modelo LSTM para {metal}...")
        try:
            lstm_results = forecaster.train_lstm(metal, epochs=100)
            
            print(f"3.6 Realizando previsão LSTM para {metal}...")
            lstm_forecast = forecaster.forecast_lstm(metal, periods=forecast_periods)
            has_lstm = True
        except Exception as e:
            print(f"  AVISO: Não foi possível treinar o modelo LSTM: {e}")
            lstm_forecast = None
            has_lstm = False
        
        # 3.7 Comparar modelos
        print(f"\n3.7 Comparando modelos para {metal}...")
        model_comparison = forecaster.compare_models(
            metal,
            arima_forecast=arima_forecast,
            var_forecast=var_forecast if has_var else None,
            lstm_forecast=lstm_forecast if has_lstm else None,
            periods=forecast_periods,
            train_size=train_size
        )
        comparison_results[metal] = model_comparison
        
        # 3.8 Exibir resultados da comparação
        print(f"\nResultados da comparação para {metal}:")
        
        if 'metrics' in model_comparison and model_comparison['metrics']:
            print("\nMétricas de erro:")
            for model_name, metrics in model_comparison['metrics'].items():
                print(f"  {model_name}:")
                for metric_name, value in metrics.items():
                    print(f"    {metric_name}: {value:.2f}")
            
            if 'best_model' in model_comparison:
                best_model = model_comparison['best_model']
                print(f"\nMelhor modelo: {best_model}")
        else:
            print("  Não há métricas disponíveis (não há dados de teste suficientes)")
    
    # 4. Criar tabela comparativa final
    print("\n4. Criando tabela comparativa final...")
    create_comparison_table(comparison_results, os.path.join(results_dir, 'model_comparison_table.csv'))
    
    # 5. Criar visualização comparativa final
    print("\n5. Criando visualização comparativa final...")
    create_comparison_visualization(comparison_results, os.path.join(results_dir, 'model_comparison_all.png'))
    
    print("\nComparação de modelos concluída!")
    print("Resultados e visualizações salvas em:", results_dir)

def create_comparison_table(results, output_path):
    """
    Cria tabela comparativa de métricas para todos os metais e modelos.
    
    Parameters:
    -----------
    results : dict
        Resultados da comparação para cada metal
    output_path : str
        Caminho para salvar a tabela
    """
    # Preparar dados para tabela
    table_data = []
    
    for metal, comparison in results.items():
        if 'metrics' in comparison and comparison['metrics']:
            for model, metrics in comparison['metrics'].items():
                row = {
                    'Metal': metal,
                    'Model': model
                }
                row.update(metrics)
                table_data.append(row)
    
    # Criar DataFrame
    table_df = pd.DataFrame(table_data)
    
    # Reordenar colunas
    column_order = ['Metal', 'Model', 'MSE', 'RMSE', 'MAE']
    table_df = table_df[column_order]
    
    # Salvar tabela
    table_df.to_csv(output_path, index=False)
    
    # Exibir tabela
    print("\nTabela comparativa:")
    print(table_df)

def create_comparison_visualization(results, output_path):
    """
    Cria visualização comparativa de RMSE para todos os metais e modelos.
    
    Parameters:
    -----------
    results : dict
        Resultados da comparação para cada metal
    output_path : str
        Caminho para salvar a visualização
    """
    # Extrair RMSE para cada metal e modelo
    metals = []
    arima_rmse = []
    var_rmse = []
    lstm_rmse = []
    
    for metal, comparison in results.items():
        if 'metrics' in comparison and comparison['metrics']:
            metals.append(metal)
            
            if 'ARIMA' in comparison['metrics']:
                arima_rmse.append(comparison['metrics']['ARIMA']['RMSE'])
            else:
                arima_rmse.append(np.nan)
            
            if 'VAR' in comparison['metrics']:
                var_rmse.append(comparison['metrics']['VAR']['RMSE'])
            else:
                var_rmse.append(np.nan)
            
            if 'LSTM' in comparison['metrics']:
                lstm_rmse.append(comparison['metrics']['LSTM']['RMSE'])
            else:
                lstm_rmse.append(np.nan)
    
    # Criar gráfico de barras
    x = np.arange(len(metals))
    width = 0.25
    
    plt.figure(figsize=(14, 8))
    
    bar1 = plt.bar(x - width, arima_rmse, width, label='ARIMA', color='blue')
    bar2 = plt.bar(x, var_rmse, width, label='VAR', color='green')
    bar3 = plt.bar(x + width, lstm_rmse, width, label='LSTM', color='red')
    
    plt.xlabel('Metal')
    plt.ylabel('RMSE (Raiz do Erro Quadrático Médio)')
    plt.title('Comparação de Modelos de Previsão por Metal')
    plt.xticks(x, metals)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionar rótulos de valores
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
    
    add_labels(bar1)
    add_labels(bar2)
    add_labels(bar3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    main()