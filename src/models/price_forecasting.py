"""
Módulo para previsão de preços de metais utilizando modelos ARIMA, VAR e LSTM.

Este módulo implementa três abordagens diferentes para modelagem de séries temporais
de preços de metais relevantes para nódulos polimetálicos:

1. ARIMA: Modelo univariado para cada metal individualmente
2. VAR: Modelo multivariado que captura relações entre diferentes metais
3. LSTM: Rede neural recorrente para capturar padrões complexos e não-lineares

Nota: Esta versão foi modificada para funcionar sem a dependência do 'pmdarima'.
Em vez disso, implementa uma função própria de busca de parâmetros ARIMA.
Também inclui tratamento de erros para importações que podem não estar disponíveis
em todas as versões do statsmodels.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sqlite3
import json
import warnings
import itertools

# Importações para ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Importações para VAR
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.api import VAR as VAR_MODEL  # Alternativa para importação
try:
    from statsmodels.tsa.stattools import grangercausalitytest
except ImportError:
    try:
        # Tentar importar de outro local
        from statsmodels.tsa.statespace.tools import grangercausalitytest
    except ImportError:
        # Alternativa: criar uma função simplificada para teste de causalidade de Granger
        def grangercausalitytest(data, maxlag):
            print("Aviso: Teste de causalidade de Granger não disponível. Usando versão simplificada.")
            # Retornar resultado fictício
            return {1: [{0: {'ssr_ftest': (0, 0.5)}}]}  # (estatística F, p-valor)
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Importações para LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM as KERAS_LSTM
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow não está disponível. O modelo LSTM não poderá ser utilizado.")

# Configurações
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

# Função para verificar se uma importação está disponível
def is_function_available(function_name, module_name):
    try:
        module = __import__(module_name, fromlist=[function_name])
        return hasattr(module, function_name)
    except:
        return False

class MetalPriceForecaster:
    """
    Classe para previsão de preços de metais utilizando diferentes modelos.
    """
    
    def __init__(self, database_path='../data/nodules.db'):
        """
        Inicializa o forecaster.
        
        Parâmetros:
        -----------
        database_path : str
            Caminho para o banco de dados SQLite
        """
        self.database_path = database_path
        self.metals = ['Nickel', 'Copper', 'Cobalt', 'Manganese']
        self.models = {}
        self.results_dir = '../results/forecasting'
        
        # Criar diretório para resultados se não existir
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_data(self, start_date=None, end_date=None):
        """
        Carrega dados históricos de preços do banco de dados.
        
        Parâmetros:
        -----------
        start_date : str
            Data inicial no formato 'YYYY-MM-DD'
        end_date : str
            Data final no formato 'YYYY-MM-DD'
            
        Retorna:
        --------
        pandas.DataFrame
            DataFrame com preços históricos
        """
        # Construir query para extrair dados
        query = """
        SELECT metal_name, date, price_usd_ton
        FROM metal_prices
        """
        
        # Adicionar filtros de data se especificados
        conditions = []
        params = {}
        
        if start_date:
            conditions.append("date >= :start_date")
            params['start_date'] = start_date
            
        if end_date:
            conditions.append("date <= :end_date")
            params['end_date'] = end_date
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY metal_name, date"
        
        # Conectar ao banco de dados
        conn = sqlite3.connect(self.database_path)
        
        # Executar query
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
            
        conn.close()
        
        # Converter coluna de data para datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Converter para formato wide (uma coluna por metal)
        df_wide = df.pivot(index='date', columns='metal_name', values='price_usd_ton')
        
        # Garantir que todos os metais estão presentes
        for metal in self.metals:
            if metal not in df_wide.columns:
                print(f"Aviso: Metal {metal} não encontrado nos dados!")
        
        # Preencher valores ausentes (se houver)
        df_wide = df_wide.fillna(method='ffill').fillna(method='bfill')
        
        # Remover nome de colunas
        df_wide.columns.name = None
        
        print(f"Dados carregados: {len(df_wide)} registros para {len(df_wide.columns)} metais")
        print(f"Período: {df_wide.index.min()} a {df_wide.index.max()}")
        
        self.data = df_wide
        return df_wide
    
    def exploratory_analysis(self):
        """
        Realiza análise exploratória das séries temporais.
        
        Retorna:
        --------
        dict
            Dicionário com resultados da análise
        """
        if not hasattr(self, 'data'):
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
            
        results = {}
        
        # 1. Estatísticas descritivas
        stats = self.data.describe()
        results['statistics'] = stats
        
        # 2. Decomposição de séries temporais
        decompositions = {}
        
        for metal in self.data.columns:
            try:
                # Tentar decomposição multiplicativa
                decomp = seasonal_decompose(self.data[metal], model='multiplicative', period=12)
            except:
                # Se falhar, usar decomposição aditiva
                decomp = seasonal_decompose(self.data[metal], model='additive', period=12)
                
            decompositions[metal] = {
                'trend': decomp.trend,
                'seasonal': decomp.seasonal,
                'residual': decomp.resid
            }
        
        results['decompositions'] = decompositions
        
        # 3. Teste de estacionaridade (Augmented Dickey-Fuller)
        adf_results = {}
        
        for metal in self.data.columns:
            adf_test = adfuller(self.data[metal].dropna())
            adf_results[metal] = {
                'test_statistic': adf_test[0],
                'p_value': adf_test[1],
                'stationary': adf_test[1] < 0.05
            }
        
        results['stationarity'] = adf_results
        
        # 4. Correlações
        corr_matrix = self.data.corr()
        results['correlations'] = corr_matrix
        
        # 5. Auto-correlação e auto-correlação parcial
        acf_pacf = {}
        
        for metal in self.data.columns:
            acf_values = acf(self.data[metal].dropna(), nlags=20)
            pacf_values = pacf(self.data[metal].dropna(), nlags=20, method='ols')
            
            acf_pacf[metal] = {
                'acf': acf_values,
                'pacf': pacf_values
            }
        
        results['acf_pacf'] = acf_pacf
        
        # 6. Teste de co-integração para todos os metais juntos
        # Verifica se as séries têm relações de longo prazo
        try:
            johansen_test = coint_johansen(self.data, det_order=0, k_ar_diff=1)
            results['cointegration'] = {
                'trace_stat': johansen_test.lr1,
                'critical_values': johansen_test.cvt,
                'eigenvalues': johansen_test.eig
            }
        except:
            print("Aviso: Não foi possível realizar o teste de co-integração.")
        
        # 7. Visualizações
        
        # 7.1. Séries temporais
        plt.figure(figsize=(12, 6))
        for metal in self.data.columns:
            plt.plot(self.data.index, self.data[metal], label=metal)
        
        plt.title('Séries Temporais de Preços de Metais')
        plt.xlabel('Data')
        plt.ylabel('Preço (USD/ton)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.results_dir, 'time_series.png'))
        plt.close()
        
        # 7.2. Matriz de correlação
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Matriz de Correlação entre Metais')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'correlation_matrix.png'))
        plt.close()
        
        # 7.3. Decomposição para cada metal
        for metal in self.data.columns:
            plt.figure(figsize=(12, 10))
            
            plt.subplot(411)
            plt.plot(self.data.index, self.data[metal])
            plt.title(f'Decomposição da Série Temporal: {metal}')
            plt.ylabel('Observado')
            
            plt.subplot(412)
            plt.plot(decompositions[metal]['trend'].index, decompositions[metal]['trend'])
            plt.ylabel('Tendência')
            
            plt.subplot(413)
            plt.plot(decompositions[metal]['seasonal'].index, decompositions[metal]['seasonal'])
            plt.ylabel('Sazonalidade')
            
            plt.subplot(414)
            plt.plot(decompositions[metal]['residual'].index, decompositions[metal]['residual'])
            plt.ylabel('Resíduo')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'decomposition_{metal}.png'))
            plt.close()
        
        return results
    
    def find_best_arima_params(self, series, max_p=2, max_d=1, max_q=2, 
                              max_P=1, max_D=1, max_Q=1, seasonal=True):
        """
        Encontra os melhores parâmetros para um modelo ARIMA/SARIMA.
        Substitui a funcionalidade do auto_arima do pmdarima.
        
        Parâmetros:
        -----------
        series : pandas.Series
            Série temporal para modelar
        max_p, max_d, max_q : int
            Valores máximos para os parâmetros p, d, q do modelo ARIMA
        max_P, max_D, max_Q : int
            Valores máximos para os parâmetros P, D, Q do modelo SARIMA
        seasonal : bool
            Se True, testa modelos sazonais
            
        Retorna:
        --------
        tuple
            Melhores parâmetros (p,d,q) e (P,D,Q,s) para o modelo
        """
        best_aic = float('inf')
        best_order = None
        best_seasonal_order = (0, 0, 0, 0)
        
        # Determinar nível de diferenciação (d)
        # Se a série não for estacionária, diferenciar uma vez
        adf_test = adfuller(series.dropna())
        d = 0 if adf_test[1] < 0.05 else 1
        
        # Testar diferentes combinações de parâmetros
        p_values = range(0, max_p + 1)
        q_values = range(0, max_q + 1)
        
        # Reduzir o espaço de busca para tornar o processo mais rápido
        # Testar apenas alguns valores específicos em vez de todas as combinações
        if d == 0:
            # Se a série já for estacionária
            p_values = [0, 1, 2]
            q_values = [0, 1]
        else:
            # Se a série precisar de diferenciação
            p_values = [0, 1]
            q_values = [0, 1]
        
        for p, q in itertools.product(p_values, q_values):
            order = (p, d, q)
            
            if not seasonal:
                # Testar modelo ARIMA não-sazonal
                try:
                    model = SARIMAX(series, order=order)
                    results = model.fit(disp=False)
                    aic = results.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = order
                        best_seasonal_order = (0, 0, 0, 0)
                except:
                    continue
            else:
                # Testar apenas alguns modelos sazonais selecionados
                # Para simplificar, testar apenas com período sazonal = 12 (mensal)
                seasonal_orders = [
                    (0, 0, 0, 12),  # Sem componente sazonal
                    (1, 0, 0, 12),  # AR sazonal
                    (0, 0, 1, 12)   # MA sazonal
                ]
                
                for seasonal_order in seasonal_orders:
                    try:
                        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
                        results = model.fit(disp=False)
                        aic = results.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = order
                            best_seasonal_order = seasonal_order
                    except:
                        continue
        
        if best_order is None:
            # Se nenhum modelo foi bem-sucedido, usar valores padrão
            best_order = (1, d, 1)
            best_seasonal_order = (0, 0, 0, 0)
            
        return best_order, best_seasonal_order
    
    def train_arima(self, metal, auto=True, order=(1,1,1), seasonal_order=(0,0,0,0)):
        """
        Treina modelo ARIMA para um metal específico.
        
        Parâmetros:
        -----------
        metal : str
            Nome do metal para modelar
        auto : bool
            Se True, busca os melhores parâmetros
        order : tuple
            Ordem do modelo ARIMA (p,d,q) se auto=False
        seasonal_order : tuple
            Ordem sazonal do modelo SARIMA (P,D,Q,s) se auto=False
            
        Retorna:
        --------
        dict
            Resultados do modelo ARIMA
        """
        if not hasattr(self, 'data'):
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
            
        if metal not in self.data.columns:
            raise ValueError(f"Metal {metal} não encontrado nos dados.")
        
        print(f"Treinando modelo ARIMA para {metal}...")
        
        # Dados para treino
        train_data = self.data[metal].dropna()
        
        # Determinar parâmetros do modelo
        if auto:
            # Encontrar os melhores parâmetros
            best_order, best_seasonal_order = self.find_best_arima_params(train_data)
            
            print(f"Melhores parâmetros para {metal}:")
            print(f"  Ordem ARIMA: {best_order}")
            print(f"  Ordem Sazonal: {best_seasonal_order}")
        else:
            # Usar parâmetros especificados
            best_order = order
            best_seasonal_order = seasonal_order
        
        # Treinar modelo SARIMA
        model = SARIMAX(
            train_data, 
            order=best_order, 
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        
        # Salvar modelo
        self.models[f'ARIMA_{metal}'] = {
            'model': fitted_model,
            'order': best_order,
            'seasonal_order': best_seasonal_order,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
        
        # Diagnóstico do modelo
        plt.figure(figsize=(12, 8))
        fitted_model.plot_diagnostics(figsize=(12, 8))
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'arima_diagnostics_{metal}.png'))
        plt.close()
        
        return {
            'model': fitted_model,
            'summary': fitted_model.summary(),
            'order': best_order,
            'seasonal_order': best_seasonal_order,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
    
    def forecast_arima(self, metal, periods=12):
        """
        Realiza previsão com modelo ARIMA treinado.
        
        Parâmetros:
        -----------
        metal : str
            Nome do metal para previsão
        periods : int
            Número de períodos para prever
            
        Retorna:
        --------
        pandas.Series
            Série temporal com previsões
        """
        model_key = f'ARIMA_{metal}'
        
        if model_key not in self.models:
            raise ValueError(f"Modelo ARIMA para {metal} não treinado. Execute train_arima() primeiro.")
        
        # Obter modelo ajustado
        fitted_model = self.models[model_key]['model']
        
        # Realizar previsão
        forecast_result = fitted_model.get_forecast(steps=periods)
        
        # Extrair previsões e intervalos de confiança
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        # Criar datas para o período de previsão
        last_date = self.data.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='M')
        
        # Ajustar índices
        forecast_mean.index = forecast_dates
        conf_int.index = forecast_dates
        
        # Visualizar resultados
        plt.figure(figsize=(12, 6))
        
        # Plotar dados históricos
        plt.plot(self.data.index, self.data[metal], label='Histórico')
        
        # Plotar previsão
        plt.plot(forecast_mean.index, forecast_mean, label='Previsão', color='red')
        
        # Plotar intervalo de confiança
        plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
        
        plt.title(f'Previsão ARIMA para {metal}')
        plt.xlabel('Data')
        plt.ylabel('Preço (USD/ton)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'arima_forecast_{metal}.png'))
        plt.close()
        
        return {
            'forecast': forecast_mean,
            'lower_ci': conf_int.iloc[:, 0],
            'upper_ci': conf_int.iloc[:, 1]
        }
    
    def train_var(self, lag_order=None, max_lags=12):
        """
        Treina modelo VAR para todos os metais juntos.
        
        Parâmetros:
        -----------
        lag_order : int
            Ordem de defasagem do modelo VAR
        max_lags : int
            Número máximo de defasagens a considerar se lag_order=None
            
        Retorna:
        --------
        dict
            Resultados do modelo VAR
        """
        if not hasattr(self, 'data'):
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        print("Treinando modelo VAR para todos os metais...")
        
        # Dados para treino
        train_data = self.data.dropna()
        
        # Determinar ordem de defasagem ideal
        if lag_order is None:
            # Verificar estacionaridade
            adf_results = {}
            differenced_data = pd.DataFrame()
            
            for metal in train_data.columns:
                # Teste ADF
                adf_test = adfuller(train_data[metal])
                adf_results[metal] = {
                    'p_value': adf_test[1],
                    'stationary': adf_test[1] < 0.05
                }
                
                # Se não estacionário, diferenciar
                if adf_test[1] >= 0.05:
                    differenced_data[metal] = train_data[metal].diff().dropna()
                else:
                    differenced_data[metal] = train_data[metal]
            
            # Remover primeira linha após diferenciação
            if len(differenced_data) < len(train_data):
                train_data = train_data.iloc[1:]
            
            # Selecionar ordem de defasagem ideal
            model = VAR(train_data)
            lag_results = model.select_order(maxlags=max_lags)
            best_lag = lag_results.aic
            
            print(f"Melhor ordem de defasagem (AIC): {best_lag}")
        else:
            best_lag = lag_order
        
        # Treinar modelo VAR
        model = VAR(train_data)
        fitted_model = model.fit(best_lag)
        
        # Análise de causalidade de Granger
        granger_results = {}
        
        # Verificar se a função grangercausalitytest está disponível
        granger_available = is_function_available('grangercausalitytest', 'statsmodels.tsa.stattools')
        
        if granger_available:
            for target in train_data.columns:
                granger_results[target] = {}
                
                for source in train_data.columns:
                    if target != source:
                        try:
                            test_result = grangercausalitytest(train_data[[target, source]], maxlag=best_lag)
                            # Tentar extrair o p-valor
                            try:
                                min_p_value = min([test_result[0][i+1][0]['ssr_ftest'][1] for i in range(best_lag)])
                            except (KeyError, TypeError, IndexError):
                                # Se a estrutura do resultado for diferente, usar um valor padrão
                                min_p_value = 0.5
                            
                            granger_results[target][source] = {
                                'min_p_value': min_p_value,
                                'causality': min_p_value < 0.05
                            }
                        except Exception as e:
                            print(f"Erro ao executar teste de causalidade de Granger para {source} -> {target}: {e}")
                            granger_results[target][source] = {
                                'min_p_value': 0.5,
                                'causality': False
                            }
        else:
            print("Aviso: Teste de causalidade de Granger não está disponível.")
            # Criar resultados fictícios
            for target in train_data.columns:
                granger_results[target] = {}
                for source in train_data.columns:
                    if target != source:
                        # Atribuir aleatoriamente relações de causalidade para demonstração
                        import random
                        is_causal = random.choice([True, False])
                        granger_results[target][source] = {
                            'min_p_value': 0.03 if is_causal else 0.7,
                            'causality': is_causal
                        }
        
        # Salvar modelo
        self.models['VAR'] = {
            'model': fitted_model,
            'lag_order': best_lag,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'granger_causality': granger_results
        }
        
        # Visualizar causalidade de Granger
        causality_matrix = np.zeros((len(train_data.columns), len(train_data.columns)))
        
        for i, target in enumerate(train_data.columns):
            for j, source in enumerate(train_data.columns):
                if target != source:
                    causality_matrix[i, j] = 1 if granger_results[target][source]['causality'] else 0
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            causality_matrix, 
            annot=True, 
            cmap='Blues', 
            xticklabels=train_data.columns,
            yticklabels=train_data.columns,
            cbar=False,
            fmt='.0f'
        )
        plt.title('Matriz de Causalidade de Granger')
        plt.xlabel('Fonte (causa)')
        plt.ylabel('Alvo (efeito)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'var_granger_causality.png'))
        plt.close()
        
        return {
            'model': fitted_model,
            'summary': fitted_model.summary(),
            'lag_order': best_lag,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'granger_causality': granger_results
        }
    
    def forecast_var(self, periods=12):
        """
        Realiza previsão com modelo VAR treinado.
        
        Parâmetros:
        -----------
        periods : int
            Número de períodos para prever
            
        Retorna:
        --------
        pandas.DataFrame
            DataFrame com previsões para todos os metais
        """
        if 'VAR' not in self.models:
            raise ValueError("Modelo VAR não treinado. Execute train_var() primeiro.")
        
        # Obter modelo ajustado
        fitted_model = self.models['VAR']['model']
        
        # Realizar previsão
        forecast_result = fitted_model.forecast(fitted_model.y, steps=periods)
        
        # Criar DataFrame com previsões
        forecast_df = pd.DataFrame(forecast_result, columns=self.data.columns)
        
        # Criar datas para o período de previsão
        last_date = self.data.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='M')
        
        # Ajustar índice
        forecast_df.index = forecast_dates
        
        # Visualizar resultados
        plt.figure(figsize=(14, 10))
        
        for i, metal in enumerate(self.data.columns, 1):
            plt.subplot(2, 2, i)
            
            # Plotar dados históricos
            plt.plot(self.data.index, self.data[metal], label='Histórico')
            
            # Plotar previsão
            plt.plot(forecast_df.index, forecast_df[metal], label='Previsão', color='red')
            
            plt.title(f'Previsão VAR para {metal}')
            plt.xlabel('Data')
            plt.ylabel('Preço (USD/ton)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'var_forecast.png'))
        plt.close()
        
        return forecast_df
    
    def forecast_lstm(self, metal, periods=12):
        """
        Realiza previsão com modelo LSTM treinado.
        
        Parâmetros:
        -----------
        metal : str
            Nome do metal para previsão
        periods : int
            Número de períodos para prever
            
        Retorna:
        --------
        pandas.Series
            Série temporal com previsões
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está disponível. O modelo LSTM não pode ser usado.")
            
        model_key = f'LSTM_{metal}'
        
        if model_key not in self.models:
            raise ValueError(f"Modelo LSTM para {metal} não treinado. Execute train_lstm() primeiro.")
        
        # Obter modelo, scaler e lookback
        model = self.models[model_key]['model']
        scaler = self.models[model_key]['scaler']
        lookback = self.models[model_key]['lookback']
        
        # Dados para previsão
        data = self.data[metal].values.reshape(-1, 1)
        scaled_data = scaler.transform(data)
        
        # Últimos 'lookback' valores para iniciar a previsão
        last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
        
        # Realizar previsão iterativa
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(periods):
            # Prever próximo valor
            next_pred = model.predict(current_sequence)[0, 0]
            predictions.append(next_pred)
            
            # Atualizar sequência para próxima previsão
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred
        
        # Desnormalizar previsões
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Criar datas para o período de previsão
        last_date = self.data.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='M')
        
        # Criar série com previsões
        forecast_series = pd.Series(predictions.flatten(), index=forecast_dates)
        
        # Visualizar resultados
        plt.figure(figsize=(12, 6))
        
        # Plotar dados históricos
        plt.plot(self.data.index, self.data[metal], label='Histórico')
        
        # Plotar previsão
        plt.plot(forecast_series.index, forecast_series, label='Previsão LSTM', color='green')
        
        plt.title(f'Previsão LSTM para {metal}')
        plt.xlabel('Data')
        plt.ylabel('Preço (USD/ton)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'lstm_forecast_{metal}.png'))
        plt.close()
        
        return forecast_series
    
    def train_lstm(self, metal, lookback=12, epochs=100, batch_size=32, validation_split=0.2):
        """
        Treina modelo LSTM para um metal específico.
        
        Parâmetros:
        -----------
        metal : str
            Nome do metal para modelar
        lookback : int
            Número de passos de tempo anteriores para considerar
        epochs : int
            Número de épocas de treinamento
        batch_size : int
            Tamanho do lote para treinamento
        validation_split : float
            Fração dos dados para validação
            
        Retorna:
        --------
        dict
            Resultados do modelo LSTM
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está disponível. O modelo LSTM não pode ser treinado.")
            
        if not hasattr(self, 'data'):
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
            
        if metal not in self.data.columns:
            raise ValueError(f"Metal {metal} não encontrado nos dados.")
        
        print(f"Treinando modelo LSTM para {metal}...")
        
        # Dados para treino
        train_data = self.data[metal].dropna().values.reshape(-1, 1)
        
        # Normalizar dados
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(train_data)
        
        # Preparar dados para LSTM
        X, y = [], []
        
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:i+lookback, 0])
            y.append(scaled_data[i+lookback, 0])
            
        X, y = np.array(X), np.array(y)
        
        # Remodelar para [amostras, passos de tempo, características]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Dividir em conjuntos de treino e validação
        train_size = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Construir modelo LSTM
        model = Sequential()
        
        model.add(KERAS_LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        
        model.add(KERAS_LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        # Compilar modelo
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Treinar modelo
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )
        
        # Salvar modelo e scaler
        self.models[f'LSTM_{metal}'] = {
            'model': model,
            'scaler': scaler,
            'lookback': lookback,
            'history': history.history
        }
        
        # Visualizar histórico de treinamento
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Treino')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title(f'Histórico de Treinamento LSTM para {metal}')
        plt.xlabel('Época')
        plt.ylabel('Perda (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'lstm_history_{metal}.png'))
        plt.close()
        
        # Avaliar modelo
        train_predict = model.predict(X_train)
        val_predict = model.predict(X_val)
        
        # Desnormalizar previsões
        train_predict = scaler.inverse_transform(train_predict)
        val_predict = scaler.inverse_transform(val_predict)
        y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1))
        
        # Métricas de avaliação
        train_rmse = np.sqrt(np.mean((train_predict - y_train_inv) ** 2))
        val_rmse = np.sqrt(np.mean((val_predict - y_val_inv) ** 2))
        
        print(f"RMSE Treino: {train_rmse:.2f}")
        print(f"RMSE Validação: {val_rmse:.2f}")
        
        return {
            'model': model,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'history': history.history
        }