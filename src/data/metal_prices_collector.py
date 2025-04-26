# src/data/metal_prices_collector.py

import os
import requests
import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta
import time
import numpy as np
import matplotlib.pyplot as plt

class MetalPricesCollector:
    """
    Coletor de dados históricos de preços de metais relevantes para nódulos polimetálicos:
    níquel, cobre, cobalto e manganês.
    """
    
    def __init__(self, database_path='data/nodules.db', data_dir='data/raw/metal_prices'):
        """
        Inicializa o coletor de preços de metais.
        
        Parâmetros:
        -----------
        database_path : str
            Caminho para o banco de dados SQLite
        data_dir : str
            Diretório para armazenar os dados brutos
        """
        self.database_path = database_path
        self.data_dir = data_dir
        
        # Criar diretório para os dados se não existir
        os.makedirs(data_dir, exist_ok=True)
        
        # Metais de interesse para nódulos polimetálicos
        self.metals = {
            'nickel': {
                'name': 'Nickel',
                'symbol': 'Ni',
                'base_price': 16000  # USD/tonelada (preço base para simulação)
            },
            'copper': {
                'name': 'Copper',
                'symbol': 'Cu',
                'base_price': 9000
            },
            'cobalt': {
                'name': 'Cobalt',
                'symbol': 'Co',
                'base_price': 30000
            },
            'manganese': {
                'name': 'Manganese',
                'symbol': 'Mn',
                'base_price': 1700
            }
        }
    
    def generate_historical_prices(self, start_date='2010-01-01', end_date=None):
        """
        Gera dados históricos sintéticos de preços de metais.
        
        Em uma implementação real, esta função seria substituída por uma
        que obtém dados reais de APIs como a da London Metal Exchange.
        
        Parâmetros:
        -----------
        start_date : str
            Data inicial no formato 'YYYY-MM-DD'
        end_date : str
            Data final no formato 'YYYY-MM-DD'. Se None, usa a data atual.
            
        Retorna:
        --------
        pandas.DataFrame
            DataFrame com os preços históricos
        """
        print(f"Gerando preços históricos de metais de {start_date} até hoje...")
        
        # Definir data final como hoje se não especificada
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Criar intervalo de datas (diário)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Dias úteis
        
        # Criar DataFrame vazio
        df = pd.DataFrame(index=date_range)
        df.index.name = 'date'
        
        # Para cada metal, gerar série de preços sintética
        for metal, info in self.metals.items():
            # Obter preço base para simulação
            base_price = info['base_price']
            
            # Número de dias no período
            n_days = len(date_range)
            
            # Gerar componentes da série de preços
            
            # 1. Tendência de longo prazo (com algumas reversões)
            # Usamos uma soma cumulativa de ruído para criar uma caminhada aleatória
            np.random.seed(42 + ord(metal[0]))  # Seed diferente para cada metal
            random_walk = np.random.normal(0, 0.0005, n_days).cumsum()
            
            # Adicionar algumas tendências específicas por metal
            if metal == 'nickel':
                # Níquel: Picos em 2011 e 2014, queda em 2015-2016, recuperação em 2017
                trend = np.zeros(n_days)
                date_indices = {
                    pd.Timestamp('2011-02-01'): 0.3,
                    pd.Timestamp('2014-05-01'): 0.2,
                    pd.Timestamp('2015-06-01'): -0.25,
                    pd.Timestamp('2017-01-01'): 0.15,
                    pd.Timestamp('2020-03-01'): -0.2,  # Queda COVID
                    pd.Timestamp('2021-06-01'): 0.3,   # Recuperação pós-COVID
                }
                
                for date, value in date_indices.items():
                    if date in df.index:
                        idx = df.index.get_loc(date)
                        # Aplicar mudança gradualmente nos próximos 120 dias
                        for i in range(min(120, n_days - idx)):
                            trend[idx + i] = value * (1 - i/120)
            
            elif metal == 'copper':
                # Cobre: Mais estável, mas com picos em 2011, 2018 e 2021
                trend = np.zeros(n_days)
                date_indices = {
                    pd.Timestamp('2011-02-01'): 0.25,
                    pd.Timestamp('2018-06-01'): 0.15,
                    pd.Timestamp('2020-03-01'): -0.15,  # Queda COVID
                    pd.Timestamp('2021-05-01'): 0.3,    # Recuperação forte
                }
                
                for date, value in date_indices.items():
                    if date in df.index:
                        idx = df.index.get_loc(date)
                        for i in range(min(90, n_days - idx)):
                            trend[idx + i] = value * (1 - i/90)
            
            elif metal == 'cobalt':
                # Cobalto: Muito volátil, com grande pico em 2018 (baterias)
                trend = np.zeros(n_days)
                date_indices = {
                    pd.Timestamp('2017-09-01'): 0.5,
                    pd.Timestamp('2018-03-01'): 0.8,
                    pd.Timestamp('2019-01-01'): -0.4,
                    pd.Timestamp('2020-03-01'): -0.2,  # Queda COVID
                    pd.Timestamp('2021-12-01'): 0.35,  # Demanda de baterias
                }
                
                for date, value in date_indices.items():
                    if date in df.index:
                        idx = df.index.get_loc(date)
                        for i in range(min(100, n_days - idx)):
                            trend[idx + i] = value * (1 - i/100)
            
            elif metal == 'manganese':
                # Manganês: Menos volátil, com pico em 2016
                trend = np.zeros(n_days)
                date_indices = {
                    pd.Timestamp('2016-08-01'): 0.3,
                    pd.Timestamp('2020-03-01'): -0.1,  # Queda COVID menor
                    pd.Timestamp('2021-02-01'): 0.15,  # Recuperação moderada
                }
                
                for date, value in date_indices.items():
                    if date in df.index:
                        idx = df.index.get_loc(date)
                        for i in range(min(80, n_days - idx)):
                            trend[idx + i] = value * (1 - i/80)
            
            # 2. Componente sazonal
            # Muitos metais têm padrões sazonais devido a demanda industrial
            t = np.arange(n_days)
            seasonal = 0.05 * np.sin(2 * np.pi * t / 252)  # Ciclo anual (252 dias úteis)
            
            # 3. Componente auto-regressivo (AR(1))
            ar = np.zeros(n_days)
            ar[0] = 0
            phi = 0.98  # Parâmetro auto-regressivo alto = alta persistência
            for i in range(1, n_days):
                ar[i] = phi * ar[i-1] + np.random.normal(0, 0.01)
            
            # 4. Ruído aleatório
            noise = np.random.normal(0, 0.015, n_days)
            
            # Combinar componentes
            price_factors = 1 + trend + seasonal + ar + noise + random_walk
            
            # Calcular preços
            prices = base_price * price_factors
            
            # Garantir que preços nunca fiquem negativos
            prices = np.maximum(prices, base_price * 0.3)
            
            # Adicionar ao DataFrame
            df[metal] = prices
        
        # Salvar arquivo CSV
        output_path = os.path.join(self.data_dir, 'historical_metal_prices.csv')
        df.to_csv(output_path)
        print(f"Dados históricos de preços salvos em: {output_path}")
        
        # Criar visualizações
        self._create_price_visualizations(df)
        
        return df
    
    def _create_price_visualizations(self, df):
        """
        Cria visualizações dos preços históricos dos metais.
        
        Parâmetros:
        -----------
        df : pandas.DataFrame
            DataFrame com os preços históricos
        """
        try:
            # Criar diretório para visualizações se não existir
            vis_dir = 'results/figures/metal_prices'
            os.makedirs(vis_dir, exist_ok=True)
            
            # 1. Preços absolutos
            plt.figure(figsize=(12, 8))
            
            for metal, info in self.metals.items():
                plt.plot(df.index, df[metal], label=f"{info['name']} (USD/ton)")
            
            plt.title('Preços Históricos de Metais')
            plt.xlabel('Data')
            plt.ylabel('Preço (USD/ton)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Salvar figura
            plt.savefig(os.path.join(vis_dir, 'metal_prices_absolute.png'), dpi=300)
            plt.close()
            
            # 2. Preços relativos (normalizado pelo preço inicial)
            plt.figure(figsize=(12, 8))
            
            for metal, info in self.metals.items():
                # Normalizar pelo primeiro preço
                rel_prices = df[metal] / df[metal].iloc[0]
                plt.plot(df.index, rel_prices, label=f"{info['name']}")
            
            plt.title('Preços Relativos de Metais (Normalizado)')
            plt.xlabel('Data')
            plt.ylabel('Preço Relativo (Primeiro dia = 1.0)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Salvar figura
            plt.savefig(os.path.join(vis_dir, 'metal_prices_relative.png'), dpi=300)
            plt.close()
            
            # 3. Variação anual
            # Calcular retorno anual
            annual_returns = df.resample('Y').last()
            for metal in self.metals.keys():
                annual_returns[f"{metal}_return"] = annual_returns[metal].pct_change() * 100
            
            # Remover primeiro ano (sem retorno calculável)
            annual_returns = annual_returns.iloc[1:]
            
            # Plotar retornos anuais
            plt.figure(figsize=(14, 8))
            
            bar_width = 0.2
            x = np.arange(len(annual_returns))
            
            for i, metal in enumerate(self.metals.keys()):
                plt.bar(x + i*bar_width - 0.3, 
                       annual_returns[f"{metal}_return"], 
                       width=bar_width, 
                       label=self.metals[metal]['name'])
            
            plt.title('Variação Anual de Preços de Metais')
            plt.xlabel('Ano')
            plt.ylabel('Variação Percentual (%)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(x, [d.year for d in annual_returns.index])
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Salvar figura
            plt.savefig(os.path.join(vis_dir, 'metal_prices_annual_change.png'), dpi=300)
            plt.close()
            
            print(f"Visualizações salvas em: {vis_dir}")
            
        except Exception as e:
            print(f"Erro ao criar visualizações: {e}")
    
    def save_to_database(self, df):
        """
        Salva os preços históricos de metais no banco de dados.
        
        Parâmetros:
        -----------
        df : pandas.DataFrame
            DataFrame com os preços históricos
            
        Retorna:
        --------
        dict
            Resumo da inserção por metal
        """
        print("Salvando preços de metais no banco de dados...")
        
        summary = {}
        
        try:
            # Conectar ao banco de dados
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Verificar se a tabela existe
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metal_prices'")
            if not cursor.fetchone():
                print("  Tabela 'metal_prices' não encontrada no banco de dados.")
                print("  Verifique se o script de configuração foi executado.")
                return {'error': 'Table not found'}
            
            # Para cada metal, inserir os preços
            for metal, info in self.metals.items():
                metal_name = info['name']
                
                # Resetar o index para ter a data como coluna
                metal_df = df.reset_index()
                
                # Criar tabela temporária para os dados
                temp_table = f"temp_metal_price_{metal}"
                
                # Preparar dados para inserção
                temp_df = pd.DataFrame({
                    'metal_name': metal_name,
                    'date': metal_df['date'].dt.strftime('%Y-%m-%d'),
                    'price_usd_ton': metal_df[metal],
                    'source': 'Synthetic Data'
                })
                
                # Salvar em tabela temporária
                temp_df.to_sql(temp_table, conn, if_exists='replace', index=False)
                
                # Inserir na tabela principal apenas registros novos
                cursor.execute(f"""
                    INSERT OR REPLACE INTO metal_prices (metal_name, date, price_usd_ton, source)
                    SELECT metal_name, date, price_usd_ton, source
                    FROM {temp_table}
                    """)
                
                # Obter número de registros inseridos
                records_saved = cursor.rowcount
                
                # Remover tabela temporária
                cursor.execute(f"DROP TABLE {temp_table}")
                
                summary[metal_name] = records_saved
                print(f"  {metal_name}: {records_saved} registros salvos")
            
            # Commit e fechar conexão
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            print(f"Erro SQLite ao salvar dados: {e}")
            summary['error'] = str(e)
        except Exception as e:
            print(f"Erro inesperado ao salvar dados: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def collect_and_process(self, years_back=13):
        """
        Coleta e processa preços históricos de metais.
        
        Parâmetros:
        -----------
        years_back : int
            Número de anos para coletar dados históricos
            
        Retorna:
        --------
        dict
            Resumo da coleta e processamento
        """
        # Definir período
        end_date = datetime.now()
        start_date = datetime(end_date.year - years_back, 1, 1)
        
        # Gerar dados históricos
        df = self.generate_historical_prices(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Salvar no banco de dados
        summary = self.save_to_database(df)
        
        return {
            'period': f"{start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}",
            'records_generated': len(df),
            'metals': list(self.metals.keys()),
            'database_summary': summary
        }

if __name__ == "__main__":
    # Criar e executar coletor
    collector = MetalPricesCollector()
    summary = collector.collect_and_process()
    
    print("\nResumo da coleta de preços de metais:")
    print(f"Período: {summary['period']}")
    print(f"Registros gerados: {summary['records_generated']}")
    print("Metais processados:", ", ".join(summary['metals']))
    
    if 'error' in summary['database_summary']:
        print(f"ERRO: {summary['database_summary']['error']}")
    else:
        print("Registros salvos no banco de dados:")
        for metal, count in summary['database_summary'].items():
            print(f"  {metal}: {count}")