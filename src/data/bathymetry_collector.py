# src/data/bathymetry_collector.py

import os
import numpy as np
import pandas as pd
import sqlite3
import time
from datetime import datetime
import matplotlib.pyplot as plt

class BathymetryCollector:
    """
    Coletor de dados batimétricos do Oceano Atlântico para análise de nódulos polimetálicos.
    
    Versão simplificada que gera dados sintéticos para áreas do Atlântico com potencial para 
    nódulos polimetálicos.
    """
    
    def __init__(self, database_path='data/nodules.db', data_dir='data/raw/bathymetry'):
        """
        Inicializa o coletor de dados batimétricos.
        
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
        
        # Áreas de interesse no Atlântico com potencial para nódulos polimetálicos
        self.areas_of_interest = {
            'rio_grande_rise': {
                'name': 'Rio Grande Rise',
                'bounds': [-35, -25, -32, -28],  # [min_lon, min_lat, max_lon, max_lat]
                'description': 'Elevação submarina no Atlântico Sul com potencial para nódulos polimetálicos'
            },
            'mid_atlantic_ridge': {
                'name': 'Mid-Atlantic Ridge',
                'bounds': [-45, -15, -35, 0],
                'description': 'Cordilheira meso-oceânica que se estende pelo Atlântico'
            },
            'bermuda_rise': {
                'name': 'Bermuda Rise',
                'bounds': [-70, 25, -60, 35],
                'description': 'Região elevada no Atlântico Norte próxima às Bermudas'
            },
            'azores_plateau': {
                'name': 'Azores Plateau',
                'bounds': [-35, 35, -20, 45],
                'description': 'Plataforma submarina no Atlântico Norte próxima aos Açores'
            }
        }
    
    def generate_synthetic_data(self, area_key=None, resolution=0.1):
        """
        Gera dados batimétricos sintéticos para as áreas de interesse.
        
        Parâmetros:
        -----------
        area_key : str
            Chave da área de interesse. Se None, gera dados para todas as áreas.
        resolution : float
            Resolução dos dados em graus
            
        Retorna:
        --------
        dict
            Dicionário com caminhos para os arquivos gerados
        """
        print("Gerando dados batimétricos sintéticos...")
        
        # Definir áreas para geração
        areas = [self.areas_of_interest[area_key]] if area_key else list(self.areas_of_interest.values())
        
        generated_files = {}
        
        for area in areas:
            print(f"Processando área: {area['name']}")
            
            # Criar nome de arquivo
            area_name = area['name'].lower().replace(' ', '_')
            file_path = os.path.join(self.data_dir, f"{area_name}_bathymetry.csv")
            
            # Verificar se o arquivo já existe
            if os.path.exists(file_path):
                print(f"  Arquivo já existe: {file_path}")
                generated_files[area['name']] = file_path
                continue
            
            try:
                # Extrair limites da área
                min_lon, min_lat, max_lon, max_lat = area['bounds']
                
                # Criar grade de coordenadas
                lons = np.arange(min_lon, max_lon + resolution, resolution)
                lats = np.arange(min_lat, max_lat + resolution, resolution)
                
                # Inicializar lista para os dados
                data = []
                
                # Gerar profundidades para cada ponto
                for lat in lats:
                    for lon in lons:
                        # Base depth (oceano profundo: ~4000m de profundidade)
                        depth = -4000
                        
                        # Adicionar características batimétricas conforme a área
                        if area['name'] == 'Rio Grande Rise':
                            # Simular elevação submarina
                            distance = np.sqrt((lon - np.mean([min_lon, max_lon]))**2 + 
                                             (lat - np.mean([min_lat, max_lat]))**2)
                            elevation = 3000 * np.exp(-distance**2 / 2)
                            depth += elevation
                            
                        elif area['name'] == 'Mid-Atlantic Ridge':
                            # Simular cordilheira meso-oceânica
                            ridge = 2500 * np.exp(-((lon - np.mean([min_lon, max_lon]))**2) / 2)
                            depth += ridge
                            
                        elif area['name'] == 'Bermuda Rise':
                            # Simular monte submarino
                            distance = np.sqrt((lon - (min_lon + 5))**2 + (lat - (min_lat + 5))**2)
                            seamount = 3500 * np.exp(-distance**2 / 5)
                            depth += seamount
                            
                        elif area['name'] == 'Azores Plateau':
                            # Simular plataforma com montes submarinos
                            plateau = 2000
                            
                            # Adicionar monte submarino
                            distance = np.sqrt((lon - (min_lon + 7))**2 + (lat - (min_lat + 5))**2)
                            seamount = 1500 * np.exp(-distance**2 / 3)
                            depth += plateau + seamount
                        
                        # Adicionar ruído para tornar mais realista
                        noise = 200 * np.random.randn()
                        depth += noise
                        
                        # Adicionar ao conjunto de dados
                        data.append({
                            'latitude': lat,
                            'longitude': lon,
                            'depth_m': -depth  # Converter elevação para profundidade (valores positivos)
                        })
                
                # Converter para DataFrame
                df = pd.DataFrame(data)
                
                # Salvar em CSV
                df.to_csv(file_path, index=False)
                generated_files[area['name']] = file_path
                print(f"  Dados gerados e salvos em: {file_path}")
                
                # Criar visualização simples
                self._create_simple_visualization(df, area_name)
                
            except Exception as e:
                print(f"  Erro ao gerar dados para {area['name']}: {e}")
            
            # Pausa entre gerações
            time.sleep(0.5)
        
        return generated_files
    
    def _create_simple_visualization(self, df, area_name):
        """
        Cria uma visualização simples dos dados batimétricos.
        
        Parâmetros:
        -----------
        df : pandas.DataFrame
            DataFrame com os dados batimétricos
        area_name : str
            Nome da área para o título
        """
        try:
            # Criar diretório para visualizações se não existir
            vis_dir = 'results/figures/bathymetry'
            os.makedirs(vis_dir, exist_ok=True)
            
            # Criar scatter plot colorido por profundidade
            plt.figure(figsize=(10, 8))
            
            # Usar scatter plot com colormap
            sc = plt.scatter(df['longitude'], df['latitude'], 
                           c=df['depth_m'], cmap='viridis', 
                           s=10, alpha=0.7)
            
            # Configurar rótulos e título
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(f'Mapa Batimétrico: {area_name}')
            
            # Adicionar barra de cores
            cbar = plt.colorbar(sc)
            cbar.set_label('Profundidade (m)')
            
            # Ajustar limites dos eixos
            plt.xlim(df['longitude'].min(), df['longitude'].max())
            plt.ylim(df['latitude'].min(), df['latitude'].max())
            
            # Adicionar grade
            plt.grid(alpha=0.3)
            
            # Salvar figura
            plt.tight_layout()
            file_path = os.path.join(vis_dir, f"{area_name}_bathymetry.png")
            plt.savefig(file_path, dpi=300)
            plt.close()
            
            print(f"  Visualização salva em: {file_path}")
            
        except Exception as e:
            print(f"Erro ao criar visualização: {e}")
    
    def save_to_database(self, file_path):
        """
        Salva os dados batimétricos no banco de dados.
        
        Parâmetros:
        -----------
        file_path : str
            Caminho para o arquivo CSV com os dados batimétricos
            
        Retorna:
        --------
        int
            Número de registros salvos
        """
        print(f"Salvando dados batimétricos de {file_path} no banco de dados...")
        
        try:
            # Ler arquivo CSV
            df = pd.read_csv(file_path)
            source = os.path.basename(file_path)
            
            # Conectar ao banco de dados
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Verificar se a tabela existe
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bathymetry'")
            if not cursor.fetchone():
                # Criar tabela se não existir
                cursor.execute("""
                CREATE TABLE bathymetry (
                    bathymetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    latitude DECIMAL(10,6) NOT NULL,
                    longitude DECIMAL(10,6) NOT NULL,
                    depth_m DECIMAL(8,2) NOT NULL,
                    source VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(latitude, longitude)
                )
                """)
                
                # Criar índice
                cursor.execute("CREATE INDEX idx_bathymetry_location ON bathymetry(latitude, longitude)")
                print("  Tabela 'bathymetry' criada com sucesso")
            
            # Criar tabela temporária para os dados
            temp_table = f"temp_bathymetry_{int(time.time())}"
            df['source'] = source
            df['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            df.to_sql(temp_table, conn, if_exists='replace', index=False)
            
            # Inserir na tabela principal apenas registros novos
            cursor.execute(f"""
            INSERT OR IGNORE INTO bathymetry (latitude, longitude, depth_m, source, created_at)
            SELECT latitude, longitude, depth_m, source, created_at
            FROM {temp_table}
            """)
            
            # Obter número de registros inseridos
            records_saved = cursor.rowcount
            
            # Remover tabela temporária
            cursor.execute(f"DROP TABLE {temp_table}")
            
            # Commit e fechar conexão
            conn.commit()
            conn.close()
            
            print(f"  Salvos {records_saved} novos registros de batimetria no banco de dados")
            return records_saved
            
        except sqlite3.Error as e:
            print(f"Erro SQLite ao salvar dados: {e}")
            return 0
        except Exception as e:
            print(f"Erro inesperado ao salvar dados: {e}")
            return 0
    
    def collect_and_process_all(self):
        """
        Gera dados batimétricos sintéticos para todas as áreas e salva no banco de dados.
        
        Retorna:
        --------
        dict
            Resumo da geração e processamento
        """
        summary = {}
        
        # Gerar dados
        generated_files = self.generate_synthetic_data()
        
        # Salvar cada arquivo no banco de dados
        for area_name, file_path in generated_files.items():
            try:
                # Salvar no banco de dados
                records_saved = self.save_to_database(file_path)
                
                summary[area_name] = {
                    'file': file_path,
                    'records_saved': records_saved
                }
            except Exception as e:
                summary[area_name] = {'error': str(e)}
        
        return summary

if __name__ == "__main__":
    # Criar e executar coletor
    collector = BathymetryCollector()
    summary = collector.collect_and_process_all()
    
    print("\nResumo da coleta de dados batimétricos:")
    for area, info in summary.items():
        if 'error' in info:
            print(f"  {area}: ERRO - {info['error']}")
        else:
            print(f"  {area}:")
            print(f"    Arquivo: {os.path.basename(info['file'])}")
            print(f"    Registros salvos: {info['records_saved']}")