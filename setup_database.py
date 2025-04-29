# setup_database.py
# Script para verificar, criar e popular o banco de dados para análise de nódulos polimetálicos

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("=" * 80)
print("CONFIGURAÇÃO DO BANCO DE DADOS PARA ANÁLISE DE NÓDULOS POLIMETÁLICOS")
print("=" * 80)

# Obter caminho absoluto para o diretório raiz do projeto
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir  # Assume que o script está na raiz do projeto

# Definir caminho do banco de dados
data_dir = os.path.join(project_root, 'data')
db_path = os.path.join(data_dir, 'nodules.db')

print(f"Diretório raiz do projeto: {project_root}")
print(f"Caminho do banco de dados: {db_path}")

# Criar diretório de dados se não existir
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Diretório de dados criado: {data_dir}")

# Verificar se o banco de dados já existe
db_exists = os.path.exists(db_path)
if db_exists:
    print(f"Banco de dados já existe em: {db_path}")
    action = input("Deseja sobrescrever o banco existente? (s/n): ")
    if action.lower() != 's':
        print("Operação cancelada pelo usuário.")
        sys.exit(0)

# Criar/Conectar ao banco de dados
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Criando tabelas no banco de dados...")

# Remover tabelas antigas se existirem
print("Removendo tabelas antigas se existirem...")
cursor.execute("DROP TABLE IF EXISTS metal_prices;")
cursor.execute("DROP TABLE IF EXISTS bathymetry;")
cursor.execute("DROP TABLE IF EXISTS nodule_samples;")
cursor.execute("DROP TABLE IF EXISTS exploration_contracts;")
cursor.execute("DROP TABLE IF EXISTS ocean_conditions;")

# Criação das tabelas
# 1. Tabela de preços de metais
cursor.execute('''
CREATE TABLE IF NOT EXISTS metal_prices (
    price_id INTEGER PRIMARY KEY AUTOINCREMENT,
    metal_name VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    price_usd_ton DECIMAL(10,2) NOT NULL,
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(metal_name, date)
);
''')

# 2. Tabela de dados batimétricos
cursor.execute('''
DROP TABLE IF EXISTS bathymetry;
''')
cursor.execute('''
CREATE TABLE bathymetry (
    bathymetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude DECIMAL(10,6) NOT NULL,
    longitude DECIMAL(10,6) NOT NULL,
    depth_m DECIMAL(8,2) NOT NULL,
    source VARCHAR(100),
    area VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
''')

# 3. Tabela de amostras de nódulos
cursor.execute('''
CREATE TABLE IF NOT EXISTS nodule_samples (
    sample_id VARCHAR(50) PRIMARY KEY,
    latitude DECIMAL(10,6) NOT NULL,
    longitude DECIMAL(10,6) NOT NULL,
    depth_m DECIMAL(8,2),
    area VARCHAR(100),
    ni_pct DECIMAL(5,2),
    cu_pct DECIMAL(5,2),
    co_pct DECIMAL(5,2),
    mn_pct DECIMAL(5,2),
    fe_pct DECIMAL(5,2),
    density_kg_m2 DECIMAL(8,2),
    nodule_size_mm DECIMAL(6,2),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
''')

print("Tabelas criadas com sucesso!")

print("\nPopulando banco de dados com dados de exemplo...")

# Inserir dados de exemplo
# 1. Dados batimétricos
print("Inserindo dados batimétricos...")

areas = ['rio_grande_rise', 'mid_atlantic_ridge', 'bermuda_rise', 'azores_plateau']
bathymetry_data = []

for area in areas:
    # Coordenadas aproximadas por área
    if area == 'rio_grande_rise':
        base_lat, base_lon = -30.0, -35.0
        base_depth = 4500.0
    elif area == 'mid_atlantic_ridge':
        base_lat, base_lon = -10.0, -40.0
        base_depth = 4800.0
    elif area == 'bermuda_rise':
        base_lat, base_lon = 32.0, -65.0
        base_depth = 5000.0
    elif area == 'azores_plateau':
        base_lat, base_lon = 38.0, -28.0
        base_depth = 3800.0
    
    # Gerar múltiplos pontos para cada área
    for i in range(20):
        lat_offset = random.uniform(-2.0, 2.0)
        lon_offset = random.uniform(-2.0, 2.0)
        depth_offset = random.uniform(-200.0, 200.0)
        
        bathymetry_data.append((
            base_lat + lat_offset,
            base_lon + lon_offset,
            base_depth + depth_offset,
            'synthetic',
            area
        ))

# Inserir dados de batimetria
cursor.executemany('''
INSERT INTO bathymetry (latitude, longitude, depth_m, source, area)
VALUES (?, ?, ?, ?, ?)
''', bathymetry_data)
print(f"Inseridos {len(bathymetry_data)} registros de batimetria")

# 2. Inserir dados de amostras de nódulos
print("Inserindo dados de amostras de nódulos...")

nodule_data = []
area_params = {
    'rio_grande_rise': {'ni': 1.2, 'cu': 1.0, 'co': 0.22, 'mn': 22.0, 'density': 10.0},
    'mid_atlantic_ridge': {'ni': 1.1, 'cu': 0.9, 'co': 0.20, 'mn': 20.0, 'density': 8.0},
    'bermuda_rise': {'ni': 1.0, 'cu': 0.8, 'co': 0.18, 'mn': 18.0, 'density': 7.0},
    'azores_plateau': {'ni': 0.9, 'cu': 0.7, 'co': 0.16, 'mn': 17.0, 'density': 6.0}
}

sample_id = 1
for area, params in area_params.items():
    # Filtrar pontos de batimetria para esta área
    area_points = [p for p in bathymetry_data if p[4] == area]
    
    # Usar pontos existentes como base para amostras
    for i, point in enumerate(area_points[:10]):  # Limitar a 10 amostras por área
        variation = random.uniform(-0.1, 0.1)  # Variação de ±10%
        
        nodule_data.append((
            f"{area[:3].upper()}{sample_id:03d}",
            point[0],  # latitude
            point[1],  # longitude
            point[2],  # depth
            area,
            params['ni'] * (1 + variation),
            params['cu'] * (1 + variation),
            params['co'] * (1 + variation),
            params['mn'] * (1 + variation),
            5.0 + random.uniform(0, 2.0),  # fe_pct
            params['density'] * (1 + variation),
            40.0 + random.uniform(-10, 10),  # nodule_size
            f"Amostra sintética para {area}"
        ))
        sample_id += 1

# Inserir dados de nódulos
cursor.executemany('''
INSERT INTO nodule_samples (
    sample_id, latitude, longitude, depth_m, area, 
    ni_pct, cu_pct, co_pct, mn_pct, fe_pct, 
    density_kg_m2, nodule_size_mm, notes
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', nodule_data)
print(f"Inseridos {len(nodule_data)} registros de amostras de nódulos")

# 3. Inserir dados de preços de metais
print("Inserindo dados históricos de preços de metais...")

end_date = datetime.now()
metals = ['Nickel', 'Copper', 'Cobalt', 'Manganese']
base_prices = {'Nickel': 15000, 'Copper': 8000, 'Cobalt': 30000, 'Manganese': 1700}
price_data = []

# Gerar 60 meses de dados
for i in range(60):
    date = (end_date - timedelta(days=30*i)).strftime('%Y-%m-%d')
    month = i % 12  # Mês do ano (0-11)
    year_factor = 1 + (i // 12) * 0.05  # Fator de crescimento por ano
    
    for metal in metals:
        # Componente sazonal
        season = 0.1 * np.sin(month/12 * 2 * np.pi)
        # Componente de tendência
        trend = 0.2 * (1 - (i / 60))  # Tendência decrescente ao longo do tempo
        # Componente aleatória
        noise = random.uniform(-0.05, 0.05)
        
        # Preço final
        price = base_prices[metal] * year_factor * (1 + trend + season + noise)
        
        price_data.append((metal, date, price, 'synthetic'))

# Inserir dados de preços
cursor.executemany('''
INSERT INTO metal_prices (metal_name, date, price_usd_ton, source)
VALUES (?, ?, ?, ?)
''', price_data)
print(f"Inseridos {len(price_data)} registros de preços de metais")

# Commit alterações e fechar conexão
conn.commit()
conn.close()

print("\nBanco de dados configurado com sucesso!")
print(f"Caminho do banco: {db_path}")
print("Contagens de registros:")
print(f"  - Batimetria: {len(bathymetry_data)} registros")
print(f"  - Amostras de nódulos: {len(nodule_data)} registros")
print(f"  - Preços de metais: {len(price_data)} registros")

print("\nAgora você pode executar os scripts de análise:")
print("  1. python notebooks/05_price_forecasting.py")
print("  2. python notebooks/06_integrated_analysis.py")