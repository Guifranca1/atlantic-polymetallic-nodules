"""
Análise Econômica Preliminar para Nódulos Polimetálicos no Atlântico

Este script realiza uma análise econômica preliminar combinando os dados batimétricos e 
de preços de metais para avaliar a viabilidade econômica da exploração de nódulos 
polimetálicos no Atlântico.
"""

# Importar bibliotecas
import os
import sys
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

# Configurar visualizações
plt.style.use('seaborn-v0_8-whitegrid')  # Versão compatível
sns.set(style="whitegrid", font_scale=1.2)

# Criar diretórios para figuras, se não existirem
os.makedirs('../results/figures', exist_ok=True)

# Adicionar o diretório raiz ao path
sys.path.append(os.path.abspath('../'))

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

# Função para conectar ao banco de dados e executar consultas
def query_database(query, params=None, dp_path=db_path):
    """Executa uma consulta no banco de dados SQLite e retorna os resultados como um DataFrame."""
    # Converter para caminho absoluto
    import os
    abs_path = os.path.abspath(db_path)
    
    # Verificar se o arquivo existe
    if not os.path.exists(abs_path):
        print(f"ERRO: Banco de dados não encontrado em: {abs_path}")
        print("Caminhos possíveis para procurar:")
        # Listar todos os arquivos .db em diretórios possíveis
        for root, dirs, files in os.walk('..'):
            for file in files:
                if file.endswith('.db'):
                    print(f"  - {os.path.join(root, file)}")
        raise FileNotFoundError(f"Banco de dados não encontrado: {abs_path}")
    
    # Se chegou aqui, o arquivo existe
    conn = sqlite3.connect(abs_path)
    
    if params:
        df = pd.read_sql_query(query, conn, params=params)
    else:
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df

print("1. Carregando dados do banco de dados...")

# Carregar dados de preços de metais
price_query = """
SELECT metal_name, date, price_usd_ton
FROM metal_prices
ORDER BY metal_name, date
"""

prices_df = query_database(price_query)
prices_df['date'] = pd.to_datetime(prices_df['date'])

# Converter para formato wide (uma coluna por metal)
prices_wide = prices_df.pivot(index='date', columns='metal_name', values='price_usd_ton')
prices_wide.columns.name = None

print(f"Dados de preços carregados: {len(prices_df)} registros para {prices_df['metal_name'].nunique()} metais")
print(f"Período: {prices_df['date'].min()} a {prices_df['date'].max()}")

# Mostrar primeiros registros
print("\nPrimeiros registros de preços:")
print(prices_wide.head())

# Carregar dados batimétricos para as áreas de interesse
bathymetry_query = """
SELECT latitude, longitude, depth_m, source
FROM bathymetry
"""

bathymetry_df = query_database(bathymetry_query)

print(f"\nDados batimétricos carregados: {len(bathymetry_df)} pontos")
print(f"Profundidade média: {bathymetry_df['depth_m'].mean():.1f} m")
print(f"Profundidade mínima: {bathymetry_df['depth_m'].min():.1f} m")
print(f"Profundidade máxima: {bathymetry_df['depth_m'].max():.1f} m")

# Mostrar primeiros registros
print("\nPrimeiros registros batimétricos:")
print(bathymetry_df.head())

print("\n2. Realizando análise dos preços de metais...")

# Plotar séries temporais de preços
plt.figure(figsize=(14, 8))

for metal in prices_wide.columns:
    plt.plot(prices_wide.index, prices_wide[metal], label=metal)

plt.title('Evolução dos Preços de Metais Relevantes para Nódulos Polimetálicos')
plt.xlabel('Data')
plt.ylabel('Preço (USD/ton)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/metal_prices_evolution.png')
plt.close()
print("Figura salva: metal_prices_evolution.png")

# Calcular estatísticas por metal
price_stats = pd.DataFrame({
    'Preço Médio (USD/ton)': prices_wide.mean(),
    'Preço Mínimo (USD/ton)': prices_wide.min(),
    'Preço Máximo (USD/ton)': prices_wide.max(),
    'Desvio Padrão': prices_wide.std(),
    'Coeficiente de Variação (%)': (prices_wide.std() / prices_wide.mean()) * 100
})

print("\nEstatísticas de preços:")
print(price_stats.sort_values('Preço Médio (USD/ton)', ascending=False))

# Calcular correlações entre os preços dos metais
price_corr = prices_wide.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(price_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlação entre Preços de Metais')
plt.tight_layout()
plt.savefig('../results/figures/metal_prices_correlation.png')
plt.close()
print("Figura salva: metal_prices_correlation.png")

# Calcular retornos mensais
monthly_returns = prices_wide.resample('M').last().pct_change().dropna()

# Plotar distribuição dos retornos mensais
plt.figure(figsize=(14, 10))

for i, metal in enumerate(monthly_returns.columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(monthly_returns[metal], bins=30, kde=True)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title(f'Distribuição dos Retornos Mensais: {metal}')
    plt.xlabel('Retorno Mensal')
    plt.ylabel('Frequência')
    
plt.tight_layout()
plt.savefig('../results/figures/metal_returns_distribution.png')
plt.close()
print("Figura salva: metal_returns_distribution.png")

# Calcular estatísticas dos retornos
return_stats = pd.DataFrame({
    'Retorno Médio Mensal (%)': monthly_returns.mean() * 100,
    'Volatilidade Mensal (%)': monthly_returns.std() * 100,
    'Retorno/Risco': monthly_returns.mean() / monthly_returns.std(),
    'Assimetria': monthly_returns.apply(stats.skew),
    'Curtose': monthly_returns.apply(stats.kurtosis)
})

print("\nEstatísticas de retornos:")
print(return_stats.sort_values('Retorno/Risco', ascending=False))

print("\n3. Analisando dados batimétricos...")

# Identificar áreas principais presentes nos dados
areas = bathymetry_df['source'].str.extract(r'(rio_grande_rise|mid_atlantic_ridge|bermuda_rise|azores_plateau)')[0].fillna('other')
bathymetry_df['area'] = areas

# Resumo estatístico por área
depth_by_area = bathymetry_df.groupby('area')['depth_m'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
depth_by_area.columns = ['Área', 'Pontos', 'Profundidade Média (m)', 'Desvio Padrão (m)', 'Profundidade Mínima (m)', 'Profundidade Máxima (m)']
depth_by_area = depth_by_area.sort_values('Profundidade Média (m)')

print("\nProfundidade por área:")
print(depth_by_area)

# Plotar distribuição de profundidades por área
plt.figure(figsize=(14, 8))

for area in bathymetry_df['area'].unique():
    if area != 'other':
        sns.kdeplot(bathymetry_df[bathymetry_df['area'] == area]['depth_m'], label=area)

plt.title('Distribuição de Profundidades por Área de Interesse')
plt.xlabel('Profundidade (m)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/depth_distribution.png')
plt.close()
print("Figura salva: depth_distribution.png")

# Criar scatter plot colorido por profundidade para cada área
areas_to_plot = [area for area in bathymetry_df['area'].unique() if area != 'other']
n_areas = len(areas_to_plot)
n_cols = 2
n_rows = (n_areas + n_cols - 1) // n_cols

plt.figure(figsize=(14, 4 * n_rows))

for i, area in enumerate(areas_to_plot, 1):
    area_data = bathymetry_df[bathymetry_df['area'] == area]
    
    plt.subplot(n_rows, n_cols, i)
    sc = plt.scatter(area_data['longitude'], area_data['latitude'], 
                    c=area_data['depth_m'], cmap='viridis', 
                    s=10, alpha=0.7)
    
    plt.title(f'{area.replace("_", " ").title()}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(sc, label='Profundidade (m)')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/figures/depth_maps.png')
plt.close()
print("Figura salva: depth_maps.png")

print("\n4. Realizando análise econômica preliminar...")

# Parâmetros típicos para nódulos polimetálicos no Atlântico
nodule_params = pd.DataFrame([
    {'area': 'rio_grande_rise', 'density_kg_m2': 10.0, 'ni_pct': 1.2, 'cu_pct': 1.0, 'co_pct': 0.22, 'mn_pct': 22.0},
    {'area': 'mid_atlantic_ridge', 'density_kg_m2': 8.0, 'ni_pct': 1.1, 'cu_pct': 0.9, 'co_pct': 0.20, 'mn_pct': 20.0},
    {'area': 'bermuda_rise', 'density_kg_m2': 7.0, 'ni_pct': 1.0, 'cu_pct': 0.8, 'co_pct': 0.18, 'mn_pct': 18.0},
    {'area': 'azores_plateau', 'density_kg_m2': 6.0, 'ni_pct': 0.9, 'cu_pct': 0.7, 'co_pct': 0.16, 'mn_pct': 17.0}
])

print("\nParâmetros dos nódulos por área:")
print(nodule_params)

# Calcular média de preços para o último ano
last_year = prices_wide.last('365D')
metal_prices = {
    'Nickel': last_year['Nickel'].mean(),
    'Copper': last_year['Copper'].mean(),
    'Cobalt': last_year['Cobalt'].mean(),
    'Manganese': last_year['Manganese'].mean()
}

print("\nPreços médios do último ano (USD/ton):")
for metal, price in metal_prices.items():
    print(f"  {metal}: ${price:.2f}")

# Calcular valor por m² para cada área
value_per_m2 = []

for _, row in nodule_params.iterrows():
    area = row['area']
    density = row['density_kg_m2']
    
    # Valor por metal (em USD por m²)
    ni_value = density * (row['ni_pct'] / 100) * metal_prices['Nickel'] / 1000
    cu_value = density * (row['cu_pct'] / 100) * metal_prices['Copper'] / 1000
    co_value = density * (row['co_pct'] / 100) * metal_prices['Cobalt'] / 1000
    mn_value = density * (row['mn_pct'] / 100) * metal_prices['Manganese'] / 1000
    
    total_value = ni_value + cu_value + co_value + mn_value
    
    value_per_m2.append({
        'area': area,
        'density_kg_m2': density,
        'ni_value_usd_m2': ni_value,
        'cu_value_usd_m2': cu_value,
        'co_value_usd_m2': co_value,
        'mn_value_usd_m2': mn_value,
        'total_value_usd_m2': total_value
    })

value_df = pd.DataFrame(value_per_m2)

print("\nValor estimado por área (USD/m²):")
print(value_df)

# Visualizar valor por m² para cada área
plt.figure(figsize=(14, 8))

# Preparar dados para gráfico de barras empilhadas
areas = value_df['area']
ni_values = value_df['ni_value_usd_m2']
cu_values = value_df['cu_value_usd_m2']
co_values = value_df['co_value_usd_m2']
mn_values = value_df['mn_value_usd_m2']

# Criar barras empilhadas
bar_width = 0.6
x = np.arange(len(areas))

plt.bar(x, ni_values, bar_width, label='Níquel', color='#1f77b4')
plt.bar(x, cu_values, bar_width, bottom=ni_values, label='Cobre', color='#ff7f0e')
plt.bar(x, co_values, bar_width, bottom=ni_values+cu_values, label='Cobalto', color='#2ca02c')
plt.bar(x, mn_values, bar_width, bottom=ni_values+cu_values+co_values, label='Manganês', color='#d62728')

# Configurar rótulos e título
plt.xlabel('Área')
plt.ylabel('Valor (USD/m²)')
plt.title('Valor Estimado de Nódulos Polimetálicos por Área')
plt.xticks(x, [area.replace('_', ' ').title() for area in areas])
plt.legend()

# Adicionar rótulos de valor total
for i, total in enumerate(value_df['total_value_usd_m2']):
    plt.text(i, total + 0.5, f'${total:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/figures/nodule_value_by_area.png')
plt.close()
print("Figura salva: nodule_value_by_area.png")

# Adicionar profundidade média para cada área
depth_means = bathymetry_df.groupby('area')['depth_m'].mean()
value_df = value_df.merge(depth_means.reset_index(), on='area', how='left')

# Calcular relação valor/profundidade
value_df['value_depth_ratio'] = value_df['total_value_usd_m2'] / value_df['depth_m']

# Ordenar por relação valor/profundidade
value_df = value_df.sort_values('value_depth_ratio', ascending=False)

print("\nValor vs. profundidade:")
print(value_df)

# Visualizar relação entre valor e profundidade
plt.figure(figsize=(12, 8))

x = value_df['depth_m']
y = value_df['total_value_usd_m2']
sizes = value_df['density_kg_m2'] * 20  # Tamanho do ponto proporcional à densidade

sc = plt.scatter(x, y, s=sizes, alpha=0.7, c=range(len(value_df)), cmap='viridis')

# Adicionar rótulos para cada ponto
for i, row in value_df.iterrows():
    plt.annotate(row['area'].replace('_', ' ').title(), 
                xy=(row['depth_m'], row['total_value_usd_m2']),
                xytext=(5, 5), textcoords='offset points')

plt.xlabel('Profundidade Média (m)')
plt.ylabel('Valor Estimado (USD/m²)')
plt.title('Valor vs. Profundidade para Áreas de Nódulos Polimetálicos')
plt.grid(True, alpha=0.3)
plt.colorbar(sc, label='Ranking (Melhor para Pior)')
plt.tight_layout()
plt.savefig('../results/figures/value_vs_depth.png')
plt.close()
print("Figura salva: value_vs_depth.png")

# Estimar custos de extração baseados na profundidade
# Fórmula simplificada: custo base + fator de profundidade
base_cost_per_ton = 300  # USD por tonelada
depth_cost_factor = 0.05  # USD por metro de profundidade por tonelada

value_df['extraction_cost_usd_ton'] = base_cost_per_ton + value_df['depth_m'] * depth_cost_factor

# Calcular valor por tonelada de nódulos
value_df['total_value_usd_ton'] = value_df['total_value_usd_m2'] * 1000 / value_df['density_kg_m2']

# Calcular margem por tonelada
value_df['margin_usd_ton'] = value_df['total_value_usd_ton'] - value_df['extraction_cost_usd_ton']
value_df['margin_pct'] = (value_df['margin_usd_ton'] / value_df['total_value_usd_ton']) * 100

# Ordenar por margem
value_df = value_df.sort_values('margin_usd_ton', ascending=False)

# Exibir resultados
result_columns = ['area', 'depth_m', 'density_kg_m2', 'total_value_usd_m2', 
                 'total_value_usd_ton', 'extraction_cost_usd_ton', 'margin_usd_ton', 'margin_pct']

print("\nResultados econômicos (margem por área):")
pd.set_option('display.float_format', '${:.2f}'.format)
print(value_df[result_columns])
pd.reset_option('display.float_format')

# Visualizar margem por área
plt.figure(figsize=(12, 6))

x = np.arange(len(value_df))
width = 0.35

plt.bar(x - width/2, value_df['total_value_usd_ton'], width, label='Valor (USD/ton)', color='green')
plt.bar(x + width/2, value_df['extraction_cost_usd_ton'], width, label='Custo (USD/ton)', color='red')

plt.xlabel('Área')
plt.ylabel('USD por Tonelada')
plt.title('Valor vs. Custo de Extração por Área')
plt.xticks(x, [area.replace('_', ' ').title() for area in value_df['area']])
plt.legend()

# Adicionar rótulos de margem
for i, row in value_df.reset_index().iterrows():
    plt.text(i, row['total_value_usd_ton'] + 50, 
            f"Margem: ${row['margin_usd_ton']:.0f}", 
            ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('../results/figures/margin_by_area.png')
plt.close()
print("Figura salva: margin_by_area.png")

print("\n5. Realizando análise de sensibilidade...")

# Criar cenários de preços
scenarios = {
    'Base': {'Nickel': 1.0, 'Copper': 1.0, 'Cobalt': 1.0, 'Manganese': 1.0},
    'Metais de Bateria +20%': {'Nickel': 1.2, 'Copper': 1.0, 'Cobalt': 1.2, 'Manganese': 1.0},
    'Metais Industriais +20%': {'Nickel': 1.0, 'Copper': 1.2, 'Cobalt': 1.0, 'Manganese': 1.2},
    'Todos +10%': {'Nickel': 1.1, 'Copper': 1.1, 'Cobalt': 1.1, 'Manganese': 1.1},
    'Todos -10%': {'Nickel': 0.9, 'Copper': 0.9, 'Cobalt': 0.9, 'Manganese': 0.9}
}

# Calcular margens para cada cenário e área
scenario_results = []

for scenario_name, price_factors in scenarios.items():
    # Ajustar preços conforme o cenário
    adjusted_prices = {
        metal: price * price_factors[metal]
        for metal, price in metal_prices.items()
    }
    
    for _, row in nodule_params.iterrows():
        area = row['area']
        density = row['density_kg_m2']
        
        # Calcular valor por metal (em USD por m²)
        ni_value = density * (row['ni_pct'] / 100) * adjusted_prices['Nickel'] / 1000
        cu_value = density * (row['cu_pct'] / 100) * adjusted_prices['Copper'] / 1000
        co_value = density * (row['co_pct'] / 100) * adjusted_prices['Cobalt'] / 1000
        mn_value = density * (row['mn_pct'] / 100) * adjusted_prices['Manganese'] / 1000
        
        total_value_m2 = ni_value + cu_value + co_value + mn_value
        total_value_ton = total_value_m2 * 1000 / density
        
        # Obter profundidade média da área
        depth = bathymetry_df[bathymetry_df['area'] == area]['depth_m'].mean()
        
        # Calcular custo de extração
        extraction_cost = base_cost_per_ton + depth * depth_cost_factor
        
        # Calcular margem
        margin = total_value_ton - extraction_cost
        margin_pct = (margin / total_value_ton) * 100
        
        # Adicionar aos resultados
        scenario_results.append({
            'scenario': scenario_name,
            'area': area,
            'total_value_usd_ton': total_value_ton,
            'extraction_cost_usd_ton': extraction_cost,
            'margin_usd_ton': margin,
            'margin_pct': margin_pct
        })

scenario_df = pd.DataFrame(scenario_results)

# Visualizar margens por cenário para cada área
for area in nodule_params['area'].unique():
    area_data = scenario_df[scenario_df['area'] == area]
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    plt.bar(x, area_data['margin_usd_ton'], color='teal')
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Cenário')
    plt.ylabel('Margem (USD/ton)')
    plt.title(f'Sensibilidade da Margem de Lucro a Preços de Metais: {area.replace("_", " ").title()}')
    plt.xticks(x, area_data['scenario'])
    
    # Adicionar rótulos de margem
    for i, row in area_data.reset_index().iterrows():
        plt.text(i, row['margin_usd_ton'] + (50 if row['margin_usd_ton'] >= 0 else -50), 
                f"${row['margin_usd_ton']:.0f}", 
                ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'../results/figures/margin_sensitivity_{area}.png')
    plt.close()
    print(f"Figura salva: margin_sensitivity_{area}.png")

# Criar uma tabela comparativa da viabilidade por área e cenário
pivot_margins = scenario_df.pivot_table(
    index='area', 
    columns='scenario', 
    values='margin_usd_ton'
)

# Formatar os nomes das áreas para melhor visualização
pivot_margins.index = [area.replace('_', ' ').title() for area in pivot_margins.index]

print("\nMargens por área e cenário (USD/ton):")
pd.set_option('display.float_format', '${:.0f}'.format)
print(pivot_margins)
pd.reset_option('display.float_format')

print("\nConclusões da análise econômica preliminar:")
print("""
1. A área do Rio Grande Rise apresenta o maior potencial econômico, combinando densidade
   relativamente alta de nódulos com teores de metais superiores.

2. A profundidade tem um impacto significativo nos custos de extração, favorecendo áreas menos profundas.

3. Os preços dos metais, especialmente níquel e cobalto, são os principais determinantes
   da viabilidade econômica.

4. A análise de sensibilidade indica que a viabilidade econômica é bastante sensível a variações
   nos preços dos metais, com algumas áreas se tornando inviáveis em cenários de preços mais baixos.

5. A relação valor/profundidade é um indicador útil para priorizar áreas de exploração.
""")

print("\nPróximos passos:")
print("""
1. Desenvolver modelos de previsão de preços (ARIMA, VAR, LSTM) para projetar cenários futuros.
2. Incorporar custos logísticos e de processamento mais detalhados.
3. Realizar análise de redes complexas para otimizar rotas logísticas.
4. Avaliar impactos ambientais e regulatórios.
""")