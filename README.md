<h1 align="center"> Viabilidade Econômica da Exploração de Nódulos Polimetálicos no Atlântico 🌊🔷 </h1>

## Sobre o Projeto

Este projeto analisa a viabilidade econômica da exploração de nódulos polimetálicos no Oceano Atlântico, utilizando metodologias avançadas de redes complexas e transferência de entropia. Os nódulos polimetálicos são concreções rochosas encontradas no fundo oceânico que contêm concentrações valiosas de metais como manganês, níquel, cobre e cobalto, essenciais para tecnologias verdes e baterias.

### Objetivos

1. Modelar rotas logísticas ótimas entre depósitos de nódulos, plataformas de extração e portos
2. Prever flutuações nos preços dos metais relevantes utilizando múltiplos modelos
3. Otimizar a seleção de locais de mineração com base em métricas de risco-retorno
4. Analisar a viabilidade econômica global considerando fatores interconectados

## Metodologia

A metodologia central deste projeto é a **Transferência de Entropia com Defasagem (LETE)**, adaptada para o contexto da mineração oceânica. Esta técnica quantifica as relações causais entre variáveis complexas e demonstra maior robustez em ambientes de alta volatilidade, comparada aos métodos tradicionais baseados em correlação.

### Componentes Principais

- **Análise de Redes Complexas**: Modelagem das relações entre locais de mineração, infraestrutura e variáveis econômicas
- **Previsão de Preços**: Comparação de modelos ARIMA, VAR e LSTM para projetar preços futuros de metais
- **Otimização de Portfólio**: Seleção de locais de mineração utilizando LETE vs. abordagem tradicional de Markowitz
- **Análise Integrada**: Síntese de todos os componentes para avaliação global de viabilidade

### Bases Utilizadas

-
-
-
-

## Estrutura do Repositório

data/ - Dados brutos e processados

raw/ - Dados brutos
processed/ - Dados processados

notebooks/ - Jupyter notebooks para análise

01_data_exploration.ipynb
02_network_analysis.ipynb
03_portfolio_optimization.ipynb
04_economic_analysis.py
price_forecasting_comparison.py
portfolio_optimization.py
integrated_analysis.py

results/ - Resultados das análises

figures/ - Gráficos e visualizações
tables/ - Tabelas de resultados

sql/ - Scripts SQL para processamento de dados

schema.sql

src/ - Código-fonte modular

data/ - Módulos para coleta e processamento de dados
models/ - Módulos para modelagem
network/ - Módulos para análise de redes
optimization/ - Módulos para otimização

docs/ - Documentação adicional
paper/ - Material para publicação acadêmica
requirements.txt - Dependências do projeto
setup.py - Script de configuração
README.md - Este arquivo

## Como Usar

### Instalação

1. Clone o repositório:
 ```bash
git clone https://github.com/seu-usuario/atlantic-polymetallic-nodules.git
cd atlantic-polymetallic-nodules

2. Crie um ambiente virtual e instale as dependências:
 ```bash
python -m venv venv #talvez seja necessário utilizar o nodules por conta de algumas ferramentas (como por exemplo o tensor flow)
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure o banco de dados
 ```bash
python src/data/setup_database.py all
```

### Execução

1. Para executar a análise completa:
 ```bash
python run_analysis.py
```

ou execute cada componente individualmente:
 ```bash
#Configurar banco de dados

python src/data/setup_database.py

#Gerar dados simulados

python src/data/metal_prices_collector.py
python src/data/bathymetry_collector.py

#Executar análises

cd notebooks
python price_forecasting_comparison.py
python portfolio_optimization.py
python integrated_analysis.py
```

### Resultados e Visualizações

Os resultados são armazenados no diretório results/ e incluem:
- Previsões de preços de metais (ARIMA, VAR, LSTM) ✅
- Matrizes LETE e visualizações de redes complexas ✅
- Rotas logísticas ótimas ✅
- Fronteiras eficientes para otimização de portfólio ✅
- Análise de sensibilidade e cenários ✅
- Conclusões integradas sobre viabilidade econômica ✅

Algumas visualizações de exemplo:
- Matriz LETE para transferência de informação entre variáveis
- Rede complexa de locais de mineração, plataformas e portos
- Fronteira eficiente para alocação de recursos
- Previsão de preços para metais críticos

## Em andamento:
- Integração com os dados reais 🔄
- Correção de alguns problemas nos scripts 🔄
- Otimização dos scripts 🔄

## Resultados Principais 

Os resultados preliminares com os dados mostram:
- 
- 
- 
- 

## Próximos Passos:

- Integração com dados reais de exploração ⏭️
- Refinamento dos modelos de preço utilizando séries temporais mais longas ⏭️
- Inclusão de fatores ambientais e regulatórios na análise ⏭️
- Desenvolvimento de modelo de simulação para diferentes cenários de extração ⏭️

_

Licença
Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

Contato:
Guilherme França - franca.guilherme@outlook.pt
