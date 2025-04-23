# Viabilidade Econômica da Exploração de Nódulos Polimetálicos no Atlântico

Este repositório contém o código, dados e análises relacionados à pesquisa sobre a viabilidade econômica da exploração de nódulos polimetálicos no Oceano Atlântico, utilizando metodologias de redes complexas e transferência de entropia.

## Visão Geral

Os nódulos polimetálicos são depósitos minerais encontrados no fundo do oceano que contêm concentrações valiosas de metais como manganês, níquel, cobre e cobalto. Este projeto visa analisar a viabilidade econômica da exploração desses recursos no Oceano Atlântico, com foco em:

1. Modelagem de rotas logísticas ótimas entre depósitos de nódulos, plataformas de extração e portos usando redes complexas
2. Análise do impacto econômico no preço dos minérios com o aumento da oferta através de modelos ARIMA, VAR e LSTM
3. Otimização de portfólios de locais de mineração baseada em métricas de risco-retorno

## Metodologia

A metodologia central deste projeto baseia-se na adaptação da Transferência de Entropia com Defasagem (LETE) para o contexto da mineração oceânica. Este método, originalmente aplicado em análises de portfólios financeiros, demonstra maior robustez em condições de alta volatilidade em comparação com métodos tradicionais baseados em correlação.

### Principais Componentes:

- **Construção de Redes Complexas**: Modelagem de relações entre locais de mineração, rotas logísticas e infraestrutura portuária
- **Análise LETE**: Quantificação da transferência de informação entre variáveis econômicas e geológicas
- **Modelagem de Preços**: Implementação de modelos ARIMA, VAR e LSTM para prever impactos no mercado de metais
- **Otimização de Portfólio**: Adaptação de técnicas de otimização de portfólio para seleção de locais de mineração

## Estrutura do Repositório

- `data/`: Conjuntos de dados brutos e processados
- `sql/`: Scripts SQL para processamento de dados
- `notebooks/`: Jupyter notebooks com análises exploratórias e visualizações
- `src/`: Código-fonte Python modularizado
- `results/`: Resultados, figuras e tabelas
- `docs/`: Documentação adicional
- `paper/`: Materiais relacionados à publicação acadêmica

## Requisitos e Instalação

Para replicar o ambiente de análise:

```bash
# Usando conda
conda env create -f environment.yml
conda activate atlantic-nodules

# Ou usando pip
pip install -r requirements.txt
