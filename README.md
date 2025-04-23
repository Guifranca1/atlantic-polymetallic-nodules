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

## Status do Projeto

**Fase atual**: Configuração inicial e coleta de dados

## Status do Projeto

### Concluído:
- ✅ Configuração inicial do repositório e ambiente de desenvolvimento
- ✅ Implementação do módulo LETE para análise de transferência de informação
- ✅ Scripts para coleta de dados simulados
- ✅ Implementação da otimização de portfólio baseada em LETE
- ✅ Notebooks de demonstração para análise de redes complexas e otimização

### Em andamento:
- 🔄 Coleta de dados reais sobre nódulos polimetálicos no Atlântico
- 🔄 Análise de impactos de preço usando modelos ARIMA, VAR e LSTM

### Próximas etapas:
- ⏭️ Integração de dados geoespaciais para otimização de rotas logísticas
- ⏭️ Implementação de análise de sensibilidade mais abrangente
- ⏭️ Modelagem de impactos ambientais e regulatórios

## Primeiros Resultados

Os testes iniciais usando dados simulados demonstraram que:

1. A metodologia LETE permite identificar de forma robusta relações causais entre variáveis econômicas e geológicas, mesmo em condições de alta volatilidade
2. A otimização de portfólio baseada em LETE oferece uma alternativa mais robusta à abordagem tradicional de Markowitz para seleção de portfólios
3. A viabilidade econômica da exploração é significativamente influenciada pelas variações nos preços dos metais, com diferentes locais apresentando sensibilidades distintas

Estes resultados preliminares fornecem uma base sólida para aplicação a dados reais e desenvolvimento de um modelo mais completo para avaliação de viabilidade econômica.

## Contato

Guilherme França - franca.guilherme@outlook.pt
