"""
Módulo para otimização de portfólio de locais de mineração de nódulos polimetálicos,
baseado na metodologia LETE (Lagged Effective Transfer Entropy).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import networkx as nx

class MiningPortfolioOptimizer:
    """
    Classe para otimização de portfólio de locais de mineração usando LETE.
    
    Esta classe adapta a metodologia de Markowitz para incluir a Transferência 
    de Entropia com Defasagem (LETE) como medida de dependência entre locais de 
    mineração, ao invés da correlação tradicional.
    """
    
    def __init__(self, lete_matrix, expected_returns, risk_aversion=1.0):
        """
        Inicializa o otimizador de portfólio.
        
        Parâmetros:
        -----------
        lete_matrix : DataFrame ou array 2D
            Matriz LETE que quantifica a transferência de informação entre locais
        expected_returns : dict ou Series
            Retornos esperados para cada local de mineração
        risk_aversion : float
            Coeficiente de aversão ao risco (lambda)
        """
        # Converter para DataFrame se necessário
        if not isinstance(lete_matrix, pd.DataFrame):
            self.lete_matrix = pd.DataFrame(lete_matrix)
        else:
            self.lete_matrix = lete_matrix
            
        # Converter para Series se necessário
        if isinstance(expected_returns, dict):
            self.expected_returns = pd.Series(expected_returns)
        else:
            self.expected_returns = expected_returns
            
        # Garantir que os índices coincidam
        self.expected_returns = self.expected_returns.loc[self.lete_matrix.index]
        
        self.risk_aversion = risk_aversion
        self.n_sites = len(self.lete_matrix)
        
    def portfolio_risk(self, weights):
        """
        Calcula o risco do portfólio baseado na matriz LETE.
        
        Parâmetros:
        -----------
        weights : array-like
            Pesos de cada local de mineração no portfólio
            
        Retorna:
        --------
        float
            Risco do portfólio
        """
        # Converter pesos para array
        weights = np.array(weights)
        
        # Calcular risco usando a matriz LETE no lugar da matriz de covariância
        risk = weights.T @ self.lete_matrix.values @ weights
        
        return risk
    
    def portfolio_return(self, weights):
        """
        Calcula o retorno esperado do portfólio.
        
        Parâmetros:
        -----------
        weights : array-like
            Pesos de cada local de mineração no portfólio
            
        Retorna:
        --------
        float
            Retorno esperado do portfólio
        """
        # Converter pesos para array
        weights = np.array(weights)
        
        # Calcular retorno esperado
        ret = weights @ self.expected_returns.values
        
        return ret
    
    def objective_function(self, weights):
        """
        Função objetivo a ser minimizada.
        
        Combina retorno esperado e risco de acordo com a aversão ao risco.
        
        Parâmetros:
        -----------
        weights : array-like
            Pesos de cada local de mineração no portfólio
            
        Retorna:
        --------
        float
            Valor da função objetivo (negativo do retorno ajustado ao risco)
        """
        portfolio_ret = self.portfolio_return(weights)
        portfolio_risk = self.portfolio_risk(weights)
        
        # Maximizar retorno e minimizar risco
        # Usamos o negativo pois a função será minimizada
        return -portfolio_ret + self.risk_aversion * portfolio_risk
    
    def optimize(self, allow_negative_weights=False):
        """
        Executa a otimização do portfólio.
        
        Parâmetros:
        -----------
        allow_negative_weights : bool
            Se verdadeiro, permite pesos negativos (posições vendidas)
            
        Retorna:
        --------
        dict
            Dicionário com os resultados da otimização
        """
        # Definir restrições
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]  # Soma dos pesos = 1
        
        # Definir limites
        if allow_negative_weights:
            bounds = None
        else:
            bounds = [(0, 1) for _ in range(self.n_sites)]  # Pesos entre 0 e 1
        
        # Pesos iniciais (igualmente distribuídos)
        initial_weights = np.ones(self.n_sites) / self.n_sites
        
        # Executar otimização
        result = minimize(
            self.objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Preparar resultados
        optimal_weights = result['x']
        optimal_return = self.portfolio_return(optimal_weights)
        optimal_risk = self.portfolio_risk(optimal_weights)
        
        # Criar Series com os pesos ótimos
        weights_series = pd.Series(
            optimal_weights, 
            index=self.lete_matrix.index,
            name='weight'
        )
        
        # Ordenar por peso
        weights_series = weights_series.sort_values(ascending=False)
        
        return {
            'weights': weights_series,
            'expected_return': optimal_return,
            'risk': optimal_risk,
            'success': result['success'],
            'message': result['message']
        }
    
    def optimize_multiple_portfolios(self, risk_aversions):
        """
        Otimiza múltiplos portfólios com diferentes níveis de aversão ao risco.
        
        Parâmetros:
        -----------
        risk_aversions : list
            Lista de coeficientes de aversão ao risco
            
        Retorna:
        --------
        DataFrame
            DataFrame com os resultados para cada nível de aversão ao risco
        """
        results = []
        
        for risk_aversion in risk_aversions:
            # Atualizar aversão ao risco
            self.risk_aversion = risk_aversion
            
            # Otimizar portfólio
            result = self.optimize()
            
            # Adicionar aversão ao risco ao resultado
            result['risk_aversion'] = risk_aversion
            
            # Adicionar à lista de resultados
            results.append(result)
        
        # Extrair métricas principais
        summary = pd.DataFrame({
            'risk_aversion': [r['risk_aversion'] for r in results],
            'expected_return': [r['expected_return'] for r in results],
            'risk': [r['risk'] for r in results]
        })
        
        # Adicionar pesos como colunas adicionais
        for site in self.lete_matrix.index:
            summary[f'weight_{site}'] = [r['weights'].get(site, 0) for r in results]
        
        return summary
    
    def calculate_efficient_frontier(self, n_points=20):
        """
        Calcula a fronteira eficiente.
        
        Parâmetros:
        -----------
        n_points : int
            Número de pontos na fronteira eficiente
            
        Retorna:
        --------
        DataFrame
            DataFrame com os pontos da fronteira eficiente
        """
        # Gerar sequência de coeficientes de aversão ao risco
        # Usa escala logarítmica para melhor cobertura
        risk_aversions = np.logspace(-1, 2, n_points)
        
        # Otimizar portfólios para cada coeficiente
        efficient_frontier = self.optimize_multiple_portfolios(risk_aversions)
        
        return efficient_frontier
    
    def plot_efficient_frontier(self, ax=None):
        """
        Plota a fronteira eficiente.
        
        Parâmetros:
        -----------
        ax : matplotlib.axes.Axes
            Eixo onde plotar. Se None, cria um novo.
            
        Retorna:
        --------
        matplotlib.axes.Axes
            Eixo com o gráfico
        """
        import matplotlib.pyplot as plt
        
        # Calcular fronteira eficiente
        ef = self.calculate_efficient_frontier()
        
        # Criar eixo se necessário
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotar fronteira eficiente
        ax.plot(ef['risk'], ef['expected_return'], 'o-', linewidth=2)
        
        # Adicionar rótulos
        ax.set_xlabel('Risco (LETE)')
        ax.set_ylabel('Retorno Esperado')
        ax.set_title('Fronteira Eficiente baseada em LETE')
        
        # Adicionar grid
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def create_portfolio_network(self, weights, threshold=0.01):
        """
        Cria uma rede do portfólio para visualização.
        
        Parâmetros:
        -----------
        weights : Series
            Pesos ótimos dos locais no portfólio
        threshold : float
            Threshold para incluir um local no gráfico
            
        Retorna:
        --------
        networkx.DiGraph
            Grafo direcionado representando o portfólio
        """
        # Criar grafo dirigido
        G = nx.DiGraph()
        
        # Filtrar locais com peso acima do threshold
        selected_sites = weights[weights > threshold].index
        
        # Adicionar nós para locais selecionados
        for site in selected_sites:
            G.add_node(site, weight=weights[site])
        
        # Adicionar arestas entre locais selecionados
        for source in selected_sites:
            for target in selected_sites:
                if source != target:
                    lete_value = self.lete_matrix.loc[source, target]
                    if lete_value > 0:
                        G.add_edge(source, target, weight=lete_value)
        
        return G
    
    def plot_portfolio_network(self, weights, threshold=0.01, ax=None):
        """
        Plota a rede do portfólio.
        
        Parâmetros:
        -----------
        weights : Series
            Pesos ótimos dos locais no portfólio
        threshold : float
            Threshold para incluir um local no gráfico
        ax : matplotlib.axes.Axes
            Eixo onde plotar. Se None, cria um novo.
            
        Retorna:
        --------
        matplotlib.axes.Axes
            Eixo com o gráfico
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        
        # Criar rede
        G = self.create_portfolio_network(weights, threshold)
        
        # Criar eixo se necessário
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        # Definir layout
        pos = nx.spring_layout(G, seed=42)
        
        # Obter pesos dos nós para definir tamanho
        node_weights = [G.nodes[node]['weight'] * 5000 for node in G.nodes()]
        
        # Obter pesos das arestas para definir largura
        edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        
        # Desenhar nós com tamanho proporcional ao peso no portfólio
        nx.draw_networkx_nodes(G, pos, node_size=node_weights, 
                              node_color='skyblue', alpha=0.8, ax=ax)
        
        # Desenhar arestas com largura proporcional ao LETE
        nx.draw_networkx_edges(G, pos, width=edge_weights, 
                              edge_color='gray', alpha=0.6,
                              arrows=True, arrowsize=15,
                              connectionstyle='arc3,rad=0.1',
                              ax=ax)
        
        # Adicionar rótulos
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # Adicionar título
        ax.set_title('Rede do Portfólio Otimizado')
        
        # Remover eixos
        ax.axis('off')
        
        # Adicionar legenda para tamanho dos nós
        sizes = sorted(node_weights)
        labels = sorted(G.nodes(), key=lambda n: G.nodes[n]['weight'])
        
        # Criar legenda
        for i, (size, label) in enumerate(zip(sizes, labels)):
            weight = G.nodes[label]['weight']
            ax.scatter([], [], s=size, label=f'{label}: {weight:.1%}', color='skyblue', alpha=0.8)
        
        ax.legend(scatterpoints=1, frameon=True, labelspacing=1, title='Pesos do Portfólio')
        
        return ax