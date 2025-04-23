# src/network/lete.py

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer

class LETECalculator:
    """
    Implementation of Lagged Effective Transfer Entropy (LETE) for spatial and economic data.
    Adapted from the methodology described in de Castro's work on portfolio construction.
    """
    
    def __init__(self, k=1, l=1, bins=5, shuffle_iterations=100, discretization='quantile'):
        """
        Initialize LETE calculator.
        
        Parameters:
        -----------
        k : int
            Markov order for target series
        l : int
            Markov order for source series
        bins : int
            Number of bins for discretization
        shuffle_iterations : int
            Number of random shuffles for effective transfer entropy calculation
        discretization : str
            Method for discretization ('quantile', 'uniform', or 'kmeans')
        """
        self.k = k
        self.l = l
        self.bins = bins
        self.shuffle_iterations = shuffle_iterations
        self.discretization = discretization
    
    def discretize(self, data):
        """
        Discretize continuous data into bins.
        
        Parameters:
        -----------
        data : array-like
            Continuous data to discretize
            
        Returns:
        --------
        array-like
            Discretized data
        """
        # Reshape for sklearn
        reshaped_data = data.reshape(-1, 1)
        
        # Initialize discretizer
        discretizer = KBinsDiscretizer(
            n_bins=self.bins, 
            encode='ordinal', 
            strategy=self.discretization
        )
        
        # Fit and transform
        return discretizer.fit_transform(reshaped_data).flatten()
    
    def calculate_te(self, source, target):
        """
        Calculate basic Transfer Entropy from source to target.
        
        Parameters:
        -----------
        source : array-like
            Source time series
        target : array-like
            Target time series
            
        Returns:
        --------
        float
            Transfer entropy value
        """
        # Ensure arrays
        source = np.asarray(source)
        target = np.asarray(target)
        
        # Discretize data
        source_d = self.discretize(source)
        target_d = self.discretize(target)
        
        # Create lagged variables
        target_future = target_d[self.k:]
        target_past = np.zeros((len(target_d) - self.k, self.k))
        
        # Corrigido: Preencher target_past de forma consistente
        for i in range(self.k):
            target_past[:, i] = target_d[i:len(target_d)-self.k+i]
            
        source_past = np.zeros((len(source_d) - self.k, self.l))
        
        # Corrigido: Preencher source_past de forma consistente
        for i in range(self.l):
            source_past[:, i] = source_d[i:len(source_d)-self.k+i]
        
        # Calculate joint and conditional probabilities
        # This is a simplified implementation - in production, we would use more efficient methods
        
        # Target future conditioned on target past
        h_future_past = self._conditional_entropy(target_future, target_past)
        
        # Target future conditioned on both target past and source past
        h_future_past_source = self._conditional_entropy(target_future, np.hstack((target_past, source_past)))
        
        # Transfer entropy is the difference
        te = h_future_past - h_future_past_source
        
        return max(0, te)  # Ensure non-negative
    
    def calculate_lete(self, source, target, lag=1):
        """
        Calculate Lagged Effective Transfer Entropy from source to target.
        
        Parameters:
        -----------
        source : array-like
            Source time series
        target : array-like
            Target time series
        lag : int
            Lag period for source series
            
        Returns:
        --------
        float
            LETE value
        """
        # Original transfer entropy
        te = self.calculate_te(source, target)
        
        # Calculate randomized transfer entropy
        rte_values = []
        for _ in range(self.shuffle_iterations):
            # Shuffle source to break causality
            shuffled_source = np.random.permutation(source)
            rte = self.calculate_te(shuffled_source, target)
            rte_values.append(rte)
        
        # Average randomized TE
        rte_mean = np.mean(rte_values)
        
        # Calculate LETE (effective TE)
        lete = te - rte_mean
        
        return max(0, lete)
    
    def calculate_lete_matrix(self, data):
        """
        Calculate LETE matrix for multivariate data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Multivariate data with columns as variables
            
        Returns:
        --------
        pandas.DataFrame
            Matrix of LETE values
        """
        n_vars = data.shape[1]
        lete_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    lete_matrix[i, j] = self.calculate_lete(
                        source=data.iloc[:, i].values,
                        target=data.iloc[:, j].values
                    )
        
        return pd.DataFrame(lete_matrix, index=data.columns, columns=data.columns)
    
    def _conditional_entropy(self, x, y):
        """
        Calculate conditional entropy H(X|Y).
        
        Parameters:
        -----------
        x : array-like
            Target variable
        y : array-like
            Conditioning variable(s)
            
        Returns:
        --------
        float
            Conditional entropy value
        """
        # This is a simplified implementation
        # For production, we would use more efficient estimators
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        # Create joint state representation
        y_states = np.zeros(len(y))
        bins_per_dim = self.bins
        
        for i in range(y.shape[1]):
            y_states += y[:, i] * (bins_per_dim ** i)
            
        # Calculate joint entropy H(X,Y)
        joint_states = np.vstack((x, y_states)).T
        joint_counts = np.array(np.unique(joint_states, axis=0, return_counts=True)[1])
        joint_probs = joint_counts / len(x)
        h_joint = entropy(joint_probs, base=2)
        
        # Calculate entropy H(Y)
        y_counts = np.array(np.unique(y_states, return_counts=True)[1])
        y_probs = y_counts / len(y_states)
        h_y = entropy(y_probs, base=2)
        
        # Conditional entropy H(X|Y) = H(X,Y) - H(Y)
        return h_joint - h_y