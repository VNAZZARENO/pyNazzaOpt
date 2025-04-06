import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

class StrategyFactorAnalysis:
    """
    Analyze the relationship between option strategy performance and economic factors
    
    This class provides tools to:
    1. Analyze correlations between strategy returns and various economic factors
    2. Build regression models to explain strategy performance 
    3. Identify optimal conditions for different strategies
    4. Create factor-based strategy selection models
    """
    
    def __init__(self, strategy_returns, factors_df):
        """
        Initialize the factor analysis
        
        Parameters:
        -----------
        strategy_returns : dict or pd.Series
            Dictionary mapping strategy names to return series, or a single return series
        factors_df : pd.DataFrame
            DataFrame containing economic factor data, indexed by date
        """
        # Process strategy returns
        if isinstance(strategy_returns, dict):
            self.strategy_returns = strategy_returns
            self.multi_strategy = True
        else:
            self.strategy_returns = {'Strategy': strategy_returns}
            self.multi_strategy = False
            
        # Store factor data
        self.factors_df = factors_df
        
        # Create aligned dataset for analysis
        self.data = self._align_data()
        
        # Dictionary to store regression models
        self.models = {}
        
    def _align_data(self):
        """Align strategy returns with factor data on common dates"""
        # Create empty DataFrame to store aligned data
        aligned_data = pd.DataFrame(index=self.factors_df.index)
        
        # Add strategy returns
        for name, returns in self.strategy_returns.items():
            # Convert to returns if needed
            if not isinstance(returns, pd.Series):
                returns = pd.Series(returns)
                
            # Calculate returns if we have values instead
            if returns.min() > 0.1:  # Likely portfolio values not returns
                returns = returns.pct_change()
                
            # Add to aligned data on common dates
            for date in aligned_data.index:
                if date in returns.index:
                    aligned_data.loc[date, f'{name}_return'] = returns.loc[date]
        
        # Add factor data
        for column in self.factors_df.columns:
            aligned_data[column] = self.factors_df[column]
            
        # Drop rows with missing values
        return aligned_data.dropna()
    
    def calculate_correlations(self):
        """
        Calculate correlations between strategy returns and economic factors
        
        Returns:
        --------
        pd.DataFrame
            Correlation matrix between strategy returns and factors
        """
        if self.data.empty:
            print("No aligned data available for correlation analysis")
            return pd.DataFrame()
        
        return self.data.corr()
    
    def plot_factor_correlations(self, top_n=10):
        """
        Plot correlation of factors with strategy returns
        
        Parameters:
        -----------
        top_n : int
            Number of top factors to display (by absolute correlation)
        """
        if self.data.empty:
            print("No data available for correlation analysis")
            return
        
        correlations = self.calculate_correlations()
        
        # For multi-strategy, create subplots
        if self.multi_strategy:
            strategies = [col for col in self.data.columns if '_return' in col]
            n_strategies = len(strategies)
            
            fig, axes = plt.subplots(n_strategies, 1, figsize=(12, 5 * n_strategies))
            if n_strategies == 1:
                axes = [axes]
                
            for i, strategy in enumerate(strategies):
                # Get correlations for this strategy
                strat_corr = correlations[strategy].drop(strategies)
                
                # Sort by absolute correlation and take top N
                top_factors = strat_corr.abs().sort_values(ascending=False).head(top_n).index
                plot_corr = strat_corr.loc[top_factors].sort_values()
                
                # Plot
                ax = axes[i]
                bars = ax.barh(plot_corr.index, plot_corr.values)
                
                # Color bars
                for j, bar in enumerate(bars):
                    bar.set_color('green' if plot_corr.values[j] > 0 else 'red')
                
                ax.set_title(f"Top Factors Correlated with {strategy.replace('_return', '')}")
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax.set_xlim(-1, 1)
                ax.grid(axis='x', alpha=0.3)
        else:
            # Single strategy case
            strategy = [col for col in self.data.columns if '_return' in col][0]
            strat_corr = correlations[strategy].drop([strategy])
            
            # Sort by absolute correlation and take top N
            top_factors = strat_corr.abs().sort_values(ascending=False).head(top_n).index
            plot_corr = strat_corr.loc[top_factors].sort_values()
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(plot_corr.index, plot_corr.values)
            
            # Color bars
            for j, bar in enumerate(bars):
                bar.set_color('green' if plot_corr.values[j] > 0 else 'red')
            
            ax.set_title(f"Top Factors Correlated with Strategy Returns")
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlim(-1, 1)
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def build_factor_model(self, strategy=None, factors=None, vif_threshold=10):
        """
        Build regression model to explain strategy returns using economic factors
        
        Parameters:
        -----------
        strategy : str, optional
            Strategy name if multiple strategies are present
        factors : list, optional
            List of factors to include in the model (default: all factors)
        vif_threshold : float
            Threshold for Variance Inflation Factor to detect multicollinearity
        
        Returns:
        --------
        statsmodels.regression.linear_model.RegressionResultsWrapper
            Regression model results
        """
        if self.data.empty:
            print("No data available for regression analysis")
            return None
        
        # Determine which strategy to model
        if strategy is None:
            strategy_cols = [col for col in self.data.columns if '_return' in col]
            if len(strategy_cols) == 0:
                print("No strategy return data found")
                return None
            strategy = strategy_cols[0]
        elif f"{strategy}_return" in self.data.columns:
            strategy = f"{strategy}_return"
        else:
            print(f"Strategy '{strategy}' not found in data")
            return None
        
        # Determine factors to include
        if factors is None:
            factors = [col for col in self.data.columns if '_return' not in col]
        else:
            # Validate factor names
            valid_factors = [f for f in factors if f in self.data.columns]
            if len(valid_factors) < len(factors):
                print(f"Some factors not found: {set(factors) - set(valid_factors)}")
            factors = valid_factors
        
        if not factors:
            print("No valid factors for regression")
            return None
        
        # Check for multicollinearity using VIF
        X = self.data[factors]
        X = sm.add_constant(X)
        
        # Calculate VIF for each factor
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        # Remove factors with high VIF (multicollinearity)
        high_vif = vif_data[vif_data["VIF"] > vif_threshold]["Variable"].tolist()
        if 'const' in high_vif:
            high_vif.remove('const')
            
        if high_vif:
            print(f"Removing factors with high multicollinearity: {high_vif}")
            factors = [f for f in factors if f not in high_vif]
            X = self.data[factors]
            X = sm.add_constant(X)
        
        # Build regression model
        y = self.data[strategy]
        model = sm.OLS(y, X).fit()
        
        # Store model
        self.models[strategy] = model
        
        # Print summary
        print(model.summary())
        
        return model
    
    def plot_model_coefficients(self, strategy=None):
        """
        Plot coefficients from regression model
        
        Parameters:
        -----------
        strategy : str, optional
            Strategy name to plot coefficients for
        """
        if not self.models:
            print("No models built yet. Use build_factor_model() first.")
            return
        
        if strategy is None:
            # Use first model
            strategy = list(self.models.keys())[0]
        
        if strategy not in self.models:
            print(f"No model found for '{strategy}'. Build a model first.")
            return
        
        model = self.models[strategy]
        
        # Extract coefficients (excluding constant)
        coefs = model.params.drop('const') if 'const' in model.params.index else model.params
        errors = model.bse.drop('const') if 'const' in model.bse.index else model.bse
        
        # Sort by absolute value
        sorted_idx = coefs.abs().sort_values().index
        coefs = coefs.loc[sorted_idx]
        errors = errors.loc[sorted_idx]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot coefficients
        bars = ax.barh(coefs.index, coefs.values, height=0.6, xerr=errors, 
                      error_kw={'ecolor': 'black', 'capsize': 5, 'elinewidth': 1})
        
        # Color bars
        for i, bar in enumerate(bars):
            bar.set_color('green' if coefs.values[i] > 0 else 'red')
        
        ax.set_title(f"Factor Model Coefficients: {strategy.replace('_return', '')}")
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_rolling_factor_impact(self, strategy=None, factors=None, window=30):
        """
        Plot rolling regression coefficients to show how factor impact changes over time
        
        Parameters:
        -----------
        strategy : str, optional
            Strategy name if multiple strategies are present
        factors : list, optional
            List of factors to include (limit to 5-6 for readability)
        window : int
            Rolling window size in days
        """
        if self.data.empty:
            print("No data available for analysis")
            return
        
        # Determine which strategy to model
        if strategy is None:
            strategy_cols = [col for col in self.data.columns if '_return' in col]
            if len(strategy_cols) == 0:
                print("No strategy return data found")
                return
            strategy = strategy_cols[0]
        elif f"{strategy}_return" in self.data.columns:
            strategy = f"{strategy}_return"
        else:
            print(f"Strategy '{strategy}' not found in data")
            return
        
        # Determine factors to include
        if factors is None:
            # Use factors from existing model if available
            if strategy in self.models:
                factors = [col for col in self.models[strategy].params.index 
                          if col != 'const' and col in self.data.columns]
                if len(factors) > 8:  # Limit for readability
                    factors = factors[:8]
            else:
                # Use top correlated factors
                correlations = self.calculate_correlations()[strategy].drop([col for col in self.data.columns if '_return' in col])
                factors = correlations.abs().sort_values(ascending=False).head(8).index.tolist()
        
        # Calculate rolling regressions
        rolling_coefs = {}
        dates = self.data.index[window-1:]
        
        for i in range(window, len(self.data) + 1):
            window_data = self.data.iloc[i-window:i]
            X = window_data[factors]
            X = sm.add_constant(X)
            y = window_data[strategy]
            
            try:
                model = sm.OLS(y, X).fit()
                for factor in factors:
                    if factor not in rolling_coefs:
                        rolling_coefs[factor] = []
                    rolling_coefs[factor].append(model.params[factor])
            except:
                # Skip if regression fails for this window
                for factor in factors:
                    if factor not in rolling_coefs:
                        rolling_coefs[factor] = []
                    rolling_coefs[factor].append(np.nan)
        
        # Convert to DataFrame
        rolling_df = pd.DataFrame(rolling_coefs, index=dates)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for factor in factors:
            rolling_df[factor].plot(ax=ax, label=factor)
        
        ax.set_title(f"{window}-Day Rolling Factor Coefficients: {strategy.replace('_return', '')}")
        ax.set_ylabel("Coefficient Value")
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def identify_optimal_conditions(self, strategy=None, percentile=75):
        """
        Identify optimal economic conditions for a strategy
        
        Parameters:
        -----------
        strategy : str, optional
            Strategy name if multiple strategies are present
        percentile : int
            Percentile of returns to consider as "good performance" (e.g., 75 = top 25%)
        
        Returns:
        --------
        pd.DataFrame
            Summary of factor values during good and bad performance periods
        """
        if self.data.empty:
            print("No data available for analysis")
            return None
        
        # Determine which strategy to analyze
        if strategy is None:
            strategy_cols = [col for col in self.data.columns if '_return' in col]
            if len(strategy_cols) == 0:
                print("No strategy return data found")
                return None
            strategy = strategy_cols[0]
        elif f"{strategy}_return" in self.data.columns:
            strategy = f"{strategy}_return"
        else:
            print(f"Strategy '{strategy}' not found in data")
            return None
        
        # Identify good and bad performance periods
        returns = self.data[strategy]
        good_threshold = returns.quantile(percentile/100)
        
        good_periods = self.data[returns >= good_threshold]
        bad_periods = self.data[returns < good_threshold]
        
        # Factors to analyze
        factors = [col for col in self.data.columns if '_return' not in col]
        
        # Calculate statistics
        summary = pd.DataFrame(index=factors)
        
        # Good periods
        summary['good_mean'] = good_periods[factors].mean()
        summary['good_median'] = good_periods[factors].median()
        summary['good_std'] = good_periods[factors].std()
        
        # Bad periods
        summary['bad_mean'] = bad_periods[factors].mean()
        summary['bad_median'] = bad_periods[factors].median()
        summary['bad_std'] = bad_periods[factors].std()
        
        # Ratio (good / bad)
        summary['mean_ratio'] = summary['good_mean'] / summary['bad_mean']
        
        # T-test for significance
        from scipy import stats
        p_values = []
        for factor in factors:
            t, p = stats.ttest_ind(good_periods[factor].dropna(), 
                                  bad_periods[factor].dropna(), 
                                  equal_var=False)
            p_values.append(p)
        
        summary['p_value'] = p_values
        summary['significant'] = summary['p_value'] < 0.05
        
        # Sort by significance and then by ratio
        summary = summary.sort_values(['significant', 'p_value'], ascending=[False, True])
        
        return summary
    
    def visualize_optimal_conditions(self, strategy=None, top_n=10):
        """
        Visualize the differences in factor values between good and bad periods
        
        Parameters:
        -----------
        strategy : str, optional
            Strategy name if multiple strategies are present
        top_n : int
            Number of top factors to display
        """
        summary = self.identify_optimal_conditions(strategy)
        
        if summary is None or summary.empty:
            return
        
        # Filter for significant factors and take top N by smallest p-value
        sig_factors = summary[summary['significant']].head(top_n).index.tolist()
        
        if not sig_factors:
            print("No significant factors found")
            sig_factors = summary.head(top_n).index.tolist()
        
        # Prepare data for plot
        plot_data = []
        for factor in sig_factors:
            plot_data.append({
                'Factor': factor,
                'Good': summary.loc[factor, 'good_mean'],
                'Bad': summary.loc[factor, 'bad_mean'],
                'Ratio': summary.loc[factor, 'mean_ratio'],
                'p-value': summary.loc[factor, 'p_value']
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot 1: Absolute values
        plot_df = plot_df.sort_values('Factor')  # Sort alphabetically for easier comparison
        
        x = np.arange(len(plot_df))
        width = 0.35
        
        ax1.bar(x - width/2, plot_df['Good'], width, label='Good Periods')
        ax1.bar(x + width/2, plot_df['Bad'], width, label='Bad Periods')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(plot_df['Factor'], rotation=45, ha='right')
        ax1.set_title('Factor Values in Good vs Bad Periods')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Ratio of good/bad
        plot_df = plot_df.sort_values('Ratio')
        
        bars = ax2.barh(plot_df['Factor'], plot_df['Ratio'])
        
        # Color bars by significance
        for i, (bar, p_value) in enumerate(zip(bars, plot_df['p-value'])):
            if p_value < 0.01:
                bar.set_color('darkgreen' if plot_df['Ratio'].iloc[i] > 1 else 'darkred')
            elif p_value < 0.05:
                bar.set_color('green' if plot_df['Ratio'].iloc[i] > 1 else 'red')
            else:
                bar.set_color('lightgreen' if plot_df['Ratio'].iloc[i] > 1 else 'lightcoral')
        
        ax2.axvline(x=1, color='black', linestyle='--')
        ax2.set_title('Ratio of Factor Values (Good/Bad)')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def build_strategy_selection_model(self, threshold_percentile=75):
        """
        Build a decision model for selecting between strategies based on economic factors
        
        Parameters:
        -----------
        threshold_percentile : int
            Percentile threshold to classify "good" performance
        
        Returns:
        --------
        dict
            Dictionary containing selection models for each strategy
        """
        if not self.multi_strategy:
            print("Need multiple strategies for selection model")
            return None
        
        strategy_cols = [col for col in self.data.columns if '_return' in col]
        
        if len(strategy_cols) < 2:
            print("Need at least two strategies for selection model")
            return None
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        
        models = {}
        features = [col for col in self.data.columns if '_return' not in col]
        
        for strategy in strategy_cols:
            print(f"\nBuilding selection model for {strategy}...")
            
            # Classify returns as good (1) or bad (0)
            threshold = self.data[strategy].quantile(threshold_percentile/100)
            self.data[f'{strategy}_good'] = (self.data[strategy] >= threshold).astype(int)
            
            # Split data (use simple time-based split)
            split_idx = int(len(self.data) * 0.7)
            train = self.data.iloc[:split_idx]
            test = self.data.iloc[split_idx:]
            
            # Train model
            X_train = train[features]
            y_train = train[f'{strategy}_good']
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            X_test = test[features]
            y_test = test[f'{strategy}_good']
            
            predictions = model.predict(X_test)
            
            print(classification_report(y_test, predictions))
            
            # Save model
            models[strategy] = {
                'model': model,
                'feature_importance': pd.Series(
                    model.feature_importances_, index=features
                ).sort_values(ascending=False)
            }
            
            # Clean up
            del self.data[f'{strategy}_good']
        
        return models
    
    def plot_strategy_selection_factors(self, models):
        """
        Plot feature importance for strategy selection models
        
        Parameters:
        -----------
        models : dict
            Dictionary of models from build_strategy_selection_model
        """
        if not models:
            print("No models provided")
            return
        
        n_models = len(models)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 5 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (strategy, model_data) in enumerate(models.items()):
            # Plot top 10 factors
            importance = model_data['feature_importance'].head(10)
            importance.sort_values().plot(kind='barh', ax=axes[i])
            
            axes[i].set_title(f"Top Factors for {strategy.replace('_return', '')}")
            axes[i].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def suggest_current_strategy(self, models, current_factors):
        """
        Suggest which strategy to use based on current economic factors
        
        Parameters:
        -----------
        models : dict
            Dictionary of models from build_strategy_selection_model
        current_factors : dict
            Dictionary of current factor values
        
        Returns:
        --------
        str
            Name of recommended strategy
        """
        if not models:
            print("No models provided")
            return None
        
        # Prepare data
        features = list(models[list(models.keys())[0]]['feature_importance'].index)
        missing = set(features) - set(current_factors.keys())
        
        if missing:
            print(f"Missing factors: {missing}")
            return None
        
        # Create feature vector
        X = pd.DataFrame([{f: current_factors[f] for f in features}])
        
        # Score all strategies
        scores = {}
        for strategy, model_data in models.items():
            # Predict probability of good performance
            proba = model_data['model'].predict_proba(X)[0][1]
            scores[strategy.replace('_return', '')] = proba
        
        # Find best strategy
        best_strategy = max(scores.items(), key=lambda x: x[1])
        
        # Print all scores
        print("Strategy probabilities of outperformance:")
        for strategy, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{strategy}: {score:.2%}")
        
        return best_strategy[0]