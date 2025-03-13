# Part 2: Cluster Analysis

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from itertools import combinations


def read_csv_2(data_file):
    df = pd.read_csv(data_file)
    df = df.drop(['Channel', 'Region'], axis=1)
    return df

def summary_statistics(df):
    stats = pd.DataFrame()
    stats['mean'] = df.mean().round().astype(int)
    stats['std'] = df.std().round().astype(int)
    stats['min'] = df.min()
    stats['max'] = df.max()
    return stats

def standardize(df):
    return pd.DataFrame((df - df.mean()) / df.std(), columns=df.columns)

def kmeans(df, k):
    # Added n_init parameter to suppress the warning
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    y = pd.Series(km.fit_predict(df))
    return y

def kmeans_plus(df, k):
    km = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init=10)
    y = pd.Series(km.fit_predict(df))
    return y

def agglomerative(df, k):
    agg = AgglomerativeClustering(n_clusters=k)
    y = pd.Series(agg.fit_predict(df))
    return y

def clustering_score(X, y):
    return silhouette_score(X, y)

def cluster_evaluation(df):
    results = []
    k_values = [3, 5, 10]
    
    # Original data
    for k in k_values:
        # Run kmeans 10 times for each k
        for _ in range(10):
            # Create a new KMeans instance with a different random_state for each run
            km = KMeans(n_clusters=k, n_init=1, random_state=np.random.randint(1000))
            y_kmeans = pd.Series(km.fit_predict(df))
            score_kmeans = clustering_score(df, y_kmeans)
            results.append({
                'Algorithm': 'Kmeans',
                'data': 'Original',
                'k': k,
                'Silhouette Score': score_kmeans
            })
        
        y_agg = agglomerative(df, k)
        score_agg = clustering_score(df, y_agg)
        results.append({
            'Algorithm': 'Agglomerative',
            'data': 'Original',
            'k': k,
            'Silhouette Score': score_agg
        })
    
    # Standardized data
    df_std = standardize(df)
    for k in k_values:
        # Run kmeans 10 times for each k
        for _ in range(10):
            # Create a new KMeans instance with a different random_state for each run
            km = KMeans(n_clusters=k, n_init=1, random_state=np.random.randint(1000))
            y_kmeans = pd.Series(km.fit_predict(df_std))
            score_kmeans = clustering_score(df_std, y_kmeans)
            results.append({
                'Algorithm': 'Kmeans',
                'data': 'Standardized',
                'k': k,
                'Silhouette Score': score_kmeans
            })
        
        y_agg = agglomerative(df_std, k)
        score_agg = clustering_score(df_std, y_agg)
        results.append({
            'Algorithm': 'Agglomerative',
            'data': 'Standardized',
            'k': k,
            'Silhouette Score': score_agg
        })
    
    return pd.DataFrame(results)

def best_clustering_score(rdf):
    return rdf['Silhouette Score'].max()

def scatter_plots(df):
    # Standardize the data
    df_std = standardize(df)
    
    # Run KMeans with k=3
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(df_std)
    
    # Get all pairs of attributes
    attributes = df.columns
    pairs = list(combinations(attributes, 2))
    
    # Set colors for clusters
    colors = ['blue', 'red', 'green']
    
    # Create scatter plots for each pair
    for i, (attr1, attr2) in enumerate(pairs):
        plt.figure(figsize=(8, 6))
        
        # Plot points with different colors for different clusters
        for cluster_id in range(3):
            cluster_points = df_std[labels == cluster_id]
            plt.scatter(
                cluster_points[attr1], 
                cluster_points[attr2], 
                label=f'Cluster {cluster_id}',
                color=colors[cluster_id],
                alpha=0.7
            )
        
        plt.xlabel(attr1)
        plt.ylabel(attr2)
        plt.title(f'Cluster Visualization: {attr1} vs {attr2}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot - use a clear naming pattern that the test script expects
        plt.savefig(f'cluster_plot_{attr1}_{attr2}.png')
        plt.close()