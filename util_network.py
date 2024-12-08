import networkx as nx
import matplotlib.pyplot as plt
import statistics
import numpy as np
import sys
import time
from typing import Dict, Any, Tuple
from collections import defaultdict
import seaborn as sns
import pandas as pd
from itertools import combinations as iter_combinations


### NETWORK CREATION FUNCTIONS ###
# Extract the largest connected component of undirected graph
def extract_largest_undirected_cc(G):
    if nx.is_connected(G):
        print('\nNetwork is already one large connected component')
        return G
    else:
        print('\nNetwork is not one large connected component, extracting...')
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G_largest_cc = G.subgraph(largest_cc_nodes).copy()
        print(f"Network after extraction: {G_largest_cc.number_of_nodes()} nodes, {G_largest_cc.number_of_edges()} edges")
        return G_largest_cc
    
# Extract the largest weakly connected component of directed graph
def extract_largest_directed_weakly_cc(G):
    if nx.is_weakly_connected(G):
        print('\nNetwork is already one large weakly connected component')
        return G
    else:
        print('\nNetwork is not one large weakly connected component, extracting...')
        largest_scc = max(nx.weakly_connected_components(G), key=len)
        G_largest_scc = G.subgraph(largest_scc).copy()
        print(f"Network after extraction: {G_largest_scc.number_of_nodes()} nodes, {G_largest_scc.number_of_edges()} edges")
        return G_largest_scc



### NETWORK ANALYSIS FUNCTIONS ###

# Function to plot distribution of given values (histogram and log-log scatter plot)
def plot_undirected_hist_loglog_distribution(values, num_bins, title_distribution_type, title_network_name):
    hist, bin_edges = np.histogram(values, bins=num_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Histogram plot
    ax1.hist(values, bins=range(min(values), max(values) + 2), color='blue', edgecolor='black')
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Histogram {title_distribution_type} Distribution of {title_network_name}')
    ax1.grid(True)

    # Scatter plot with log-log scale
    ax2.scatter(bin_edges[:-1], hist, color='blue', edgecolor='black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f'Log-Log {title_distribution_type} Distribution of {title_network_name}')
    ax2.set_xlabel('Degree')
    ax2.set_ylabel('Count')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_directed_hist_loglog_distribution(values_in, values_out, num_bins, title_network_name):
    hist_in, bin_edges_in = np.histogram(values_in, bins=num_bins)
    hist_out, bin_edges_out = np.histogram(values_out, bins=num_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Combined histogram
    ax1.hist(values_in, bins=num_bins, alpha=0.5, color='blue', 
             edgecolor='black', label='In-Degree')
    ax1.hist(values_out, bins=num_bins, alpha=0.5, color='red', 
             edgecolor='black', label='Out-Degree')
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Degree Distributions of {title_network_name}')
    ax1.legend()
    ax1.grid(True)

    # Combined log-log scatter
    ax2.scatter(bin_edges_in[:-1], hist_in, color='blue', edgecolor='black', label='In-Degree')
    ax2.scatter(bin_edges_out[:-1], hist_out, color='red', edgecolor='black', label='Out-Degree')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Degree')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Log-Log Degree Distributions of {title_network_name}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def calculate_statistics(values, title):
    print(f"Statistics for {title}:")
    print(f"Mean: {np.mean(values):.2f}, Median: {int(np.median(values))}, Mode: {statistics.mode(values)}, Min: {np.min(values)}, Max: {np.max(values)}")

def print_status(message: str, same_line: bool = True) -> None:
    """Print a status message with optional line updating"""
    if same_line:
        sys.stdout.write('\r' + message)
        sys.stdout.flush()
    else:
        print(message)

def compute_hits(G: nx.DiGraph, max_iter: int = 100, tol: float = 1.0e-6) -> Tuple[Dict, Dict]:
    """
    Compute HITS (Hyperlink-Induced Topic Search) scores for a directed graph.
    Returns both hub and authority scores, which are particularly meaningful
    for directed networks.
    
    Hub score: Measures how well a node points to good authorities
    Authority score: Measures how many good hubs point to the node
    """
    print_status("Computing HITS scores...")
    try:
        return nx.hits(G, max_iter=max_iter, tol=tol)
    except nx.PowerIterationFailedConvergence:
        print_status("HITS failed to converge, using normalized scores from partial results...", False)
        # Compute simple hub/authority scores based on degree
        auth = nx.in_degree_centrality(G)
        hub = nx.out_degree_centrality(G)
        return hub, auth

def analyze_directed_network(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Comprehensive analysis of directed network metrics, specifically designed
    for directed graphs with proper handling of edge directions and path definitions.
    """
    metrics = {}
    start_time = time.time()
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    print_status(f"Analyzing directed graph with {n_nodes} nodes and {n_edges} edges...", False)
    
    # Basic network properties
    metrics['density'] = nx.density(G)
    metrics['reciprocity'] = nx.reciprocity(G)  # Proportion of mutual connections
    
    # Degree-based measures
    print_status("Computing degree-based metrics...")
    metrics['in_degree_centrality'] = nx.in_degree_centrality(G)
    metrics['out_degree_centrality'] = nx.out_degree_centrality(G)
    
    # Degree correlation patterns (assortativity)
    print_status("Computing degree correlation patterns...")
    assortativity_types = [('in', 'in'), ('in', 'out'), ('out', 'in'), ('out', 'out')]
    for x, y in assortativity_types:
        metrics[f'{x}_{y}_assortativity'] = nx.degree_assortativity_coefficient(G, x=x, y=y)
    
    # Directed closeness centrality (both incoming and outgoing)
    print_status("Computing bidirectional closeness centrality...")
    try:
        # Out-closeness: how easily this node can reach others
        metrics['out_closeness_centrality'] = nx.closeness_centrality(G)
        # In-closeness: how easily this node can be reached by others
        metrics['in_closeness_centrality'] = nx.closeness_centrality(G.reverse())
    except nx.NetworkXError as e:
        print_status(f"Error in closeness computation: {str(e)}", False)
    
    # Betweenness centrality (considers directed paths)
    print_status("Computing betweenness centrality (this may take a while)...")
    start_between = time.time()
    metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
    print_status(f"Completed betweenness centrality in {time.time() - start_between:.1f}s", False)
    
    # PageRank (designed specifically for directed graphs)
    print_status("Computing PageRank...")
    start_page = time.time()
    try:
        metrics['pagerank'] = nx.pagerank(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print_status("PageRank failed to converge, using simplified calculation...", False)
        # Fall back to a simpler calculation with more iterations
        metrics['pagerank'] = nx.pagerank(G, max_iter=2000, tol=1e-4)
    print_status(f"Completed PageRank in {time.time() - start_page:.1f}s", False)
    
    # HITS Algorithm (specifically designed for directed graphs)
    print_status("Computing HITS scores...")
    start_hits = time.time()
    metrics['hub_scores'], metrics['authority_scores'] = compute_hits(G)
    print_status(f"Completed HITS in {time.time() - start_hits:.1f}s", False)
    
    # Eigenvector centrality (with careful handling for directed graphs)
    print_status("Computing eigenvector centrality...")
    start_eigen = time.time()
    try:
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(
            G,
            max_iter=1000,
            tol=1e-4,
            weight=None
        )
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError) as e:
        print_status(f"Eigenvector centrality failed: {str(e)}", False)
        print_status("Using authority scores as substitute for eigenvector centrality", False)
        metrics['eigenvector_centrality'] = metrics['authority_scores']
    
    # Add summary statistics
    metrics['summary'] = {
        'avg_in_degree': np.mean(list(metrics['in_degree_centrality'].values())),
        'avg_out_degree': np.mean(list(metrics['out_degree_centrality'].values())),
        'density': metrics['density'],
        'reciprocity': metrics['reciprocity'],
        'execution_time': time.time() - start_time
    }
    
    print_status(f"\nAnalysis completed in {metrics['summary']['execution_time']:.1f} seconds", False)
    
    return metrics

def format_centrality_stats(metrics_dict):
    """
    Create summary statistics for centrality measures in a readable format
    """
    stats = {}
    for metric_name, values in metrics_dict.items():
        if isinstance(values, dict) and metric_name != 'summary':
            # Calculate key statistics for the metric
            values_list = list(values.values())
            stats[metric_name] = {
                'mean': np.mean(values_list),
                'max': max(values_list),
                'min': min(values_list),
                # Find node with maximum value
                'max_node': max(values.items(), key=lambda x: x[1])[0]
            }
    return stats

def print_network_analysis(metrics):
    """
    Print network analysis results in a clear, organized format
    """
    # Network Overview
    print("\n" + "="*50)
    print("NETWORK OVERVIEW")
    print("="*50)
    n_nodes = len(metrics['in_degree_centrality'])
    print(f"Number of Nodes: {n_nodes:,}")
    print(f"Network Density: {metrics['density']:.4f}")
    print(f"Reciprocity: {metrics['reciprocity']:.4f}")
    
    # Assortativity Patterns
    print("\n" + "="*50)
    print("ASSORTATIVITY PATTERNS")
    print("="*50)
    assortativity_types = ['in_in', 'in_out', 'out_in', 'out_out']
    for atype in assortativity_types:
        print(f"{atype}_assortativity: {metrics[f'{atype}_assortativity']:.4f}")
    
    # Centrality Measures
    print("\n" + "="*50)
    print("CENTRALITY MEASURES")
    print("="*50)
    
    stats = format_centrality_stats(metrics)
    
    centrality_metrics = [
        ('in_degree_centrality', 'Incoming Connections'),
        ('out_degree_centrality', 'Outgoing Connections'),
        ('betweenness_centrality', 'Path Centrality'),
        ('pagerank', 'PageRank'),
        ('hub_scores', 'Hub Score'),
        ('authority_scores', 'Authority Score'),
        ('eigenvector_centrality', 'Eigenvector Centrality'),
        ('in_closeness_centrality', 'Incoming Closeness'),
        ('out_closeness_centrality', 'Outgoing Closeness')
    ]
    
    for metric_key, metric_name in centrality_metrics:
        if metric_key in stats:
            stat = stats[metric_key]
            print(f"\n{metric_name}:")
            print(f"  Average: {stat['mean']:.4f}")
            print(f"  Maximum: {stat['max']:.4f} (Node {stat['max_node']})")
            print(f"  Minimum: {stat['min']:.4f}")
    
    # Most Central Nodes
    print("\n" + "="*50)
    print("TOP NODES BY DIFFERENT METRICS")
    print("="*50)
    
    for metric_key, metric_name in centrality_metrics:
        if metric_key in metrics:
            # Get top 5 nodes for each metric
            top_nodes = sorted(metrics[metric_key].items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:5]
            print(f"\nTop 5 nodes by {metric_name}:")
            for node, value in top_nodes:
                print(f"  Node {node}: {value:.4f}")

    # Execution Information
    if 'summary' in metrics:
        print("\n" + "="*50)
        print("EXECUTION INFORMATION")
        print("="*50)
        print(f"Total execution time: {metrics['summary']['execution_time']:.2f} seconds")


def analyze_affiliations(G, top_n=20):
    """
    Performs a comprehensive analysis of affiliation patterns, combining both 
    direct combinations and pairwise co-occurrences.
    """
    # First, collect both combinations and co-occurrences in one pass
    combinations = defaultdict(int)
    cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
    all_affiliations = set()
    
    # Single pass through the network to collect all data
    for _, data in G.nodes(data=True):
        if 'affiliation' in data and data['affiliation']:
            # Get sorted affiliations for this character
            affiliations = sorted(data['affiliation'])
            all_affiliations.update(affiliations)
            
            # Count full combinations (if more than one affiliation)
            if len(affiliations) > 1:
                combinations[tuple(affiliations)] += 1
            
            # Count all pairwise co-occurrences
            for aff1, aff2 in iter_combinations(affiliations, 2):
                cooccurrence_matrix[aff1][aff2] += 1
                cooccurrence_matrix[aff2][aff1] += 1
    
    # Convert defaultdict matrix to pandas DataFrame
    all_affiliations = sorted(all_affiliations)
    matrix = pd.DataFrame(0, index=all_affiliations, columns=all_affiliations)
    
    for aff1 in all_affiliations:
        for aff2 in all_affiliations:
            matrix.loc[aff1, aff2] = cooccurrence_matrix[aff1][aff2]
    
    # Get top affiliations by total co-occurrences
    totals = matrix.sum()
    top_affiliations = totals.nlargest(top_n).index
    matrix_subset = matrix.loc[top_affiliations, top_affiliations]
    
    # Create visualization with both matrix and top combinations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot heatmap in first subplot
    sns.heatmap(matrix_subset, 
                cmap='YlOrRd',
                square=True,
                annot=True,
                fmt='g',
                ax=ax1)
    ax1.set_title('Affiliation Co-occurrences')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Create horizontal bar plot of top combinations in second subplot
    top_combos = sorted(combinations.items(), key=lambda x: x[1], reverse=True)[:10]
    combo_names = [' & '.join(combo) for combo, _ in top_combos]
    combo_counts = [count for _, count in top_combos]
    
    sns.barplot(y=combo_names, x=combo_counts, ax=ax2)
    ax2.set_title('Top Affiliation Combinations')
    ax2.set_xlabel('Number of Characters')
    
    plt.tight_layout()
    
    # Prepare text summary of findings
    top_pairs = []
    for i in range(len(matrix_subset.index)):
        for j in range(i + 1, len(matrix_subset.columns)):
            count = matrix_subset.iloc[i, j]
            if count > 0:
                top_pairs.append((
                    matrix_subset.index[i],
                    matrix_subset.columns[j],
                    count
                ))
    
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return {
        'figure': fig,
        'cooccurrence_matrix': matrix_subset,
        'top_combinations': dict(top_combos),
        'top_pairs': top_pairs[:10]
    }

def print_analysis_results(results):
    """
    Prints a comprehensive summary of the affiliation analysis.
    """
    print("\nTop Individual Affiliation Pairs:")
    print("-" * 50)
    for aff1, aff2, count in results['top_pairs']:
        print(f"'{aff1}' & '{aff2}': {int(count)} characters")
    
    print("\nTop Complete Affiliation Combinations:")
    print("-" * 50)
    for combo, count in results['top_combinations'].items():
        print(f"{' & '.join(combo)}: {count} characters")


