import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional

def calculate_safety_score(employee: dict, situation_seed: dict) -> float:
    """Re-calculate safety score to visualize the public/private split logic.
    Matches the updated logic in forward_forecaster.py with continuous buffers."""
    sanction_strength = situation_seed.get('sanction_strength', 0.0)
    visibility = situation_seed.get('visibility', 'private')
    performance = employee.get('performance', 0.6)
    tenure = employee.get('tenure', 0) / 10.0  # Normalize to 0-1
    level = employee.get('level', 'Manager')
    
    is_public = 1.0 if visibility == 'public' else 0.0
    
    # Base safety decreases with sanctions and visibility
    base_safety = 1.0 - (sanction_strength * 0.6 + is_public * 0.4)
    
    # Continuous performance buffer (instead of binary)
    performance_buffer = (performance - 0.5) * 0.4
    
    # Tenure buffer
    tenure_buffer = tenure * 0.15
    
    # Level buffer
    level_buffer = {'C-Suite': 0.15, 'Director': 0.08, 'Manager': 0.0}.get(level, 0.0)
    
    # Combine all buffers
    safety_score = base_safety + performance_buffer + tenure_buffer + level_buffer
    return np.clip(safety_score, 0.0, 1.0)

def analyze_public_private_divergence(hidden_state: dict, forecast: dict):
    """
    Analyze and visualize the gap between private sentiment (implied by safety) 
    and public forecast results.
    """
    employees = hidden_state.get('employees', [])
    situation_seed = hidden_state.get('situation_seed', {})
    individual_sentiments = forecast.get('individual_sentiments', [])
    
    data = []
    for i, sentiment in enumerate(individual_sentiments):
        emp = employees[i]
        probs = sentiment.get('probabilities', {})
        
        # Calculate Safety
        safety = calculate_safety_score(emp, situation_seed)
        
        # Get Final Public Sentiment (Support Probability)
        public_support = probs.get('support', 0.0)
        public_neutral = probs.get('neutral', 0.0)
        public_oppose = probs.get('oppose', 0.0)
        
        # We don't have the raw "baseline" (private) score saved in the forecast JSON,
        # but we can infer "Compliance" if Safety is low and Public Support/Neutral is high 
        # for someone who typically might oppose (e.g., high tenure/conflict).
        # For visualization, we'll plot Safety vs. "Non-Opposition" (Neutral + Support).
        
        data.append({
            'Employee ID': emp['employee_id'],
            'Department': emp.get('department', 'Unknown'),
            'Level': emp.get('level', 'Unknown'),
            'Safety Score': safety,
            'Public Support': public_support,
            'Public Neutral': public_neutral,
            'Public Oppose': public_oppose,
            'Non-Opposition (Compliance)': public_support + public_neutral,
            'Performance': emp.get('performance', 0.0)
        })
    
    df = pd.DataFrame(data)
    
    # Print diagnostic info
    print(f"\nDiagnostic Info:")
    print(f"  Sanction Strength: {situation_seed.get('sanction_strength', 0.0):.3f}")
    print(f"  Visibility: {situation_seed.get('visibility', 'unknown')}")
    print(f"  Safety Score Range: {df['Safety Score'].min():.3f} to {df['Safety Score'].max():.3f}")
    print(f"  Employees with Safety < 0.5 (gradual suppression): {(df['Safety Score'] < 0.5).sum()}/{len(df)}")
    print(f"  Safety Score Std Dev: {df['Safety Score'].std():.3f} (higher = more variance)")
    
    # 1. Safety Score vs. Opposition
    # Hypothesis: Low Safety should correlate with Low Opposition (Silence/Compliance)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Safety vs Opposition
    sns.scatterplot(data=df, x='Safety Score', y='Public Oppose', 
                    hue='Department', size='Performance', sizes=(50, 300), palette='viridis', alpha=0.8, ax=ax1)
    
    # Add threshold line and Zone of Silence (gradual suppression starts at 0.5)
    ax1.axvline(x=0.5, color='r', linestyle='--', label='Suppression Threshold (0.5)')
    ax1.axvspan(0, 0.5, alpha=0.1, color='red', label='Zone of Gradual Suppression')
    
    ax1.set_title("Impact of Psychological Safety on Public Opposition")
    ax1.set_xlabel("Safety Score\n(Low = Dangerous to Speak Up, High = Safe)")
    ax1.set_ylabel("Public Probability of Opposition")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Safety vs Compliance (Non-Opposition)
    sns.scatterplot(data=df, x='Safety Score', y='Non-Opposition (Compliance)', 
                    hue='Department', size='Performance', sizes=(50, 300), palette='viridis', alpha=0.8, ax=ax2)
    
    ax2.axvline(x=0.5, color='r', linestyle='--', label='Suppression Threshold (0.5)')
    ax2.axvspan(0, 0.5, alpha=0.1, color='red', label='Zone of Gradual Suppression')
    
    ax2.set_title("Safety vs. Public Compliance (Silence Effect)")
    ax2.set_xlabel("Safety Score")
    ax2.set_ylabel("Non-Opposition Probability\n(Neutral + Support = Compliance)")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df

def plot_influence_network(hidden_state: dict, forecast: dict):
    """
    Visualize the influence graph as a hierarchical tree organized by organizational level.
    Top-down: C-Suite -> Directors -> Managers
    """
    graphs = hidden_state.get('graphs', {})
    reports_to = np.array(graphs.get('reports_to', []))
    employees = hidden_state.get('employees', [])
    
    # Create graph
    G = nx.DiGraph()
    N = hidden_state.get('num_employees', 0)
    
    # Add nodes with sentiment color and level info
    individual_sentiments = forecast.get('individual_sentiments', [])
    node_colors = []
    node_levels = {}
    
    for i in range(N):
        emp = employees[i] if i < len(employees) else {}
        level = emp.get('level', 'Manager')
        node_levels[i] = level
        
        probs = individual_sentiments[i].get('probabilities', {})
        # Color based on dominant sentiment
        if probs['support'] > 0.5:
            color = '#2ecc71' # Green
        elif probs['oppose'] > 0.5:
            color = '#e74c3c' # Red
        else:
            color = '#95a5a6' # Grey
        
        G.add_node(i, color=color, level=level)
        node_colors.append(color)
    
    # Add edges (Reports To hierarchy) with sentiment flow coloring
    # Color edge by source node's sentiment to show "influence flow"
    edge_colors = []
    for i in range(N):
        for j in range(N):
            if reports_to[i, j] == 1: # i manages j
                G.add_edge(i, j)
                edge_colors.append(node_colors[i]) # Source color
    
    # Create hierarchical layout: organize by level
    # Level positions: C-Suite at top (y=2), Directors (y=1), Managers (y=0)
    level_y_positions = {'C-Suite': 2.0, 'Director': 1.0, 'Manager': 0.0}
    
    # Group nodes by level
    nodes_by_level = {'C-Suite': [], 'Director': [], 'Manager': []}
    for i in range(N):
        level = node_levels.get(i, 'Manager')
        if level in nodes_by_level:
            nodes_by_level[level].append(i)
    
    # Position nodes: distribute horizontally within each level
    pos = {}
    for level, node_list in nodes_by_level.items():
        if not node_list:
            continue
        y = level_y_positions.get(level, 0.0)
        # Distribute nodes evenly across width
        width = max(1, len(node_list))
        for idx, node_id in enumerate(node_list):
            x = (idx - (width - 1) / 2) * 2.0  # Center nodes, spacing of 2.0
            pos[node_id] = (x, y)
    
    # Draw the graph
    plt.figure(figsize=(14, 10))
    
    # Draw edges first (so nodes appear on top)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, alpha=0.5, 
                           width=1.5, arrowsize=20, arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9, 
                          edgecolors='black', linewidths=2)
    
    # Draw labels with employee info
    labels = {}
    for i in range(N):
        emp = employees[i] if i < len(employees) else {}
        dept = emp.get('department', '?')[:3]  # Abbreviate department
        labels[i] = f"{i}\n{dept}"
    
    nx.draw_networkx_labels(G, pos, labels, font_color='white', font_weight='bold', font_size=8)
    
    # Add level labels on the left
    for level, y_pos in level_y_positions.items():
        if nodes_by_level.get(level):
            plt.text(-8, y_pos, level, fontsize=12, fontweight='bold', 
                    verticalalignment='center', horizontalalignment='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Support'),
        Patch(facecolor='#95a5a6', label='Neutral'),
        Patch(facecolor='#e74c3c', label='Oppose')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Organizational Sentiment Network (Hierarchical by Level)\n" +
              "Green=Support, Red=Oppose, Grey=Neutral | Edges colored by source sentiment",
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
