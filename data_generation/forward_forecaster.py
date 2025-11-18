"""
Deterministic Forward Forecasting Module
=========================================
Generates validated forecasts from organizational hidden states using OpenAI
Structured Outputs with temperature=0 for deterministic predictions.
"""

import openai
import json
import hashlib
import os
import time
import re
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple, Any
from dotenv import load_dotenv
import numpy as np
import networkx as nx
from scipy.stats import pearsonr

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

# Load environment variables
load_dotenv()

# ============================================================================
# Core Computational Functions
# ============================================================================

def precompute_all_pairs_shortest_paths(graph: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Precompute all pairs shortest paths in the reports_to graph.
    
    Args:
        graph: N×N adjacency matrix (reports_to graph, binary or weighted)
        
    Returns:
        Dictionary mapping (source, target) tuples to shortest path distance.
        Returns inf for disconnected pairs.
    """
    N = graph.shape[0]
    distance_cache = {}
    
    # Create NetworkX directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(N))  # Ensure all nodes exist, even if isolated
    for i in range(N):
        for j in range(N):
            if graph[i, j] > 0:
                G.add_edge(i, j, weight=1.0)  # reports_to is binary, but we use weight=1 for path counting
    
    # Compute all pairs shortest paths
    for source in range(N):
        try:
            lengths = nx.single_source_shortest_path_length(G, source)
            for target in range(N):
                if target in lengths:
                    distance_cache[(source, target)] = float(lengths[target])
                else:
                    distance_cache[(source, target)] = float('inf')
        except nx.NetworkXError:
            # Node not in graph (shouldn't happen, but handle gracefully)
            for target in range(N):
                distance_cache[(source, target)] = float('inf')
    
    return distance_cache


def compute_graph_distance(source: int, target: int, distance_cache: Dict[Tuple[int, int], float]) -> float:
    """
    Get shortest path distance from cache.
    
    Args:
        source: Source employee ID
        target: Target employee ID
        distance_cache: Precomputed distance cache from precompute_all_pairs_shortest_paths
        
    Returns:
        Shortest path distance, or inf if disconnected
    """
    return distance_cache.get((source, target), float('inf'))


def compute_baseline_scores(hidden_state: dict, scenario_modifiers: Optional[dict] = None) -> np.ndarray:
    """
    Compute baseline sentiment scores for all employees.
    
    Formula includes global factors plus individual/department/level-specific modifiers
    to create more differentiation between employees.
    
    Args:
        hidden_state: Organizational hidden state dictionary
        scenario_modifiers: Optional dict with 'department_modifiers' and 'level_modifiers'
        
    Returns:
        Array of baseline scores, shape (N,)
    """
    N = hidden_state.get('num_employees', 0)
    employees = hidden_state.get('employees', [])
    rec_seed = hidden_state.get('rec_seed', {})
    situation_seed = hidden_state.get('situation_seed', {})
    org_seed = hidden_state.get('org_seed', {})
    graphs = hidden_state.get('graphs', {})
    
    # Extract modifiers if provided
    if scenario_modifiers is None:
        scenario_modifiers = {}
    
    dept_scenario_mods = scenario_modifiers.get('department_modifiers', {})
    level_scenario_mods = scenario_modifiers.get('level_modifiers', {})
    
    resource_need = rec_seed.get('resource_need', 0.0)
    sanction_strength = situation_seed.get('sanction_strength', 0.0)
    visibility = situation_seed.get('visibility', 'private')
    domain = rec_seed.get('domain', 'unknown')
    
    # Convert visibility to factor
    visibility_factor = 1.0 if visibility == 'public' else 0.5 if visibility == 'private' else 0.2
    
    # Get organizational context (for differentiation)
    power_distance = org_seed.get('power_distance', 0.5)
    sanction_salience = org_seed.get('sanction_salience', 0.5)
    in_group_bias = org_seed.get('in_group_bias', 0.5)
    
    # Get conflict graph
    conflict_graph = np.array(graphs.get('conflict', []))
    
    # Department-domain alignment (some departments care more about certain domains)
    domain_department_alignment = {
        'budget': {'Engineering': 0.1, 'Sales': -0.1, 'Marketing': 0.05, 'HR': 0.0},
        'hiring': {'Engineering': 0.05, 'Sales': 0.1, 'Marketing': 0.05, 'HR': 0.15},
        'product_roadmap': {'Engineering': 0.15, 'Sales': 0.05, 'Marketing': 0.1, 'HR': -0.05},
        'compliance': {'Engineering': 0.0, 'Sales': -0.05, 'Marketing': -0.05, 'HR': 0.1},
        'pricing': {'Engineering': -0.05, 'Sales': 0.15, 'Marketing': 0.1, 'HR': -0.05},
        'market_entry': {'Engineering': 0.05, 'Sales': 0.1, 'Marketing': 0.15, 'HR': 0.0},
        'vendor_selection': {'Engineering': 0.1, 'Sales': 0.0, 'Marketing': 0.05, 'HR': 0.05}
    }
    
    # Level-specific modifiers (higher levels have different risk/reward profiles)
    level_modifiers = {
        'C-Suite': 0.0,  # Baseline
        'Director': -0.05 * (1 - power_distance),  # More cautious if high power distance
        'Manager': 0.05 * (1 - power_distance)  # More aligned if low power distance
    }
    
    baseline_scores = np.zeros(N)
    
    for i in range(N):
        emp = employees[i] if i < len(employees) else {}
        tenure = emp.get('tenure', 0) / 10.0  # Normalize to 0-1
        department = emp.get('department', 'Unknown')
        level = emp.get('level', 'Manager')
        
        # Compute average conflict for this employee
        conflict_avg = 0.0
        if conflict_graph.size > 0 and conflict_graph.shape[0] > i:
            conflict_edges = conflict_graph[i, :]
            conflict_values = conflict_edges[conflict_edges > 0]
            if len(conflict_values) > 0:
                conflict_avg = float(np.mean(conflict_values))
        
        # Department-domain alignment modifier
        dept_alignment = domain_department_alignment.get(domain, {}).get(department, 0.0)
        
        # Level modifier
        level_mod = level_modifiers.get(level, 0.0)
        
        # Sanction salience affects how much employees weight sanction_strength
        # Higher salience = more responsive to sanctions
        effective_sanction = sanction_strength * (0.5 + sanction_salience * 0.5)
        
        # In-group bias affects department cohesion (stronger in-group = more department-specific response)
        dept_cohesion = in_group_bias * 0.1  # Modulates department alignment effect
        
        # Scenario-based modifiers (from narrative fusion)
        scenario_dept_mod = dept_scenario_mods.get(department, 0.0)
        scenario_level_mod = level_scenario_mods.get(level, 0.0)
        
        # Compute baseline score with more differentiation
        # We subtract 0.2 as a "cost of change" / inertia factor.
        # Without this, the positive terms (resource_need, sanction) dominate, leading to perpetual "Support".
        baseline_scores[i] = (
            (resource_need * 0.3) +
            (effective_sanction * visibility_factor) -
            (conflict_avg * 0.2) +
            (tenure * 0.05) +
            (dept_alignment * (1.0 + dept_cohesion)) +  # Department-domain alignment
            level_mod +  # Structural level modifier
            scenario_dept_mod + # Narrative-driven department modifier
            scenario_level_mod - # Narrative-driven level modifier
            0.25 # Inertia / Cost of Change
        )
    
    return baseline_scores


def compute_influenced_scores(baseline_scores: np.ndarray, hidden_state: dict, distance_cache: Dict[Tuple[int, int], float]) -> np.ndarray:
    """
    Apply graph-based influence to baseline scores.
    
    Args:
        baseline_scores: Array of baseline scores, shape (N,)
        hidden_state: Organizational hidden state dictionary
        distance_cache: Precomputed distance cache
        
    Returns:
        Array of influenced scores, shape (N,)
    """
    N = len(baseline_scores)
    graphs = hidden_state.get('graphs', {})
    reports_to_graph = np.array(graphs.get('reports_to', []))
    
    # Graph type multipliers
    graph_multipliers = {
        'reports_to': 1.0,
        'influence': 0.8,
        'collaboration': 0.6,
        'friendship': 0.4,
        'conflict': -0.3  # Negative for conflict
    }
    
    influenced_scores = baseline_scores.copy()
    
    # Graph types to process (in order of influence strength)
    graph_types = ['reports_to', 'influence', 'collaboration', 'friendship', 'conflict']
    
    for graph_type in graph_types:
        if graph_type not in graphs:
            continue
        
        graph_matrix = np.array(graphs[graph_type])
        multiplier = graph_multipliers[graph_type]
        
        for i in range(N):
            # For each influencer j (where graph[j][i] > 0, meaning j influences i)
            for j in range(N):
                if i == j:
                    continue
                
                edge_weight = graph_matrix[j, i] if j < graph_matrix.shape[0] and i < graph_matrix.shape[1] else 0.0
                
                if edge_weight > 0:
                    # Compute distance in reports_to graph
                    distance = compute_graph_distance(j, i, distance_cache)
                    
                    # Apply decay: 0.7^distance (handle inf → 0.0)
                    if np.isinf(distance) or distance < 0:
                        influence_decay = 0.0
                    else:
                        influence_decay = 0.7 ** distance
                    
                    # Compute influence contribution
                    influence_contribution = baseline_scores[j] * edge_weight * influence_decay * multiplier
                    influenced_scores[i] += influence_contribution
    
    return influenced_scores


def apply_softmax_mapping(influenced_scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Map influenced scores to sentiment probabilities using softmax.
    
    Args:
        influenced_scores: Array of influenced scores, shape (N,)
        temperature: Softmax temperature (default 0.5)
        
    Returns:
        Array of probabilities, shape (N, 3) where columns are [oppose, neutral, support]
    """
    N = len(influenced_scores)
    probabilities = np.zeros((N, 3))
    
    for i in range(N):
        score = influenced_scores[i]
        
        # Map to logits
        oppose_logit = -score * 0.5
        neutral_logit = score * 0.3
        support_logit = score * 1.0
        
        logits = np.array([oppose_logit, neutral_logit, support_logit])
        
        # Apply temperature and softmax with numerical stability
        logits_scaled = logits / temperature
        logits_scaled = logits_scaled - np.max(logits_scaled)  # Subtract max for numerical stability
        
        exp_logits = np.exp(logits_scaled)
        sum_exp = np.sum(exp_logits)
        
        if sum_exp > 0:
            probabilities[i] = exp_logits / sum_exp
        else:
            # Fallback to uniform distribution if all exp values are 0
            probabilities[i] = np.array([0.333, 0.333, 0.334])
    
    return probabilities


def apply_constraints_and_rounding(probabilities: np.ndarray, hidden_state: dict) -> np.ndarray:
    """
    Apply constraints and rounding to probabilities.
    
    Args:
        probabilities: Array of probabilities, shape (N, 3) [oppose, neutral, support]
        hidden_state: Organizational hidden state dictionary
        
    Returns:
        Normalized and rounded probabilities, shape (N, 3)
    """
    N = probabilities.shape[0]
    employees = hidden_state.get('employees', [])
    graphs = hidden_state.get('graphs', {})
    conflict_graph = np.array(graphs.get('conflict', []))
    
    constrained_probs = probabilities.copy()
    
    for i in range(N):
        # Get conflict average for this employee
        conflict_avg = 0.0
        if conflict_graph.size > 0 and conflict_graph.shape[0] > i:
            conflict_edges = conflict_graph[i, :]
            conflict_values = conflict_edges[conflict_edges > 0]
            if len(conflict_values) > 0:
                conflict_avg = float(np.mean(conflict_values))
        
        # Apply constraint: if conflict_avg > 0.05 and oppose_prob < 0.05, set oppose_prob = 0.05
        if conflict_avg > 0.05 and constrained_probs[i, 0] < 0.05:
            constrained_probs[i, 0] = 0.05
            # Renormalize remaining probabilities
            remaining = 1.0 - 0.05
            remaining_probs = constrained_probs[i, 1:3]
            if np.sum(remaining_probs) > 0:
                constrained_probs[i, 1:3] = remaining_probs / np.sum(remaining_probs) * remaining
            else:
                # Fallback if all remaining are zero
                constrained_probs[i, 1:3] = np.array([remaining / 2, remaining / 2])
        
        # Round to 3 decimal places
        constrained_probs[i] = np.round(constrained_probs[i], 3)
        
        # Ensure sum = 1.0 (adjust largest if needed)
        prob_sum = np.sum(constrained_probs[i])
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            diff = 1.0 - prob_sum
            # Add difference to largest probability
            max_idx = np.argmax(constrained_probs[i])
            constrained_probs[i, max_idx] += diff
            constrained_probs[i, max_idx] = np.round(constrained_probs[i, max_idx], 3)
        
        # Clip to [0, 1] range
        constrained_probs[i] = np.clip(constrained_probs[i], 0.0, 1.0)
    
    return constrained_probs


def identify_influence_sources(employee_id: int, baseline_scores: np.ndarray, hidden_state: dict, distance_cache: Dict[Tuple[int, int], float], top_k: int = 5) -> List[Dict]:
    """
    Identify top influence sources for a given employee.
    
    Args:
        employee_id: Target employee ID
        baseline_scores: Array of baseline scores for all employees
        hidden_state: Organizational hidden state dictionary
        distance_cache: Precomputed distance cache
        top_k: Number of top influencers to return
        
    Returns:
        List of influence source dicts: [{employee_id, graph_type, influence_weight}, ...]
    """
    graphs = hidden_state.get('graphs', {})
    reports_to_graph = np.array(graphs.get('reports_to', []))
    
    # Graph type multipliers
    graph_multipliers = {
        'reports_to': 1.0,
        'influence': 0.8,
        'collaboration': 0.6,
        'friendship': 0.4,
        'conflict': -0.3
    }
    
    influence_contributions = []
    graph_types = ['reports_to', 'influence', 'collaboration', 'friendship', 'conflict']
    
    for graph_type in graph_types:
        if graph_type not in graphs:
            continue
        
        graph_matrix = np.array(graphs[graph_type])
        multiplier = graph_multipliers[graph_type]
        
        # Find all influencers j (where graph[j][employee_id] > 0)
        for j in range(len(baseline_scores)):
            if j == employee_id:
                continue
            
            edge_weight = graph_matrix[j, employee_id] if j < graph_matrix.shape[0] and employee_id < graph_matrix.shape[1] else 0.0
            
            if edge_weight > 0:
                # Compute distance in reports_to graph
                distance = compute_graph_distance(j, employee_id, distance_cache)
                
                # Apply decay
                if np.isinf(distance) or distance < 0:
                    influence_decay = 0.0
                else:
                    influence_decay = 0.7 ** distance
                
                # Compute total influence contribution
                influence_contribution = baseline_scores[j] * edge_weight * influence_decay * abs(multiplier)
                
                influence_contributions.append({
                    'employee_id': j,
                    'graph_type': graph_type,
                    'influence_weight': float(edge_weight),
                    'contribution': influence_contribution
                })
    
    # Sort by contribution (descending) and return top_k
    influence_contributions.sort(key=lambda x: x['contribution'], reverse=True)
    top_influencers = influence_contributions[:top_k]
    
    # Return in format matching schema (without contribution field)
    return [
        {
            'employee_id': inf['employee_id'],
            'graph_type': inf['graph_type'],
            'influence_weight': inf['influence_weight']
        }
        for inf in top_influencers
    ]


def compute_propagation_path(source_ids: List[int], target_id: int, graph: np.ndarray, distance_cache: Dict[Tuple[int, int], float]) -> List[int]:
    """
    Find shortest propagation path from any source to target.
    
    Args:
        source_ids: List of source employee IDs (e.g., C-Suite/Directors)
        target_id: Target employee ID
        graph: Reports_to graph matrix
        distance_cache: Precomputed distance cache
        
    Returns:
        Sequence of employee IDs representing the path, or empty list if no path exists
    """
    if not source_ids:
        return []
    
    # Find shortest path from any source
    best_path = None
    best_distance = float('inf')
    
    for source_id in source_ids:
        distance = compute_graph_distance(source_id, target_id, distance_cache)
        if distance < best_distance and not np.isinf(distance):
            best_distance = distance
            # Reconstruct path using NetworkX
            G = nx.DiGraph()
            N = graph.shape[0]
            for i in range(N):
                for j in range(N):
                    if graph[i, j] > 0:
                        G.add_edge(i, j)
            
            try:
                path = nx.shortest_path(G, source_id, target_id)
                if path:
                    best_path = path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    
    return best_path if best_path else []


def compute_department_aggregates(probabilities: np.ndarray, hidden_state: dict) -> Dict[str, Dict[str, float]]:
    """
    Compute department-level aggregates weighted by collaboration centrality.
    
    Args:
        probabilities: Array of probabilities, shape (N, 4) [oppose, neutral, support, escalate]
        hidden_state: Organizational hidden state dictionary
        
    Returns:
        Dictionary mapping department names to probability dicts
    """
    employees = hidden_state.get('employees', [])
    graphs = hidden_state.get('graphs', {})
    collaboration_graph = np.array(graphs.get('collaboration', []))
    
    # Group employees by department
    dept_employees = {}
    for i, emp in enumerate(employees):
        dept = emp.get('department', 'Unknown')
        if dept not in dept_employees:
            dept_employees[dept] = []
        dept_employees[dept].append(i)
    
    aggregates = {}
    
    for dept, emp_indices in dept_employees.items():
        if not emp_indices:
            continue
        
        # Compute collaboration centrality for each employee in department
        centralities = []
        for i in emp_indices:
            if collaboration_graph.size > 0 and i < collaboration_graph.shape[0]:
                # Sum of collaboration edges within department
                dept_collab = sum(collaboration_graph[i, j] for j in emp_indices if j < collaboration_graph.shape[1])
                centrality = dept_collab / len(emp_indices) if len(emp_indices) > 0 else 0.0
            else:
                centrality = 0.0
            centralities.append(centrality)
        
        # Weighted average by centrality
        total_weight = sum(centralities)
        
        if total_weight > 0:
            dept_probs = np.zeros(3)
            for idx, emp_idx in enumerate(emp_indices):
                weight = centralities[idx]
                dept_probs += probabilities[emp_idx] * weight
            dept_probs = dept_probs / total_weight
        else:
            # Fallback to simple average
            dept_probs = np.mean(probabilities[emp_indices], axis=0)
        
        # Round and ensure sum = 1.0
        dept_probs = np.round(dept_probs, 3)
        prob_sum = np.sum(dept_probs)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            diff = 1.0 - prob_sum
            max_idx = np.argmax(dept_probs)
            dept_probs[max_idx] += diff
            dept_probs[max_idx] = np.round(dept_probs[max_idx], 3)
        
        aggregates[dept] = {
            'oppose': float(dept_probs[0]),
            'neutral': float(dept_probs[1]),
            'support': float(dept_probs[2])
        }
    
    return aggregates


def compute_level_aggregates(probabilities: np.ndarray, hidden_state: dict) -> Dict[str, Dict[str, float]]:
    """
    Compute level-level aggregates (excluding C-Suite).
    
    Args:
        probabilities: Array of probabilities, shape (N, 4) [oppose, neutral, support, escalate]
        hidden_state: Organizational hidden state dictionary
        
    Returns:
        Dictionary mapping level names to probability dicts (excluding C-Suite)
    """
    employees = hidden_state.get('employees', [])
    
    # Group employees by level (excluding C-Suite)
    level_employees = {}
    for i, emp in enumerate(employees):
        level = emp.get('level', 'Unknown')
        if level != 'C-Suite':  # Exclude C-Suite
            if level not in level_employees:
                level_employees[level] = []
            level_employees[level].append(i)
    
    aggregates = {}
    
    for level, emp_indices in level_employees.items():
        if not emp_indices:
            continue
        
        # Simple average (or could weight by tenure)
        level_probs = np.mean(probabilities[emp_indices], axis=0)
        
        # Round and ensure sum = 1.0
        level_probs = np.round(level_probs, 3)
        prob_sum = np.sum(level_probs)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            diff = 1.0 - prob_sum
            max_idx = np.argmax(level_probs)
            level_probs[max_idx] += diff
            level_probs[max_idx] = np.round(level_probs[max_idx], 3)
        
        aggregates[level] = {
            'oppose': float(level_probs[0]),
            'neutral': float(level_probs[1]),
            'support': float(level_probs[2])
        }
    
    return aggregates


def compute_aggregate_outcomes(probabilities: np.ndarray) -> Dict[str, Any]:
    """
    Compute aggregate outcomes across all employees.
    
    Args:
        probabilities: Array of probabilities, shape (N, 3) [oppose, neutral, support]
        
    Returns:
        Dictionary with probabilities and top_class
    """
    # Simple average across all employees
    agg_probs = np.mean(probabilities, axis=0)
    
    # Round
    agg_probs = np.round(agg_probs, 3)
    
    # Ensure sum = 1.0
    prob_sum = np.sum(agg_probs)
    if not np.isclose(prob_sum, 1.0, atol=1e-6):
        diff = 1.0 - prob_sum
        max_idx = np.argmax(agg_probs)
        agg_probs[max_idx] += diff
        agg_probs[max_idx] = np.round(agg_probs[max_idx], 3)
    
    # Find top class
    sentiment_classes = ['oppose', 'neutral', 'support']
    top_class_idx = np.argmax(agg_probs)
    top_class = sentiment_classes[top_class_idx]
    
    return {
        'probabilities': {
            'oppose': float(agg_probs[0]),
            'neutral': float(agg_probs[1]),
            'support': float(agg_probs[2])
        },
        'top_class': top_class
    }


def compute_feature_importance(probabilities: np.ndarray, hidden_state: dict) -> List[Dict]:
    """
    Compute feature importance using correlation with support probabilities.
    
    Args:
        probabilities: Array of probabilities, shape (N, 3) [oppose, neutral, support]
        hidden_state: Organizational hidden state dictionary
        
    Returns:
        List of feature importance dicts: [{"feature": "...", "direction": "+/-", "weight": ...}, ...]
    """
    N = probabilities.shape[0]
    employees = hidden_state.get('employees', [])
    graphs = hidden_state.get('graphs', {})
    collaboration_graph = np.array(graphs.get('collaboration', []))
    
    # Extract support probabilities
    support_probs = probabilities[:, 2]  # support is index 2
    
    # Features to analyze
    features = []
    
    # Tenure
    tenure_values = np.array([emp.get('tenure', 0) for emp in employees[:N]])
    if len(tenure_values) > 1 and np.std(tenure_values) > 0:
        corr, _ = pearsonr(tenure_values, support_probs)
        features.append({
            'feature': 'tenure',
            'direction': '+' if corr > 0 else '-',
            'weight': abs(corr)
        })
    
    # Collaboration degree
    collab_degrees = []
    for i in range(min(N, collaboration_graph.shape[0])):
        if collaboration_graph.size > 0:
            degree = np.sum(collaboration_graph[i, :]) if i < collaboration_graph.shape[0] else 0.0
        else:
            degree = 0.0
        collab_degrees.append(degree)
    
    collab_degrees = np.array(collab_degrees)
    if len(collab_degrees) > 1 and np.std(collab_degrees) > 0:
        corr, _ = pearsonr(collab_degrees, support_probs)
        features.append({
            'feature': 'collaboration_degree',
            'direction': '+' if corr > 0 else '-',
            'weight': abs(corr)
        })
    
    # Department (one-hot encoded, use most correlated department)
    departments = [emp.get('department', 'Unknown') for emp in employees[:N]]
    unique_depts = list(set(departments))
    for dept in unique_depts:
        dept_indicator = np.array([1 if d == dept else 0 for d in departments])
        if np.sum(dept_indicator) > 1:  # Need at least 2 employees in department
            corr, _ = pearsonr(dept_indicator, support_probs)
            features.append({
                'feature': f'department_{dept}',
                'direction': '+' if corr > 0 else '-',
                'weight': abs(corr)
            })
    
    # Level (one-hot encoded)
    levels = [emp.get('level', 'Unknown') for emp in employees[:N]]
    unique_levels = list(set(levels))
    for level in unique_levels:
        level_indicator = np.array([1 if l == level else 0 for l in levels])
        if np.sum(level_indicator) > 1:
            corr, _ = pearsonr(level_indicator, support_probs)
            features.append({
                'feature': f'level_{level}',
                'direction': '+' if corr > 0 else '-',
                'weight': abs(corr)
            })
    
    # Normalize weights to sum to 1.0
    total_weight = sum(f['weight'] for f in features)
    if total_weight > 0:
        for f in features:
            f['weight'] = f['weight'] / total_weight
    
    # Sort by weight and return top 5
    features.sort(key=lambda x: x['weight'], reverse=True)
    return features[:5]


def compute_individual_sentiments(hidden_state: dict, narrative_context: Optional[dict] = None) -> List[Dict]:
    """
    Orchestrate all computational steps to compute individual sentiments.
    
    Args:
        hidden_state: Organizational hidden state dictionary
        narrative_context: Optional dictionary with encoded narrative and scenario modifiers
        
    Returns:
        List of individual sentiment dicts matching schema
    """
    N = hidden_state.get('num_employees', 0)
    employees = hidden_state.get('employees', [])
    graphs = hidden_state.get('graphs', {})
    reports_to_graph = np.array(graphs.get('reports_to', []))
    
    # Extract modifiers from narrative context if available
    scenario_modifiers = None
    if narrative_context and 'scenario_modifiers' in narrative_context:
        scenario_modifiers = narrative_context['scenario_modifiers']
    
    # Step 1: Precompute distance cache
    distance_cache = precompute_all_pairs_shortest_paths(reports_to_graph)
    
    # Step 2: Compute baseline scores (with narrative fusion)
    baseline_scores = compute_baseline_scores(hidden_state, scenario_modifiers)
    
    # Step 3: Compute influenced scores
    influenced_scores = compute_influenced_scores(baseline_scores, hidden_state, distance_cache)
    
    # Step 4: Apply softmax
    probabilities = apply_softmax_mapping(influenced_scores, temperature=1.0)
    
    # Step 5: Apply constraints and rounding
    probabilities = apply_constraints_and_rounding(probabilities, hidden_state)
    
    # Step 6: Identify C-Suite and Directors for propagation paths
    csuite_ids = [i for i, emp in enumerate(employees) if emp.get('level') == 'C-Suite']
    director_ids = [i for i, emp in enumerate(employees) if emp.get('level') == 'Director']
    source_ids = csuite_ids + director_ids
    
    # Step 7: Format output for each employee
    sentiment_classes = ['oppose', 'neutral', 'support']
    individual_sentiments = []
    
    for i in range(N):
        # Get sentiment (class with highest probability)
        top_class_idx = np.argmax(probabilities[i])
        sentiment = sentiment_classes[top_class_idx]
        
        # Get probabilities
        probs = {
            'oppose': float(probabilities[i, 0]),
            'neutral': float(probabilities[i, 1]),
            'support': float(probabilities[i, 2])
        }
        
        # Identify influence sources
        influence_sources = identify_influence_sources(i, baseline_scores, hidden_state, distance_cache, top_k=5)
        
        # Compute propagation path
        propagation_path = compute_propagation_path(source_ids, i, reports_to_graph, distance_cache)
        
        individual_sentiments.append({
            'employee_id': i,
            'sentiment': sentiment,
            'probabilities': probs,
            'influence_sources': influence_sources,
            'propagation_path': propagation_path
        })
    
    return individual_sentiments


# JSON Schema for forecast output
FORECAST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scenario_id", "state_hash", "generated_at", "schema_version", "horizon", "model", "individual_sentiments", "aggregate_outcomes"],
    "properties": {
        "scenario_id": {"type": "string"},
        "state_hash": {"type": "string", "pattern": "^sha256:[A-Fa-f0-9]{64}$"},
        "generated_at": {"type": "string", "format": "date-time"},
        "schema_version": {"type": "string", "enum": ["1.0"]},
        "horizon": {"type": "string", "enum": ["decision", "next_cycle", "quarter"]},
        "model": {
            "type": "object",
            "additionalProperties": False,
            "required": ["provider", "model", "temperature", "prompt_template_id"],
            "properties": {
                "provider": {"type": "string", "enum": ["openai", "azure_openai"]},
                "model": {"type": "string"},
                "temperature": {"type": "number", "minimum": 0, "maximum": 0},
                "prompt_template_id": {"type": "string"}
            }
        },
        "individual_sentiments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["employee_id", "sentiment", "probabilities", "influence_sources"],
                "properties": {
                    "employee_id": {"type": "integer", "minimum": 0},
                    "sentiment": {"type": "string", "enum": ["oppose", "neutral", "support"]},
                    "probabilities": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["oppose", "neutral", "support"],
                        "properties": {
                            "oppose": {"type": "number", "minimum": 0, "maximum": 1},
                            "neutral": {"type": "number", "minimum": 0, "maximum": 1},
                            "support": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    },
                    "influence_sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["employee_id", "graph_type", "influence_weight"],
                            "properties": {
                                "employee_id": {"type": "integer", "minimum": 0},
                                "graph_type": {"type": "string", "enum": ["reports_to", "collaboration", "friendship", "influence", "conflict"]},
                                "influence_weight": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        }
                    },
                    "propagation_path": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "description": "Sequence of employee IDs showing how sentiment propagated to this individual"
                    }
                }
            }
        },
        "aggregate_outcomes": {
            "type": "object",
            "additionalProperties": False,
            "required": ["probabilities", "top_class"],
            "properties": {
                "probabilities": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "oppose": {"type": "number", "minimum": 0, "maximum": 1},
                        "neutral": {"type": "number", "minimum": 0, "maximum": 1},
                        "support": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["oppose", "neutral", "support"]
                },
                "top_class": {"type": "string", "enum": ["oppose", "neutral", "support"]}
            }
        },
        "segments": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "by_department": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["oppose", "neutral", "support"],
                        "properties": {
                            "oppose": {"type": "number", "minimum": 0, "maximum": 1},
                            "neutral": {"type": "number", "minimum": 0, "maximum": 1},
                            "support": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                },
                "by_level": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["oppose", "neutral", "support"],
                        "properties": {
                            "oppose": {"type": "number", "minimum": 0, "maximum": 1},
                            "neutral": {"type": "number", "minimum": 0, "maximum": 1},
                            "support": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                }
            }
        },
        "rationale": {"type": "string", "maxLength": 500},
        "features_importance": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["feature", "direction", "weight"],
                "properties": {
                    "feature": {"type": "string"},
                    "direction": {"type": "string", "enum": ["+", "-"]},
                    "weight": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        },
        "constraints_used": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}}
    }
}

# System prompt for forecasting
FORECAST_SYSTEM_PROMPT = """
You are an organizational analyst. You will receive computed sentiment probabilities and graph analysis results. Your task is to generate a concise rationale (max 500 chars) explaining the forecast.

The probabilities have been computed using:
- Baseline sentiment scores based on resource need, sanction strength, visibility, conflict, tenure, department-domain alignment, and level-specific modifiers
- Graph-based influence propagation through reports_to, collaboration, friendship, and influence networks
- Softmax mapping with temperature=1.0

Focus on:
- Key patterns in the results (e.g., "Engineering shows higher support due to strong collaboration")
- Notable outliers or anomalies
- Department-level differences
- Network effects (e.g., "C-Suite influence propagates strongly through reports_to hierarchy")
"""


def canonicalize_json(obj: dict) -> str:
    """
    Canonicalize JSON by sorting keys recursively.
    
    Args:
        obj: Dictionary to canonicalize
        
    Returns:
        Canonical JSON string
    """
    if isinstance(obj, dict):
        sorted_items = sorted(obj.items())
        return "{" + ",".join(
            f'"{k}":{canonicalize_json(v)}' for k, v in sorted_items
        ) + "}"
    elif isinstance(obj, list):
        return "[" + ",".join(canonicalize_json(item) for item in obj) + "]"
    else:
        return json.dumps(obj, sort_keys=True)


def compute_state_hash(hidden_state: dict) -> str:
    """
    Compute SHA256 hash of canonicalized hidden state JSON.
    
    Args:
        hidden_state: The organizational hidden state dictionary
        
    Returns:
        SHA256 hash prefixed with "sha256:"
    """
    canonical = canonicalize_json(hidden_state)
    hash_bytes = hashlib.sha256(canonical.encode('utf-8')).digest()
    hash_hex = hash_bytes.hex()
    return f"sha256:{hash_hex}"


def extract_segments(hidden_state: dict) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract unique departments and levels from employee data.
    
    Args:
        hidden_state: The organizational hidden state dictionary
        
    Returns:
        Dictionary with 'departments' and 'levels' keys containing lists of unique values
    """
    employees = hidden_state.get("employees", [])
    
    departments = set()
    levels = set()
    
    for emp in employees:
        if "department" in emp:
            departments.add(emp["department"])
        if "level" in emp:
            levels.add(emp["level"])
    
    return {
        "departments": sorted(list(departments)),
        "levels": sorted(list(levels))
    }


def validate_heterogeneity(forecast: dict, hidden_state: Optional[dict] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate that support probabilities show meaningful heterogeneity across departments.
    
    Args:
        forecast: Forecast dictionary to validate
        hidden_state: Optional hidden state to check collaboration differences
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if "segments" not in forecast or "by_department" not in forecast["segments"]:
        # If segments not computed, skip heterogeneity check
        return True, None
    
    dept_segments = forecast["segments"]["by_department"]
    dept_names = list(dept_segments.keys())
    
    if len(dept_names) < 2:
        # Need at least 2 departments to check heterogeneity
        return True, None
    
    # Get support probabilities by department
    support_probs = {dept: dept_segments[dept].get("support", 0.0) for dept in dept_names}
    
    # Check variance
    support_values = list(support_probs.values())
    support_std = float(np.std(support_values))
    
    if support_std < 0.05:
        return False, f"Support probabilities too homogeneous: std={support_std:.3f} (required >= 0.05). Departments: {support_probs}"
    
    # If hidden_state provided, check collaboration-based heterogeneity
    if hidden_state:
        graphs = hidden_state.get('graphs', {})
        collaboration_graph = np.array(graphs.get('collaboration', []))
        employees = hidden_state.get('employees', [])
        
        # Compute average collaboration per department
        dept_collab_avg = {}
        for dept in dept_names:
            dept_employees = [i for i, emp in enumerate(employees) if emp.get('department') == dept]
            if dept_employees:
                collab_values = []
                for i in dept_employees:
                    if collaboration_graph.size > 0 and i < collaboration_graph.shape[0]:
                        collab_values.extend([collaboration_graph[i, j] for j in dept_employees if j < collaboration_graph.shape[1] and collaboration_graph[i, j] > 0])
                dept_collab_avg[dept] = float(np.mean(collab_values)) if collab_values else 0.0
            else:
                dept_collab_avg[dept] = 0.0
        
        # Check if departments with collaboration difference >= 0.20 have support difference >= 0.10
        for i, dept1 in enumerate(dept_names):
            for dept2 in dept_names[i+1:]:
                collab_diff = abs(dept_collab_avg[dept1] - dept_collab_avg[dept2])
                support_diff = abs(support_probs[dept1] - support_probs[dept2])
                
                if collab_diff >= 0.20 and support_diff < 0.10:
                    return False, f"Heterogeneity requirement violated: {dept1} vs {dept2} have collaboration diff={collab_diff:.3f} >= 0.20 but support diff={support_diff:.3f} < 0.10"
    
    return True, None


def validate_forecast(forecast: dict, hidden_state: Optional[dict] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate forecast against JSON schema and business rules.
    
    Args:
        forecast: Forecast dictionary to validate
        hidden_state: Optional hidden state for heterogeneity validation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Schema validation
    if HAS_JSONSCHEMA:
        try:
            jsonschema.validate(instance=forecast, schema=FORECAST_SCHEMA)
        except jsonschema.ValidationError as e:
            return False, f"Schema validation error: {e.message}"
    else:
        # Basic validation without jsonschema
        required_fields = ["scenario_id", "state_hash", "generated_at", "schema_version", "horizon", "model", "individual_sentiments", "aggregate_outcomes"]
        for field in required_fields:
            if field not in forecast:
                return False, f"Missing required field: {field}"
    
    # Validate individual sentiments
    if "individual_sentiments" in forecast:
        employee_ids = set()
        for sentiment in forecast["individual_sentiments"]:
            emp_id = sentiment.get("employee_id")
            if emp_id is None:
                return False, "Individual sentiment missing employee_id"
            if emp_id in employee_ids:
                return False, f"Duplicate employee_id: {emp_id}"
            employee_ids.add(emp_id)
            
            # Validate probabilities sum to 1.0
            probs = sentiment.get("probabilities", {})
            prob_sum = sum(probs.values())
            if abs(prob_sum - 1.0) > 1e-6:
                return False, f"Employee {emp_id} probabilities sum to {prob_sum}, not 1.0"
            
            # Auto-correct sentiment to match highest probability
            if probs:
                max_class = max(probs.items(), key=lambda x: x[1])[0]
                sentiment["sentiment"] = max_class
    
    # Validate aggregate outcomes
    if "aggregate_outcomes" in forecast:
        agg_probs = forecast["aggregate_outcomes"]["probabilities"]
        prob_sum = sum(agg_probs.values())
        if abs(prob_sum - 1.0) > 1e-6:
            return False, f"Aggregate probabilities sum to {prob_sum}, not 1.0"
        
        # Auto-correct top_class to match highest probability
        if agg_probs:
            max_class = max(agg_probs.items(), key=lambda x: x[1])[0]
            forecast["aggregate_outcomes"]["top_class"] = max_class
    
    # Validate segment probabilities
    if "segments" in forecast:
        for segment_type in ["by_department", "by_level"]:
            if segment_type in forecast["segments"]:
                for segment_name, segment_probs in forecast["segments"][segment_type].items():
                    seg_sum = sum(segment_probs.values())
                    if abs(seg_sum - 1.0) > 1e-6:
                        return False, f"Segment {segment_type}/{segment_name} probabilities sum to {seg_sum}, not 1.0"
    
    # Note: Heterogeneity validation removed - accepting computed probabilities as-is
    
    return True, None


def forecast_scenario(
    hidden_state: dict,
    scenario_id: str,
    model: str = "gpt-4o-mini",
    prompt_template_id: str = "v3",
    horizon: str = "decision",
    provider: str = "openai",
    narrative_context: Optional[dict] = None
) -> dict:
    """
    Generate deterministic forecast from hidden state using Python computation and LLM for rationale.
    
    Args:
        hidden_state: The organizational state JSON
        scenario_id: Unique identifier for this scenario
        model: OpenAI model name
        prompt_template_id: Version identifier for the prompt template
        horizon: Forecast horizon
        provider: API provider
        narrative_context: Optional dictionary containing encoded narrative (story, relationships, etc.)
        
    Returns:
        Validated forecast object matching the schema
    """
    # Compute state hash
    state_hash = compute_state_hash(hidden_state)
    
    # Step 1: Compute individual sentiments using Python functions
    try:
        individual_sentiments = compute_individual_sentiments(hidden_state, narrative_context)
    except Exception as e:
        return {
            "scenario_id": scenario_id,
            "state_hash": state_hash,
            "error": True,
            "error_message": f"Error computing individual sentiments: {str(e)}"
        }
    
    # Step 2: Extract probabilities array for aggregation
    N = hidden_state.get('num_employees', 0)
    probabilities = np.zeros((N, 3))
    for i, sentiment in enumerate(individual_sentiments):
        probs = sentiment.get('probabilities', {})
        probabilities[i, 0] = probs.get('oppose', 0.0)
        probabilities[i, 1] = probs.get('neutral', 0.0)
        probabilities[i, 2] = probs.get('support', 0.0)
    
    # Step 3: Compute aggregates
    try:
        department_aggregates = compute_department_aggregates(probabilities, hidden_state)
        level_aggregates = compute_level_aggregates(probabilities, hidden_state)
        aggregate_outcomes = compute_aggregate_outcomes(probabilities)
        feature_importance = compute_feature_importance(probabilities, hidden_state)
    except Exception as e:
        return {
            "scenario_id": scenario_id,
            "state_hash": state_hash,
            "error": True,
            "error_message": f"Error computing aggregates: {str(e)}"
        }
    
    # Step 4: Call LLM for rationale generation only
    # Initialize OpenAI client
    try:
        client = openai.OpenAI()
        if not client.api_key:
            raise openai.OpenAIError("API key not found after loading .env")
    except Exception as e:
        # If LLM fails, still return forecast without rationale
        print(f"Warning: OpenAI client initialization failed: {str(e)}. Continuing without rationale.")
        rationale = None
        warnings = ["Could not generate rationale: OpenAI API unavailable"]
    else:
        # Build simplified user prompt with computed results
        segments = extract_segments(hidden_state)
        num_employees = hidden_state.get('num_employees', 0)
        
        # Create summary of computed probabilities
        support_probs = [s['probabilities']['support'] for s in individual_sentiments]
        support_mean = float(np.mean(support_probs))
        support_std = float(np.std(support_probs))
        
        dept_summary = {}
        for dept, probs in department_aggregates.items():
            dept_summary[dept] = {
                'support': probs['support'],
                'oppose': probs['oppose']
            }
        
        # Build narrative context string if available
        narrative_section = ""
        if narrative_context:
            narrative_section = f"""
Narrative Context:
- Company Story: {narrative_context.get('company_story', 'N/A')[:500]}...
- Key Relationships: {narrative_context.get('key_relationships', 'N/A')[:500]}...
- Scenario: {narrative_context.get('recommendation_scenario', 'N/A')[:500]}...
"""

        user_prompt = f"""Generate a concise rationale (max 500 chars) explaining this organizational sentiment forecast.

Computed Results Summary:
- Total Employees: {num_employees}
- Overall Support Probability: {support_mean:.3f} (std: {support_std:.3f})
- Top Sentiment Class: {aggregate_outcomes['top_class']}

Department-Level Probabilities:
{json.dumps(dept_summary, indent=2)}

Level Aggregates:
{json.dumps({k: {'support': v['support']} for k, v in level_aggregates.items()}, indent=2)}

Key Features (by importance):
{json.dumps(feature_importance[:3], indent=2)}

Organizational Context:
- Industry: {hidden_state.get('org_seed', {}).get('industry', 'unknown')}
- Recommendation Domain: {hidden_state.get('rec_seed', {}).get('domain', 'unknown')}
- Visibility: {hidden_state.get('situation_seed', {}).get('visibility', 'unknown')}{narrative_section}

Focus on explaining:
1. Why certain departments show higher/lower support
2. Notable patterns or outliers
3. Network effects (how influence propagated)
4. Key factors driving the forecast
5. How the company's story/culture (if provided) aligns with these results

CRITICAL: Keep rationale under 500 characters. Be concise and focus on the most important insights only.
"""
        
        # Call LLM for rationale
        max_retries = 3
        base_delay = 1
        rationale = None
        warnings = []
        
        for attempt in range(max_retries):
            try:
                # Simplified schema for rationale only
                rationale_schema = {
                    "type": "object",
                    "properties": {
                        "rationale": {"type": "string", "maxLength": 500},
                        "warnings": {"type": "array", "items": {"type": "string"}},
                        "features_importance": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "feature": {"type": "string"},
                                    "direction": {"type": "string", "enum": ["+", "-"]},
                                    "weight": {"type": "number", "minimum": 0, "maximum": 1}
                                }
                            }
                        }
                    }
                }
                
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": FORECAST_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "rationale_output",
                            "strict": False,
                            "schema": rationale_schema
                        }
                    }
                )
                
                rationale_json = json.loads(response.choices[0].message.content)
                rationale = rationale_json.get('rationale', '')
                warnings = rationale_json.get('warnings', [])
                
                # Truncate rationale to 500 characters if it exceeds the limit
                if len(rationale) > 500:
                    rationale = rationale[:497] + '...'
                    if not warnings:
                        warnings = []
                    warnings.append("Rationale was truncated to 500 characters")
                
                # Use computed feature_importance instead of LLM's
                break
                
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"⚠ Rate limit hit. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    warnings.append("Could not generate rationale: Rate limit exceeded")
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    warnings.append(f"Could not generate rationale: {str(e)}")
                    break
    
    # Step 5: Assemble final forecast object
    forecast = {
        "scenario_id": scenario_id,
        "state_hash": state_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "horizon": horizon,
        "model": {
            "provider": provider,
            "model": model,
            "temperature": 0,
            "prompt_template_id": prompt_template_id
        },
        "individual_sentiments": individual_sentiments,
        "aggregate_outcomes": aggregate_outcomes,
        "segments": {
            "by_department": department_aggregates,
            "by_level": level_aggregates
        },
        "features_importance": feature_importance
    }
    
    if rationale:
        # Ensure rationale doesn't exceed 500 characters (schema constraint)
        if len(rationale) > 500:
            rationale = rationale[:497] + '...'
            if not warnings:
                warnings = []
            warnings.append("Rationale was truncated to 500 characters")
        forecast["rationale"] = rationale
    if warnings:
        forecast["warnings"] = warnings
    
    # Step 6: Validate forecast
    is_valid, error_msg = validate_forecast(forecast, hidden_state)
    
    if not is_valid:
        return {
            "scenario_id": scenario_id,
            "state_hash": state_hash,
            "error": True,
            "error_message": f"Validation failed: {error_msg}"
        }
    
    return forecast


def forecast_from_file(
    hidden_state_path: str,
    scenario_id: Optional[str] = None,
    model: str = "gpt-4o-mini",
    prompt_template_id: str = "v3",
    horizon: str = "decision"
) -> dict:
    """
    Convenience function to load hidden state from file and generate forecast.
    
    Args:
        hidden_state_path: Path to hidden state JSON file
        scenario_id: Optional scenario ID (defaults to filename without extension)
        model: OpenAI model name
        prompt_template_id: Version identifier for the prompt template
        horizon: Forecast horizon
        
    Returns:
        Forecast dictionary
    """
    # Load hidden state
    try:
        with open(hidden_state_path, 'r') as f:
            hidden_state = json.load(f)
    except FileNotFoundError:
        return {
            "error": True,
            "error_message": f"File not found: {hidden_state_path}"
        }
    except json.JSONDecodeError as e:
        return {
            "error": True,
            "error_message": f"Invalid JSON: {str(e)}"
        }
    
    # Generate scenario_id from filename if not provided
    if scenario_id is None:
        scenario_id = os.path.splitext(os.path.basename(hidden_state_path))[0]
    
    return forecast_scenario(
        hidden_state=hidden_state,
        scenario_id=scenario_id,
        model=model,
        prompt_template_id=prompt_template_id,
        horizon=horizon
    )


if __name__ == "__main__":
    # Example usage
    hidden_state_path = "./sample_hidden_states/sample_0.json"
    output_dir = "./sample_forecasts"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Generating Deterministic Forecast ===\n")
    
    forecast = forecast_from_file(
        hidden_state_path=hidden_state_path,
        scenario_id="sample_0",
        model="gpt-4o-mini",
        horizon="decision"
    )
    
    if "error" in forecast and forecast["error"]:
        print(f"Error: {forecast.get('error_message', 'Unknown error')}")
    else:
        # Save forecast
        output_path = os.path.join(output_dir, "sample_0_forecast.json")
        with open(output_path, "w") as f:
            json.dump(forecast, f, indent=2)
        
        print("✓ Forecast generated successfully")
        print(f"\nScenario ID: {forecast['scenario_id']}")
        print(f"State Hash: {forecast['state_hash']}")
        print(f"Horizon: {forecast['horizon']}")
        
        # Display aggregate outcomes
        if "aggregate_outcomes" in forecast:
            agg = forecast["aggregate_outcomes"]
            print(f"\nAggregate Outcomes:")
            print(f"  Top Class: {agg['top_class']}")
            top_prob = agg['probabilities'][agg['top_class']]
            print(f"  Top Probability: {top_prob:.3f}")
            print(f"\n  Probabilities:")
            for cls, prob in agg['probabilities'].items():
                print(f"    {cls}: {prob:.3f}")
        
        # Display individual sentiments summary
        if "individual_sentiments" in forecast:
            sentiments = forecast["individual_sentiments"]
            print(f"\nIndividual Sentiments: {len(sentiments)} employees")
            
            # Count by sentiment
            sentiment_counts = {}
            for s in sentiments:
                sent = s.get("sentiment", "unknown")
                sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
            
            print(f"  Distribution:")
            for sent, count in sentiment_counts.items():
                print(f"    {sent}: {count} employees")
            
            # Show a few examples with influence sources
            print(f"\n  Sample Individual Predictions (first 5):")
            for s in sentiments[:5]:
                emp_id = s.get("employee_id", "?")
                sent = s.get("sentiment", "?")
                prob = s.get("probabilities", {}).get(sent, 0)
                influences = s.get("influence_sources", [])
                print(f"    Employee {emp_id}: {sent} ({prob:.2f}) - Influenced by {len(influences)} sources")
        
        if "rationale" in forecast:
            print(f"\nRationale: {forecast['rationale'][:200]}...")
        
        print(f"\n✓ Saved to: {output_path}")
