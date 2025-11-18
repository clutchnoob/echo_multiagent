# Agentic System Implementation Plan

## Problem Statement
The current LLM-based approach produces homogeneous probabilities (support: 0.698-0.711, range: 0.013) because the LLM cannot reliably execute multi-step graph-theoretic calculations. We need to move all computational logic into Python functions, using the LLM only for rationale generation.

## Architecture Overview
- **Python Functions**: Perform all deterministic calculations (baseline scores, graph influence, softmax, aggregation)
- **LLM**: Generate rationale only, using computed results as context
- **Result**: Deterministic, heterogeneous probabilities that reflect actual graph structure

---

## Phase 1: Core Computational Functions

### 1.1 Graph Utilities
**File**: `forward_forecaster.py`  
**Functions to create**:
- `precompute_all_pairs_shortest_paths(graph: np.ndarray) -> Dict[Tuple[int, int], int]`
  - Use NetworkX or scipy for efficient shortest path computation
  - Handle disconnected nodes (return inf or max_distance)
  - Cache results for O(n²) space, O(n³) time upfront

- `compute_graph_distance(source: int, target: int, graph: np.ndarray, distance_cache: Dict) -> float`
  - Lookup in precomputed cache
  - Return inf if disconnected, convert to numeric decay value

### 1.2 Baseline Score Computation
**Function**: `compute_baseline_scores(hidden_state: dict) -> np.ndarray`
- Input: hidden_state dict
- Output: array of baseline scores (one per employee)
- Formula: `(resource_need × 0.3) + (sanction_strength × visibility_factor) - (conflict_avg × 0.2) + (tenure × 0.05)`
- Handle edge cases: zero conflict (use 0), tenure normalization

### 1.3 Graph Influence Application
**Function**: `compute_influenced_scores(baseline_scores: np.ndarray, hidden_state: dict, distance_cache: Dict) -> np.ndarray`
- For each employee i:
  - Start with baseline_score[i]
  - For each graph type (reports_to, collaboration, friendship, influence, conflict):
    - For each influencer j (where graph[j][i] > 0):
      - Get edge weight from graph matrix
      - Compute distance in reports_to graph (use cache)
      - Apply decay: `0.7^distance` (handle inf → 0.0)
      - Apply graph type multiplier:
        - reports_to: 1.0
        - influence: 0.8
        - collaboration: 0.6
        - friendship: 0.4
        - conflict: -0.3 (negative)
      - Add: `baseline_score[j] × edge_weight × decay × multiplier`
- Return influenced_scores array

### 1.4 Softmax and Probability Mapping
**Function**: `apply_softmax_mapping(influenced_scores: np.ndarray, temperature: float = 0.5) -> np.ndarray`
- Input: influenced_scores (N×1), temperature
- For each employee:
  - Map to logits:
    - oppose_logit = -influenced_score × 0.5
    - neutral_logit = influenced_score × 0.3
    - support_logit = influenced_score × 1.0
    - escalate_logit = influenced_score × 0.2
  - Apply softmax with temperature
  - Return (N×4) array: [oppose, neutral, support, escalate] per employee

### 1.5 Constraints and Rounding
**Function**: `apply_constraints_and_rounding(probabilities: np.ndarray, hidden_state: dict) -> np.ndarray`
- Input: (N×4) probabilities array, hidden_state
- For each employee:
  - If conflict_avg > 0.05 and oppose_prob < 0.05:
    - Set oppose_prob = 0.05
    - Renormalize remaining probabilities
  - Round to 3 decimal places
  - Ensure sum = 1.0 (adjust largest if needed)
- Return normalized, rounded probabilities

### 1.6 Influence Sources Identification
**Function**: `identify_influence_sources(employee_id: int, hidden_state: dict, distance_cache: Dict, top_k: int = 5) -> List[Dict]`
- For each graph type, find top influencers (highest edge weights × influence contribution)
- Return list of {employee_id, graph_type, influence_weight}
- Sort by total influence contribution (not just edge weight)

### 1.7 Propagation Path Computation
**Function**: `compute_propagation_path(source_ids: List[int], target_id: int, graph: np.ndarray, distance_cache: Dict) -> List[int]`
- Find shortest path from any source (C-Suite/Directors) to target
- Use NetworkX shortest_path or reconstruct from distance cache
- Return sequence of employee IDs

---

## Phase 2: Aggregation Functions

### 2.1 Department Aggregates
**Function**: `compute_department_aggregates(probabilities: np.ndarray, hidden_state: dict) -> Dict[str, Dict[str, float]]`
- For each department:
  - Get department employees
  - Compute collaboration centrality: `sum(collaboration[i][j] for j in dept) / len(dept)`
  - Weighted average of probabilities by centrality
  - Handle zero total_weight (fallback to simple average)
- Return: `{"Engineering": {"oppose": 0.05, ...}, ...}`

### 2.2 Level Aggregates
**Function**: `compute_level_aggregates(probabilities: np.ndarray, hidden_state: dict) -> Dict[str, Dict[str, float]]`
- For each level (Director, Manager - EXCLUDE C-Suite):
  - Get level employees
  - Weight by tenure or simple average
  - Exclude C-Suite explicitly
- Return: `{"Director": {"oppose": 0.05, ...}, ...}`

### 2.3 Aggregate Outcomes
**Function**: `compute_aggregate_outcomes(probabilities: np.ndarray) -> Dict[str, Any]`
- Simple average across all employees
- Find top_class (highest probability)
- Return: `{"probabilities": {...}, "top_class": "support"}`

---

## Phase 3: Feature Importance

### 3.1 Correlation-Based Feature Importance
**Function**: `compute_feature_importance(probabilities: np.ndarray, hidden_state: dict) -> List[Dict]`
- For each feature (tenure, level, department, collaboration_degree, etc.):
  - Compute Pearson correlation with support probability
  - Use absolute correlation as weight
  - Normalize to sum to 1.0
- Return: `[{"feature": "tenure", "direction": "+", "weight": 0.35}, ...]`
- Top 3-5 features only

---

## Phase 4: Main Orchestration Function

### 4.1 Individual Sentiment Computation
**Function**: `compute_individual_sentiments(hidden_state: dict) -> List[Dict]`
- Orchestrates all computational steps:
  1. Precompute distance cache
  2. Compute baseline scores
  3. Compute influenced scores
  4. Apply softmax
  5. Apply constraints and rounding
  6. For each employee:
     - Identify influence sources
     - Compute propagation path
     - Format output: {employee_id, sentiment, probabilities, influence_sources, propagation_path}
- Return list of individual sentiment dicts

### 4.2 Main Forecast Function Update
**Function**: `forecast_scenario()` - Refactor
- **Step 1**: Call `compute_individual_sentiments()` → get deterministic probabilities
- **Step 2**: Call aggregation functions → get segments and aggregate outcomes
- **Step 3**: Call `compute_feature_importance()` → get feature importance
- **Step 4**: Call LLM with simplified prompt:
  - Input: computed probabilities, graph structure summary, key insights
  - Task: Generate rationale explaining the forecast
  - Output: rationale, features_importance (optional - can use computed), warnings
- **Step 5**: Assemble final forecast object
- **Step 6**: Validate (including heterogeneity check)

---

## Phase 5: LLM Integration Updates

### 5.1 Simplified System Prompt
**New `FORECAST_SYSTEM_PROMPT`**:
```
You are an organizational analyst. You will receive computed sentiment probabilities and graph analysis results. Your task is to generate a concise rationale (max 500 chars) explaining the forecast.

The probabilities have been computed using:
- Baseline sentiment scores based on resource need, sanction strength, visibility, conflict, and tenure
- Graph-based influence propagation through reports_to, collaboration, friendship, and influence networks
- Softmax mapping with temperature=0.5

Focus on:
- Key patterns in the results (e.g., "Engineering shows higher support due to strong collaboration")
- Notable outliers or anomalies
- Department-level differences
- Network effects (e.g., "C-Suite influence propagates strongly through reports_to hierarchy")
```

### 5.2 User Prompt for Rationale
**New `user_prompt`**:
- Include computed probabilities summary
- Include department/level aggregates
- Include key graph metrics (centrality, distance from C-Suite)
- Ask for rationale, warnings, and feature importance insights

### 5.3 LLM Schema Update
**Simplified schema** for rationale generation:
- Only require: `rationale`, `features_importance` (optional), `warnings`
- Remove computational fields (already computed)

---

## Phase 6: Validation and Quality Checks

### 6.1 Heterogeneity Validation
**Function**: `validate_heterogeneity(forecast: dict) -> Tuple[bool, Optional[str]]`
- Check support probability variance across departments
- If departments have collaboration_avg difference ≥ 0.20, ensure support difference ≥ 0.10
- Return (is_valid, error_msg)

### 6.2 Enhanced Validation
**Update `validate_forecast()`**:
- Add heterogeneity check
- Verify probabilities reflect graph structure (correlation checks)
- Check that influence_sources match actual graph edges

---

## Phase 7: Error Handling and Edge Cases

### 7.1 Disconnected Graphs
- Handle inf distances: set influence_decay = 0.0
- Log warnings for disconnected employees

### 7.2 Zero Division
- Check total_weight > 0 before division in aggregation
- Fallback to simple average if centrality is zero

### 7.3 Numerical Stability
- Use np.clip for probabilities to stay in [0, 1]
- Use np.isclose for floating point comparisons
- Handle overflow in softmax (subtract max before exp)

---

## Phase 8: Testing and Verification

### 8.1 Unit Tests
- Test each computational function independently
- Test edge cases (zero conflict, disconnected graphs, single employee)
- Verify probabilities sum to 1.0
- Verify heterogeneity requirements

### 8.2 Integration Test
- Run full pipeline on sample hidden state
- Verify output matches schema
- Verify probabilities are heterogeneous (std > 0.05 for support)
- Compare with previous LLM-only output

### 8.3 Performance Test
- Measure computation time for N=22, N=50, N=100
- Verify distance cache reduces complexity

---

## Phase 9: Documentation and Notebook Updates

### 9.1 Code Documentation
- Add docstrings to all new functions
- Document formula parameters (weights, temperature, decay)
- Explain graph interpretation

### 9.2 Notebook Updates
- Update `pipeline_demo.ipynb` to reflect agentic system
- Add visualization of computed probabilities vs. graph structure
- Show heterogeneity metrics

---

## Implementation Order

1. **Phase 1.1-1.3**: Graph utilities and baseline/influence computation (foundation)
2. **Phase 1.4-1.6**: Softmax, constraints, influence sources (core calculation)
3. **Phase 2**: Aggregation functions (department, level, aggregate)
4. **Phase 3**: Feature importance (correlation-based)
5. **Phase 4**: Orchestration function (tie everything together)
6. **Phase 5**: LLM integration (simplified prompts)
7. **Phase 6**: Validation (heterogeneity checks)
8. **Phase 7**: Error handling (edge cases)
9. **Phase 8**: Testing (unit + integration)
10. **Phase 9**: Documentation (docstrings + notebook)

---

## Success Criteria

1. ✅ Support probabilities have std > 0.05 (meaningful variance)
2. ✅ Probabilities correlate with graph structure (collaboration, distance, etc.)
3. ✅ Department aggregates show differences when collaboration differs
4. ✅ All probabilities sum to 1.0
5. ✅ Influence sources match actual graph edges
6. ✅ Computation is deterministic (same input → same output)
7. ✅ Performance: < 5 seconds for N=22, < 30 seconds for N=100

---

## Dependencies

- **numpy**: Array operations, softmax, correlation
- **scipy** or **networkx**: Shortest path computation
- **openai**: LLM for rationale (reduced token usage)

---

## Notes

- Keep existing schema structure (no breaking changes)
- Maintain backward compatibility with existing validation
- Use type hints for all functions
- Follow PEP8 style guide
- Add logging for computational steps (optional, for debugging)

