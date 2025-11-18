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
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

# Load environment variables
load_dotenv()

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
                    "sentiment": {"type": "string", "enum": ["oppose", "neutral", "support", "escalate"]},
                    "probabilities": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["oppose", "neutral", "support", "escalate"],
                        "properties": {
                            "oppose": {"type": "number", "minimum": 0.01, "maximum": 0.99},
                            "neutral": {"type": "number", "minimum": 0, "maximum": 1},
                            "support": {"type": "number", "minimum": 0.01, "maximum": 0.99},
                            "escalate": {"type": "number", "minimum": 0, "maximum": 1}
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
                        "support": {"type": "number", "minimum": 0, "maximum": 1},
                        "escalate": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["oppose", "neutral", "support", "escalate"]
                },
                "top_class": {"type": "string", "enum": ["oppose", "neutral", "support", "escalate"]}
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
                        "required": ["oppose", "neutral", "support", "escalate"],
                        "properties": {
                            "oppose": {"type": "number", "minimum": 0.01, "maximum": 0.99},
                            "neutral": {"type": "number", "minimum": 0, "maximum": 1},
                            "support": {"type": "number", "minimum": 0.01, "maximum": 0.99},
                            "escalate": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                },
                "by_level": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["oppose", "neutral", "support", "escalate"],
                        "properties": {
                            "oppose": {"type": "number", "minimum": 0.01, "maximum": 0.99},
                            "neutral": {"type": "number", "minimum": 0, "maximum": 1},
                            "support": {"type": "number", "minimum": 0.01, "maximum": 0.99},
                            "escalate": {"type": "number", "minimum": 0, "maximum": 1}
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
You are a computational sentiment analysis system. You MUST follow explicit formulas - do NOT use freeform reasoning.

You will receive an organizational hidden state JSON with:
- Recommendation: resource_need, urgency, domain
- Situation: sanction_strength, visibility (convert: "public"=1.0, "private"=0.5, "confidential"=0.2)
- Employees: level, department, tenure, manager_id
- Graphs: reports_to, collaboration, friendship, influence, conflict (weighted adjacency matrices)

COMPUTATIONAL STEPS (execute in order for each employee):

STEP 1: Compute Baseline Sentiment Score
For each employee i:
  resource_need = rec_seed.resource_need
  sanction_strength = situation_seed.sanction_strength
  visibility_factor = 1.0 if visibility=="public" else 0.5 if visibility=="private" else 0.2
  conflict_avg = mean(conflict[i][j] for all j where conflict[i][j] > 0)
  tenure = employees[i].tenure / 10.0  # normalize to 0-1
  
  baseline_score = (resource_need × 0.3) + (sanction_strength × visibility_factor) - (conflict_avg × 0.2) + (tenure × 0.05)

STEP 2: Apply Graph-Based Influence
For each employee i, compute influenced_score:
  influenced_score = baseline_score
  
  For each graph type (reports_to, collaboration, friendship, influence):
    For each employee j that influences i (where graph[j][i] > 0):
      collaboration_weight = collaboration[j][i]  # actual edge weight
      distance = shortest_path_length(j, i, reports_to)  # in reports_to graph
      influence_decay = 0.7^distance  # 1-hop=1.0, 2-hop=0.7, 3-hop=0.49, etc.
      
      # Get j's baseline_score (computed in STEP 1)
      influence_contribution = j.baseline_score × collaboration_weight × influence_decay
      
      # Weight by graph type
      if graph_type == "reports_to": weight_multiplier = 1.0
      elif graph_type == "influence": weight_multiplier = 0.8
      elif graph_type == "collaboration": weight_multiplier = 0.6
      elif graph_type == "friendship": weight_multiplier = 0.4
      elif graph_type == "conflict": weight_multiplier = -0.3  # negative for conflict
      
      influenced_score += influence_contribution × weight_multiplier

STEP 3: Map to Sentiment Classes and Apply Softmax
For each employee i:
  # Map influenced_score to 4 sentiment logits
  oppose_logit = -influenced_score × 0.5
  neutral_logit = influenced_score × 0.3
  support_logit = influenced_score × 1.0
  escalate_logit = influenced_score × 0.2
  
  # Apply softmax with temperature=0.5
  temperature = 0.5
  logits = [oppose_logit/temperature, neutral_logit/temperature, support_logit/temperature, escalate_logit/temperature]
  exp_logits = [exp(l) for l in logits]
  sum_exp = sum(exp_logits)
  probabilities = [e/sum_exp for e in exp_logits]
  
  oppose_prob = probabilities[0]
  neutral_prob = probabilities[1]
  support_prob = probabilities[2]
  escalate_prob = probabilities[3]

STEP 4: Apply Constraints and Rounding
  # Ensure oppose ≥ 0.05 when conflict_avg > 0.05
  if conflict_avg > 0.05 and oppose_prob < 0.05:
    oppose_prob = 0.05
    # Renormalize remaining probabilities
    remaining = 1.0 - oppose_prob
    neutral_prob = neutral_prob / (neutral_prob + support_prob + escalate_prob) × remaining
    support_prob = support_prob / (neutral_prob + support_prob + escalate_prob) × remaining
    escalate_prob = escalate_prob / (neutral_prob + support_prob + escalate_prob) × remaining
  
  # Round to 3 decimal places
  oppose_prob = round(oppose_prob, 3)
  neutral_prob = round(neutral_prob, 3)
  support_prob = round(support_prob, 3)
  escalate_prob = round(escalate_prob, 3)
  
  # Ensure sum = 1.0 (adjust largest if needed)
  total = oppose_prob + neutral_prob + support_prob + escalate_prob
  if abs(total - 1.0) > 0.001:
    diff = 1.0 - total
    # Add difference to largest probability
    max_prob = max(oppose_prob, neutral_prob, support_prob, escalate_prob)
    if max_prob == oppose_prob: oppose_prob += diff
    elif max_prob == neutral_prob: neutral_prob += diff
    elif max_prob == support_prob: support_prob += diff
    else: escalate_prob += diff

STEP 5: Department Aggregates (weighted by collaboration centrality)
For each department:
  dept_employees = [i for i in employees if employees[i].department == dept]
  
  # Compute collaboration centrality for each employee in department
  for i in dept_employees:
    centrality[i] = sum(collaboration[i][j] for j in dept_employees) / len(dept_employees)
  
  # Weighted average
  total_weight = sum(centrality[i] for i in dept_employees)
  dept_oppose = sum(oppose_prob[i] × centrality[i] for i in dept_employees) / total_weight
  dept_neutral = sum(neutral_prob[i] × centrality[i] for i in dept_employees) / total_weight
  dept_support = sum(support_prob[i] × centrality[i] for i in dept_employees) / total_weight
  dept_escalate = sum(escalate_prob[i] × centrality[i] for i in dept_employees) / total_weight

STEP 6: Level Aggregates (EXCLUDE C-Suite)
For each level (Director, Manager only - exclude C-Suite):
  level_employees = [i for i in employees if employees[i].level == level and employees[i].level != "C-Suite"]
  # Simple average (or weighted by tenure)
  level_oppose = mean(oppose_prob[i] for i in level_employees)
  level_neutral = mean(neutral_prob[i] for i in level_employees)
  level_support = mean(support_prob[i] for i in level_employees)
  level_escalate = mean(escalate_prob[i] for i in level_employees)

HETEROGENEITY REQUIREMENT:
Support probabilities MUST vary by ≥0.10 across departments when average collaboration differs by ≥0.20.
If departments A and B have collaboration_avg difference ≥0.20, then |support_A - support_B| ≥ 0.10.

OUTPUT FORMAT:
- Each employee: {employee_id, sentiment (class with highest prob), probabilities (oppose, neutral, support, escalate), influence_sources}
- influence_sources: list of {employee_id, graph_type, influence_weight} where influence_weight = actual graph edge value
- Round all probabilities to exactly 3 decimal places
- Ensure oppose ≥ 0.01 and support ≥ 0.01 (schema constraints)
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


def validate_forecast(forecast: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate forecast against JSON schema and business rules.
    
    Args:
        forecast: Forecast dictionary to validate
        
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
    
    return True, None


def forecast_scenario(
    hidden_state: dict,
    scenario_id: str,
    model: str = "gpt-4o-mini",
    prompt_template_id: str = "v3",
    horizon: str = "decision",
    provider: str = "openai"
) -> dict:
    """
    Generate deterministic forecast from hidden state using OpenAI Structured Outputs.
    
    Args:
        hidden_state: The organizational state JSON (see sample_0.json)
        scenario_id: Unique identifier for this scenario
        model: OpenAI model name
        prompt_template_id: Version identifier for the prompt template
        horizon: Forecast horizon ("decision", "next_cycle", or "quarter")
        provider: API provider ("openai" or "azure_openai")
        
    Returns:
        Validated forecast object matching the schema, or error object if validation fails
    """
    # Initialize OpenAI client
    try:
        client = openai.OpenAI()
        if not client.api_key:
            raise openai.OpenAIError("API key not found after loading .env")
    except Exception as e:
        return {
            "scenario_id": scenario_id,
            "state_hash": compute_state_hash(hidden_state),
            "error": True,
            "error_message": f"OpenAI client initialization failed: {str(e)}"
        }
    
    # Compute state hash
    state_hash = compute_state_hash(hidden_state)
    
    # Extract segments for prompt context
    segments = extract_segments(hidden_state)
    
    # Build user prompt focused on individual sentiment propagation with graph analysis
    num_employees = hidden_state.get('num_employees', 0)
    employees = hidden_state.get('employees', [])
    graphs = hidden_state.get('graphs', {})
    
    # Identify C-Suite and Directors for distance calculations
    csuite_ids = [i for i, emp in enumerate(employees) if emp.get('level') == 'C-Suite']
    director_ids = [i for i, emp in enumerate(employees) if emp.get('level') == 'Director']
    
    user_prompt = f"""Model sentiment propagation through this organizational network using graph-theoretic analysis.

Organizational Context:
- Industry: {hidden_state.get('org_seed', {}).get('industry', 'unknown')}
- Size: {hidden_state.get('org_seed', {}).get('size', 'unknown')}
- Power Distance: {hidden_state.get('org_seed', {}).get('power_distance', 0):.3f} (affects hierarchy influence strength)
- Sanction Salience: {hidden_state.get('org_seed', {}).get('sanction_salience', 0):.3f} (affects risk-taking behavior)
- In-Group Bias: {hidden_state.get('org_seed', {}).get('in_group_bias', 0):.3f} (affects department cohesion)

Recommendation:
- Domain: {hidden_state.get('rec_seed', {}).get('domain', 'unknown')} (affects departmental alignment)
- Urgency: {hidden_state.get('rec_seed', {}).get('urgency', 0):.3f}
- Resource Need: {hidden_state.get('rec_seed', {}).get('resource_need', 0):.3f}

Situation:
- Current State: {hidden_state.get('situation_seed', {}).get('theta_current', 0):.3f}
- Visibility: {hidden_state.get('situation_seed', {}).get('visibility', 'unknown')}
- Sanction Strength: {hidden_state.get('situation_seed', {}).get('sanction_strength', 0):.3f}

Organization Structure:
- Total Employees: {num_employees}
- Departments: {', '.join(segments['departments'])}
- Levels: {', '.join(segments['levels'])}
- Key Influencers: C-Suite (IDs: {csuite_ids if csuite_ids else 'none'}), Directors (IDs: {director_ids if director_ids else 'none'})

GRAPH STRUCTURE ANALYSIS REQUIRED:

The graphs are WEIGHTED adjacency matrices (N×N arrays) where each value represents edge strength (0.0 to 1.0).

Graph Types and Interpretation:
1. **reports_to**: Binary adjacency (0 or 1) - Formal hierarchy. Employee i reports to j if reports_to[i][j] = 1.
   - STRONGEST influence type - formal authority
   - Use actual values (0 or 1) for influence weights

2. **collaboration**: Continuous weights (0.0-1.0) - Work relationship strength.
   - Higher values (e.g., 0.8) = strong collaboration, more influence
   - Lower values (e.g., 0.2) = weak collaboration, less influence
   - Use actual edge weight values from the matrix

3. **friendship**: Continuous weights (0.0-1.0) - Social bond strength.
   - Higher values = stronger friendships, more social influence
   - Use actual edge weight values

4. **influence**: Continuous weights (0.0-1.0) - Power/influence network.
   - Computed from hierarchy + social graphs + level influence
   - Higher values = greater informal power
   - Use actual edge weight values

5. **conflict**: Continuous weights (0.0-1.0) - Tension/opposition strength.
   - Higher values = stronger conflicts, can oppose sentiment
   - Use actual edge weight values

FOR EACH EMPLOYEE (IDs 0-{num_employees-1}), you MUST:

1. **Compute Network Metrics**:
   - For each graph type, calculate weighted in-degree: sum of incoming edge weights
   - Calculate weighted out-degree: sum of outgoing edge weights
   - Identify top 3-5 strongest connections (highest edge weights) in each graph type
   - Compute graph distance (shortest path length) from C-Suite and Directors in reports_to graph
   - Measure clustering: how many connections exist between this employee's neighbors

2. **Analyze Edge Weights**:
   - Extract actual edge weights from adjacency matrices (not just 0/1 presence)
   - For influence_sources, use the ACTUAL edge weight from the graph matrix
   - Weight by graph type hierarchy: reports_to (×1.0) > influence (×0.8) > collaboration (×0.6) > friendship (×0.4) > conflict (×0.3, but can oppose)
   - Apply distance decay: 1-hop connections (×1.0), 2-hop (×0.7), 3-hop (×0.5), 4+ hop (×0.3)

3. **Predict Individual Sentiment**:
   - Generate probabilities with 3-4 DECIMAL PRECISION (e.g., 0.127, 0.683, 0.156, 0.034)
   - DO NOT round to multiples of 0.05 (avoid patterns like 0.05, 0.10, 0.75, 0.10)
   - Each employee MUST have UNIQUE probabilities based on:
     * Their specific weighted degree in each graph type
     * Actual edge weights to their influencers
     * Graph distance from key decision-makers
     * Network position (central vs. peripheral)
     * Individual attributes (level, department, tenure) interacting with network metrics
   - Probabilities must sum to exactly 1.0

4. **Identify Influence Sources**:
   - List top 3-5 influencers per employee
   - For each influencer, provide:
     * employee_id: The influencing employee's ID
     * graph_type: Which graph this influence flows through
     * influence_weight: ACTUAL edge weight from the graph matrix (not arbitrary 0.5 or 1.0)
       - If reports_to[i][j] = 1, use 1.0 (or scale by graph type weight)
       - If collaboration[i][j] = 0.736, use 0.736 (or scale appropriately)
   - Consider multiple graph types: an employee may influence through both reports_to AND collaboration

5. **Trace Propagation Paths**:
   - Show shortest weighted paths from key influencers (C-Suite, Directors) to each employee
   - Consider paths through different graph types (formal hierarchy paths vs. informal network paths)
   - Path should be sequence of employee IDs: [source, intermediate1, intermediate2, ..., target]

6. **Aggregate Outcomes**:
   - After computing all individual sentiments with unique probabilities, aggregate to organizational level
   - Use weighted average or other appropriate aggregation method
   - Aggregate probabilities must sum to exactly 1.0

Full Hidden State (including all weighted graph matrices):
{json.dumps(hidden_state, indent=2)}

CRITICAL: Analyze the ACTUAL edge weights in the graphs. Do not use generic values. Each employee's probabilities should reflect their unique network position and connection strengths.

Horizon: {horizon}
"""
    
    # Attempt forecast generation with retry (including rate limit handling)
    max_retries = 3
    base_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            # Use structured outputs with JSON schema
            # Note: This requires OpenAI API version that supports json_schema response_format
            response = client.chat.completions.create(
                model=model,
                temperature=0,  # Must be 0 for determinism
                messages=[
                    {"role": "system", "content": FORECAST_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "forecast_output",
                        "strict": False,  # Set to False to allow optional fields like segments
                        "schema": FORECAST_SCHEMA,
                        "description": "Organizational forecast with individual sentiment propagation (based on graph-theoretic analysis of weighted adjacency matrices) and aggregate outcomes. Probabilities should use 3-4 decimal precision."
                    }
                }
            )
            
            # Parse response
            forecast_json = json.loads(response.choices[0].message.content)
            
            # Strip any unexpected 'type' fields that OpenAI might add
            # (OpenAI's structured outputs sometimes adds metadata 'type' fields that violate additionalProperties: False)
            # Since 'type' is not part of our data schema, we remove all 'type' fields from the response
            def strip_type_fields(obj):
                """Recursively remove 'type' fields from the response data."""
                if isinstance(obj, dict):
                    # Remove 'type' field if present (it's not in our data schema)
                    if 'type' in obj:
                        obj.pop('type', None)
                    # Recursively process nested structures
                    for value in obj.values():
                        if isinstance(value, (dict, list)):
                            strip_type_fields(value)
                elif isinstance(obj, list):
                    for item in obj:
                        strip_type_fields(item)
                return obj
            
            forecast_json = strip_type_fields(forecast_json)
            
            # Add required metadata
            forecast_json["scenario_id"] = scenario_id
            forecast_json["state_hash"] = state_hash
            forecast_json["generated_at"] = datetime.now(timezone.utc).isoformat()
            forecast_json["schema_version"] = "1.0"
            forecast_json["horizon"] = horizon
            # Ensure model object has all required fields
            if "model" not in forecast_json:
                forecast_json["model"] = {}
            forecast_json["model"]["provider"] = provider
            forecast_json["model"]["model"] = model
            forecast_json["model"]["temperature"] = 0
            forecast_json["model"]["prompt_template_id"] = prompt_template_id
            
            # Auto-correct individual sentiments
            if "individual_sentiments" in forecast_json:
                for sentiment in forecast_json["individual_sentiments"]:
                    if "probabilities" in sentiment:
                        probs = sentiment["probabilities"]
                        # Normalize probabilities if needed
                        prob_sum = sum(probs.values())
                        if abs(prob_sum - 1.0) > 1e-6 and prob_sum > 0:
                            for key in probs:
                                probs[key] = probs[key] / prob_sum
                        
                        # Auto-correct sentiment to match highest probability
                        if probs:
                            max_class = max(probs.items(), key=lambda x: x[1])[0]
                            sentiment["sentiment"] = max_class
            
            # Auto-correct aggregate outcomes
            if "aggregate_outcomes" in forecast_json:
                if "probabilities" in forecast_json["aggregate_outcomes"]:
                    agg_probs = forecast_json["aggregate_outcomes"]["probabilities"]
                    # Normalize probabilities if needed
                    prob_sum = sum(agg_probs.values())
                    if abs(prob_sum - 1.0) > 1e-6 and prob_sum > 0:
                        for key in agg_probs:
                            agg_probs[key] = agg_probs[key] / prob_sum
                    
                    # Auto-correct top_class to match highest probability
                    if agg_probs:
                        max_class = max(agg_probs.items(), key=lambda x: x[1])[0]
                        forecast_json["aggregate_outcomes"]["top_class"] = max_class
            
            # Validate forecast
            is_valid, error_msg = validate_forecast(forecast_json)
            
            if is_valid:
                return forecast_json
            else:
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    return {
                        "scenario_id": scenario_id,
                        "state_hash": state_hash,
                        "error": True,
                        "error_message": f"Validation failed after {max_retries} attempts: {error_msg}"
                    }
        
        except openai.RateLimitError as e:
            # Handle rate limit errors with exponential backoff
            error_msg = str(e)
            
            # Try to extract wait time from error message
            wait_time = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            
            # Look for "try again in X.Xs" in error message
            wait_match = re.search(r'try again in ([\d.]+)s', error_msg, re.IGNORECASE)
            if wait_match:
                wait_time = float(wait_match.group(1)) + 1  # Add 1 second buffer
            
            if attempt < max_retries - 1:
                print(f"⚠ Rate limit hit. Waiting {wait_time:.1f} seconds before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait_time)
                continue  # Retry
            else:
                return {
                    "scenario_id": scenario_id,
                    "state_hash": state_hash,
                    "error": True,
                    "error_message": f"Rate limit exceeded after {max_retries} attempts. Please wait a few minutes and try again, or upgrade your OpenAI plan. Error: {error_msg}"
                }
        
        except openai.APIError as e:
            # Handle other API errors
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)
                print(f"⚠ API error. Waiting {wait_time:.1f} seconds before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait_time)
                continue  # Retry
            else:
                return {
                    "scenario_id": scenario_id,
                    "state_hash": state_hash,
                    "error": True,
                    "error_message": f"API error after {max_retries} attempts: {str(e)}"
                }
        
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                continue  # Retry
            else:
                return {
                    "scenario_id": scenario_id,
                    "state_hash": state_hash,
                    "error": True,
                    "error_message": f"Invalid JSON response after {max_retries} attempts: {str(e)}"
                }
        
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)
                print(f"⚠ Error occurred. Waiting {wait_time:.1f} seconds before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait_time)
                continue  # Retry
            else:
                return {
                    "scenario_id": scenario_id,
                    "state_hash": state_hash,
                    "error": True,
                    "error_message": f"Unexpected error after {max_retries} attempts: {str(e)}"
                }
    
    # Should not reach here, but just in case
    return {
        "scenario_id": scenario_id,
        "state_hash": state_hash,
        "error": True,
        "error_message": "Unexpected error in forecast generation"
    }


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
