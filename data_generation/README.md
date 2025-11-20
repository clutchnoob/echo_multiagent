# Data Generation Pipeline

This directory contains the **agentic** synthetic data generation pipeline for the ECHO multiagent learning project. The pipeline creates labeled training datasets by generating synthetic organizations and using a **hybrid approach**: deterministic Python-based computation for sentiment forecasting, with LLM-generated rationales.

## Architecture Overview

The system uses an **agentic architecture** where:
- **Python Functions**: Perform all deterministic calculations (baseline scores, graph influence propagation, softmax mapping, aggregation)
- **LLM**: Generates qualitative rationales only, using pre-computed results as context
- **Result**: Deterministic, heterogeneous probabilities that reflect actual graph structure and organizational dynamics

This approach ensures reproducibility and eliminates the homogeneity issues of pure LLM-based forecasting.

## Directory Structure

```
data_generation/
├── hidden_state_generation.py    # Organization modeling and graph generation
├── encoder_layer.py              # Narrative synthesis + scenario modifier extraction
├── forward_forecaster.py         # Deterministic forecasting engine + LLM rationale
├── visualize_results.py          # Analysis and visualization tools
├── pipeline_demo.ipynb           # Complete workflow demonstration
├── sample_hidden_states/         # Example synthetic organizations
│   ├── demo_state.json
│   └── sample_0.json
├── sample_forecasts/             # Example labeled training data
│   └── demo_forecast.json
└── sample_encoder_input/         # Example narrative encodings
    └── demo_encoded.json
```

## Module Overview

### 1. `hidden_state_generation.py`

**Purpose**: Generates synthetic organizations with realistic multi-graph network structures and individual personality traits.

**Key Components**:

- **Seed Functions**: Sample random organizational, recommendation, and situational parameters
  - `sample_org_seed()`: Industry, size, cultural parameters (`power_distance`, `sanction_salience`, `in_group_bias`, `past_change_success`)
  - `sample_rec_seed()`: Recommendation domain, urgency, resource needs, theta alignment
  - `sample_situation_seed()`: Current state, visibility, sanctions, provocation flags

- **`OrganizationHiddenState` Class**: Core organization generator
  - Automatically generates employee roster across hierarchical levels (C-Suite, Director, Manager)
  - Creates 5 weighted adjacency matrices:
    - `reports_to`: Hierarchical reporting structure (binary)
    - `collaboration`: Working relationships (weighted)
    - `friendship`: Social bonds (binary, derived from collaboration)
    - `influence`: Power networks using PageRank (weighted)
    - `conflict`: Tensions and competition (weighted)
  - **Individual Traits**: Each employee has:
    - `sanction_salience`: Individual sensitivity to sanctions (Beta distribution around org mean)
    - `in_group_bias`: Department loyalty (Beta distribution around org mean)
    - `openness`: Receptivity to change (Beta distribution, mean=0.5)
    - `performance`: Performance rating (Normal distribution, mean=0.6, std=0.2)
  - Exports to canonical JSON format

**Example Usage**:

```python
from hidden_state_generation import (
    sample_org_seed, sample_rec_seed, sample_situation_seed,
    OrganizationHiddenState
)

# Generate seeds
org_seed = sample_org_seed(seed=42)
rec_seed = sample_rec_seed(seed=43)
sit_seed = sample_situation_seed(seed=44)

# Create organization
org = OrganizationHiddenState(
    org_seed=org_seed,
    rec_seed=rec_seed,
    situation_seed=sit_seed,
    departments=['Engineering', 'Sales', 'Marketing', 'HR'],
    avg_span_of_control=5
)

# Access employee roster with individual traits
print(org.employees[['employee_id', 'level', 'department', 'performance', 'openness']])

# Save to file
with open('my_org.json', 'w') as f:
    f.write(org.to_json_encoding())
```

**Key Methods**:
- `_generate_roster()`: Creates employees with individual personality traits
- `_generate_reports_to_graph()`: Creates hierarchical reporting structure
- `_generate_collaboration_graph()`: Models working relationships with edge weights
- `_generate_friendship_graph()`: Derives social bonds from collaboration and demographics
- `_generate_influence_graph()`: Computes power networks using PageRank
- `_generate_conflict_graph()`: Identifies tensions based on competition and distance
- `to_json_encoding()`: Exports complete state to JSON

---

### 2. `encoder_layer.py`

**Purpose**: Converts structured organizational JSON into natural language narratives and extracts scenario-specific modifiers.

**Key Functions**:

- `create_encoded_narrative(hidden_state_data: dict) -> dict`: Main orchestration function
  - Generates company story from organizational context
  - Analyzes key relationships using graph structure
  - Creates recommendation scenario narrative
  - **Extracts scenario modifiers**: Uses LLM to extract `department_modifiers` and `level_modifiers` (numerical adjustments -0.3 to +0.3) based on narrative context
  - Returns structured JSON with narrative text and numerical modifiers

**Output Structure**:
```json
{
  "company_story": "Narrative about industry, size, culture...",
  "company_profile": "Structured company information...",
  "key_relationships": "Analysis of influencers, collaborations, conflicts...",
  "recommendation_scenario": "Summary of the recommendation and its context...",
  "scenario_modifiers": {
    "department_modifiers": {"Engineering": 0.15, "Sales": -0.1, ...},
    "level_modifiers": {"C-Suite": 0.0, "Director": 0.05, ...}
  }
}
```

**Example Usage**:

```python
from encoder_layer import create_encoded_narrative
import json

# Load hidden state
with open('my_org.json', 'r') as f:
    hidden_state = json.load(f)

# Generate narrative with modifiers
narrative_data = create_encoded_narrative(hidden_state)

# The modifiers are fused into the forecasting computation
# (see forward_forecaster.py for how they're used)
```

**Note**: The `scenario_modifiers` are extracted from the narrative but **not** included in the LLM prompt for rationale generation (to avoid data leakage). They are used only in the deterministic Python computation.

---

### 3. `forward_forecaster.py`

**Purpose**: Core deterministic forecasting engine that computes sentiment propagation using graph theory and organizational psychology, with LLM-generated rationales.

**Architecture**: 
- **Deterministic Computation**: All probabilities computed in Python using NumPy/NetworkX
- **LLM Integration**: GPT-4o-mini generates qualitative rationales from pre-computed results
- **Reproducibility**: Temperature=0, deterministic seeds, state hashing

**Key Computational Functions**:

1. **`compute_baseline_scores(hidden_state, scenario_modifiers)`**
   - Calculates initial sentiment scores for each employee
   - Factors include:
     - Individual traits: `openness`, `performance`, `sanction_salience`, `in_group_bias`
     - Organizational context: `past_change_success` (cynicism), `industry` risk profiles
     - Recommendation alignment: `theta_ideal - theta_current`
     - Urgency, resource needs, conflict, tenure, department alignment
   - Returns: Array of baseline scores (N,)

2. **`compute_influenced_scores(baseline_scores, hidden_state, distance_cache)`**
   - Applies graph-based influence propagation
   - Features:
     - **Complex Contagion**: Skeptics require multiple strong supporters to shift opinion
     - **Opinion Leaders**: High PageRank employees have amplified influence (up to 1.5x)
     - **Performance-based Resistance**: High performers resist peer pressure
     - **Distance Decay**: Influence decays as `0.5^distance` in hierarchy
     - **Normalization**: Prevents influence from overwhelming baseline
   - Graph types: `reports_to`, `influence`, `collaboration`, `friendship`, `conflict`
   - Returns: Array of influenced scores (N,)

3. **`apply_softmax_mapping(influenced_scores, hidden_state, base_temperature=1.0)`**
   - Maps continuous scores to probability distributions using level-dependent temperature
   - **Level-Based Variance**: Higher levels have more variance (strong opinions, disagreement)
     - C-Suite: temperature = 1.8x (high variance - executives have divergent opinions)
     - Directors: temperature = 1.3x (medium-high variance - VPs disagree)
     - Managers: temperature = 0.6x (low variance - lower levels are more uniform/neutral)
   - Three sentiment classes: `oppose`, `neutral`, `support`
   - Returns: Array of probabilities (N, 3)

4. **`apply_constraints_and_rounding(probabilities, hidden_state)`**
   - **Public vs. Private Logic**: Implements "Organizational Silence"
     - Calculates `safety_score` based on `sanction_strength`, `visibility`, `performance`, `tenure`, `level`
     - **Gradual Suppression**: If `safety_score < 0.5`, suppresses "Oppose" probabilities
     - Suppression increases smoothly from 0% (at safety=0.5) to 80% (at safety=0.0)
     - Shifts suppressed probability to "Neutral" (silence) and "Support" (compliance)
   - Business rules: Conflict-based constraints
   - Returns: Normalized and rounded probabilities (N, 3)

5. **`compute_individual_sentiments(hidden_state, narrative_context)`**
   - Orchestrates the full computation pipeline
   - Extracts `scenario_modifiers` from `narrative_context` and fuses into baseline
   - Returns: List of individual sentiment dicts matching schema

6. **`forecast_scenario(hidden_state, scenario_id, model, narrative_context, ...)`**
   - Main entry point for generating forecasts
   - Runs deterministic computation, then generates LLM rationale
   - Returns: Complete forecast JSON with probabilities and rationale

**Example Usage**:

```python
from forward_forecaster import forecast_scenario
import json

# Load hidden state
with open('my_org.json', 'r') as f:
    hidden_state = json.load(f)

# Optional: Load narrative context (includes scenario_modifiers)
with open('narrative.json', 'r') as f:
    narrative_context = json.load(f)

# Generate forecast
forecast = forecast_scenario(
    hidden_state=hidden_state,
    scenario_id='demo_scenario',
    model='gpt-4o-mini',
    horizon='decision',
    narrative_context=narrative_context  # Optional: fuses modifiers into computation
)

# Access results
for sentiment in forecast['individual_sentiments']:
    print(f"Employee {sentiment['employee_id']}: {sentiment['sentiment']} "
          f"(prob: {sentiment['probabilities'][sentiment['sentiment']]:.2f})")

print(f"Rationale: {forecast['rationale']}")
```

**Output Structure**:
```json
{
  "scenario_id": "demo_scenario",
  "state_hash": "sha256:...",
  "generated_at": "2025-11-19T...",
  "schema_version": "1.0",
  "horizon": "decision",
  "model": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0
  },
  "individual_sentiments": [
    {
      "employee_id": 0,
      "sentiment": "support",
      "probabilities": {
        "oppose": 0.191,
        "neutral": 0.316,
        "support": 0.493
      },
      "influence_sources": [...],
      "propagation_path": [0]
    }
  ],
  "aggregate_outcomes": {
    "probabilities": {"oppose": 0.083, "neutral": 0.216, "support": 0.510},
    "top_class": "support"
  },
  "note": "aggregate_outcomes reflects CEO's sentiment, not company average",
  "segments": {
    "by_department": {...},
    "by_level": {...}
  },
  "rationale": "LLM-generated explanation of the results...",
  "features_importance": [...]
}
```

**Key Features**:
- **Deterministic**: Same inputs → same outputs (reproducible)
- **Heterogeneous**: Probabilities reflect actual graph structure and individual differences
- **Realistic**: Models complex contagion, organizational silence, opinion leaders
- **Validated**: Automatic schema validation and probability normalization

---

### 4. `visualize_results.py`

**Purpose**: Analysis and visualization tools for understanding forecast dynamics.

**Key Functions**:

- **`analyze_public_private_divergence(hidden_state, forecast)`**
  - Creates scatter plots showing the gap between psychological safety and public expression
  - **Left Plot**: Safety Score vs. Public Opposition (shows suppression effect)
  - **Right Plot**: Safety Score vs. Public Compliance (shows silence/compliance)
  - Identifies "silenced" employees (low safety + high compliance)
  - Point size represents `performance` rating
  - Returns: DataFrame with analysis data

- **`plot_influence_network(hidden_state, forecast)`**
  - Visualizes organizational hierarchy as a top-down tree
  - **Layout**: C-Suite (top) → Directors (middle) → Managers (bottom)
  - **Node Colors**: Green=Support, Red=Oppose, Grey=Neutral
  - **Edge Colors**: Colored by source node's sentiment (shows influence flow)
  - **Labels**: Employee ID and department abbreviation
  - Shows how sentiment propagates through the hierarchy

**Example Usage**:

```python
from visualize_results import analyze_public_private_divergence, plot_influence_network

# Analyze safety vs. expression
df_analysis = analyze_public_private_divergence(hidden_state, forecast)

# Visualize network
plot_influence_network(hidden_state, forecast)
```

---

### 5. `pipeline_demo.ipynb`

**Purpose**: Interactive Jupyter notebook demonstrating the complete data generation workflow.

**Sections**:
1. **Setup**: Import libraries and configure environment
2. **Generate Hidden State**: Create synthetic organization with custom parameters
3. **Encode Narrative (Optional)**: Generate narrative context and scenario modifiers
4. **Generate Forecast**: Run deterministic computation with LLM rationale
5. **Analyze & Visualize**: Explore Public vs. Private sentiment gaps and influence networks

**How to Use**:
```bash
jupyter notebook pipeline_demo.ipynb
```

Run cells sequentially to see the complete pipeline in action.

---

## Quick Start Guide

### 1. Install Dependencies

```bash
pip install openai numpy pandas networkx matplotlib seaborn python-dotenv jsonschema scipy jupyter
```

### 2. Set Up API Key

Create a `.env` file in the project root:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 3. Generate Training Data

**Option A: Use the notebook** (recommended for first-time users)
```bash
jupyter notebook pipeline_demo.ipynb
```

**Option B: Use Python scripts**
```python
from hidden_state_generation import *
from forward_forecaster import forecast_scenario
from encoder_layer import create_encoded_narrative
import json

# Generate organization
org = OrganizationHiddenState(
    org_seed=sample_org_seed(seed=1),
    rec_seed=sample_rec_seed(seed=2),
    situation_seed=sample_situation_seed(seed=3)
)
hidden_state = json.loads(org.to_json_encoding())

# Optional: Generate narrative
narrative = create_encoded_narrative(hidden_state)

# Generate forecast (deterministic computation + LLM rationale)
labeled_data = forecast_scenario(
    hidden_state=hidden_state,
    scenario_id='train_001',
    model='gpt-4o-mini',
    narrative_context=narrative  # Optional: fuses modifiers
)

# Save
with open('training_data/example_001.json', 'w') as f:
    json.dump(labeled_data, f, indent=2)
```

---

## Key Features & Capabilities

### Organizational Psychology Modeling

- **Individual Heterogeneity**: Each employee has unique `openness`, `performance`, `sanction_salience`, `in_group_bias` (sampled from distributions)
- **Organizational Cynicism**: `past_change_success` creates global resistance to change
- **Industry Risk Profiles**: Different industries have different baseline risk tolerance
- **Theta Alignment** (Research-Based Asymmetric Model):
  - **Loss Aversion** (Kahneman & Tversky): Misalignment weighted 2.7x stronger than alignment (0.8 vs 0.3)
  - **Tribal Amplification** (Social Identity Theory): High `in_group_bias` amplifies resistance to misalignment by up to 1.5x
  - **Hierarchical Resistance**: Managers show strongest resistance (1.3x), Directors moderate (1.1x), C-Suite least (0.9x)
  - **Tenure Sunk Costs**: Long-tenured employees more invested in status quo (up to 1.4x for 10 years)
  - **Urgency Dampening**: Time pressure has diminishing returns when proposals conflict with core beliefs

### Social Influence Dynamics

- **Complex Contagion**: Skeptics require multiple strong supporters (threshold model)
- **Opinion Leaders**: High PageRank employees have amplified influence
- **Performance-based Resistance**: High performers resist peer pressure (idiosyncrasy credits)
- **Distance Decay**: Influence weakens with hierarchical distance (`0.5^distance`)
- **Symmetric Softmax Mapping**: No built-in bias toward support; opposition and support are equally likely for neutral scores (reflects research on status quo stability)
- **Level-Dependent Variance**: Organizational hierarchy affects opinion variance
  - **C-Suite** (temperature=1.8x): High variance - executives have strong, divergent opinions (VPs disagree a lot)
  - **Directors** (temperature=1.3x): Medium-high variance - some disagreement among VPs
  - **Managers** (temperature=0.6x): Low variance - lower levels are more uniform and less opinionated on strategic matters

### Public vs. Private Sentiment

- **Safety Score**: Calculated from `sanction_strength`, `visibility`, `performance`, `tenure`, `level`
- **Gradual Suppression**: Smooth transition from no suppression (safety=0.5) to maximum suppression (safety=0.0)
- **Organizational Silence**: Low-safety employees suppress "Oppose" → appear "Neutral" or "Support"
- **Compliance Modeling**: Distinguishes between genuine support and feigned compliance

### Computational Fusion

- **Narrative Modifiers**: LLM-extracted `department_modifiers` and `level_modifiers` are fused into baseline computation
- **No Data Leakage**: Numerical modifiers are excluded from LLM rationale prompts
- **Hybrid Approach**: Best of both worlds (deterministic computation + narrative context)

---

## Configuration Options

### Organization Parameters

**Industry Types**:
- `tech`, `finance`, `healthcare`, `manufacturing`, `government`

**Size Categories**:
- `small` (5-10 employees)
- `mid` (10-15 employees)
- `large` (15-25 employees)
- `enterprise` (25-30 employees)

**Cultural Parameters** (0-1 continuous):
- `power_distance`: Hierarchy acceptance (high = more top-down influence)
- `sanction_salience`: Organizational fear of negative consequences (high = more risk-averse)
- `in_group_bias`: Department loyalty (high = more within-group clustering)
- `past_change_success`: Historical change success rate (low = high cynicism)

**Individual Employee Traits** (sampled per employee):
- `sanction_salience`: Individual sensitivity (Beta distribution around org mean)
- `in_group_bias`: Individual department loyalty (Beta distribution around org mean)
- `openness`: Receptivity to change (Beta distribution, mean=0.5)
- `performance`: Performance rating (Normal distribution, mean=0.6, std=0.2)

### Recommendation Parameters

**Domains**:
- `product_roadmap`, `hiring`, `budget`, `compliance`, `pricing`, `market_entry`, `vendor_selection`

**Continuous Parameters** (0-1):
- `urgency`: Time pressure for decision
- `resource_need`: Resource intensity of recommendation
- `theta_ideal`: Optimal policy position

### Situation Parameters

- `theta_current`: Current organizational position (0-1)
- `visibility`: Information transparency (`private`, `public`)
- `sanction_strength`: Consequence severity (0-1)
- `provocation_flag`: Recent destabilizing events (0 or 1)

### Model Parameters

**Models**:
- `gpt-4o-mini`: Faster, cheaper, good quality (recommended for bulk generation)
- `gpt-4o`: Higher quality, more expensive

**Horizons**:
- `decision`: Immediate decision response
- `next_cycle`: Next planning cycle
- `quarter`: Next quarterly review

---

## Output Files

### Hidden State JSON Structure

```json
{
  "num_employees": 22,
  "org_seed": {
    "industry": "tech",
    "size": "large",
    "power_distance": 0.054,
    "sanction_salience": 0.220,
    "in_group_bias": 0.184,
    "past_change_success": 0.623
  },
  "rec_seed": {
    "domain": "budget",
    "urgency": 0.124,
    "resource_need": 0.755,
    "theta_ideal": 0.448
  },
  "situation_seed": {
    "theta_current": 0.092,
    "visibility": "private",
    "sanction_strength": 0.632,
    "provocation_flag": 0
  },
  "employees": [
    {
      "employee_id": 0,
      "level": "C-Suite",
      "department": "Engineering",
      "tenure": 3,
      "sanction_salience": 0.966,
      "in_group_bias": 0.291,
      "openness": 0.523,
      "performance": 0.742,
      "manager_id": -1
    }
  ],
  "graphs": {
    "reports_to": [[N×N binary matrix]],
    "collaboration": [[N×N weighted matrix]],
    "friendship": [[N×N binary matrix]],
    "influence": [[N×N weighted matrix]],
    "conflict": [[N×N weighted matrix]]
  }
}
```

### Forecast JSON Structure

See the `forward_forecaster.py` section above for complete structure. Key fields:
- `individual_sentiments`: Individual-level labels with probabilities
- `influence_sources`: Top influencers for each employee
- `propagation_path`: Sentiment flow sequence from leadership
- `aggregate_outcomes`: Organization-wide label distribution
- `rationale`: LLM-generated explanation
- `features_importance`: Correlation-based feature analysis

---

## Troubleshooting

### Common Issues

**1. OpenAI API Key Not Found**
```
Error: OpenAI API key not found after loading .env
```
**Solution**: Create a `.env` file in the project root with your API key:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

**2. Import Errors**
```
ModuleNotFoundError: No module named 'networkx'
```
**Solution**: Install missing dependencies:
```bash
pip install networkx pandas numpy matplotlib seaborn openai python-dotenv jsonschema scipy
```

**3. Validation Errors**
```
Validation failed: Rationale too long
```
**Solution**: The system automatically truncates long rationales. Check the `warnings` field in output.

**4. Low Variance in Probabilities**
If you see homogeneous probabilities, check:
- Are individual traits (`openness`, `performance`) being generated correctly?
- Is the influence propagation too strong? (Try reducing graph multipliers)
- Is the baseline score formula too dominated by global factors?

---

## Best Practices

### Generating Training Datasets

1. **Use Diverse Seeds**: Vary seeds to create diverse organizational structures
2. **Balance Scenarios**: Generate examples across different industries, sizes, and cultures
3. **Validate Labels**: Check `warnings` field and probability distributions
4. **Track State Hashes**: Use state hashes to avoid duplicate examples
5. **Save Metadata**: Keep track of generation parameters for reproducibility

### Example: Batch Generation

```python
import json
from hidden_state_generation import *
from forward_forecaster import forecast_scenario
from encoder_layer import create_encoded_narrative

# Generate 100 diverse training examples
training_data = []
for i in range(100):
    # Vary seeds for diversity
    org = OrganizationHiddenState(
        org_seed=sample_org_seed(seed=i * 3),
        rec_seed=sample_rec_seed(seed=i * 3 + 1),
        situation_seed=sample_situation_seed(seed=i * 3 + 2)
    )
    hidden_state = json.loads(org.to_json_encoding())
    
    # Optional: Generate narrative
    narrative = create_encoded_narrative(hidden_state)
    
    # Generate labels (deterministic computation)
    labeled = forecast_scenario(
        hidden_state=hidden_state,
        scenario_id=f'train_{i:04d}',
        model='gpt-4o-mini',
        narrative_context=narrative
    )
    
    # Save
    training_data.append(labeled)
    print(f"Generated {i+1}/100: {labeled['state_hash'][:16]}...")

# Save dataset
with open('training_dataset.json', 'w') as f:
    json.dump(training_data, f, indent=2)
```

---

## Performance Considerations

### Generation Speed

- **Organization Generation**: ~0.1 seconds per organization
- **Narrative Encoding**: ~10-20 seconds (LLM calls)
- **Forecast Computation**: ~0.1 seconds (deterministic Python)
- **LLM Rationale Generation**: ~2-5 seconds (gpt-4o-mini)
- **Total per example**: ~12-25 seconds

### Cost Estimates (OpenAI API)

For a 20-employee organization:
- **Narrative Encoding**: ~3,000 input tokens, ~1,500 output tokens
- **Rationale Generation**: ~4,000 input tokens, ~500 output tokens

**Cost per example** (approximate):
- `gpt-4o-mini`: $0.003 - $0.006
- `gpt-4o`: $0.04 - $0.08

**For 1,000 training examples**:
- `gpt-4o-mini`: $3-6
- `gpt-4o`: $40-80

### Optimization Tips

1. Use `gpt-4o-mini` for bulk generation
2. Skip narrative encoding if you don't need scenario modifiers
3. Cache generated states and reuse for different recommendations
4. Parallelize generation across multiple API keys (if available)
5. Use smaller organizations (10-15 employees) for faster iteration

---

## Development Notes

### Code Style

- Type hints used where appropriate
- Docstrings follow NumPy style
- JSON output uses 2-space indentation
- Random seeds for reproducibility

### Key Design Decisions

1. **Agentic Architecture**: Computation in Python, LLM for rationale only
2. **Individual Heterogeneity**: Beta/Normal distributions for personality traits
3. **Complex Contagion**: Threshold model prevents unrealistic cascades
4. **Gradual Suppression**: Smooth safety-based suppression (not binary)
5. **Computational Fusion**: Narrative modifiers fused into computation, not LLM prompt

### Testing

Manual validation includes:
- Schema validation against `FORECAST_SCHEMA`
- Probability sum checks (must equal 1.0)
- Graph symmetry checks (undirected graphs)
- Hash consistency checks
- Safety score variance checks

---

## References

- **NetworkX**: Graph analysis library used for influence computation and PageRank
- **OpenAI Structured Outputs**: [Documentation](https://platform.openai.com/docs/guides/structured-outputs)
- **JSON Schema**: [Specification](https://json-schema.org/)
- **Complex Contagion**: Centola & Macy (2007) - "Complex Contagions and the Weakness of Long Ties"
- **Organizational Silence**: Morrison & Milliken (2000) - "Organizational Silence: A Barrier to Change"

---

## Contact & Support

For issues related to the data generation pipeline, please check:
1. This README for common solutions
2. Example notebooks in `pipeline_demo.ipynb`
3. Sample outputs in `sample_forecasts/` and `sample_hidden_states/`
4. Implementation plan in `AGENTIC_IMPLEMENTATION_PLAN.md`

For project-level questions, see the main `README.md` in the project root.
