# Data Generation Pipeline

This directory contains the synthetic data generation pipeline for the ECHO multiagent learning project. The pipeline creates labeled training datasets by generating synthetic organizations and using LLMs to annotate sentiment propagation patterns.

## Directory Structure

```
data_generation/
├── hidden_state_generation.py    # Organization modeling and graph generation
├── encoder_layer.py              # Optional narrative synthesis
├── forward_forecaster.py         # LLM-based label generation
├── pipeline_demo.ipynb           # Complete workflow demonstration
├── sample_hidden_states/         # Example synthetic organizations
│   ├── demo_state.json
│   └── sample_0.json
└── sample_forecasts/             # Example labeled training data
    └── demo_forecast.json
```

## Module Overview

### 1. `hidden_state_generation.py`

**Purpose**: Generates synthetic organizations with realistic multi-graph network structures.

**Key Components**:

- **Seed Functions**: Sample random organizational, recommendation, and situational parameters
  - `sample_org_seed()`: Industry, size, cultural parameters
  - `sample_rec_seed()`: Recommendation domain, urgency, resource needs
  - `sample_situation_seed()`: Current state, visibility, sanctions

- **`OrganizationHiddenState` Class**: Core organization generator
  - Automatically generates employee roster across hierarchical levels
  - Creates 5 weighted adjacency matrices (reports_to, collaboration, friendship, influence, conflict)
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

# Save to file
with open('my_org.json', 'w') as f:
    f.write(org.to_json_encoding())

# Visualize graphs
org.visualize_graph('collaboration')
```

**Key Methods**:
- `_generate_roster()`: Distributes employees across levels and departments
- `_generate_reports_to_graph()`: Creates hierarchical reporting structure
- `_generate_collaboration_graph()`: Models working relationships with edge weights
- `_generate_friendship_graph()`: Derives social bonds from collaboration and demographics
- `_generate_influence_graph()`: Computes power networks using PageRank
- `_generate_conflict_graph()`: Identifies tensions based on competition and distance
- `to_json_encoding()`: Exports complete state to JSON
- `visualize_graph()`: Plots adjacency matrix heatmaps

**Output Format**: JSON with organizational parameters, employee roster, and 5 adjacency matrices.

**Run Standalone**:
```bash
python hidden_state_generation.py
```
This generates `sample_hidden_states/sample_0.json` and displays a visualization.

---

### 2. `encoder_layer.py`

**Purpose**: Optional module that converts structured organizational JSON into natural language narratives using GPT-4o.

**Key Function**:
- `encode_hidden_state_to_text(file_path: str) -> str`: Takes a hidden state JSON file and returns a structured narrative summary

**Output Structure** (JSON):
```json
{
  "company_description": "Narrative about industry, size, culture...",
  "key_relationships": "Analysis of influencers, collaborations, conflicts...",
  "recommendation_context": "Summary of the recommendation and its context..."
}
```

**Example Usage**:

```python
from encoder_layer import encode_hidden_state_to_text

# Generate narrative from hidden state
narrative_json = encode_hidden_state_to_text('./sample_hidden_states/sample_0.json')

# Parse and use
import json
narrative = json.loads(narrative_json)
print(narrative['company_description'])
```

**Requirements**:
- OpenAI API key set in `.env` file
- GPT-4o model access

**Run Standalone**:
```bash
python encoder_layer.py
```
This reads `sample_hidden_states/sample_0.json` and generates `sample_encoder_input/sample_0_encoded.json`.

**Note**: This is an optional preprocessing step. The main forecaster can work directly with the hidden state JSON.

---

### 3. `forward_forecaster.py`

**Purpose**: Core labeling module that uses GPT-4o with structured outputs to generate sentiment propagation labels for training data.

**Key Functions**:

- **`forecast_scenario(hidden_state, scenario_id, model, prompt_template_id, horizon, provider)`**
  - Main label generation function
  - Takes organizational hidden state as input
  - Returns validated labeled training example
  - Uses temperature=0 for deterministic outputs

- **`forecast_from_file(hidden_state_path, scenario_id, model, prompt_template_id, horizon)`**
  - Convenience wrapper that loads JSON from file
  - Automatically generates scenario_id from filename if not provided

- **`compute_state_hash(hidden_state)`**
  - Generates SHA256 hash for reproducibility
  - Uses canonical JSON serialization (sorted keys)

- **`validate_forecast(forecast)`**
  - Validates output against JSON schema
  - Checks probability normalization (sum to 1.0)
  - Auto-corrects sentiment classes to match highest probability

**Example Usage**:

```python
from forward_forecaster import forecast_from_file
import json

# Generate labeled example
labeled_data = forecast_from_file(
    hidden_state_path='./sample_hidden_states/sample_0.json',
    scenario_id='training_example_1',
    model='gpt-4o-mini',
    horizon='decision'
)

# Save training data
with open('labeled_example_1.json', 'w') as f:
    json.dump(labeled_data, f, indent=2)

# Access labels
for employee_sentiment in labeled_data['individual_sentiments']:
    emp_id = employee_sentiment['employee_id']
    sentiment = employee_sentiment['sentiment']
    probs = employee_sentiment['probabilities']
    print(f"Employee {emp_id}: {sentiment} (confidence: {probs[sentiment]:.2f})")
```

**Advanced Usage** (direct API):

```python
from forward_forecaster import forecast_scenario
import json

# Load hidden state
with open('my_org.json', 'r') as f:
    hidden_state = json.load(f)

# Generate labels with custom parameters
labeled_data = forecast_scenario(
    hidden_state=hidden_state,
    scenario_id='custom_scenario',
    model='gpt-4o-mini',  # or 'gpt-4o' for higher quality
    prompt_template_id='v3',
    horizon='decision',  # or 'next_cycle', 'quarter'
    provider='openai'
)
```

**Label Output Structure**:

```json
{
  "scenario_id": "example_1",
  "state_hash": "sha256:...",
  "generated_at": "2025-11-18T...",
  "schema_version": "1.0",
  "horizon": "decision",
  "model": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0,
    "prompt_template_id": "v3"
  },
  "individual_sentiments": [
    {
      "employee_id": 0,
      "sentiment": "support",
      "probabilities": {
        "oppose": 0.0,
        "neutral": 0.0,
        "support": 0.8,
        "escalate": 0.2
      },
      "influence_sources": [
        {
          "employee_id": 1,
          "graph_type": "reports_to",
          "influence_weight": 1.0
        }
      ],
      "propagation_path": [0]
    }
  ],
  "aggregate_outcomes": {
    "probabilities": {
      "oppose": 0.045,
      "neutral": 0.136,
      "support": 0.727,
      "escalate": 0.091
    },
    "top_class": "support"
  },
  "segments": {
    "by_department": {...},
    "by_level": {...}
  },
  "rationale": "...",
  "features_importance": [...]
}
```

**Validation & Error Handling**:
- Automatic retry with exponential backoff (3 attempts)
- Rate limit detection and graceful waiting
- Probability normalization and validation
- Schema validation with detailed error messages
- State hash for reproducibility tracking

**Run Standalone**:
```bash
python forward_forecaster.py
```
This generates labels for `sample_hidden_states/sample_0.json` and saves to `sample_forecasts/sample_0_forecast.json`.

---

### 4. `pipeline_demo.ipynb`

**Purpose**: Interactive Jupyter notebook demonstrating the complete data generation workflow.

**Sections**:
1. **Setup**: Import libraries and configure environment
2. **Stage 1**: Generate synthetic organization with custom parameters
3. **Stage 2**: (Optional) Generate narrative encoding
4. **Stage 3**: Generate labeled training examples
5. **Analysis**: Visualize and interpret generated labels
6. **Validation**: Check label quality and consistency

**How to Use**:
```bash
jupyter notebook pipeline_demo.ipynb
```

Then run cells sequentially to see the complete pipeline in action.

**What It Demonstrates**:
- How to customize organizational parameters
- Graph structure visualization
- LLM label generation process
- Label interpretation and validation
- Complete training example creation

---

## Quick Start Guide

### 1. Install Dependencies

```bash
pip install openai numpy pandas networkx matplotlib seaborn python-dotenv jsonschema jupyter
```

### 2. Set Up API Key

Create a `.env` file in the project root:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > ../.env
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
import json

# Generate organization
org = OrganizationHiddenState(
    org_seed=sample_org_seed(seed=1),
    rec_seed=sample_rec_seed(seed=2),
    situation_seed=sample_situation_seed(seed=3)
)
hidden_state = json.loads(org.to_json_encoding())

# Generate labels
labeled_data = forecast_scenario(
    hidden_state=hidden_state,
    scenario_id='train_001',
    model='gpt-4o-mini'
)

# Save
with open('training_data/example_001.json', 'w') as f:
    json.dump(labeled_data, f, indent=2)
```

**Option C: Run standalone scripts**
```bash
# Generate organization
python hidden_state_generation.py

# Generate labels
python forward_forecaster.py
```

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
- `sanction_salience`: Fear of negative consequences (high = more risk-averse)
- `in_group_bias`: Department loyalty (high = more within-group clustering)

### Recommendation Parameters

**Domains**:
- `product_roadmap`, `hiring`, `budget`, `compliance`, `pricing`, `market_entry`, `vendor_selection`

**Continuous Parameters** (0-1):
- `urgency`: Time pressure for decision
- `resource_need`: Resource intensity of recommendation
- `theta_ideal`: Optimal policy position

### Situation Parameters

- `theta_current`: Current organizational position (0-1)
- `visibility`: Information transparency (`private`, `public`, `confidential`)
- `sanction_strength`: Consequence severity (0-1)
- `provocation_flag`: Recent destabilizing events (0 or 1)

### Model Parameters

**Models**:
- `gpt-4o-mini`: Faster, cheaper, good quality
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
    "in_group_bias": 0.184
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
      "manager_id": -1
    },
    ...
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

### Labeled Training Example Structure

See the `forward_forecaster.py` section above for the complete structure.

**Key Fields for Training**:
- `individual_sentiments`: Individual-level labels for each employee
- `influence_sources`: Ground-truth influence network edges
- `propagation_path`: Sentiment flow sequence
- `aggregate_outcomes`: Organization-wide label distribution

---

## Troubleshooting

### Common Issues

**1. OpenAI API Key Not Found**
```
Error: OpenAI API key not found after loading .env
```
**Solution**: Create a `.env` file in the project root with your API key:
```bash
echo "OPENAI_API_KEY=sk-..." > ../.env
```

**2. Rate Limit Errors**
```
Error: Rate limit exceeded
```
**Solution**: The forecaster automatically retries with exponential backoff. Wait a few minutes or upgrade your OpenAI plan.

**3. Import Errors**
```
ModuleNotFoundError: No module named 'networkx'
```
**Solution**: Install missing dependencies:
```bash
pip install networkx pandas numpy matplotlib seaborn openai python-dotenv jsonschema
```

**4. Validation Errors**
```
Validation failed: Individual sentiment probabilities sum to 0.998, not 1.0
```
**Solution**: This is auto-corrected by the validation system. If errors persist, check the `warnings` field in the output.

**5. Empty Graphs**
```
Organization has 0 employees
```
**Solution**: Check your seed parameters. Ensure `size_id` is between 0-3.

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
    
    # Generate labels
    labeled = forecast_scenario(
        hidden_state=hidden_state,
        scenario_id=f'train_{i:04d}',
        model='gpt-4o-mini'
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
- **LLM Labeling**: ~5-30 seconds per example (depends on model and organization size)
  - `gpt-4o-mini`: ~5-10 seconds
  - `gpt-4o`: ~15-30 seconds

### Cost Estimates (OpenAI API)

For a 20-employee organization:
- **Input tokens**: ~4,000 tokens (hidden state + prompt)
- **Output tokens**: ~2,000 tokens (labels)

**Cost per example** (approximate):
- `gpt-4o-mini`: $0.002 - $0.005
- `gpt-4o`: $0.03 - $0.06

**For 1,000 training examples**:
- `gpt-4o-mini`: $2-5
- `gpt-4o`: $30-60

### Optimization Tips

1. Use `gpt-4o-mini` for bulk generation
2. Use `gpt-4o` for validation examples
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

### Testing

Currently no automated tests. Manual validation includes:
- Schema validation against `FORECAST_SCHEMA`
- Probability sum checks (must equal 1.0)
- Graph symmetry checks (undirected graphs)
- Hash consistency checks

### Future Improvements

- [ ] Add unit tests for graph generation
- [ ] Implement parallel batch generation
- [ ] Add label quality metrics
- [ ] Create visualization tools for training data
- [ ] Support for larger organizations (100+ employees)
- [ ] Dynamic graph generation (temporal evolution)

---

## References

- **NetworkX**: Graph analysis library used for influence computation
- **OpenAI Structured Outputs**: [Documentation](https://platform.openai.com/docs/guides/structured-outputs)
- **JSON Schema**: [Specification](https://json-schema.org/)

---

## Contact & Support

For issues related to the data generation pipeline, please check:
1. This README for common solutions
2. Example notebooks in `pipeline_demo.ipynb`
3. Sample outputs in `sample_forecasts/` and `sample_hidden_states/`

For project-level questions, see the main `README.md` in the project root.

