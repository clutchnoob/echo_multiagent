# ECHO: Emergent Consensus through Hierarchical Organizational Modeling

**A Multiagent Learning System for Organizational Sentiment Propagation Forecasting**

*MIT 6.S890: Multiagent Learning | Fall 2025*

---

## Abstract

ECHO is a multiagent learning project that aims to model and forecast how recommendations propagate through organizational networks. **This repository contains the synthetic data generation pipeline** for creating training datasets that capture organizational dynamics. By representing organizations as multi-graph systems with heterogeneous agent types and relationships, the pipeline uses large language models with structured outputs to generate labeled examples of individual-level sentiment responses and their propagation through formal hierarchies and informal social networks. These synthetic datasets will be used to train multiagent learning models that can predict organizational decision-making dynamics. **Note: The actual ECHO learning model is not yet implemented; this codebase focuses solely on generating realistic training data.**

---

## Table of Contents

- [Project Status](#project-status)
- [Motivation](#motivation)
- [Data Generation Approach](#data-generation-approach)
- [Pipeline Architecture](#pipeline-architecture)
- [Technical Details](#technical-details)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Sample Generated Data](#sample-generated-data)
- [Future Work](#future-work)
- [References & Acknowledgments](#references--acknowledgments)

---

## Project Status

**⚠️ This repository contains the data generation pipeline only.**

- ✅ **Implemented**: Synthetic organizational modeling and labeled data generation
- ✅ **Implemented**: Multi-graph representation of organizational networks
- ✅ **Implemented**: LLM-based labeling for sentiment propagation
- ❌ **Not Yet Implemented**: The ECHO multiagent learning model
- ❌ **Not Yet Implemented**: Model training and evaluation framework
- ❌ **Not Yet Implemented**: Real-world data collection and validation

**Current Goal**: Generate high-quality synthetic training data that captures realistic organizational dynamics for future model training.

---

## Motivation

### The Challenge of Organizational Decision-Making

Organizations are complex multiagent systems where decisions must navigate through multiple layers of hierarchy, competing interests, and diverse social networks. When a recommendation is proposed—whether it concerns budget allocation, strategic pivots, or policy changes—the ultimate outcome depends not only on the recommendation's merit but on how individual agents respond and influence each other through formal and informal channels.

### Why Multiagent Learning?

From a multiagent learning perspective, organizational decision-making presents several key challenges:

1. **Heterogeneous Agents**: Employees vary in their roles, influence, and relationships, creating a complex action space
2. **Network Effects**: Individual responses propagate through multiple overlapping networks (hierarchy, collaboration, friendship, conflict)
3. **Emergent Behavior**: Aggregate outcomes emerge from local interactions rather than top-down control
4. **Strategic Behavior**: Agents may not reveal their true preferences, and their responses are influenced by anticipated reactions of others

### Gap in Current Approaches

Traditional organizational behavior models often rely on:
- Survey-based methods that are retrospective and subject to response bias
- Simplified hierarchical models that ignore informal networks
- Agent-based simulations that require extensive hand-crafted rules
- Statistical models that struggle to capture the nuanced dynamics of sentiment propagation

### The Need for Synthetic Training Data

Training multiagent learning models for organizational dynamics faces a critical challenge: **lack of labeled data**. Real organizational decision-making data is scarce, proprietary, and difficult to collect at scale. 

**This data generation pipeline addresses this challenge** by:
1. Creating synthetic organizations with realistic network structures
2. Using LLMs with graph-theoretic reasoning to generate plausible sentiment propagation labels
3. Producing large-scale datasets with ground-truth influence networks and propagation paths
4. Enabling systematic exploration of diverse organizational cultures and scenarios

The generated data will be used to train the future ECHO multiagent learning model.

---

## Data Generation Approach

The pipeline generates synthetic training data by modeling organizations as **multiagent systems** with rich structural and relational information, then uses LLMs to generate labels for sentiment propagation dynamics.

**Key Idea**: Use LLMs as expert labelers (not as the final model) to create realistic training examples that capture how sentiments propagate through organizational networks.

### 1. Graph-Based Organizational Modeling

We represent organizations using **five distinct weighted adjacency matrices**, each capturing a different dimension of organizational relationships:

#### Network Types

- **`reports_to`** (Binary, Directed): Formal hierarchical reporting structure
  - Captures formal authority and power dynamics
  - Used to compute organizational distance and hierarchy-based influence

- **`collaboration`** (Weighted [0-1], Undirected): Strength of work relationships
  - Higher weights indicate stronger working relationships
  - Generated based on hierarchical proximity and departmental co-location

- **`friendship`** (Binary, Undirected): Social bonds beyond work
  - Derived from collaboration strength, level similarity, and tenure proximity
  - Represents informal influence channels

- **`influence`** (Weighted [0-1], Directed): Perceived authority and power
  - Computed from formal hierarchy, social network centrality, and level-based authority
  - Uses PageRank over social graphs to capture informal power structures

- **`conflict`** (Weighted [0-1], Undirected): Tension and competing interests
  - Models inter-departmental competition and level-based rivalry
  - Inversely related to collaboration and friendship

This multi-graph representation captures the **complexity of organizational dynamics** where agents simultaneously operate in multiple social contexts.

### 2. Hidden State Generation

Organizations are parameterized by cultural and situational factors:

**Organizational Seed (`org_seed`)**:
- Industry and size classification
- `power_distance`: Cultural acceptance of hierarchical inequality (0-1)
- `sanction_salience`: Sensitivity to negative consequences (0-1)
- `in_group_bias`: Preference for one's department/group (0-1)

**Recommendation Seed (`rec_seed`)**:
- Domain (e.g., budget, hiring, product_roadmap)
- `urgency`: Time pressure (0-1)
- `resource_need`: Resource intensity (0-1)
- `theta_ideal`: Optimal policy position (0-1)

**Situation Seed (`situation_seed`)**:
- `theta_current`: Current organizational state (0-1)
- `visibility`: Information transparency (private/public/confidential)
- `sanction_strength`: Consequences for opposition (0-1)
- `provocation_flag`: Recent destabilizing events (binary)

These parameters enable **systematic exploration** of how organizational culture and context affect decision-making.

### 3. LLM-Based Label Generation

The pipeline uses **GPT-4o with structured outputs** (temperature=0) to generate labeled training examples. The LLM acts as an expert annotator, providing consistent and detailed labels for sentiment propagation patterns.

#### Sentiment Classes

Each agent's response is classified into one of four sentiments:
- **Oppose**: Active resistance to the recommendation
- **Neutral**: Passive acceptance or indifference
- **Support**: Active advocacy for the recommendation
- **Escalate**: Intense support with willingness to champion

#### Label Generation Process

The LLM receives:
1. Complete organizational hidden state (all graphs, employee attributes)
2. Recommendation and situational context
3. Explicit computational instructions for propagation dynamics

The LLM generates labeled training examples containing:
1. **Individual-level labels**: For each employee, sentiment class and probability distribution
2. **Influence sources**: Which specific connections (and through which graphs) influenced each agent
3. **Propagation paths**: Sequence showing how sentiment flows from key influencers
4. **Aggregate outcomes**: Organization-wide sentiment distribution
5. **Segment analysis**: Breakdowns by department and hierarchical level

These labels will serve as supervised learning targets for training the future ECHO model.

### 4. Network Propagation Dynamics

The system models sentiment propagation through **graph-theoretic influence**:

```
influenced_score(i) = baseline_score(i) + 
                      Σ [influence_contribution(j→i) × graph_weight × distance_decay]
```

Where:
- **Graph weights**: reports_to (1.0) > influence (0.8) > collaboration (0.6) > friendship (0.4)
- **Distance decay**: Influence diminishes with graph distance (0.7^distance)
- **Edge weights**: Actual collaboration/influence strengths from adjacency matrices

This approach captures how **local interactions produce global patterns**, a core principle in multiagent systems.

---

## Pipeline Architecture

The data generation pipeline consists of three modular stages:

```
┌─────────────────────────────────────────────────────────────────┐
│              Synthetic Data Generation Pipeline                  │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────┐
  │  Stage 1: Hidden State Generation    │
  │  (hidden_state_generation.py)        │
  │                                       │
  │  Input: Seeds (org, rec, situation)  │
  │  Output: Multi-graph organization    │
  │    - 5 adjacency matrices             │
  │    - Employee attributes              │
  │    - Cultural parameters              │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │  Stage 2: Encoder Layer (Optional)   │
  │  (encoder_layer.py)                  │
  │                                       │
  │  Input: Hidden state JSON            │
  │  Output: Narrative summary           │
  │    - Company description              │
  │    - Key relationships                │
  │    - Recommendation context           │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │  Stage 3: LLM-Based Labeling         │
  │  (forward_forecaster.py)             │
  │                                       │
  │  Input: Hidden state + prompt        │
  │  Output: Labeled training example    │
  │    - Individual sentiment labels      │
  │    - Influence networks               │
  │    - Propagation paths                │
  │    - Aggregate outcomes               │
  └──────────────────────────────────────┘
                 │
                 ▼
         Training Dataset for
         Future ECHO Model
```

### Key Components

#### 1. `hidden_state_generation.py`
**Purpose**: Generates synthetic organizations with realistic network structures

**Core Class**: `OrganizationHiddenState`
- Distributes employees across C-Suite, Director, and Manager levels
- Constructs five relationship graphs with domain-specific logic
- Exports to canonical JSON format

**Key Methods**:
- `_generate_roster()`: Creates employee demographics
- `_generate_reports_to_graph()`: Builds hierarchical reporting structure
- `_generate_collaboration_graph()`: Models working relationships
- `_generate_friendship_graph()`: Derives social networks
- `_generate_influence_graph()`: Computes power/authority networks
- `_generate_conflict_graph()`: Identifies tensions

#### 2. `encoder_layer.py`
**Purpose**: Converts structured data to natural language narratives (optional step)

**Function**: `encode_hidden_state_to_text()`
- Uses GPT-4o to synthesize organizational context
- Generates three narrative components:
  - Company description (culture, industry, size)
  - Key relationships (influencers, collaborations, conflicts)
  - Recommendation context (strategic importance, urgency)

#### 3. `forward_forecaster.py`
**Purpose**: Generates labeled training examples using LLM-based annotation

**Main Functions**:
- `forecast_scenario()`: Core label generation with structured outputs
- `compute_state_hash()`: Ensures reproducibility via content-based hashing
- `validate_forecast()`: Schema and business rule validation

**Features**:
- Temperature=0 for deterministic label generation
- JSON Schema validation (strict mode)
- Automatic probability normalization
- Retry logic with exponential backoff for API rate limits
- Produces training examples with ground-truth labels for future model training

#### 4. `pipeline_demo.ipynb`
**Purpose**: End-to-end demonstration notebook for data generation

Demonstrates:
- Synthetic organization generation with custom parameters
- Optional narrative encoding
- LLM-based label generation and validation
- Training data analysis and interpretation
- Complete workflow for creating labeled datasets

---

## Technical Details

### Data Schema

#### Hidden State Format

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

#### Generated Training Example Format

```json
{
  "scenario_id": "demo_scenario",
  "state_hash": "sha256:3b51f8f867ab1b968f2abad441bcb6f7...",
  "generated_at": "2025-11-17T21:23:44.171059+00:00",
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
    },
    ...
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

### Propagation Algorithm

The LLM is instructed to follow a multi-step computational process:

**Step 1: Baseline Sentiment**
```
baseline_score = (resource_need × 0.3) + 
                 (sanction_strength × visibility_factor) - 
                 (conflict_avg × 0.2) + 
                 (tenure × 0.05)
```

**Step 2: Graph-Based Influence**
```
For each graph type G:
  For each influencer j → employee i:
    influence = baseline_score(j) × edge_weight(j,i) × 
                decay(distance(j,i)) × graph_multiplier(G)
    influenced_score(i) += influence
```

**Step 3: Softmax to Probabilities**
```
logits = [oppose_logit, neutral_logit, support_logit, escalate_logit]
probabilities = softmax(logits / temperature)
```

**Step 4: Constraints & Normalization**
- Ensure minimum probabilities (oppose ≥ 0.01, support ≥ 0.01)
- High conflict → minimum oppose probability ≥ 0.05
- Normalize to sum = 1.0 with 3 decimal precision

**Step 5: Segment Aggregation**
- Department-level: Weighted by collaboration centrality
- Level-level: Simple or tenure-weighted average (excluding C-Suite from segments)

### Validation & Determinism

**Schema Validation**:
- Strict JSON Schema with `additionalProperties: false`
- Type checking for all fields
- Probability sum validation (must equal 1.0 within 1e-6 tolerance)
- State hash verification using SHA-256 of canonical JSON

**Determinism Guarantees**:
- Temperature = 0 for OpenAI API calls
- Content-based hashing for reproducibility
- Canonical JSON serialization (sorted keys)
- Fixed random seeds for organization generation

**Error Handling**:
- Automatic retry with exponential backoff (3 attempts)
- Rate limit detection and graceful waiting
- Validation with auto-correction (e.g., normalizing probabilities)
- Comprehensive error messages with troubleshooting hints

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT-4o and GPT-4o-mini access)
- Git (for cloning the repository)

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/echo_multiagent.git
cd echo_multiagent
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**:
```
openai>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
networkx>=3.0
matplotlib>=3.7.0
seaborn>=0.12.0
python-dotenv>=1.0.0
jsonschema>=4.17.0
jupyter>=1.0.0
```

Or install individually:
```bash
pip install openai numpy pandas networkx matplotlib seaborn python-dotenv jsonschema jupyter
```

### Step 3: Configure OpenAI API Key

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Or export as environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Getting an API Key**:
1. Visit https://platform.openai.com/api-keys
2. Create a new API key
3. Ensure your account has access to GPT-4o models

### Step 4: Verify Installation

```bash
python -c "import openai; import networkx; import pandas; print('✓ All dependencies installed')"
```

### Directory Structure

```
echo_multiagent/
├── data_generation/
│   ├── hidden_state_generation.py    # Organization modeling
│   ├── encoder_layer.py              # Narrative synthesis
│   ├── forward_forecaster.py         # Sentiment forecasting
│   ├── pipeline_demo.ipynb           # End-to-end demo
│   ├── sample_hidden_states/         # Example organizations
│   │   ├── demo_state.json
│   │   └── sample_0.json
│   └── sample_forecasts/             # Example predictions
│       └── demo_forecast.json
├── .env                              # API keys (create this)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Usage

### Quick Start

The fastest way to see the data generation pipeline in action is to run the Jupyter notebook:

```bash
cd data_generation
jupyter notebook pipeline_demo.ipynb
```

This notebook demonstrates the complete data generation process with visualizations and detailed explanations of how synthetic organizational data and labels are created.

### Command-Line Usage

#### 1. Generate an Organization

```python
from data_generation.hidden_state_generation import (
    sample_org_seed, sample_rec_seed, sample_situation_seed,
    OrganizationHiddenState
)
import json

# Sample random seeds
org_seed = sample_org_seed(seed=42)
rec_seed = sample_rec_seed(seed=43)
sit_seed = sample_situation_seed(seed=44)

# Generate organization
org = OrganizationHiddenState(
    org_seed=org_seed,
    rec_seed=rec_seed,
    situation_seed=sit_seed,
    departments=['Engineering', 'Sales', 'Marketing', 'HR']
)

# Save to file
with open('my_organization.json', 'w') as f:
    f.write(org.to_json_encoding())

print(f"✓ Generated {org.N} employee organization")
```

#### 2. (Optional) Generate Narrative Summary

```python
from data_generation.encoder_layer import encode_hidden_state_to_text

narrative = encode_hidden_state_to_text('my_organization.json')
print(narrative)
```

#### 3. Generate Labeled Training Example

```python
from data_generation.forward_forecaster import forecast_from_file
import json

# Generate labeled example using LLM
labeled_example = forecast_from_file(
    hidden_state_path='my_organization.json',
    scenario_id='my_scenario',
    model='gpt-4o-mini',
    horizon='decision'
)

# Save labeled training data
with open('my_labeled_example.json', 'w') as f:
    json.dump(labeled_example, f, indent=2)

# Display summary
if 'aggregate_outcomes' in labeled_example:
    agg = labeled_example['aggregate_outcomes']
    print(f"\nAggregate Sentiment Label: {agg['top_class']} ({agg['probabilities'][agg['top_class']]:.1%})")
    print(f"Individual Labels: {len(labeled_example['individual_sentiments'])} employees")
```

#### 4. Run Complete Data Generation Pipeline

```python
from data_generation.hidden_state_generation import *
from data_generation.forward_forecaster import forecast_scenario
import json

# Generate synthetic organization
org_seed = sample_org_seed(seed=123)
rec_seed = sample_rec_seed(seed=456)
sit_seed = sample_situation_seed(seed=789)

org = OrganizationHiddenState(org_seed, rec_seed, sit_seed)
hidden_state = json.loads(org.to_json_encoding())

# Generate labeled training example
labeled_data = forecast_scenario(
    hidden_state=hidden_state,
    scenario_id='training_example_1',
    model='gpt-4o-mini'
)

print("✓ Training data generated")
print(f"State Hash: {labeled_data['state_hash']}")
print(f"Top Sentiment Label: {labeled_data['aggregate_outcomes']['top_class']}")
print("\n→ This labeled example is ready to be used for training the ECHO model")
```

### Running as Scripts

Each module can be run standalone:

```bash
# Generate sample organization
cd data_generation
python hidden_state_generation.py

# Generate narrative (requires organization JSON)
python encoder_layer.py

# Generate labeled examples (requires organization JSON)
python forward_forecaster.py
```

### Custom Organization Parameters

```python
# Create specific organization structure
org = OrganizationHiddenState(
    org_seed={
        'industry_id': 0,  # tech
        'size_id': 2,      # large
        'power_distance': 0.8,      # high hierarchy
        'sanction_salience': 0.3,   # low fear of sanctions
        'in_group_bias': 0.6        # moderate tribalism
    },
    rec_seed={
        'domain_id': 2,        # budget
        'urgency': 0.9,        # very urgent
        'resource_need': 0.4,  # low resources needed
        'theta_ideal': 0.7
    },
    situation_seed={
        'theta_current': 0.2,
        'visibility_flag': 1,     # public
        'sanction_strength': 0.5,
        'provocation_flag': 1     # recent provocation
    },
    departments=['Eng', 'Sales', 'Marketing', 'HR', 'Legal'],
    avg_span_of_control=6
)
```

---

## Sample Generated Data

### Example Training Data

The repository includes sample generated data in `data_generation/sample_forecasts/demo_forecast.json` (labeled examples) and `data_generation/sample_hidden_states/demo_state.json` (organizational inputs).

**Organization Profile**:
- **Industry**: Technology
- **Size**: Large (22 employees)
- **Structure**: 1 C-Suite, 4 Directors, 17 Managers
- **Departments**: Engineering (10), Sales (6), Marketing (5), HR (1)
- **Culture**: Low power distance (0.054), moderate sanction salience (0.220)

**Recommendation Context**:
- **Domain**: Budget allocation
- **Urgency**: Low (0.124)
- **Resource Need**: High (0.755)
- **Visibility**: Private

### Generated Labels

**Aggregate Sentiment Distribution**:
```
Support:   72.7% ████████████████████████████████████
Neutral:   13.6% ██████
Escalate:   9.1% ████
Oppose:     4.5% ██
```

**Top-Class Label**: SUPPORT

**Label Characteristics**:
1. Strong C-Suite support (Employee 0: 80% support, 20% escalate) drives organizational consensus
2. Directors show varied responses based on departmental alignment:
   - Sales Directors: Mixed (75% support vs. 70% support)
   - Marketing Director: Strong support (70%)
3. Propagation paths show sentiment flows from C-Suite → Directors → Managers
4. Engineering department shows highest support (due to domain alignment with budget)
5. Collaboration strength is the strongest predictor of sentiment alignment

**Ground-Truth Influence Network**:
- Primary influencer: Employee 0 (C-Suite, CEO)
- Secondary influencers: Employees 1, 2, 3, 4 (Directors)
- Influence flows through multiple channels: reports_to (strongest), collaboration, influence graphs
- Average propagation path length: 1-2 hops from C-Suite

These labeled examples capture realistic organizational dynamics that the future ECHO model will learn to predict.

### Output Interpretation

**Individual Sentiments**:
```json
{
  "employee_id": 5,
  "sentiment": "support",
  "probabilities": {
    "oppose": 0.0,
    "neutral": 0.091,
    "support": 0.818,
    "escalate": 0.091
  },
  "influence_sources": [
    {"employee_id": 1, "graph_type": "reports_to", "influence_weight": 1.0},
    {"employee_id": 8, "graph_type": "collaboration", "influence_weight": 0.85}
  ],
  "propagation_path": [0, 1, 5]
}
```

**Interpretation**:
- Employee 5 strongly supports the recommendation (81.8% confidence)
- Primary influence comes from their manager (Employee 1) via formal hierarchy
- Secondary influence from collaborator (Employee 8)
- Sentiment propagated: CEO → Director → Employee 5

**Segment Analysis**:
```json
{
  "by_department": {
    "Engineering": {"oppose": 0.02, "neutral": 0.10, "support": 0.75, "escalate": 0.13},
    "Sales": {"oppose": 0.08, "neutral": 0.18, "support": 0.68, "escalate": 0.06},
    "Marketing": {"oppose": 0.03, "neutral": 0.12, "support": 0.72, "escalate": 0.13}
  }
}
```

**Interpretation**:
- Engineering shows strongest support (budget domain alignment)
- Sales shows most opposition (8%) and neutrality (competing priorities)
- Marketing aligns closely with Engineering (similar resource needs)

---

## Future Work

### Priority: Build the ECHO Learning Model

**The main next step is to implement the actual multiagent learning model** that will be trained on the synthetic data generated by this pipeline. This involves:

1. **Model Architecture Design**:
   - Graph neural network for processing multi-graph organizational structures
   - Attention mechanisms for capturing influence propagation
   - Multi-task learning for individual and aggregate predictions

2. **Training Framework**:
   - Supervised learning on generated labeled examples
   - Loss functions for probability distributions and propagation paths
   - Validation and evaluation metrics

3. **Inference System**:
   - Fast inference for real-time organizational forecasting
   - Uncertainty quantification
   - Interpretable explanations of predictions

### Data Generation Improvements

1. **Scale Up**: Generate large-scale datasets (1000+ organizations, diverse scenarios)
2. **Quality Control**: Validate label quality through consistency checks
3. **Diversity**: Ensure coverage of edge cases and rare organizational structures
4. **Dynamic Scenarios**: Generate multi-round decision sequences

### Model Extensions (After Initial Implementation)

1. **Dynamic Forecasting**: Model temporal evolution of sentiments over multiple rounds
2. **Strategic Behavior**: Incorporate game-theoretic reasoning about agents hiding preferences
3. **Counterfactual Analysis**: Explore "what-if" scenarios (e.g., "What if the CEO opposed?")
4. **Multi-Modal Inputs**: Incorporate text communications, meeting transcripts
5. **Reinforcement Learning**: Train agents to optimize recommendation framing

### Validation & Real-World Application

1. **Real-World Data Collection**: Partner with organizations to collect validation data
2. **Validation Studies**: Compare model predictions against actual organizational outcomes
3. **Benchmarking**: Compare against baseline models (graph neural nets, transformers)
4. **Case Studies**: Apply to real decision-making scenarios

### Current Limitations

1. **No Learning Model Yet**: This repository only generates training data; the ECHO model is not implemented
2. **LLM-Based Labels**: Labels come from GPT-4o, not ground truth; quality depends on LLM reasoning
3. **Synthetic Data Only**: No real organizational data for validation
4. **Static Graphs**: Networks are fixed; real organizations have dynamic relationships
5. **Cultural Simplification**: Three parameters cannot fully capture organizational culture

---

## References & Acknowledgments

### Course Information

This project was developed for **MIT 6.S890: Multiagent Learning** (Fall 2025), exploring how multiagent systems principles can be applied to organizational dynamics and decision-making.

### Relevant Literature

**Multiagent Systems**:
- Wooldridge, M. (2009). *An Introduction to MultiAgent Systems*. Wiley.
- Shoham, Y., & Leyton-Brown, K. (2008). *Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations*. Cambridge University Press.

**Social Network Analysis**:
- Borgatti, S. P., et al. (2018). *Analyzing Social Networks*. SAGE Publications.
- Newman, M. (2018). *Networks*. Oxford University Press.

**Organizational Behavior**:
- March, J. G., & Simon, H. A. (1958). *Organizations*. Wiley.
- Krackhardt, D., & Hanson, J. R. (1993). "Informal Networks: The Company Behind the Chart." *Harvard Business Review*.

**LLMs for Multiagent Systems**:
- Park, J. S., et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." *UIST 2023*.
- OpenAI (2024). "Structured Outputs in the API." *OpenAI Documentation*.

### Acknowledgments

- MIT 6.S890 course staff for guidance on multiagent learning frameworks
- OpenAI for API access enabling LLM-powered forecasting
- NetworkX and scientific Python ecosystem for graph analysis tools

---

## License

This project is developed for academic purposes as part of MIT 6.S890: Multiagent Learning.

---

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on the GitHub repository or contact the project maintainers through MIT course channels.

---

**Project Status**: Active Development | **Version**: 1.0 | **Last Updated**: November 2025

