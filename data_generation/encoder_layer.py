import openai
import json
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables from a .env file
# This makes the path explicit, ensuring it finds the .env file in the same directory as the script.
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Global OpenAI Client ---
# Note: This script requires the 'python-dotenv' and 'openai' packages.
# It loads the OPENAI_API_KEY from a .env file in your project root.
try:
    client = openai.OpenAI()
    if not client.api_key:
        raise openai.OpenAIError("API key not found after loading .env")
except openai.OpenAIError as e:
    print(f"OpenAI API key error: {e}")
    client = None

# --- Phase 2: Function 1 - Company Story Generation ---
def generate_company_story(org_seed: dict) -> str:
    """Generates a creative company backstory using GPT-4o."""
    if not client:
        return "OpenAI client not initialized."

    prompt = f"""
    Based on the following company profile, write a plausible, fleshed-out company history and description.
    - Industry: {org_seed.get('industry', 'N/A')}
    - Size: {org_seed.get('size', 'N/A')}
    - Power Distance: {org_seed.get('power_distance', 0.5):.2f} (Low values mean a flatter, more egalitarian culture; high values mean a steep hierarchy.)
    - Sanction Salience: {org_seed.get('sanction_salience', 0.5):.2f} (High values suggest a risk-averse, strict culture.)
    - In-Group Bias: {org_seed.get('in_group_bias', 0.5):.2f} (High values mean strong loyalty to one's own group or department.)

    Generate a short, narrative description of the company's founding, its journey, and its current market position and culture.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7, # Allow for some creativity
            messages=[
                {"role": "system", "content": "You are a business analyst and storyteller."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in generate_company_story: {e}")
        return "Error generating company story."

# --- Phase 3: Function 2 - Basic Information Encoding ---
def encode_basic_info(org_seed: dict) -> dict:
    """Extracts and formats the factual company data."""
    return {
        "industry": org_seed.get("industry", "N/A"),
        "size_category": org_seed.get("size", "N/A"),
        "culture_profile": {
            "power_distance": org_seed.get("power_distance"),
            "sanction_salience": org_seed.get("sanction_salience"),
            "in_group_bias": org_seed.get("in_group_bias")
        }
    }

# --- Phase 4: Function 3 - LLM-Driven Relationship Analysis ---

def _get_llm_analysis_for_graph(graph_name: str, matrix: list, employees: list, context: str) -> str:
    """
    Analyzes a single graph matrix using GPT-4o to extract key insights.
    """
    if not client:
        return ""

    # Prepare the data for the prompt by converting it to a string format.
    employee_list_str = "\n".join([f"- Employee {e['employee_id']}: {e['level']}, {e['department']}" for e in employees])
    matrix_str = json.dumps(matrix)

    prompt = f"""
    You are an expert organizational analyst. Your task is to analyze the provided adjacency matrix and summarize its most significant insights.

    **Graph Context:**
    {context}

    **Employee Roster:**
    {employee_list_str}

    **Adjacency Matrix:**
    {matrix_str}

    Based on all the provided data, identify and describe the 2-3 most important patterns, key players, or relationships within this specific graph. Be concise and analytical.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are an expert organizational analyst focused on interpreting network graph data."},
                {"role": "user", "content": prompt}
            ]
        )
        return f"--- Analysis of {graph_name.replace('_', ' ').title()} ---\n{response.choices[0].message.content.strip()}"
    except Exception as e:
        print(f"Error analyzing graph '{graph_name}': {e}")
        return f"Error analyzing {graph_name}."

def analyze_graphs_with_llm(graphs: dict, employees: list) -> str:
    """
    Orchestrates the analysis of all social graphs using a dedicated LLM call for each.
    """
    # Define the specific context for each graph to guide the LLM's analysis.
    graph_contexts = {
        "reports_to": "This is a directed graph where a '1' at matrix[i][j] means Employee j reports to Employee i. Identify the key leaders and the overall hierarchical structure.",
        "collaboration": "This is a weighted, undirected graph where higher values (0-1) indicate stronger collaboration. Identify the main hubs of collaboration and any significant cross-departmental teams.",
        "friendship": "This is a binary, undirected graph where a '1' indicates a friendship. Identify any notable social cliques, especially among senior leadership.",
        "influence": "This is a weighted, directed graph where a value at matrix[i][j] represents the level of influence Employee i has over Employee j. Identify the top 2-3 most influential individuals and who they primarily affect.",
        "conflict": "This is a weighted, undirected graph where higher values (0-1) indicate more conflict or tension. Identify the most significant points of conflict, particularly between senior leaders."
    }
    
    full_analysis = []
    for name, matrix in graphs.items():
        if name in graph_contexts:
            # For each graph, get a dedicated analysis from the LLM.
            context = graph_contexts[name]
            analysis = _get_llm_analysis_for_graph(name, matrix, employees, context)
            full_analysis.append(analysis)
    
    return "\n\n".join(full_analysis)


def generate_relationship_summary(graph_summary_text: str, company_story: str) -> str:
    """Synthesizes the pre-processed graph text into a narrative using GPT-4o."""
    if not client:
        return "OpenAI client not initialized."

    prompt = f"""
    Given the following company background, provide an analytical summary of the key social dynamics based on the data points provided. Focus on the potential implications of these relationships within the context of the company.

    **Company Background:**
    {company_story}

    **Key Relationship Data:**
    {graph_summary_text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are an expert organizational analyst. Your task is to interpret raw data about employee relationships and explain their strategic importance in the context of their company."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in generate_relationship_summary: {e}")
        return "Error generating relationship summary."

# --- Phase 5: Function 4 - Recommendation Scenario Generation ---
def generate_recommendation_scenario(rec_seed: dict, situation_seed: dict, company_story: str) -> str:
    """Generates a realistic business scenario for the recommendation using GPT-4o."""
    if not client:
        return "OpenAI client not initialized."

    prompt = f"""
    Given the following company background and situational data, generate a realistic, real-world business scenario or consulting task.

    **Company Background:**
    {company_story}

    **Recommendation & Situation Data:**
    - Domain: {rec_seed.get('domain', 'N/A')}
    - Recommendation Nature (0=conservative, 1=aggressive): {rec_seed.get('theta_ideal', 0.5):.2f}
    - Urgency (0=low, 1=high): {rec_seed.get('urgency', 0.5):.2f}
    - Resource Need (0=low, 1=high): {rec_seed.get('resource_need', 0.5):.2f}
    - Company's Current Stance (0=conservative, 1=aggressive): {situation_seed.get('theta_current', 0.5):.2f}
    - Visibility: {situation_seed.get('visibility', 'private')}
    - Provocation Event Occurred (1=yes): {situation_seed.get('provocation_flag', 0)}

    Based on this data, describe a specific, plausible business problem or opportunity that a consulting firm might be asked to address. Make it concrete and narrative-driven.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0, # deterministic output
            messages=[
                {"role": "system", "content": "You are a creative business strategist. Your task is to turn abstract data points into a compelling business scenario."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in generate_recommendation_scenario: {e}")
        return "Error generating recommendation scenario."

# --- Phase 1: Main Structure and Orchestration ---
def create_encoded_narrative(hidden_state_data: dict) -> dict:
    """
    Orchestrates the multi-stage encoding process from a hidden state dictionary.
    """
    # Extract the relevant parts of the hidden state
    org_seed = hidden_state_data.get("org_seed", {})
    rec_seed = hidden_state_data.get("rec_seed", {})
    situation_seed = hidden_state_data.get("situation_seed", {})
    employees = hidden_state_data.get("employees", [])
    graphs = hidden_state_data.get("graphs", {})

    # --- Execute Pipeline ---
    # 1. Generate the company story
    company_story = generate_company_story(org_seed)
    
    # 2. Encode the basic company profile
    company_profile = encode_basic_info(org_seed)
    
    # 3. Analyze relationships
    graph_summary_text = analyze_graphs_with_llm(graphs, employees)
    key_relationships = generate_relationship_summary(graph_summary_text, company_story)
    
    # 4. Generate the recommendation scenario
    recommendation_scenario = generate_recommendation_scenario(rec_seed, situation_seed, company_story)

    # Assemble the final JSON output
    final_output = {
        "company_story": company_story,
        "company_profile": company_profile,
        "key_relationships": key_relationships,
        "recommendation_scenario": recommendation_scenario
    }
    
    return final_output

# --- Phase 6: Final Execution ---
if __name__ == "__main__":
    if not client:
        print("Exiting: OpenAI client could not be initialized.")
    else:
        # Define paths
        input_file_path = "./sample_hidden_states/sample_0.json"
        output_dir = "./sample_encoder_input"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load the input JSON
            with open(input_file_path, 'r') as f:
                hidden_state = json.load(f)
            
            # Run the full encoding pipeline
            encoded_narrative = create_encoded_narrative(hidden_state)
            
            # Save the output
            output_filename = os.path.basename(input_file_path).replace('.json', '_encoded.json')
            output_file_path = os.path.join(output_dir, output_filename)
            
            with open(output_file_path, "w") as f:
                json.dump(encoded_narrative, f, indent=4)
            
            print("--- Generated Sophisticated Narrative ---")
            print(json.dumps(encoded_narrative, indent=4))
            print(f"\nSuccessfully saved encoded summary to: {output_file_path}")

        except FileNotFoundError:
            print(f"Error: Input file not found at {input_file_path}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {input_file_path}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

