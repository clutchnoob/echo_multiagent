import openai
import json
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- System Prompt Definition ---
# This prompt provides the LLM with the context needed to interpret the JSON file.
SYSTEM_PROMPT = """
You are an expert organizational analyst. Your task is to synthesize a structured JSON object into a structured JSON output. This output should be an in-depth, analytical summary that captures the essence of the company's culture, key social dynamics, and the context of a specific recommendation.

The input JSON object you will be given has the following structure:
1.  `org_seed`: Describes the company's foundational attributes (industry, size, culture).
2.  `rec_seed`: Details a specific business recommendation being considered.
3.  `situation_seed`: Outlines the current organizational context.
4.  `employees`: A list of the key decision-makers.
5.  `graphs`: Adjacency matrices for `reports_to`, `collaboration`, `friendship`, `influence`, and `conflict`.

Your response MUST be a valid JSON object with the following three keys: "company_description", "key_relationships", and "recommendation_context". Your analysis should be thorough and insightful.

-   **`company_description`**: (String) Provide an in-depth narrative summary of the company. Synthesize the `org_seed` data to describe the company's industry and size. Elaborate on the implications of its cultural profile (e.g., "This is a large tech company with a relatively flat hierarchy (low power distance), suggesting that ideas can flow freely, but its high sanction salience indicates a culture where mistakes are not well-tolerated.").
-   **`key_relationships`**: (String) Provide a detailed analysis of the most significant social dynamics based on the `graphs`. Go beyond a surface-level list and explain *why* these relationships are important. Identify key influencers, collaborative hubs, and potential sources of conflict. For example: "The CEO (Employee 0) is the primary influencer. While Directors of Sales and Marketing (Employees 1 and 3) collaborate closely, there is significant underlying conflict between them, which could complicate budget approvals."
-   **`recommendation_context`**: (String) Provide a rich, nuanced summary of the action being considered. Synthesize the `rec_seed` and `situation_seed` to explain the recommendation in detail, its strategic importance, urgency, and cost. Discuss the potential challenges and dynamics, considering the company's current stance and any recent provocative events.
"""

def encode_hidden_state_to_text(file_path: str) -> str:
    """
    Loads a JSON hidden state, sends it to GPT-4o with a detailed system prompt,
    and returns the generated textual summary.

    Args:
        file_path: The path to the JSON hidden state file.

    Returns:
        A string containing the narrative summary of the organization.
    """
    # Note: This script requires the 'python-dotenv' and 'openai' packages.
    # It loads the OPENAI_API_KEY from a .env file in your project root.
    # Example .env file content:
    # OPENAI_API_KEY="your-api-key-here"
    try:
        client = openai.OpenAI()
        if not client.api_key:
            raise openai.OpenAIError("API key not found after loading .env")
    except openai.OpenAIError:
        return "OpenAI API key not found. Please create a .env file in the project root and add your OPENAI_API_KEY."

    # 1. Load the JSON data
    try:
        with open(file_path, 'r') as f:
            hidden_state_json = json.load(f)
    except FileNotFoundError:
        return f"Error: The file '{file_path}' was not found."
    except json.JSONDecodeError:
        return f"Error: The file '{file_path}' is not a valid JSON file."

    # 2. Call the GPT-4o API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,  # For deterministic, factual output
            response_format={"type": "json_object"}, # Enforce JSON output
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": json.dumps(hidden_state_json) # No need for indentation for the model
                }
            ]
        )
        # Parse the JSON response and pretty-print it
        parsed_json = json.loads(response.choices[0].message.content)
        return json.dumps(parsed_json, indent=4)
    except Exception as e:
        return f"An error occurred while calling the OpenAI API: {e}"

if __name__ == "__main__":
    # Define the path to the sample hidden state file
    input_file_path = "./sample_hidden_states/sample_0.json"
    
    # Define the output directory and create it if it doesn't exist
    output_dir = "./sample_encoder_input"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the textual encoding
    narrative_summary_json = encode_hidden_state_to_text(input_file_path)
    
    # Save the output to a new JSON file
    output_filename = os.path.basename(input_file_path).replace('.json', '_encoded.json')
    output_file_path = os.path.join(output_dir, output_filename)
    
    if narrative_summary_json.startswith("Error"):
        print(narrative_summary_json)
    else:
        with open(output_file_path, "w") as f:
            f.write(narrative_summary_json)
        
        print("--- Generated Organizational Narrative ---")
        print(narrative_summary_json)
        print(f"\nSuccessfully saved encoded summary to: {output_file_path}")

