#!/usr/bin/env python3
"""
AI Research Proposal Generator

This module reads all files generated in the 'some_project/doc' folder,
sends the content to an AI model, and has the AI generate a formal 
academic research proposal paper.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    print("Loading environment variables from .env file...")
    load_dotenv()
    print("Environment variables loaded successfully.")
except ImportError:
    print("Warning: python-dotenv not installed. Trying to use existing environment variables.")

# Import the existing AI clients correctly
from ai_clients import Claude37SonnetClient, DeepseekR1Client

# Define the file order for processing
FILE_ORDER = [
    "CONTEXT_CONSTRAINTS.md",
    "DIVERGENT_SOLUTIONS.md", 
    "DEEP_DIVE_MECHANISMS.md",
    "SELF_CRITIQUE_SYNERGY.md", 
    "BREAKTHROUGH_BLUEPRINT.md",
    "IMPLEMENTATION_PATH.md",
    "NOVELTY_CHECK.md",
    "ELABORATIONS.md"
]

def read_file_content(file_path: Path) -> str:
    """Read and return the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return f"[Content from {file_path.name} could not be read]"

def get_project_title(doc_folder: Path) -> str:
    """Extract the project title from the BREAKTHROUGH_BLUEPRINT.md file."""
    blueprint_path = doc_folder / "BREAKTHROUGH_BLUEPRINT.md"
    if blueprint_path.exists():
        content = read_file_content(blueprint_path)
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line.replace('# ', '')
    return "NeuroCognitive Architecture (NCA): A Brain-Inspired LLM Framework"

def prepare_prompt(file_contents: Dict[str, str], project_title: str) -> str:
    """Prepare the prompt for the AI model."""
    prompt = f"""
Create a formal academic research proposal for a project titled "{project_title}".

Use the following content from previous design documents to create a comprehensive, well-structured academic research proposal. Format it according to standard academic conventions with proper sections, citations, and academic tone.

The research proposal should include:
1. Title Page
2. Abstract
3. Introduction and Problem Statement
4. Literature Review
5. Research Questions and Objectives
6. Methodology and Technical Approach
7. Implementation Plan and Timeline
8. Expected Results and Impact
9. Conclusion
10. References

Below are the source documents to synthesize into the proposal:

"""
    
    # Add each file's content to the prompt
    for file_name in FILE_ORDER:
        if file_name in file_contents:
            section_name = file_name.replace('.md', '').replace('_', ' ').title()
            prompt += f"\n===== {section_name} =====\n"
            prompt += file_contents[file_name]
            prompt += "\n\n"

    prompt += """
Create a cohesive, professionally formatted academic research proposal that integrates these materials. 
Use formal academic language and structure. Ensure proper citation of external works where appropriate.
Focus on presenting this as a serious, innovative research initiative with clear methodology and expected outcomes.
The proposal should be comprehensive enough for submission to a major research funding organization.
"""

    return prompt

def save_proposal(content: str, output_path: Path) -> None:
    """Save the generated proposal to a file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Research proposal saved to {output_path}")

def check_environment_variables():
    """Check and display the status of environment variables."""
    print("\nEnvironment Variable Status:")
    
    # Check for Anthropic API key
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        key_preview = anthropic_key[:6] + "..." + anthropic_key[-4:] if len(anthropic_key) > 10 else "***"
        print(f"✓ ANTHROPIC_API_KEY is set: {key_preview}")
    else:
        print("❌ ANTHROPIC_API_KEY is not set in environment variables")
    
    # Check for DeepSeek API key
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        key_preview = deepseek_key[:6] + "..." + deepseek_key[-4:] if len(deepseek_key) > 10 else "***" 
        print(f"✓ DEEPSEEK_API_KEY is set: {key_preview}")
    else:
        print("❌ DEEPSEEK_API_KEY is not set in environment variables")
        
    # List all environment variables (for debugging)
    print("\nDebug: All environment variable names:")
    for i, (key, _) in enumerate(os.environ.items()):
        print(f"  {key}")
        if i >= 20:  # Limit to first 20 to avoid overwhelming output
            print(f"  ... and {len(os.environ) - 20} more")
            break

def generate_ai_proposal(model: str = "claude") -> None:
    """
    Generate a research proposal using AI.
    
    Args:
        model: The AI model to use ('claude' or 'deepseek')
    """
    # Check environment variables
    check_environment_variables()
    
    doc_folder = Path("some_project/doc")
    output_folder = Path("some_project")
    
    # Check if the folder exists
    if not doc_folder.exists():
        print("Error: 'some_project/doc' folder does not exist.")
        return
    
    # Read all the relevant files
    file_contents = {}
    for file_name in FILE_ORDER:
        file_path = doc_folder / file_name
        if file_path.exists():
            file_contents[file_name] = read_file_content(file_path)
        else:
            print(f"Warning: {file_name} not found in the doc folder.")
    
    # Get project title
    project_title = get_project_title(doc_folder)
    
    # Prepare the prompt
    prompt = prepare_prompt(file_contents, project_title)
    
    # Save the prompt for reference
    with open(output_folder / "ai_prompt.txt", 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"Prompt saved to {output_folder}/ai_prompt.txt")
    
    # Call the appropriate AI client directly with the correct method
    ai_response = None
    try:
        if model.lower() == "claude":
            print("Using Claude 3.7 Sonnet to generate research proposal...")
            claude = Claude37SonnetClient()
            
            # Print the API key status (just the beginning and end for security)
            api_key = claude.api_key
            if api_key == "missing-api-key":
                print("\nERROR: API key is set to the default 'missing-api-key' value.")
                print("Please ensure ANTHROPIC_API_KEY is correctly set in your .env file.")
                return
            
            masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
            print(f"Using API key: {masked_key}")
            
            # Format the messages correctly for the run method
            messages = [
                {"role": "system", "content": "You are an expert academic researcher and writer specializing in creating comprehensive research proposals."},
                {"role": "user", "content": prompt}
            ]
            
            # Call the run method directly with messages
            ai_response = claude.run(messages, max_tokens=20000)
            
        elif model.lower() == "deepseek":
            print("Using DeepSeek R1 to generate research proposal...")
            deepseek = DeepseekR1Client()
            
            # Print the API key status (just the beginning and end for security)
            api_key = deepseek.api_key
            if api_key == "missing-deepseek-key":
                print("\nERROR: API key is set to the default 'missing-deepseek-key' value.")
                print("Please ensure DEEPSEEK_API_KEY is correctly set in your .env file.")
                return
                
            masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
            print(f"Using API key: {masked_key}")
            
            # Format the messages correctly for the run method
            messages = [
                {"role": "system", "content": "You are an expert academic researcher and writer specializing in creating comprehensive research proposals."},
                {"role": "user", "content": prompt}
            ]
            
            # Call the run method directly with messages
            ai_response = deepseek.run(messages, max_tokens=8000)
            
        else:
            print(f"Error: Unsupported model '{model}'. Please use 'claude' or 'deepseek'.")
            return
            
    except Exception as e:
        print(f"Error calling AI API: {e}")
        print("\nSuggestion: Use cursor_proposal_generator.py instead which doesn't require API keys")
        return
    
    if ai_response:
        # Check if the response is an error message
        if ai_response.startswith("ERROR from"):
            print(f"AI API returned an error: {ai_response}")
            print("\nPlease run cursor_proposal_generator.py instead")
            return
            
        # Save the generated proposal
        save_proposal(ai_response, output_folder / "ai_research_proposal.md")
        print("Success! Research proposal generated successfully.")
    else:
        print("Failed to generate research proposal.")
        print("\nPlease run cursor_proposal_generator.py instead")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an AI-written research proposal")
    parser.add_argument('--model', choices=['claude', 'deepseek'], default='claude', 
                        help='AI model to use (claude or deepseek)')
    args = parser.parse_args()
    
    generate_ai_proposal(args.model) 