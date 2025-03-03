# Breakthrough-Idea Walkthrough Framework

This project implements an AI orchestrator system designed to walk an LLM (Large Language Model) through a structured process for generating breakthrough ideas. The system uses a carefully designed 8-stage framework that maximizes novelty while producing actionable and implementable ideas.

## Overview

The "Breakthrough-Idea Walkthrough" Framework is an eight-stage structure that guides an LLM through a sequence of prompts, each designed to build upon previous outputs. The framework progressively develops a novel idea from initial domain understanding to a complete, actionable blueprint.

## Modules

This project consists of two main modules:

1. **Orchestrator (`orchestrator.py`)**: The main module that walks an LLM through the 8-stage framework to generate breakthrough ideas
2. **Proposal Generator**: Modules that convert the generated ideas into a formal academic research proposal:
   - `ai_proposal_generator.py`: Uses API calls to Claude or DeepSeek to generate the proposal
   - `cursor_proposal_generator.py`: Creates a prompt for use with Cursor's built-in Claude (no API key required)

## How to Use

### Stage 1: Generate Breakthrough Ideas with the Orchestrator

1. Run the orchestrator script with your preferred LLM:
   ```
   # Basic usage
   python orchestrator.py <claude37sonnet|deepseekr1>
   
   # Optionally provide your domain/challenge directly
   python orchestrator.py <claude37sonnet|deepseekr1> "Your domain or challenge description here"
   
   # Fully automated mode with auto-yes to all prompts
   python orchestrator.py --auto-yes <claude37sonnet|deepseekr1>
   # or use the short form
   python orchestrator.py -y <claude37sonnet|deepseekr1>
   ```

2. When prompted (if not provided via command line), describe your domain or challenge that you want breakthrough ideas for
   
3. The system will guide you through each of the 8 stages, allowing you to:
   - Proceed with each step
   - Skip steps you don't need
   - Quit the process at any point
   - Or use `--auto-yes` to automatically proceed through all steps with no interaction needed
   
4. After each step, review the AI's output and choose whether to apply the changes (which saves files to the `some_project/doc/` directory) - unless auto-yes is enabled, in which case changes are automatically applied

5. At the end, you'll have a complete breakthrough blueprint in the `some_project/doc/` directory

### Stage 2: Generate a Formal Research Proposal

After completing the orchestrator process, you can use one of the proposal generator modules to convert your breakthrough idea into a formal academic research proposal:

#### Option 1: Using API-based Proposal Generator

If you have valid API keys for Claude or DeepSeek:

```
python ai_proposal_generator.py --model <claude|deepseek>
```

This will:
1. Read all files from the `some_project/doc/` directory
2. Prepare a comprehensive prompt
3. Call the selected AI model's API
4. Save the generated research proposal to `some_project/ai_research_proposal.md`

#### Option 2: Using Cursor's Built-in Claude (No API Key Required)

If you don't have API keys or prefer to use Cursor's built-in Claude:

```
python cursor_proposal_generator.py
```

This will:
1. Read all files from the `some_project/doc/` directory
2. Prepare a comprehensive prompt
3. Save the prompt to `some_project/cursor_prompt.md`
4. Provide instructions for using the prompt with Cursor's built-in Claude

After running this script:
1. Open the prompt file: `cursor_prompt.md`
2. Copy its contents
3. Create a new chat with Claude in Cursor
4. Paste the prompt and let Claude generate your research proposal
5. Copy Claude's response and save it as your research proposal

## The 8-Stage Framework

### 1. Context & Constraints Clarification
Establishes the domain background and constraints while inviting cross-domain synergy. The AI summarizes your goals and constraints, then collects unusual references that might apply.

### 2. Divergent Brainstorm of Solutions
Generates multiple conceptually distinct solutions (at least 5), each mixing known ideas in uncommon ways. This increases the chance of finding a breakthrough approach.

### 3. Deep-Dive on Each Idea's Mechanism
For each solution, explores the underlying logic, theoretical basis, synergy with constraints, example scenarios, and pros/cons.

### 4. Self-Critique for Gaps & Synergy
The AI critiques each solution for missing details and suggests ways to merge or expand solutions to create stronger approaches.

### 5. Merged Breakthrough Blueprint
Creates a final blueprint that synthesizes the best elements from prior solutions into a coherent design that pushes beyond standard practice.

### 6. Implementation Path & Risk Minimization
Develops a practical path for implementation, focusing on starting small, proving key aspects, and expanding. Identifies resources needed and ways to mitigate risks.

### 7. Cross-Checking with Prior Knowledge
Compares the blueprint with existing projects to determine its novelty and highlight its unique aspects or advantages.

### 8. Q&A or Additional Elaborations
Allows for follow-up questions and clarifications about any aspect of the final blueprint.

## Output Files

### Orchestrator Output Files
The process creates several files in the `some_project/doc/` directory:

- `CONTEXT_CONSTRAINTS.md` - Initial domain understanding and constraints
- `DIVERGENT_SOLUTIONS.md` - Multiple distinct solution approaches
- `DEEP_DIVE_MECHANISMS.md` - Detailed exploration of each solution
- `SELF_CRITIQUE_SYNERGY.md` - Critical analysis and combination opportunities
- `BREAKTHROUGH_BLUEPRINT.md` - The final merged breakthrough idea design
- `IMPLEMENTATION_PATH.md` - Step-by-step implementation plan
- `NOVELTY_CHECK.md` - Analysis of the idea's novelty compared to existing solutions
- `ELABORATIONS.md` - Responses to follow-up questions and additional details

### Proposal Generator Output Files
Depending on which proposal generator you use:

- `ai_prompt.txt` - The prompt sent to the AI model (both generators)
- `cursor_prompt.md` - The prompt for use with Cursor's Claude (from cursor_proposal_generator.py)
- `ai_research_proposal.md` - The formal academic research proposal (from ai_proposal_generator.py)

## Environment Setup

The system requires API keys for the LLM service you choose:

- For Claude 3.7 Sonnet: Set the `ANTHROPIC_API_KEY` environment variable
- For DeepSeek R1: Set the `DEEPSEEK_API_KEY` environment variable

API keys can be set in a `.env` file in the project root directory, which will be automatically loaded.

## Key Features

1. **Structured Ideation**: Follows a carefully designed process that builds on each prior step
2. **Focus on Novelty**: Prompts are designed to encourage cross-domain connections and new combinations
3. **No Disclaimers**: The system instructs the LLM to avoid feasibility disclaimers and focus on solutions
4. **Actionable Output**: The final blueprint includes a practical implementation path
5. **Progressive Refinement**: Each step improves and builds upon previous ideas
6. **Formal Research Proposal**: Converts breakthrough ideas into a structured academic document

## Technical Requirements

- Python 3.6+
- Required packages: `anthropic`, `openai`, `python-dotenv` (see requirements.txt)

## Cross-Platform Compatibility

This tool works on both Windows and Linux/macOS systems:

- **Windows**: File paths in the AI's output may use forward slashes (/) but will be automatically converted to backslashes (\\) when saving files.
- **Linux/macOS**: Standard path handling with forward slashes.

The system uses Python's `pathlib` for platform-independent path handling, ensuring compatibility across different operating systems.

## Complete Workflow Example

1. Generate breakthrough ideas for improving education:
   ```
   python orchestrator.py claude37sonnet "Improving personalized education through AI"
   ```

2. Follow the 8-stage process, reviewing and approving outputs at each stage

3. Generate a formal research proposal:
   ```
   python ai_proposal_generator.py --model claude
   ```
   
   Or if you don't have API keys:
   ```
   python cursor_proposal_generator.py
   ```

4. The final result is a comprehensive research proposal based on your breakthrough idea, ready for academic or funding submission.

## Example

### Sample Domain
```
Improving personalized education through AI while maintaining human connection and addressing individual learning styles
```

### Sample Output Files

After running through the 8-stage process, you might get these files in your `some_project/doc/` directory:

#### BREAKTHROUGH_BLUEPRINT.md (excerpt)
```markdown
# Adaptive Learning Mesh: A Human-AI Educational Ecosystem

The Adaptive Learning Mesh (ALM) combines real-time neurobiological feedback, distributed mentor networks, and anticipatory content shaping to create a personalized education system that enhances rather than replaces human connection.

At its core, ALM uses non-invasive EEG/eye-tracking to detect micro-patterns in student engagement, which feed into a dual-pathway AI system. The first pathway optimizes content delivery and pacing in real-time, while the second pathway connects students with the ideal human mentors at precisely the right intervention points.

Unlike traditional adaptive learning systems that isolate learners, ALM deliberately creates "synchronized learning moments" where students working on similar conceptual challenges are brought together. The system's distributed nature ensures no single AI holds a complete model of any student, preserving privacy while maintaining effectiveness.
```

#### IMPLEMENTATION_PATH.md (excerpt)
```markdown
# Implementation Roadmap

## Phase 1: Core Engagement Engine (3-4 months)
- Develop lightweight EEG + eye-tracking integration
- Train baseline engagement detection models on volunteer dataset
- Create minimal content adaptation API
- Build prototype for 1-2 specific subjects (math and language)

## Phase 2: Mentor Network Framework (2-3 months)
- Develop matching algorithm for student-mentor pairing
- Create intervention triggering system based on engagement signals
- Build mentor dashboard with context awareness
- Test with small group of mentors and students

...
```

These outputs provide a comprehensive blueprint for a breakthrough idea, along with practical steps for implementation.
