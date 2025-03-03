# Breakthrough-Idea Walkthrough Framework

This project implements an AI orchestrator system designed to walk an LLM (Large Language Model) through a structured process for generating breakthrough ideas. The system uses a carefully designed 8-stage framework that maximizes novelty while producing actionable and implementable ideas.

## Overview

The "Breakthrough-Idea Walkthrough" Framework is an eight-stage structure that guides an LLM through a sequence of prompts, each designed to build upon previous outputs. The framework progressively develops a novel idea from initial domain understanding to a complete, actionable blueprint.

## How to Use

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

The process creates several files in the `some_project/doc/` directory:

- `BREAKTHROUGH_BLUEPRINT.md` - The final merged breakthrough idea design
- `IMPLEMENTATION_PATH.md` - Step-by-step implementation plan
- `NOVELTY_CHECK.md` - Analysis of the idea's novelty compared to existing solutions
- `ELABORATIONS.md` - Responses to follow-up questions and additional details

## Environment Setup

The system requires API keys for the LLM service you choose:

- For Claude 3.7 Sonnet: Set the `ANTHROPIC_API_KEY` environment variable
- For DeepSeek R1: Set the `DEEPSEEK_API_KEY` environment variable

## Key Features

1. **Structured Ideation**: Follows a carefully designed process that builds on each prior step
2. **Focus on Novelty**: Prompts are designed to encourage cross-domain connections and new combinations
3. **No Disclaimers**: The system instructs the LLM to avoid feasibility disclaimers and focus on solutions
4. **Actionable Output**: The final blueprint includes a practical implementation path
5. **Progressive Refinement**: Each step improves and builds upon previous ideas

## Technical Requirements

- Python 3.6+
- Required packages: `anthropic`, `openai` (see requirements.txt)

## Cross-Platform Compatibility

This tool works on both Windows and Linux/macOS systems:

- **Windows**: File paths in the AI's output may use forward slashes (/) but will be automatically converted to backslashes (\\) when saving files.
- **Linux/macOS**: Standard path handling with forward slashes.

The system uses Python's `pathlib` for platform-independent path handling, ensuring compatibility across different operating systems.

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
