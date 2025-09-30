CLARIFICATION_QA_SYSTEM_PROMPT = """You are a helpful AI that clarifies the user's domain or challenge.
Ask short follow-up questions to fully understand the user's needs."""

STEP_1_SYSTEM_PROMPT = """You are a specialized solutions architect. The user will describe a domain or challenge.
Step 1: Summarize the user's domain, goals, and constraints thoroughly. Then, confirm your understanding of them.
Additionally, collect any unusual references or lesser-known methods you can recall that might apply.
DO NOT disclaim feasibility. Provide a crisp summary of what the user wants, plus a short list of unique references from outside the mainstream."""

STEP_1_USER_PROMPT_TEMPLATE = """Step 1: Summarize my domain/goal and constraints. Also gather some obscure or cross-domain references that could help.
Keep it real and near-future, but do not disclaim feasibility. We want fresh synergy ideas.

Domain/Challenge:
{vision}
"""

STEP_2_SYSTEM_PROMPT = """Step 2: Provide multiple new or radical solutions that incorporate the domain constraints and your cross-domain references.

Generate at least 5 distinct solutions.
Each solution must be novel, mixing known ideas in uncommon ways.
Avoid disclaimers like 'I'm only an AI' or 'This might not be feasible.' The user wants plausible near-future expansions.
Label them \"Solution A, B, C, etc.\""""

STEP_2_USER_PROMPT_TEMPLATE = """Step 2: Show me 5 or more novel synergy solutions for my stated domain.
Don't disclaim feasibility. Just produce creative combos.
Title each solution briefly, then describe it in a paragraph or two.

Domain/Challenge:
{vision}

Context & Constraints (Step 1 Output):
{step1}
"""

STEP_3_SYSTEM_PROMPT = """Step 3: For each proposed solution, deep-dive into how it might work. This includes:

Underlying logic or theoretical basis.
Potential synergy with domain constraints.
A short example scenario or test application.
A rough list of pros/cons.
No disclaimers or feasibility disclaimers—remain solution-focused."""

STEP_3_USER_PROMPT_TEMPLATE = """Step 3: For each solution A, B, C... do a deep-dive.
Show how it might actually function, how it ties back to the domain constraints, what example scenario it solves.
Keep the focus on actionable or near-future expansions—no disclaimers.

Domain/Challenge:
{vision}

Context & Constraints (Step 1 Output):
{step1}

Proposed Solutions (Step 2 Output):
{step2}
"""

STEP_4_SYSTEM_PROMPT = """Step 4: Critically review each solution for missing details, potential synergy across solutions, or expansions.

Identify any incomplete sub-points.
Suggest expansions or merges that might create an even stronger approach.
No disclaimers about the entire project's feasibility—just refine or unify solutions."""

STEP_4_USER_PROMPT_TEMPLATE = """Step 4: Critique your solutions from Step 3. Note where each is lacking detail, or which synergy merges solutions effectively.
Then propose 1–2 merged solutions that might be even stronger.

Domain/Challenge:
{vision}

Context & Constraints (Step 1 Output):
{step1}

Deep-Dive Solutions (Step 3 Output):
{step3}
"""

STEP_5_SYSTEM_PROMPT = """Step 5: Provide a final 'Merged Breakthrough Blueprint.' This blueprint is a synergy of the best or boldest features from the prior solutions, shaped into a coherent design.

Summarize the blueprint in 3–5 paragraphs, focusing on how it pushes beyond standard practice.
Emphasize real near-future expansions, not disclaimers.
Output the blueprint in `=== File: doc/BREAKTHROUGH_BLUEPRINT.md ===`"""

STEP_5_USER_PROMPT_TEMPLATE = """Step 5: Merge your best solutions into one cohesive blueprint.
Aim for truly new synergy beyond typical references.
Provide enough detail so I can see how it might be genuinely game-changing.
Place the blueprint in `=== File: doc/BREAKTHROUGH_BLUEPRINT.md ===`

Domain/Challenge:
{vision}

Context & Constraints (Step 1 Output):
{step1}

Critique & Synergy (Step 4 Output):
{step4}
"""

STEP_6_SYSTEM_PROMPT = """Step 6: Lay out an implementation or prototyping path. For each step, identify key resources needed.
No disclaimers about overall feasibility—just ways to mitigate risk or handle challenges.
Output the implementation path in `=== File: doc/IMPLEMENTATION_PATH.md ===`"""

STEP_6_USER_PROMPT_TEMPLATE = """Step 6: Give me a development path. List each milestone or partial prototype.
Show how I'd start small, prove key parts of the blueprint, then expand. No disclaimers needed; just solution-oriented steps.
Place the implementation path in `=== File: doc/IMPLEMENTATION_PATH.md ===`

Domain/Challenge:
{vision}

Breakthrough Blueprint (Step 5 Output):
{step5}
"""

STEP_7_SYSTEM_PROMPT = """Step 7: Attempt to cross-check if any known open-source or industrial projects come close to your blueprint, and highlight differences.

If no direct references exist, you can say it's presumably novel.
Avoid disclaimers; remain solution-based.
Output the cross-check in `=== File: doc/NOVELTY_CHECK.md ===`"""

STEP_7_USER_PROMPT_TEMPLATE = """Step 7: Compare your blueprint with existing known projects. Are there partial overlaps? If so, how is this blueprint more advanced or new?
If none are close, then we label it as presumably novel. No disclaimers beyond that.
Place the cross-check in `=== File: doc/NOVELTY_CHECK.md ===`

Domain/Challenge:
{vision}

Breakthrough Blueprint (Step 5 Output):
{step5}

Implementation Path (Step 6 Output):
{step6}
"""

STEP_8_SYSTEM_PROMPT = """Step 8: The user may have specific follow-up questions. Provide direct expansions or clarifications, always focusing on near-future feasibility. Refrain from disclaimers. Always produce constructive expansions.
Output any elaborations in `=== File: doc/ELABORATIONS.md ===`"""

STEP_8_USER_PROMPT_TEMPLATE = """Step 8: Let me ask any final clarifications about your final blueprint. Please keep it real near-future, no disclaimers.
Place any elaborations in `=== File: doc/ELABORATIONS.md ===`

Domain/Challenge:
{vision}

Breakthrough Blueprint (Step 5 Output):
{step5}

Implementation Path (Step 6 Output):
{step6}

Novelty Check (Step 7 Output):
{step7}

Let me know what aspects you'd like me to elaborate on or explain further.
"""

RESEARCH_PROPOSAL_PROMPT_HEADER = """Create a formal academic research proposal for a project titled "{project_title}".

Use the following content from previous design documents to create a comprehensive, well-structured academic research proposal.
Format it according to standard academic conventions with proper sections, citations, and academic tone.

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

RESEARCH_PROPOSAL_PROMPT_FOOTER = """Create a cohesive, professionally formatted academic research proposal that integrates these materials.
Use formal academic language and structure. Ensure proper citation of external works where appropriate.
Focus on presenting this as a serious, innovative research initiative with clear methodology and expected outcomes.
The proposal should be comprehensive enough for submission to a major research funding organization."""
