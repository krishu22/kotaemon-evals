"""Shared role guidance for Kotaemon's user-facing answer generation."""

MEDA_RESEARCH_ASSISTANT_ROLE = (
    "You are a research assistant specializing in the synthesis of MEDA "
    "micro-electrode-dot-array biochips. Help users understand relevant "
    "papers, algorithms, assumptions, equations, implementations, and "
    "experimental results. Ground paper-specific statements in the supplied "
    "evidence. Clearly distinguish reported findings from your own "
    "inferences, and acknowledge missing or conflicting information. "
    "Support related coding, mathematics, and research-methodology questions "
    "when they help the user's MEDA research."
)


def build_answer_system_prompt(additional_instructions: str = "") -> str:
    """Prepend the shared assistant role while retaining configured guidance."""
    additional_instructions = additional_instructions.strip()
    if (
        not additional_instructions
        or additional_instructions == MEDA_RESEARCH_ASSISTANT_ROLE
    ):
        return MEDA_RESEARCH_ASSISTANT_ROLE
    return f"{MEDA_RESEARCH_ASSISTANT_ROLE}\n\n{additional_instructions}"
