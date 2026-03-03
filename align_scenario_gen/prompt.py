SYSTEM_PROMPT = """\
You are an expert scenario designer for decision-making research. \
You turn scenario descriptions into concrete, vivid decision scenarios."""


def build_user_prompt(description: str, choices: list[str]) -> str:
    choices_text = "\n".join(f"- {c}" for c in choices)
    return f"""\
Turn the following scenario description into a concrete decision scenario \
where the decision maker must choose between the given options.

SCENARIO DESCRIPTION:
{description}

ACTION CHOICES (use these exactly):
{choices_text}

Requirements:
- Write a vivid 2-4 paragraph narrative placing the reader as the decision maker
- The narrative must naturally lead to a decision between the given choices
- End the narrative by presenting the choice

Respond with ONLY valid JSON in this exact format:
{{
  "unstructured": "The full narrative scenario text..."
}}"""
