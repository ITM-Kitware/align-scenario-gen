SYSTEM_PROMPT = """\
You are an expert scenario designer for decision-making research. \
You turn scenario descriptions into concrete, vivid decision scenarios."""


def build_user_prompt(description: str, config: dict) -> str:
    resolved = config["_resolved"]
    frame = resolved["frame"]
    behavior = resolved["behavior"]
    choices_text = "\n".join(f"- {choice['label']}" for choice in frame["choices"])
    tensions_text = _format_active_kdmas(behavior.get("active_kdmas", {}))
    guidance_text = _format_guidance(behavior.get("generation_guidance", []))

    requirements = _requirements_for_frame(frame["prompt_template"])

    return f"""\
Turn the following scenario description into a concrete decision scenario \
where the decision maker must choose between the given options.

SCENARIO DESCRIPTION:
{description}

TARGET BEHAVIOR:
{behavior.get('description_hint', 'Surface the intended KDMA tensions clearly in the final scenario.')}

INTENDED KDMA TENSIONS:
{tensions_text}

ADDITIONAL GENERATION GUIDANCE:
{guidance_text}

ACTION CHOICES (use these exactly):
{choices_text}

Requirements:
- {requirements}
- Be concise and direct — no filler or elaboration
- Use the action choices exactly as written
- End with the decision question implied by the frame

Respond with ONLY valid JSON in this exact format:
{{
  "unstructured": "The full narrative scenario text..."
}}"""


def _requirements_for_frame(prompt_template: str) -> str:
    if prompt_template == "move_vs_wait":
        return (
            "Write exactly 3 sentences: (1) the tactical setting and threat context, "
            "(2) the casualty and exposure details, (3) the direct move-versus-wait question"
        )
    if prompt_template == "stay_vs_search":
        return (
            "Write exactly 3 sentences: (1) the mass-casualty setting, "
            "(2) the current patient's condition, (3) the evidence about other possible casualties and the direct choice"
        )
    return (
        "Write exactly 3 sentences: (1) the setting, "
        "(2) Patient A's injury and background, (3) Patient B's injury and background"
    )


def _format_active_kdmas(active_kdmas: dict) -> str:
    if not active_kdmas:
        return "- No explicit KDMA metadata provided."
    lines = []
    for kdma_name, tension in active_kdmas.items():
        favors = tension.get("favors", "unspecified option")
        strength = tension.get("strength", "unspecified")
        lines.append(f"- {kdma_name}: favors {favors} with {strength} strength")
    return "\n".join(lines)


def _format_guidance(guidance: list[str]) -> str:
    if not guidance:
        return "- No additional guidance."
    return "\n".join(f"- {item}" for item in guidance)
