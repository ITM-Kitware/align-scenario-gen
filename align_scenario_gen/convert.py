def scenario_to_record(scenario: dict, scenario_id: str, scene_index: int, resolved: dict | None = None) -> dict:
    sid = str(scene_index)
    resolved = resolved or {}
    metadata = {
        "scene_id": sid,
        "behavior_id": resolved.get("behavior_id"),
        "frame_id": resolved.get("frame_id"),
        "intended_kdma_tensions": resolved.get("behavior", {}).get("active_kdmas"),
    }
    choices = [
        {
            "action_id": f"{sid}.action_{i}",
            "action_type": "SITREP",
            "unstructured": choice["label"],
            **({"character_id": choice["character_id"]} if "character_id" in choice else {}),
            "kdma_association": None,
        }
        for i, choice in enumerate(scenario["choices"])
    ]
    return {
        "input": {
            "scenario_id": scenario_id,
            "full_state": {
                "unstructured": scenario["unstructured"],
                "meta_info": metadata,
                "scenario_complete": False,
            },
            "choices": choices,
        }
    }
