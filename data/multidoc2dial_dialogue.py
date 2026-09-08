"""Render verified raw dialogue turns instead of the reversed MRC question field."""
VERSION = "multidoc2dial-chronological-v1"


def chronological_question(row, dialogue):
    turns = dialogue["turns"]
    if isinstance(turns, dict):
        lengths = {len(values) for values in turns.values()}
        if len(lengths) != 1:
            raise ValueError("Inconsistent dialogue columns")
        turns = [dict(zip(turns, values)) for values in zip(*turns.values())]
    matches = [i for i, turn in enumerate(turns)
               if f'{dialogue["dial_id"]}_{turn["turn_id"]}' == row["id"]]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous raw dialogue turn")
    index = matches[0]
    if turns[index]["role"] != "user" or index + 1 >= len(turns):
        raise ValueError("Expected answerable current user turn")
    following = turns[index + 1]
    if (following["role"] != "agent" or following["da"] == "respond_no_solution"
            or following["utterance"] != row["utterance"]):
        raise ValueError("Reference answer differs from raw next agent turn")
    normalized = []
    for turn in turns[:index + 1]:
        if turn["role"] not in {"user", "agent"}:
            raise ValueError("Unknown dialogue role")
        normalized.append((turn["role"], turn["utterance"].replace("\n", " ").replace("\t", " ")))
    history = [f"{role}: {text}" for role, text in normalized[:-1]]
    current = normalized[-1][1]
    encoded = current + "[SEP]" + "||".join(reversed(history))
    if encoded != row["question"]:
        raise ValueError("Raw dialogue does not reconstruct pinned MRC question")
    history_text = "\n".join(history) if history else "(No prior turns.)"
    return ("Conversation history (oldest first; agent means the prior assistant):\n"
            + history_text + "\n\nCurrent user request:\n" + current
            + "\n\nAnswer the current user request using the source documents.")
