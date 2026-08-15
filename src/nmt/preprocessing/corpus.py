import json

LANG_CODES = {"Bengali": "bn", "Hindi": "hi"}


def read_split(path, field="source"):
    """Flatten the competition JSON into {pair: {ids, source, target}} lists."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    out = {}
    for pair, sections in raw.items():
        ids, sources, targets = [], [], []
        for entries in sections.values():
            for entry_id, entry in entries.items():
                ids.append(entry_id)
                sources.append(entry["source"])
                targets.append(entry.get("target"))
        out[pair] = {"ids": ids, "source": sources, "target": targets}
    return out
