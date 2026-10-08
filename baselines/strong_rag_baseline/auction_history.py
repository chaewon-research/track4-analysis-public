"""Historical Treasury auction fallback using only pre-cutoff evidence."""

import math
import re
from statistics import mean


def auction_mean6_fallback(task, entity, corpus):
    target = task.get("target") or {}

    if (
        task.get("family") != "auction_demand"
        or target.get("type") != "regression"
        or target.get("name") != "bid_to_cover_ratio"
    ):
        return None

    tenor = entity.get("tenor")
    cutoff = task.get("cutoff_date")

    if not isinstance(tenor, str) or not isinstance(cutoff, str):
        return None

    history = {}

    for doc_id, text in corpus.doc_texts.items():
        doc_date = corpus.doc_dates.get(doc_id)

        if not isinstance(doc_date, str) or doc_date > cutoff:
            continue

        heading = text.splitlines()[0] if text else ""

        if "auction" not in heading.lower():
            continue

        pattern = rf"(?<![0-9]){re.escape(tenor)}(?![0-9])"

        if not re.search(pattern, heading, re.IGNORECASE):
            continue

        for line in text.splitlines():
            cells = [cell.strip() for cell in line.split("|")]

            if len(cells) != 7:
                continue

            date = cells[0]

            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                continue

            if date > cutoff:
                continue

            try:
                ratio = float(cells[4])
            except ValueError:
                continue

            if not math.isfinite(ratio) or ratio <= 0:
                continue

            if date in history and history[date] != ratio:
                return None

            history[date] = ratio

    rows = sorted(history.items())

    if len(rows) < 6:
        return None

    return mean(value for _, value in rows[-6:])
