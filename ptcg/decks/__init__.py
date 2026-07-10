from pathlib import Path

_DECK_DIR = Path(__file__).parent


def load_deck(name: str) -> list[int]:
    """Load a 60-card deck from a CSV file (one card ID per line)."""
    path = _DECK_DIR / f"{name}.csv"
    deck = [int(line) for line in path.read_text().splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"Deck '{name}' has {len(deck)} cards, expected 60.")
    return deck
