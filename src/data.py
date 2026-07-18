"""Number surface forms across scripts, notations, and languages.

THREE axes of variation, ordered by how directly they test a *value-driven* shared helix.
The SCRIPT axis is the headline (it's the least-covered contribution vs prior work like
FARS 2605.09496, which used Latin-script prose only):

  - SCRIPT axis   (same language, same digit notation, only GLYPHS change) -- HEADLINE:
        en_digit "37"  vs  devanagari_digit "३७"  vs  arabic_indic_digit "٣٧"  vs  fullwidth_digit "３７"
        A shared helix here is almost pure evidence of value-driven geometry: the quantity
        is identical, only the code points differ.
  - NOTATION axis (same language, digits vs SPELLED-OUT words):
        en_digit "37"  vs  en_word "thirty-seven"
  - LANGUAGE axis (spelled-out words, meaning fixed, LANGUAGE varies):
        en_word "thirty-seven"  vs  es_word "treinta y siete"  vs  fr_word ...  vs  de_word ...

Every form renders the SAME integer set in the SAME order, so activation rows are paired
across forms (number i -> row i for every form), which is what makes CKA and paired
subspace alignment valid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from num2words import num2words

# --- digit-script translation tables (language stays English, only glyphs change) ---
_DIGIT_MAPS = {
    "devanagari": str.maketrans("0123456789", "०१२३४५६७८९"),
    "arabic_indic": str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"),
    "fullwidth": str.maketrans("0123456789", "０１２３４５６７８９"),
}


def _digit(n: int, script: str | None = None) -> str:
    s = str(n)
    return s if script is None else s.translate(_DIGIT_MAPS[script])


@dataclass(frozen=True)
class Form:
    key: str
    axis: str  # "script" | "notation" | "language"
    render: Callable[[int], str]
    template: str  # must contain "{x}"; carrier sentence that elicits the value


# Default carrier templates are deliberately simple. The number value is most cleanly
# represented at the LAST token of the number span (pooling="last"), which is what we use.
# en_digit is the REFERENCE form that everything is aligned against.
FORMS: dict[str, Form] = {
    # --- reference ---
    "en_digit": Form("en_digit", "script", lambda n: _digit(n), "The number {x} is"),
    # --- SCRIPT axis (headline): same language + notation, only glyphs change ---
    "devanagari_digit": Form("devanagari_digit", "script", lambda n: _digit(n, "devanagari"), "The number {x} is"),
    "arabic_indic_digit": Form("arabic_indic_digit", "script", lambda n: _digit(n, "arabic_indic"), "The number {x} is"),
    "fullwidth_digit": Form("fullwidth_digit", "script", lambda n: _digit(n, "fullwidth"), "The number {x} is"),
    # --- NOTATION axis: digits vs spelled-out words, same language ---
    "en_word": Form("en_word", "notation", lambda n: num2words(n, lang="en"), "The number {x} is"),
    # --- LANGUAGE axis: spelled-out words, language varies ---
    "es_word": Form("es_word", "language", lambda n: num2words(n, lang="es"), "El número {x} es"),
    "fr_word": Form("fr_word", "language", lambda n: num2words(n, lang="fr"), "Le nombre {x} est"),
    "de_word": Form("de_word", "language", lambda n: num2words(n, lang="de"), "Die Zahl {x} ist"),
}

# Headline default: reference + full SCRIPT axis first, then one NOTATION and two LANGUAGE
# points as secondary contrast. Cross-script sharing is the least-covered contribution.
DEFAULT_FORMS = [
    "en_digit",           # reference
    "devanagari_digit",   # script
    "arabic_indic_digit", # script
    "fullwidth_digit",    # script
    "en_word",            # notation
    "es_word",            # language
    "fr_word",            # language
]


def build_prompts(form_key: str, numbers: list[int]) -> list[tuple[int, str, str]]:
    """Return list of (number, rendered_number_string, full_prompt)."""
    form = FORMS[form_key]
    out = []
    for n in numbers:
        rendered = form.render(n)
        prompt = form.template.format(x=rendered)
        out.append((n, rendered, prompt))
    return out
