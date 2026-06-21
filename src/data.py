"""Number surface forms across languages and scripts.

Two axes of variation we care about:
  - SCRIPT/FORM axis (hold language ~fixed, vary the glyphs/format):
        en_digit "37"  vs  devanagari_digit "३७"  vs  arabic_indic_digit "٣٧"  vs  en_word "thirty-seven"
  - LANGUAGE axis (hold meaning fixed, vary the language of the number word):
        en_word "thirty-seven"  vs  es_word "treinta y siete"  vs  fr_word ...  vs  de_word ...

Every form renders the SAME integer set, so activation rows are paired across forms
(number i -> row i for every form), which is what makes CKA and paired alignment valid.
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
    axis: str  # "script" or "language"
    render: Callable[[int], str]
    template: str  # must contain "{x}"; carrier sentence that elicits the value


# Default carrier templates are deliberately simple. The number value is most cleanly
# represented at the LAST token of the number span (pooling="last"), which is what we use.
FORMS: dict[str, Form] = {
    "en_digit": Form("en_digit", "script", lambda n: _digit(n), "The number {x} is"),
    "en_word": Form("en_word", "script", lambda n: num2words(n, lang="en"), "The number {x} is"),
    "devanagari_digit": Form("devanagari_digit", "script", lambda n: _digit(n, "devanagari"), "The number {x} is"),
    "arabic_indic_digit": Form("arabic_indic_digit", "script", lambda n: _digit(n, "arabic_indic"), "The number {x} is"),
    "es_word": Form("es_word", "language", lambda n: num2words(n, lang="es"), "El número {x} es"),
    "fr_word": Form("fr_word", "language", lambda n: num2words(n, lang="fr"), "Le nombre {x} est"),
    "de_word": Form("de_word", "language", lambda n: num2words(n, lang="de"), "Die Zahl {x} ist"),
}

DEFAULT_FORMS = ["en_digit", "en_word", "devanagari_digit", "es_word", "fr_word"]


def build_prompts(form_key: str, numbers: list[int]) -> list[tuple[int, str, str]]:
    """Return list of (number, rendered_number_string, full_prompt)."""
    form = FORMS[form_key]
    out = []
    for n in numbers:
        rendered = form.render(n)
        prompt = form.template.format(x=rendered)
        out.append((n, rendered, prompt))
    return out
