# =============================================================================
# Testu-normalizazioa TTS-rako
# =============================================================================
# Pocket TTS-ek testua "irakurtzen den moduan" behar du: zenbakiak, ehunekoak
# eta ikurrak hitzez. Euskarazko zenbakiak hemen bertan sortzen dira
# (num2words-ek ez du euskara onartzen); gaztelaniarako num2words erabiltzen da.
# =============================================================================

import re

from num2words import num2words

# ---------------------------------------------------------------------------
# Euskarazko zenbakiak (sistema hogeitarra)
# ---------------------------------------------------------------------------
_EU_UNITS = [
    "zero", "bat", "bi", "hiru", "lau", "bost", "sei", "zazpi", "zortzi",
    "bederatzi", "hamar", "hamaika", "hamabi", "hamahiru", "hamalau",
    "hamabost", "hamasei", "hamazazpi", "hemezortzi", "hemeretzi",
]
_EU_TWENTIES = {1: "hogei", 2: "berrogei", 3: "hirurogei", 4: "laurogei"}
_EU_HUNDREDS = {
    1: "ehun", 2: "berrehun", 3: "hirurehun", 4: "laurehun", 5: "bostehun",
    6: "seiehun", 7: "zazpiehun", 8: "zortziehun", 9: "bederatziehun",
}


def _eu_below_100(n: int) -> str:
    if n < 20:
        return _EU_UNITS[n]
    tw, rest = divmod(n, 20)
    if rest == 0:
        return _EU_TWENTIES[tw]
    return f"{_EU_TWENTIES[tw]}ta {_EU_UNITS[rest]}"


def _eu_below_1000(n: int) -> str:
    h, rest = divmod(n, 100)
    if h == 0:
        return _eu_below_100(rest)
    if rest == 0:
        return _EU_HUNDREDS[h]
    return f"{_EU_HUNDREDS[h]} eta {_eu_below_100(rest)}"


def _eu_join(head: str, rest: int) -> str:
    """'mila' + gainerakoa: ehunekoak edo <100 bakarrik badira 'eta' tartean."""
    if rest == 0:
        return head
    tail = _eu_below_1000(rest)
    if rest < 100 or rest % 100 == 0:
        return f"{head} eta {tail}"
    return f"{head} {tail}"


def eu_number(n: int) -> str:
    if n < 0:
        return "minus " + eu_number(-n)
    if n < 1000:
        return _eu_below_1000(n)
    if n < 1_000_000:
        th, rest = divmod(n, 1000)
        head = "mila" if th == 1 else f"{_eu_below_1000(th)} mila"
        return _eu_join(head, rest)
    if n < 1_000_000_000_000:
        mi, rest = divmod(n, 1_000_000)
        head = "milioi bat" if mi == 1 else f"{eu_number(mi)} milioi"
        if rest == 0:
            return head
        if rest < 1000:
            return _eu_join(head, rest)
        return f"{head} {eu_number(rest)}"
    return " ".join(_EU_UNITS[int(d)] for d in str(n))


def es_number(n: int) -> str:
    return num2words(n, lang="es")


# ---------------------------------------------------------------------------
# Testu osoa
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"(?<![\w])(%\s?)?(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(\s?%)?")

_SYMBOLS = {
    "eu": {"€": " euro", "$": " dolar", "&": " eta ", "+": " gehi ", "=": " berdin "},
    "es": {"€": " euros", "$": " dólares", "&": " y ", "+": " más ", "=": " igual a "},
}
_WORDS = {"eu": {"comma": "koma"}, "es": {"comma": "coma"}}


def _say_number(lang: str, integer: int) -> str:
    return eu_number(integer) if lang == "eu" else es_number(integer)


def _es_apocope(words: str) -> str:
    """'veintiuno años' -> 'veintiún años', 'uno euros' -> 'un euros'."""
    if words.endswith("veintiuno"):
        return words[:-len("veintiuno")] + "veintiún"
    if words == "uno" or words.endswith(" uno"):
        return words[:-3] + "un"
    return words


def _replace_number(lang: str, m: re.Match) -> str:
    integer = int(m.group(2).replace(".", ""))
    decimals = m.group(3)
    percent = m.group(1) or m.group(4)
    words = _say_number(lang, integer)
    if decimals:
        words += f" {_WORDS[lang]['comma']} " + (
            _say_number(lang, int(decimals)) if not decimals.startswith("0")
            else " ".join(_say_number(lang, int(d)) for d in decimals)
        )
    elif lang == "es" and not percent and re.match(r"\s+[^\W\d_]", m.string[m.end():]):
        words = _es_apocope(words)
    if percent:
        words = f"ehuneko {words}" if lang == "eu" else f"{words} por ciento"
    return words


def normalize(text: str, lang: str) -> str:
    text = text.replace(" ", " ")
    # Hamartarrak puntuarekin ("3.5") komara pasa, milakoak ez ukitzeko
    text = re.sub(r"(?<=\d)\.(?=\d{1,2}(?!\d))", ",", text)
    for sym, word in _SYMBOLS.get(lang, {}).items():
        text = text.replace(sym, word)
    text = _NUM_RE.sub(lambda m: _replace_number(lang, m), text)
    return " ".join(text.split())
