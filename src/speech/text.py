"""Conservative spoken identifiers and speech-friendly formatting."""

import re
from ..redaction import normalize_digits

DIGITS = {word: str(i) for i, forms in enumerate((('zero','शून्य','जीरो'), ('one','एक'), ('two','दो'),
    ('three','तीन'), ('four','चार'), ('five','पांच','पाँच'), ('six','छह','छः'), ('seven','सात'),
    ('eight','आठ'), ('nine','नौ'))) for word in forms}
WORDS = '|'.join(sorted(DIGITS, key=len, reverse=True))
RUN = re.compile(r'(?<!\w)(?:' + WORDS + r')(?:[\s,\-]+(?:' + WORDS + r')){1,}(?!\w)', re.I)


def normalize_identifiers(text, *, expecting_identifier=False):
    text = normalize_digits(text)
    relevant = expecting_identifier or bool(re.search(r'order|phone|last four|pincode|pin code|digits|ऑर्डर|अंक|पिनकोड|फ़ोन', text, re.I))
    if not relevant:
        return text, False
    ambiguous = bool(re.search(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine)\b(?:[\s,]+(?:to|too|for|oh)\b)|\b(?:to|too|for|oh)\b[\s,]+(?:zero|one|two|three|four|five|six|seven|eight|nine)\b', text, re.I))
    def replace(match):
        values = re.findall(WORDS, match[0], re.I)
        return ''.join(DIGITS[word.casefold()] for word in values)
    return RUN.sub(replace, text), ambiguous


def spoken_reply(text, language='en'):
    text = re.sub(r'₹\s*([\d,]+(?:\.\d+)?)', lambda m: m[1] + (' रुपये' if language == 'hi' else ' rupees'), text)
    def identifier(match):
        digits = re.sub(r'\D', '', match[2])
        names = ('शून्य','एक','दो','तीन','चार','पाँच','छह','सात','आठ','नौ') if language == 'hi' else ('zero','one','two','three','four','five','six','seven','eight','nine')
        groups = [' '.join(names[int(c)] for c in digits[i:i+4]) for i in range(0,len(digits),4)]
        return match[1] + ', '.join(groups)
    text = re.sub(r'((?:order(?: ID| number)?|RRN|reference number|ऑर्डर नंबर)\s*[:#]?\s*)(\d[\d -]{6,14}\d)', identifier, text, flags=re.I)
    text = re.sub(r'\*\*|`|#{1,3}\s*', '', text)
    return text.strip()
