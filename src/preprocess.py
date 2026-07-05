"""Text cleaning for Hinglish cyber-crime complaint narratives."""

import re
import unicodedata

import pandas as pd

HINGLISH_NORM = {
    r'\bnhi\b': 'nahi',
    r'\bnahin\b': 'nahi',
    r'\bnhin\b': 'nahi',
    r'\bnai\b': 'nahi',
    r'\bhay\b': 'hai',
    r'\bhain\b': 'hai',
    r'\bkia\b': 'kiya',
    r'\bap\b': 'aap',
    r'\bmuje\b': 'mujhe',
    r'\bmuze\b': 'mujhe',
    r'\bmujhse\b': 'mujhse',
    r'\bmra\b': 'mera',
    r'\bmere\b': 'mere',
    r'\bkaro\b': 'karo',
    r'\bkarna\b': 'karna',
    r'\bwo\b': 'woh',
    r'\bvoh\b': 'woh',
    r'\bvo\b': 'woh',
    r'\bunhe\b': 'unhe',
    r'\bunko\b': 'unhe',
    r'\bbh\b': 'bhi',
    r'\bkr\b': 'kar',
    r'\bphir\b': 'phir',
    r'\bfir\b': 'phir',
    r'\babi\b': 'abhi',
    r'\bpaisa\b': 'paise',
    r'\bpesa\b': 'paise',
    r'\bpese\b': 'paise',
    r'\bbar\b': 'baar',
    r'\bgya\b': 'gaya',
    r'\bgaye\b': 'gaye',
    r'\bdiya\b': 'diya',
    r'\bdia\b': 'diya',
    r'\bliya\b': 'liya',
    r'\blia\b': 'liya',
    r'\brha\b': 'raha',
    r'\brhi\b': 'rahi',
    r'\bfroud\b': 'fraud',
    r'\bfraud\b': 'fraud',
    r'\bfruad\b': 'fraud',
    r'\bfaud\b': 'fraud',
    r'\bfrauded\b': 'fraud',
    r'\bfrudster\b': 'fraudster',
    r'\bfraudstar\b': 'fraudster',
    r'\bfraudulent\b': 'fraudulent',
    r'\bharrased\b': 'harassed',
    r'\bharased\b': 'harassed',
    r'\bharrasment\b': 'harassment',
    r'\brecieved\b': 'received',
    r'\brecived\b': 'received',
    r'\brecivied\b': 'received',
    r'\baccout\b': 'account',
    r'\baccount\b': 'account',
    r'\baccound\b': 'account',
    r'\btransation\b': 'transaction',
    r'\btransection\b': 'transaction',
    r'\bblackmaling\b': 'blackmailing',
    r'\bvictim\b': 'victim',
    r'\bvictm\b': 'victim',
    r'\bimmediately\b': 'immediately',
    r'\bimmidiately\b': 'immediately',
    r'\bimmeditely\b': 'immediately',
}

_HINGLISH_COMPILED = [
    (re.compile(pat, re.IGNORECASE), replacement)
    for pat, replacement in HINGLISH_NORM.items()
]


def clean_text(
    text: str,
    apply_hinglish_norm: bool = True,
    apply_entity_tags: bool = False,
) -> str:
    """Clean a single complaint: normalize unicode, strip boilerplate, Hinglish typos."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)

    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    text = text.replace('\r\n', ' ').replace('\r', ' ')

    text = re.sub(
        r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[a-zA-Z0-9.-]+\.(com|org|net|in|co\.in|gov\.in|edu)[^\s]*',
        '[URL]', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,7}\b',
        '[EMAIL]', text
    )

    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if upper_ratio > 0.6:
            text = text.lower()
    text = text.lower()

    text = re.sub(
        r'^[\s,]*(?:'
        r'(?:respected|dear)[\s,]+(?:sir|ma\'?am|madam|officer|team|cyber\s*crime\s*(?:team|cell|branch|department|portal)?)'
        r'|sir[\s,]+'
        r'|hello[\s,]+(?:sir|ma\'?am|team)?'
        r'|hi[\s,]+(?:sir|ma\'?am)?'
        r'|greetings[\s,]*'
        r'|to[\s,]+(?:whomsoever|whom)[\s\w,]+'
        r'|(?:good\s+)?(?:morning|afternoon|evening)[\s,]*(?:sir|ma\'?am)?'
        r')[\s,.:]*',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'[\s,.:]*(?:'
        r'please\s+(?:help\s+(?:me|us)?|take\s+(?:immediate\s+)?action|do\s+(?:the\s+)?needful|look\s+into\s+(?:this|the\s+matter)|investigate|solve\s+(?:this|the\s+matter)?)'
        r'|kindly\s+(?:help|take\s+action|look\s+into)'
        r'|(?:thanking\s+you|thank\s+you|thanks)[\s,]*(?:in\s+advance)?'
        r'|(?:warm\s+)?regards[\s,]*'
        r'|your\s+(?:sincerely|faithfully|obediently|truly)'
        r')[\s.]*$',
        '', text, flags=re.IGNORECASE
    )

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    text = re.sub(r'[-_=*]{3,}', ' ', text)
    text = re.sub(r'[*#@~^]{2,}', ' ', text)

    for _ in range(2):
        text = re.sub(r'\b(\w{2,})\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    if apply_hinglish_norm:
        for pattern, replacement in _HINGLISH_COMPILED:
            text = pattern.sub(replacement, text)

    text = re.sub(r'\s+', ' ', text).strip()

    if apply_entity_tags:
        text = _inject_entity_tags(text)

    return text


def _inject_entity_tags(text: str) -> str:
    replacements = [
        (r'\b(?:paytm|phonepe|google\s*pay|gpay|bhim|amazon\s*pay|mobikwik|freecharge)\b', '[FIN_APP]'),
        (r'\b(?:whatsapp|facebook|instagram|telegram|youtube|twitter|snapchat|tiktok|linkedin|discord)\b', '[SOCIAL]'),
        (r'\b(?:sbi|hdfc|icici|kotak|axis\s*bank|pnb|canara|union\s*bank)\b', '[BANK]'),
        (r'\b(?:anydesk|teamviewer)\b', '[REMOTE]'),
    ]
    for pat, tag in replacements:
        text = re.sub(pat, tag, text, flags=re.IGNORECASE)
    return text


def smart_truncate(text: str, max_words: int = 200) -> str:
    """Keep first and last halves of long texts with a [TRUNC] marker."""
    words = text.split()
    if len(words) <= max_words:
        return text
    half = max_words // 2
    return ' '.join(words[:half]) + ' [TRUNC] ' + ' '.join(words[-half:])
