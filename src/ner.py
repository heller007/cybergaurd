"""
src/ner.py

Rule-based Named Entity Recognition for Indian cyber-crime complaint text.

All patterns are keyword/phrase based — no digit patterns because all PII 
(phone numbers, account numbers, amounts in digits) has been pre-stripped 
from the dataset.

Produces:
  1. A 25-dim binary feature vector (0/1 presence of each entity type)
  2. Annotated text (optional) with entity spans replaced by type tags
  3. Entity list (type, matched_text) for interpretability

These features discriminate categories far better than the model alone:
  - ransomware_kw  → 48% prevalence in Ransomware class, ~0% elsewhere
  - crypto         → 48% prevalence in Cryptocurrency class
  - email_mention  → 100% prevalence in Cyber Attack class
  - threatening    → 93% prevalence in RapeGang class
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# ENTITY DEFINITIONS
# Each entry: (feature_name, regex_pattern, replacement_tag)
# ═══════════════════════════════════════════════════════════════════════════

ENTITY_PATTERNS: List[Tuple[str, str, str]] = [

    # ── Financial platforms & instruments ──────────────────────────────────
    ("otp_mention",
     r"\botp\b",
     "[OTP]"),

    ("upi_mention",
     r"\bupi\b|\bunified\s+payment\s+interface\b",
     "[UPI]"),

    ("fin_app",
     r"\bpaytm\b|\bphonepe\b|\bgoogle\s*pay\b|\bgpay\b|\bbhim\b"
     r"|\bamazon\s*pay\b|\bmobikwik\b|\bfreecharge\b|\bjuspay\b"
     r"|\bcred\b|\bslice\b|\bkwik\s*pay\b",
     "[FIN_APP]"),

    ("bank_name",
     r"\bsbi\b|\bstate\s+bank\b|\bhdfc\b|\bicici\b|\bkotak\b"
     r"|\baxis\s+bank\b|\bpnb\b|\bpunjab\s+national\b|\bcanara\b"
     r"|\bunion\s+bank\b|\bbank\s+of\s+baroda\b|\bbob\b"
     r"|\bindian\s+bank\b|\bcentral\s+bank\b|\byesbank\b|\bindusind\b"
     r"|\brbl\s+bank\b|\bidfc\b|\bfederal\s+bank\b",
     "[BANK]"),

    ("atm_card",
     r"\batm\b|\bdebit\s+card\b|\bcredit\s+card\b"
     r"|\bcard\s+(?:detail|number|clone|fraud|block)\b"
     r"|\bcloned?\s+card\b",
     "[ATM_CARD]"),

    ("sim_swap",
     r"\bsim\s+swap\b|\bsim\s+block\b|\bnew\s+sim\b|\bsim\s+port\b"
     r"|\bsim\s+hijack\b|\bsim\s+cloning\b",
     "[SIM_SWAP]"),

    # ── Social & communication platforms ──────────────────────────────────
    ("social_media",
     r"\bwhatsapp\b|\bfacebook\b|\binstagram\b|\btelegram\b"
     r"|\byoutube\b|\btwitter\b|\bsnapchat\b|\btiktok\b"
     r"|\bkoo\b|\bsharechat\b|\bmx\s+taka\s+tak\b|\bjosh\b"
     r"|\blinkedin\b|\bpinterest\b|\bred\s*dit\b|\bdiscord\b"
     r"|\bclubhouse\b|\bimo\b|\bhike\b|\bviber\b",
     "[SOCIAL_MEDIA]"),

    ("dating_app",
     r"\btinder\b|\bbumble\b|\bhinge\b|\bokc\b|\bokcupid\b"
     r"|\btruly\s+madly\b|\bquackquack\b|\bwoo\b|\baisle\b"
     r"|\bdating\s+(?:app|site|website)\b|\bonline\s+dating\b",
     "[DATING_APP]"),

    ("email_mention",
     r"\bemail\b|\be-mail\b|\bmail\s+id\b|\bgmail\b|\byahoo\s*mail\b"
     r"|\boutlook\b|\bhotmail\b|\bproton\s*mail\b|\btutanota\b"
     r"|\btemp\s*mail\b|\bthrowaway\s*mail\b",
     "[EMAIL]"),

    # ── Malware & attack keywords ──────────────────────────────────────────
    ("ransomware_kw",
     r"\bransomware\b|\bfiles?\s+encrypt(?:ed|ion)?\b"
     r"|\bencrypt(?:ed|ion)?\s+files?\b"
     r"|\bransom\s*(?:note|demand|payment|amount)?\b"
     r"|\bdecrypt(?:ion|ed)?\b|\bdecryption\s+key\b"
     r"|\bfiles?\s+(?:locked|inaccessible)\b"
     r"|\blocked\s+files?\b|\breadme\.txt\b"
     r"|\bhow\s+to\s+decrypt\b",
     "[RANSOMWARE]"),

    ("malware_kw",
     r"\bmalware\b|\bvirus\b|\btrojan\b|\bspyware\b|\bkeylogger\b"
     r"|\brootkit\b|\bworm\b|\badware\b|\bbotnet\b"
     r"|\binfected?\s+(?:file|device|system|computer)\b",
     "[MALWARE]"),

    ("phishing_kw",
     r"\bphishing\b|\bfake\s+(?:link|website|portal|page)\b"
     r"|\bclicked?\s+(?:a\s+)?link\b|\bmalicious\s+link\b"
     r"|\bfraudulent\s+(?:link|website|email)\b"
     r"|\bfake\s+(?:sbi|hdfc|icici|paytm|google)\s+(?:website|portal|page)\b",
     "[PHISHING]"),

    ("anydesk_remote",
     r"\banydesk\b|\bteamviewer\b|\bremote\s+(?:access|desktop|control)\b"
     r"|\bscreen\s+shar(?:e|ing)\b|\bremote\s+session\b"
     r"|\binstalled?\s+(?:anydesk|teamviewer|remote)\b",
     "[REMOTE_ACCESS]"),

    ("sql_injection",
     r"\bsql\s+injection\b|\bsql\b.*\battack\b|\binjection\s+attack\b"
     r"|\bdatabase\s+(?:hack|breach|inject)\b",
     "[SQL_INJECTION]"),

    ("ddos_attack",
     r"\bddos\b|\bdos\s+attack\b|\bdenial.of.service\b"
     r"|\bdistributed\s+denial\b|\bserver\s+down\b|\bserver\s+crash\b",
     "[DDOS]"),

    # ── Content types ──────────────────────────────────────────────────────
    ("nude_content",
     r"\bnude\b|\bnaked\b|\bexplicit\s+(?:photos?|videos?|content|images?)\b"
     r"|\bobscene\s+(?:photos?|videos?|content|material)\b"
     r"|\bmorphed\s+(?:photos?|videos?|images?)\b|\bporn(?:ography)?\b"
     r"|\bsexually\s+explicit\b|\bsexual\s+content\b",
     "[NUDE_CONTENT]"),

    ("sextortion",
     r"\bvideo\s+call\b|\bintimate\s+video\b|\bprivate\s+video\b"
     r"|\bnude\s+video\b|\bcompromising\s+video\b|\bsex\s+video\b"
     r"|\bsextortion\b|\bscreen\s*shot\s*(?:of\s+)?video\b",
     "[SEXTORTION]"),

    ("blackmail_threat",
     r"\bblackmail(?:ing)?\b|\bblackmailer\b|\bextort(?:ion)?\b"
     r"|\bthreat(?:en(?:ed|ing)?)?\b|\bthreats?\b"
     r"|\bdemand(?:ing)?\s+money\b|\bpay\s+(?:or|else)\b"
     r"|\bpay\s+(?:me|him|her|them)\s+(?:money|amount|rs)\b"
     r"|\bviral\s+(?:video|photo|content)\b|\bshare\s+(?:video|photo)\b",
     "[BLACKMAIL]"),

    # ── Fraud types ────────────────────────────────────────────────────────
    ("loan_app_fraud",
     r"\bloan\s+app\b|\binstant\s+loan\b|\beasy\s+loan\b"
     r"|\bonline\s+loan\b|\bquick\s+loan\b|\bpersonal\s+loan\s+app\b"
     r"|\bdigital\s+loan\b|\bloan\s+fraud\b|\bfake\s+loan\b"
     r"|\bloan\s+(?:recover|agent|recovery|harassment)\b"
     r"|\bcredit\s+(?:app|pearl|wallet|bee|fair)\b",
     "[LOAN_APP]"),

    ("job_fraud",
     r"\bjob\s+offer\b|\bwork\s+from\s+home\b|\bpart.?time\s+job\b"
     r"|\bearn\s+(?:money|daily|weekly|online)\b|\bonline\s+earning\b"
     r"|\bfreelance\s+job\b|\bdata\s+entry\s+job\b|\btask\s+(?:job|fraud)\b"
     r"|\binstagram\s+(?:task|like|follow)\b|\byoutube\s+(?:task|like|subscribe)\b",
     "[JOB_FRAUD]"),

    ("matrimonial_fraud",
     r"\bmatrimonial\b|\bshaadi\.com\b|\bjeevansathi\b"
     r"|\bbharatmatrimony\b|\bmatrimony\s+site\b"
     r"|\bonline\s+marriage\b|\bfake\s+(?:bride|groom|profile)\s+(?:on|marriage)\b"
     r"|\bmarriage\s+fraud\b|\bfiance\s+fraud\b",
     "[MATRIMONIAL]"),

    ("gambling_betting",
     r"\bgambling\b|\bbetting\b|\bcasino\b|\bsatta\b|\bmatka\b"
     r"|\bteen\s+patti\b|\brummy\b|\bpoker\b|\bonline\s+game\s+(?:win|earn)\b"
     r"|\bgame\s+(?:fraud|cheat|trick)\b|\bdream11\b|\bmpl\b",
     "[GAMBLING]"),

    # ── Crypto ─────────────────────────────────────────────────────────────
    ("crypto_mention",
     r"\bbitcoin\b|\bbtc\b|\bcryptocurrenc(?:y|ies)\b|\bcrypto\b"
     r"|\bethereum\b|\beth\b|\busdt\b|\btether\b|\bbinance\b"
     r"|\bcrypto\s+wallet\b|\bwallet\s+address\b|\bdigital\s+currenc\b",
     "[CRYPTO]"),

    # ── Identification & documents ─────────────────────────────────────────
    ("identity_docs",
     r"\baadhar\b|\baadhaar\b|\bpan\s+card\b|\bkyc\b"
     r"|\bvoting\s+card\b|\bpassport\b|\bdriving\s+licen\b"
     r"|\bidentity\s+(?:theft|fraud|steal)\b|\bpersonal\s+(?:document|id)\b",
     "[IDENTITY_DOC]"),

    # ── Cyber terrorism & national security ────────────────────────────────
    ("cyber_terrorism",
     r"\bterror(?:ism|ist|istic)?\b|\bnational\s+security\b"
     r"|\banti.?national\b|\bsedition\b|\bjihad\b"
     r"|\bterrorist\s+(?:group|organization|activity)\b"
     r"|\bsovereignty\b|\bintegrity\s+of\s+india\b"
     r"|\bpro.?pakistan\b|\banti.?india\b",
     "[CYBER_TERROR]"),

    # ── Government impersonation ───────────────────────────────────────────
    ("govt_impersonation",
     r"\bpolice\b|\bcbi\b|\bnarcotics?\b|\bcustoms?\b"
     r"|\bincome\s+tax\b|\btrai\b|\bsebi\b|\brbi\b"
     r"|\benforcement\s+direct\b|\bed\s+officer\b"
     r"|\bcyber\s+cell\b|\bpretended?\s+to\s+be\s+(?:police|officer|official)\b"
     r"|\bposed?\s+as\s+(?:police|officer|official|agent)\b"
     r"|\bimpersonat(?:ing|ed?)\s+(?:police|officer|official)\b",
     "[GOVT_IMPERSONATION]"),

    # ── Data breach / hacking ──────────────────────────────────────────────
    ("data_breach",
     r"\bdata\s+(?:breach|theft|leak(?:ed)?|stolen|compromised?)\b"
     r"|\bpersonal\s+(?:data|information|details?)\s+(?:stolen|leaked|hacked)\b"
     r"|\bunauthorized\s+access\b|\baccount\s+(?:hack|comprom|breach)\b",
     "[DATA_BREACH]"),

    ("website_defacement",
     r"\bwebsite\s+(?:hack(?:ed)?|defac(?:ed?|ement)|attack)\b"
     r"|\bdefac(?:ed?|ement)\b"
     r"|\bhomepage\s+(?:hack|changed?|replac)\b"
     r"|\bweb\s*(?:page|site)\s+(?:content\s+)?(?:modified|changed?|altered)\b",
     "[WEBSITE_DEFACEMENT]"),
]


# Total feature count
N_FEATURES = len(ENTITY_PATTERNS)

# Build name → index mapping
FEATURE_INDEX: Dict[str, int] = {
    name: i for i, (name, _, _) in enumerate(ENTITY_PATTERNS)
}

# Precompile all patterns
_COMPILED = [
    (name, re.compile(pat, re.IGNORECASE | re.UNICODE), tag)
    for name, pat, tag in ENTITY_PATTERNS
]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN NER CLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NERResult:
    feature_vector: np.ndarray          # shape (N_FEATURES,), dtype float32
    entities: List[Tuple[str, str]]     # [(entity_type, matched_text), ...]
    annotated_text: str                  # text with entity spans replaced by tags


class CyberCrimeNER:
    """
    Rule-based Named Entity Recognizer for Indian cyber-crime complaints.
    
    Fast, deterministic, no external dependencies.
    Processes ~50,000 texts/second on a single CPU core.
    
    Usage:
        ner = CyberCrimeNER()
        result = ner.process("Someone called pretending to be SBI and asked for OTP")
        print(result.feature_vector)     # [0, 1, 0, 1, 0, 1, ...]
        print(result.entities)           # [('otp_mention', 'OTP'), ('bank_name', 'SBI')]
        print(result.annotated_text)     # "...pretending to be [BANK] and asked for [OTP]"
    """

    def __init__(self, annotate: bool = False):
        """
        Args:
            annotate: If True, produce annotated text (slower). 
                      Set False for training (only need feature vector).
        """
        self.annotate = annotate

    def process(self, text: str) -> NERResult:
        """Process a single text and return NERResult."""
        if not text or not isinstance(text, str):
            return NERResult(
                feature_vector=np.zeros(N_FEATURES, dtype=np.float32),
                entities=[],
                annotated_text=text or ""
            )

        text_lower = text.lower()
        feature_vector = np.zeros(N_FEATURES, dtype=np.float32)
        entities: List[Tuple[str, str]] = []
        annotated = text

        for i, (name, compiled_pat, tag) in enumerate(_COMPILED):
            # Find all matches
            matches = compiled_pat.findall(text_lower)
            if matches:
                feature_vector[i] = 1.0
                for m in set(matches):
                    entities.append((name, m.strip()))
                
                # Replace in annotated text (only if requested)
                if self.annotate:
                    annotated = compiled_pat.sub(tag, annotated)

        return NERResult(
            feature_vector=feature_vector,
            entities=entities,
            annotated_text=annotated if self.annotate else text
        )

    def process_batch(self, texts: List[str]) -> np.ndarray:
        """
        Process a list of texts.
        Returns feature matrix of shape (len(texts), N_FEATURES), dtype float32.
        Optimized for speed — skips annotated text generation.
        """
        matrix = np.zeros((len(texts), N_FEATURES), dtype=np.float32)
        for i, text in enumerate(texts):
            if text and isinstance(text, str):
                text_lower = text.lower()
                for j, (name, compiled_pat, _) in enumerate(_COMPILED):
                    if compiled_pat.search(text_lower):
                        matrix[i, j] = 1.0
        return matrix

    @staticmethod
    def feature_names() -> List[str]:
        """Return list of feature names in order."""
        return [name for name, _, _ in ENTITY_PATTERNS]

    @staticmethod
    def n_features() -> int:
        return N_FEATURES


# ═══════════════════════════════════════════════════════════════════════════
# ENTITY-SWAP AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

# Swap-lists for entity-aware augmentation
# Replace detected entity mentions with semantically equivalent alternatives
ENTITY_SWAP_POOLS = {
    "fin_app":    ["Paytm", "PhonePe", "Google Pay", "BHIM", "Amazon Pay", 
                   "Mobikwik", "Freecharge"],
    "bank_name":  ["SBI", "HDFC Bank", "ICICI Bank", "Kotak Bank", "Axis Bank",
                   "PNB", "Canara Bank", "Union Bank"],
    "social_media": ["WhatsApp", "Facebook", "Instagram", "Telegram", "YouTube",
                     "Twitter", "Snapchat"],
    "dating_app": ["Tinder", "Bumble", "Truly Madly", "Aisle", "QuackQuack",
                   "online dating app"],
}


def entity_swap_augment(text: str, ner: CyberCrimeNER = None) -> str:
    """
    Augment text by replacing entity mentions with alternatives from the same pool.
    E.g., "WhatsApp" → "Telegram", "SBI" → "HDFC Bank"
    
    This creates realistic paraphrases without changing semantic class.
    Used in src/augment.py for minority class oversampling.
    """
    import random
    if ner is None:
        ner = CyberCrimeNER(annotate=False)

    result = text
    for pool_name, alternatives in ENTITY_SWAP_POOLS.items():
        # Find the pattern for this pool
        idx = FEATURE_INDEX.get(pool_name, -1)
        if idx < 0:
            continue
        _, compiled_pat, _ = _COMPILED[idx]

        matches = compiled_pat.findall(result.lower())
        if not matches:
            continue

        # Pick a random alternative different from what's in text
        detected = matches[0].strip()
        pool = [a for a in alternatives if a.lower() != detected.lower()]
        if not pool:
            continue

        replacement = random.choice(pool)
        result = compiled_pat.sub(replacement, result, count=1)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# PRECOMPUTE + CACHE
# ═══════════════════════════════════════════════════════════════════════════

def _ner_process_chunk(chunk: List[str]) -> np.ndarray:
    """Top-level worker for multiprocessing (must be picklable)."""
    ner = CyberCrimeNER(annotate=False)
    return ner.process_batch(chunk)


def precompute_ner_features(texts: List[str],
                             cache_path: str = None,
                             n_workers: int = 8) -> np.ndarray:
    """
    Precompute NER feature matrix for a list of texts using multiprocessing.
    Optionally saves to a .npy cache file for fast reloading.
    
    Args:
        texts:       list of strings
        cache_path:  if given, save/load the matrix from this path
        n_workers:   number of CPU workers for parallel processing
    
    Returns:
        np.ndarray of shape (len(texts), N_FEATURES), dtype float32
    """
    import os
    from multiprocessing import Pool

    if cache_path and os.path.exists(cache_path):
        print(f"Loading NER features from cache: {cache_path}")
        return np.load(cache_path)

    print(f"Computing NER features for {len(texts):,} texts using {n_workers} workers...")

    # Split into chunks for parallel processing
    chunk_size = max(1, len(texts) // (n_workers * 4))
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]

    if n_workers <= 1 or len(chunks) == 1:
        results = [_ner_process_chunk(c) for c in chunks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_ner_process_chunk, chunks)

    matrix = np.vstack(results)
    print(f"NER features computed: shape={matrix.shape}")
    print(f"Mean entity density per sample: {matrix.sum(axis=1).mean():.2f} entities/text")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.save(cache_path, matrix)
        print(f"Saved to {cache_path}")

    return matrix


# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ner = CyberCrimeNER(annotate=True)

    test_cases = [
        ("All my files are encrypted. Ransom note demands 500 USD in bitcoin to decrypt them.",
         ["ransomware_kw", "crypto_mention"]),
        ("SBI called me and asked for OTP. Money deducted via UPI.",
         ["bank_name", "otp_mention", "upi_mention"]),
        ("WhatsApp video call came, nude screenshot taken, blackmailing for money.",
         ["social_media", "sextortion", "nude_content", "blackmail_threat"]),
        ("Received email from fake sender pretending to be our CEO asking fund transfer.",
         ["email_mention"]),
        ("Got job offer for work from home data entry, paid registration fee, they vanished.",
         ["job_fraud"]),
        ("Loan app recovery agents sending my morphed photos to all contacts.",
         ["loan_app_fraud", "nude_content"]),
        ("AnyDesk installed on phone, they accessed my PhonePe and transferred money.",
         ["anydesk_remote", "fin_app"]),
        ("Gambling site took money, said I won but never paid out, satta fraud.",
         ["gambling_betting"]),
        ("Website defaced by hackers, homepage changed with political message.",
         ["website_defacement"]),
        ("Someone posing as CBI officer threatened me with arrest unless I pay.",
         ["govt_impersonation", "blackmail_threat"]),
    ]

    print(f"CyberCrimeNER — {N_FEATURES} entity types\n")
    all_passed = True
    for text, expected_entities in test_cases:
        result = ner.process(text)
        found = [name for name, _ in result.entities]
        missing = [e for e in expected_entities if e not in found]
        status = "✅" if not missing else f"❌ MISSING: {missing}"
        print(f"{status}")
        print(f"  Text: {text[:80]}...")
        print(f"  Found: {found}")
        print(f"  Vector sum: {result.feature_vector.sum():.0f}/{N_FEATURES}")
        print()
        if missing:
            all_passed = False

    print(f"\nBatch test:")
    texts = [tc[0] for tc in test_cases]
    matrix = ner.process_batch(texts)
    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Avg entities per text: {matrix.sum(axis=1).mean():.2f}")

    # Test entity swap augmentation
    print(f"\nEntity swap augmentation test:")
    orig = "I received a WhatsApp message from someone claiming to be SBI official."
    aug  = entity_swap_augment(orig)
    print(f"  Original:  {orig}")
    print(f"  Augmented: {aug}")

    if all_passed:
        print("\n✅ All NER tests passed")
    else:
        print("\n⚠️  Some tests failed — check patterns above")
