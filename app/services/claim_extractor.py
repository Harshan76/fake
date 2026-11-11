import re
from typing import List, Dict, Any, Optional

CLAIM_INDICATORS = [
    r'\bclaims?\b', r'\bannounced?\b', r'\bdiscovered?\b', r'\bcures?\b',
    r'\bcauses?\b', r'\bconfirmed?\b', r'\breported?\b', r'\bproves?\b',
    r'\brequires?\b', r'\bbanned?\b', r'\bsecretly\b', r'\bpassed law\b',
    r'\bdied\b', r'\bkilled\b', r'\binvented\b', r'\breleased\b', r'\bdeclares?\b'
]

COMMON_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "of", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "us", "our", "you", "your",
    "today", "yesterday", "tomorrow", "urgently", "urgent", "forward", "share",
    "official", "officially", "secretly", "reportedly", "allegedly"
}

# Organizations, authorities, agencies, ministries
ORG_PATTERNS = [
    r'\b(?:the\s+)?([A-Z][a-zA-Z0-9\s&,-]{2,40}?\s+(?:Ministry|Department|Agency|Organization|Organisation|Commission|Council|Authority|Institute|Bureau|Federation|Committee|Association|Administration))\b',
    r'\bAccording to\s+(?:the\s+)?([A-Z][a-zA-Z0-9\s&,-]{2,40}?)(?:,|\s+in\b|\s+that\b|\s+has\b|\s+have\b|\s+announced\b|\s+reported\b)',
    r'\b([A-Z][a-zA-Z0-9\s&,-]{2,40}?)\s+(?:announced|confirmed|reported|claimed|discovered|stated|ruled)\s+that\b'
]

def extract_named_attributions(text: str) -> List[str]:
    """
    Extracts named organizations, institutions, agencies, or authorities to which claims are attributed.
    Useful for investigating fictional or fabricated entities.
    """
    if not text:
        return []

    found = []
    seen = set()

    for pat in ORG_PATTERNS:
        matches = re.finditer(pat, text, re.IGNORECASE)
        for m in matches:
            val = m.group(1).strip()
            # Clean leading noise such as 'called', 'named', 'known as', 'an organisation called'
            val = re.sub(r'^(?:an?\s+)?(?:organisation|organization|institution|agency|group)?\s*(?:called|named|known as|dubbed)\s+', '', val, flags=re.IGNORECASE).strip()
            val = re.sub(r'^(?:the|an?)\s+', '', val, flags=re.IGNORECASE).strip()
            # Clean trailing punctuation
            val = re.sub(r'[,.:;]+$', '', val).strip()
            # Avoid single generic words or excessively long sentences
            words = val.split()
            if 1 < len(words) <= 7:
                norm = val.lower()
                if norm not in seen and not any(stop in norm for stop in ["someone", "anyone", "everyone", "people"]):
                    seen.add(norm)
                    found.append(val)

    return found

def distill_claim_keywords(claim: str, max_words: int = 7) -> str:
    """
    Distills a verbose claim sentence into 4-7 salient search keywords.
    Strips conversational noise, punctuation, and stopwords for optimal search query relevance.
    """
    if not claim:
        return ""

    # Remove quotes, brackets, and special punctuation
    clean = re.sub(r'["\'\(\)\[\]\{\},:;!?\-.]', ' ', claim)
    # Strip common conversational phrases
    clean = re.sub(r'\b(forwarded as received|urgent alert|breaking news|doctors stunned|secretly leaked|shocking discovery)\b', '', clean, flags=re.I)
    
    words = clean.split()
    salient = []
    
    for w in words:
        clean_w = w.strip(".,;:?!'\"")
        w_lower = clean_w.lower()
        if len(clean_w) > 2 and w_lower not in COMMON_STOPWORDS:
            salient.append(clean_w)
            
    if not salient:
        return " ".join([w.strip(".,;:?!'\"") for w in words[:max_words]])
        
    return " ".join(salient[:max_words])

def generate_verification_queries(claim: str, full_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates targeted, multi-faceted verification search queries for a claim:
    1. Distilled core claim keywords.
    2. Fact-check and debunk query.
    3. Attribution query if an organization or authority is named.
    """
    distilled = distill_claim_keywords(claim)
    attributions = extract_named_attributions(claim)
    if full_text and not attributions:
        attributions = extract_named_attributions(full_text)

    debunk_query = f"{distilled} fact check" if distilled else f"{claim[:60]} fact check"
    
    attribution_queries = []
    for org in attributions:
        attribution_queries.append(f'"{org}"')
        if distilled:
            attribution_queries.append(f'"{org}" {distilled}')

    return {
        "distilled_query": distilled,
        "debunk_query": debunk_query,
        "attributions": attributions,
        "attribution_queries": attribution_queries
    }

def extract_key_claims(text: str) -> List[str]:
    """
    Identifies and extracts verifiable key factual claims from input text.
    Filters out noise, rhetorical questions, and conversational padding.
    """
    if not text:
        return []

    # Clean text
    clean_text = re.sub(r'\s+', ' ', text.strip())
    
    # Split by sentence boundaries (. ? ! or newlines)
    raw_sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    
    scored_claims = []
    
    for sentence in raw_sentences:
        s = sentence.strip()
        # Filter very short sentences
        if len(s) < 20:
            continue
        # Skip pure conversational phrases
        if re.match(r'^(hi|hello|please share|check this out|good morning|forwarded as received)', s, re.I):
            continue
            
        score = 0
        s_lower = s.lower()
        
        # Check claim indicator keywords
        for pattern in CLAIM_INDICATORS:
            if re.search(pattern, s_lower):
                score += 2
                
        # Check for numbers, statistics, percentages, years, currency
        if re.search(r'\b\d+(\.\d+)?%?\b|\$\d+|\b(19|20)\d{2}\b', s):
            score += 2
            
        # Check for named entities or capitalized terms (proper nouns)
        caps_words = re.findall(r'\b[A-Z][a-z]+\b', s)
        if len(caps_words) >= 2:
            score += 1.5
            
        # Check for direct or indirect quotations
        if '"' in s or "'" in s:
            score += 1
            
        # Penalize excessive question marks if it's purely rhetorical
        if s.endswith("?") and not any(re.search(p, s_lower) for p in CLAIM_INDICATORS):
            score -= 1
            
        if score > 0 or len(raw_sentences) <= 3:
            scored_claims.append((score, s))
            
        # Sort by claim significance
    scored_claims.sort(key=lambda x: x[0], reverse=True)
    
    extracted = [claim for _, claim in scored_claims[:4]]
    
    # Fallback if no specific claim reached threshold
    if not extracted and clean_text:
        extracted = [clean_text[:180] + ("..." if len(clean_text) > 180 else "")]
        
    return extracted
