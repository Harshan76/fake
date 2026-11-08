import re
from typing import List

CLAIM_INDICATORS = [
    r'\bclaims?\b', r'\bannounced?\b', r'\bdiscovered?\b', r'\bcures?\b',
    r'\bcauses?\b', r'\bconfirmed?\b', r'\breported?\b', r'\bproves?\b',
    r'\brequires?\b', r'\bbanned?\b', r'\bsecretly\b', r'\bpassed law\b',
    r'\bdied\b', r'\bkilled\b', r'\binvented\b', r'\breleased\b'
]

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
