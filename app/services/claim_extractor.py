import re
from typing import List, Dict, Any, Optional

CLAIM_ACTION_INDICATORS = [
    r'\bclaims?\b', r'\bannounced?\b', r'\bdiscovered?\b', r'\bcures?\b',
    r'\bcauses?\b', r'\bconfirmed?\b', r'\breported?\b', r'\bproves?\b',
    r'\brequires?\b', r'\bbanned?\b', r'\bpassed law\b', r'\bdied\b',
    r'\bkilled\b', r'\binvented\b', r'\breleased\b', r'\bdeclares?\b',
    r'\bapproved?\b', r'\bsigned?\b', r'\bvoted?\b', r'\bwon\b',
    r'\bdefeated\b', r'\blaunched\b', r'\bhalted\b', r'\bwarned?\b'
]

COMMON_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "of", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "us", "our", "you", "your",
    "today", "yesterday", "tomorrow", "urgently", "urgent", "forward", "share",
    "official", "officially", "secretly", "reportedly", "allegedly", "according",
    "says", "said", "also", "just", "now", "well", "very", "about",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "next", "last", "all", "any", "some", "such", "not", "into", "over", "after",
    "before", "under", "between", "than", "more", "most", "other", "another"
}

GENERIC_TITLES = {
    "prime", "minister", "president", "government", "agency", "department",
    "official", "officials", "police", "council", "court", "commission",
    "united", "states", "country", "state", "federal", "national", "international",
    "declared", "announced", "confirmed", "stated", "high", "magnitude"
}

TOPIC_KEYWORDS = {
    "health": [
        "disease", "cure", "vaccine", "health", "hospital", "doctor", "virus",
        "drug", "clinical", "treatment", "who", "fda", "cdc", "medical", "cancer",
        "covid", "patient", "infection", "symptoms", "therapy", "pandemic"
    ],
    "science": [
        "nasa", "space", "orbit", "planet", "moon", "telescope", "physics",
        "astronomy", "quantum", "asteroid", "galaxy", "biology", "scientific",
        "researchers", "study", "satellite", "isro", "esa", "earth", "climate",
        "atmosphere", "species", "fossil", "laboratory", "cosmic"
    ],
    "government": [
        "ministry", "parliament", "congress", "president", "minister", "law",
        "bill", "policy", "supreme court", "election", "vote", "treaty", "governor",
        "department", "senate", "cabinet", "administration", "order", "decree",
        "ordinance", "legislation", "ambassador", "federal"
    ],
    "sports": [
        "cup", "tournament", "championship", "league", "coach", "player", "match",
        "score", "goal", "fifa", "olympics", "nba", "nfl", "cricket", "football",
        "tennis", "medal", "trophy", "stadium", "referee", "won", "defeated"
    ],
    "technology": [
        "software", "ai", "model", "chip", "semiconductor", "algorithm", "google",
        "apple", "microsoft", "release", "update", "app", "cybersecurity",
        "hacker", "device", "hardware", "server", "data", "platform"
    ],
    "finance": [
        "shares", "stock", "market", "inflation", "interest rate", "bank", "revenue",
        "profit", "gdp", "economy", "crypto", "central bank", "investor", "dividend",
        "recession", "currency", "treasury", "sec", "fed", "monetary"
    ],
    "international": [
        "united nations", "treaty", "war", "foreign", "ambassador", "summit",
        "sanctions", "diplomacy", "nato", "eu", "border", "ceasefire", "accord"
    ]
}

ORG_PATTERNS = [
    r'\b(?:the\s+)?([A-Z][a-zA-Z0-9\s&,-]{2,45}?\s+(?:Ministry|Department|Agency|Organization|Organisation|Commission|Council|Authority|Institute|Bureau|Federation|Committee|Association|Administration|Observatory|University))\b',
    r'\bAccording to\s+(?:the\s+)?([A-Z][a-zA-Z0-9\s&,-]{2,45}?)(?:,|\s+in\b|\s+that\b|\s+has\b|\s+have\b|\s+announced\b|\s+reported\b)',
    r'\b([A-Z][a-zA-Z0-9\s&,-]{2,45}?)\s+(?:announced|confirmed|reported|claimed|discovered|stated|ruled|warned|advised)\s+that\b'
]

def classify_article_topic(text: str) -> str:
    """
    Dynamically identifies the subject domain of the article
    (health, science, government, sports, technology, finance, international, or general).
    Used to select targeted authoritative source strategies without rigid hardcoding.
    """
    if not text:
        return "general"

    text_lower = text.lower()
    topic_scores = {topic: 0 for topic in TOPIC_KEYWORDS}

    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                topic_scores[topic] += 1

    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_topics and sorted_topics[0][1] >= 2:
        return sorted_topics[0][0]

    return "general"

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
            # Clean leading noise
            val = re.sub(r'^(?:an?\s+)?(?:organisation|organization|institution|agency|group)?\s*(?:called|named|known as|dubbed)\s+', '', val, flags=re.IGNORECASE).strip()
            val = re.sub(r'^(?:the|an?)\s+', '', val, flags=re.IGNORECASE).strip()
            val = re.sub(r'[,.:;]+$', '', val).strip()
            words = val.split()
            if 1 < len(words) <= 7:
                norm = val.lower()
                if norm not in seen and not any(stop in norm for stop in ["someone", "anyone", "everyone", "people"]):
                    seen.add(norm)
                    found.append(val)

    return found

def extract_entities_and_statistics(text: str) -> Dict[str, List[str]]:
    """
    Extracts key verifiable entities: numbers, percentages, currency, dates/years, proper names.
    """
    if not text:
        return {"numbers": [], "dates": [], "entities": [], "attributions": []}

    numbers = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?%?\b|\$\d+(?:,\d+)*(?:\.\d+)?', text)
    dates = re.findall(r'\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b', text, re.IGNORECASE)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    filtered_nouns = [pn for pn in proper_nouns if len(pn) > 3 and pn.lower() not in COMMON_STOPWORDS][:6]
    attributions = extract_named_attributions(text)

    return {
        "numbers": list(set(numbers[:5])),
        "dates": list(set(dates[:4])),
        "entities": list(set(filtered_nouns)),
        "attributions": attributions
    }

def distill_claim_keywords(claim: str, max_words: int = 7) -> str:
    """
    Distills a verbose claim sentence into 4-7 salient search keywords.
    Prioritizes entities, proper nouns, numerical values, and domain-specific predicates.
    Demotes conversational noise, stopwords, and generic institutional titles.
    """
    if not claim:
        return ""

    clean = re.sub(r'["\'\(\)\[\]\{\},:;!?\-.]', ' ', claim)
    clean = re.sub(r'\b(forwarded as received|urgent alert|breaking news|doctors stunned|secretly leaked|shocking discovery)\b', '', clean, flags=re.I)
    
    words = clean.split()
    priority_words = []
    secondary_words = []
    
    for w in words:
        w_clean = w.strip('.,;:\'"?!')
        w_lower = w_clean.lower()
        if not w_clean or len(w_clean) < 2 or w_lower in COMMON_STOPWORDS:
            continue
        if re.search(r'\d', w_clean) or (w_clean[0].isupper() and w_lower not in GENERIC_TITLES):
            if w_clean not in priority_words:
                priority_words.append(w_clean)
        elif w_lower not in GENERIC_TITLES:
            if w_clean not in secondary_words:
                secondary_words.append(w_clean)
                
    combined = (priority_words + secondary_words)[:max_words]
    if not combined:
        return " ".join([w.strip('.,;:\'"?!') for w in words[:max_words]])
        
    return " ".join(combined)

def generate_adaptive_verification_queries(claim: str, topic: Optional[str] = None, full_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Universal Topic-Adaptive Query Generation:
    Generates multi-faceted, intelligent search queries for any submitted claim:
    1. Distilled core claim assertion query.
    2. Fact-check & debunk inquiry query.
    3. Topic-specific official authority query (government, health, sports, science, tech, finance).
    4. Attribution confirmation query if an institution is cited.
    """
    distilled = distill_claim_keywords(claim)
    detected_topic = topic or classify_article_topic(full_text or claim)
    attributions = extract_named_attributions(claim)
    if full_text and not attributions:
        attributions = extract_named_attributions(full_text)

    # Core assertion queries
    debunk_query = f"{distilled} fact check" if distilled else f"{claim[:60]} fact check"
    supporting_query = f"{distilled} official announcement" if distilled else f"{claim[:60]} confirmed"

    # Topic-adaptive queries
    authority_queries = []
    if detected_topic == "health":
        authority_queries.append(f"{distilled} WHO CDC health authority")
    elif detected_topic == "science":
        authority_queries.append(f"{distilled} NASA scientific research study")
    elif detected_topic == "government":
        authority_queries.append(f"{distilled} government ministry official statement")
    elif detected_topic == "sports":
        authority_queries.append(f"{distilled} official tournament results")
    elif detected_topic == "technology":
        authority_queries.append(f"{distilled} official documentation announcement")
    elif detected_topic == "finance":
        authority_queries.append(f"{distilled} central bank regulatory filing")

    # Attribution queries
    attribution_queries = []
    for org in attributions:
        attribution_queries.append(f'"{org}"')
        if distilled:
            attribution_queries.append(f'"{org}" {distilled}')

    return {
        "distilled_query": distilled,
        "debunk_query": debunk_query,
        "supporting_query": supporting_query,
        "detected_topic": detected_topic,
        "authority_queries": authority_queries,
        "attributions": attributions,
        "attribution_queries": attribution_queries
    }

def generate_verification_queries(claim: str, full_text: Optional[str] = None) -> Dict[str, Any]:
    """Backward compatibility wrapper for verifier and search pipeline."""
    return generate_adaptive_verification_queries(claim, full_text=full_text)

def extract_key_claims(text: str) -> List[str]:
    """
    Identifies and extracts verifiable key factual claims from arbitrary input text.
    Filters out conversational noise, rhetorical questions, and subjective opinions.
    """
    if not text:
        return []

    clean_text = re.sub(r'\s+', ' ', text.strip())
    raw_sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    
    scored_claims = []
    
    for sentence in raw_sentences:
        s = sentence.strip()
        if len(s) < 20:
            continue
        if re.match(r'^(hi|hello|please share|check this out|good morning|forwarded as received)', s, re.I):
            continue
            
        score = 0
        s_lower = s.lower()
        
        # Check claim action indicators
        for pattern in CLAIM_ACTION_INDICATORS:
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
            
        # Penalize excessive question marks if purely rhetorical
        if s.endswith("?") and not any(re.search(p, s_lower) for p in CLAIM_ACTION_INDICATORS):
            score -= 1
            
        if score > 0 or len(raw_sentences) <= 3:
            scored_claims.append((score, s))
            
    scored_claims.sort(key=lambda x: x[0], reverse=True)
    extracted = [claim for _, claim in scored_claims[:5]]
    
    if not extracted and clean_text:
        extracted = [clean_text[:180] + ("..." if len(clean_text) > 180 else "")]
        
    return extracted
