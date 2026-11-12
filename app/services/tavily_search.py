import re
import logging
import httpx
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from app.config import TAVILY_API_KEY
from app.services.evidence_search import fetch_source_page_text, deduplicate_syndicated_sources

logger = logging.getLogger("truthlens.tavily")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_TIMEOUT_SECONDS = 9.0

# Authoritative primary news, scientific, regulatory, and institutional domains
AUTHORITATIVE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "who.int", "cdc.gov",
    "nih.gov", "nasa.gov", "nature.com", "thelancet.com", "sciencemag.org",
    "un.org", "pib.gov.in", "thehindu.com", "indianexpress.com", "nytimes.com",
    "washingtonpost.com", "wsj.com", "bloomberg.com", "ft.com", "economist.com",
    "snopes.com", "politifact.com", "factcheck.org", "afp.com", "aljazeera.com",
    "usatoday.com", "leadstories.com", "fullfact.org", "fda.gov", "sec.gov",
    "fifa.com", "olympics.com", "isro.gov.in", "esa.int"
}

# Indicators that a source is reporting a fact-check refutation or debunk
DEBUNK_INDICATORS = [
    r'\bfact[-\s]?check\b', r'\bdebunk(?:ed)?\b', r'\bhoax\b', r'\bfalse claim\b',
    r'\bnot true\b', r'\bno evidence\b', r'\bfabricat(?:ed|ion)\b', r'\bmisleading\b',
    r'\bdisproven\b', r'\bdisproved\b', r'\bbogus\b', r'\bsatire\b', r'\bfictional\b',
    r'\brumou?r\b', r'\bnever (?:said|occurred|happened)\b',
    r'\bdid not (?:announce|say|happen)\b', r'\bdoes not exist\b', r'\bno such (?:agency|department|organization|commission)\b',
    r'\bpants on fire\b', r'\bmisattributed\b', r'\bunfounded\b', r'\bcounterfeit\b', r'\bphony\b',
    r'\bimpossible\b', r'\buntrue\b', r'\bmyth\b', r'\bunverified\b', r'\bfake news\b'
]

COMMON_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with',
    'of', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'this', 'that', 'these', 'those',
    'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your',
    'today', 'yesterday', 'tomorrow', 'official', 'officially', 'report', 'reported',
    'announced', 'stated', 'confirmed', 'revealed', 'news', 'according', 'scientists',
    'experts', 'authorities', 'officials', 'team', 'study', 'new', 'recent', 'will',
    'due', 'also', 'about', 'over', 'into', 'after', 'under', 'all', 'any'
}

GENERIC_TITLES = {
    'prime', 'minister', 'president', 'government', 'agency', 'department',
    'official', 'officials', 'police', 'council', 'court', 'commission',
    'united', 'states', 'country', 'state', 'federal', 'national', 'international',
    'declared', 'announced', 'confirmed', 'stated', 'high', 'magnitude', 'news',
    'report', 'reported', 'month', 'year', 'week', 'today', 'yesterday', 'tomorrow'
}

def is_tavily_configured() -> bool:
    """Checks whether the Tavily Search API key is present."""
    return bool(TAVILY_API_KEY and TAVILY_API_KEY.strip())

def extract_domain(url: str) -> str:
    """Extracts clean lowercase domain from URL."""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def extract_substantive_tokens(text: str) -> set:
    """Extracts clean substantive content tokens from text."""
    clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    words = clean.split()
    return set(w for w in words if len(w) > 2 and w not in COMMON_STOPWORDS)

def inspect_source_stance(source: Dict[str, Any], claim: str = "") -> str:
    """
    Evaluates whether a retrieved web source is directly contradicting/debunking a claim,
    supporting it with authoritative corroboration of the specific material assertions, or neutral.
    General background articles that merely share broad keywords (e.g. 'Moon' or 'NASA')
    without confirming the claim's core event are classified as neutral.
    """
    text_to_check = f"{source.get('title', '')} {source.get('description', '')}".lower()
    raw_content = source.get("raw_content", "")
    if raw_content:
        text_to_check += " " + raw_content[:1000].lower()
    
    # 1. Check for explicit debunk / refutation markers
    for pattern in DEBUNK_INDICATORS:
        if re.search(pattern, text_to_check):
            return "contradicts"

    # Only authoritative sources can corroborate
    is_auth = source.get("authoritative", False)
    if not is_auth:
        domain = source.get("domain") or extract_domain(source.get("url", ""))
        if any(auth in domain for auth in AUTHORITATIVE_DOMAINS) or "wikipedia.org" in domain or domain.endswith(".gov") or domain.endswith(".edu"):
            is_auth = True
    if not is_auth:
        return "neutral"

    if not claim or not claim.strip():
        return "neutral"

    # 2. Extract substantive content tokens from claim and source
    claim_tokens = extract_substantive_tokens(claim)
    if not claim_tokens:
        return "neutral"

    source_tokens = extract_substantive_tokens(text_to_check)
    overlap = claim_tokens.intersection(source_tokens)
    distinctive_overlap = overlap - GENERIC_TITLES

    # Must match substantive event predicates, not just institutional titles or country names alone
    if len(distinctive_overlap) < 2 and len(overlap) < 4:
        return "neutral"

    overlap_ratio = len(overlap) / len(claim_tokens)
    if len(distinctive_overlap) >= 3 or (len(distinctive_overlap) >= 2 and overlap_ratio >= 0.28):
        return "supports"

    return "neutral"

def search_tavily(
    query: str,
    search_depth: str = "advanced",
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes a structured search against Tavily Search API.
    Returns cleaned source records with titles, URLs, full snippets, and publication dates.
    """
    if not is_tavily_configured():
        return {
            "success": False,
            "configured": False,
            "query": query,
            "sources": [],
            "error": "TAVILY_API_KEY environment variable is not configured."
        }

    clean_query = query.strip()
    if not clean_query:
        return {"success": False, "configured": True, "sources": [], "error": "Query cannot be empty."}

    payload = {
        "api_key": TAVILY_API_KEY.strip(),
        "query": clean_query,
        "search_depth": search_depth,
        "include_answer": True,
        "include_raw_content": True,
        "max_results": max_results
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            response = client.post(TAVILY_SEARCH_URL, json=payload)
            
            if response.status_code == 401:
                logger.error("Tavily API Authentication failed (401 Unauthorized). Check TAVILY_API_KEY.")
                return {
                    "success": False,
                    "configured": True,
                    "sources": [],
                    "error": "Tavily API key is invalid or unauthorized."
                }
            elif response.status_code == 429:
                logger.warning("Tavily API Rate Limit reached (429).")
                return {
                    "success": False,
                    "configured": True,
                    "sources": [],
                    "error": "Tavily API rate limit exceeded."
                }
            elif response.status_code >= 500:
                logger.error(f"Tavily API Server Error ({response.status_code}).")
                return {
                    "success": False,
                    "configured": True,
                    "sources": [],
                    "error": f"Tavily search service temporarily unavailable (HTTP {response.status_code})."
                }

            response.raise_for_status()
            data = response.json()

        raw_results = data.get("results", [])
        answer = data.get("answer")
        clean_sources = []

        for item in raw_results:
            raw_url = item.get("url", "")
            domain = extract_domain(raw_url)
            is_auth = any(domain.endswith(auth_dom) for auth_dom in AUTHORITATIVE_DOMAINS)
            raw_content = item.get("raw_content") or ""
            
            source_obj = {
                "title": item.get("title") or "Web Search Reference",
                "publisher": domain.capitalize() if domain else "Web Source",
                "domain": domain,
                "url": raw_url,
                "description": item.get("content") or "",
                "raw_content": raw_content[:2000],
                "published_date": item.get("published_date") or None,
                "relevance_score": round(float(item.get("score", 0.0)), 3),
                "authoritative": is_auth,
                "source_type": "web_search",
                "reliability": "High" if is_auth else "Moderate"
            }
            clean_sources.append(source_obj)

        return {
            "success": True,
            "configured": True,
            "query": clean_query,
            "answer": answer,
            "sources": clean_sources,
            "error": None
        }

    except httpx.TimeoutException:
        logger.warning(f"Tavily API request timed out after {DEFAULT_TIMEOUT_SECONDS}s.")
        return {
            "success": False,
            "configured": True,
            "sources": [],
            "error": "Tavily search request timed out."
        }
    except Exception as e:
        logger.error(f"Tavily search unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "configured": True,
            "sources": [],
            "error": f"Tavily search failure: {str(e)}"
        }

def search_claims_evidence(claims: List[str], full_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Universal Dynamic Search Execution:
    Conducts topic-adaptive multi-track queries:
    1. Distilled core factual search for corroboration.
    2. Debunk / fact-check targeted search for contradictions.
    3. Topic-adaptive official authority query (government, health, sports, science, tech).
    4. Attribution confirmation query for cited organizations.
    Deduplicates syndicated sources and performs deep stance inspection.
    """
    if not is_tavily_configured() or not claims:
        return {
            "configured": is_tavily_configured(),
            "sources": [],
            "contradicting": [],
            "supporting": [],
            "answers": [],
            "attribution_findings": []
        }

    from app.services.claim_extractor import generate_adaptive_verification_queries

    collected_sources: List[Dict[str, Any]] = []
    seen_urls = set()
    answers = []
    contradicting = []
    supporting = []
    attribution_findings = []

    # Search top 2 most significant claims
    for claim in claims[:2]:
        claim_clean = claim.strip()
        if len(claim_clean) < 10:
            continue

        q_pkg = generate_adaptive_verification_queries(claim_clean, full_text=full_text)
        distilled = q_pkg.get("distilled_query") or claim_clean[:70]
        debunk_q = q_pkg.get("debunk_query") or f"{distilled} fact check"

        # 1. Distilled Corroboration query
        res_direct = search_tavily(distilled, max_results=3)
        if res_direct.get("success"):
            if res_direct.get("answer"):
                answers.append(res_direct["answer"])
            for s in res_direct.get("sources", []):
                if s["url"] and s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    collected_sources.append(s)

        # 2. Debunk / fact-check targeted search
        res_debunk = search_tavily(debunk_q, max_results=3)
        if res_debunk.get("success"):
            for s in res_debunk.get("sources", []):
                if s["url"] and s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    collected_sources.append(s)

        # 3. Topic-adaptive authority queries
        for auth_q in q_pkg.get("authority_queries", [])[:1]:
            res_auth = search_tavily(auth_q, max_results=2)
            if res_auth.get("success"):
                for s in res_auth.get("sources", []):
                    if s["url"] and s["url"] not in seen_urls:
                        seen_urls.add(s["url"])
                        collected_sources.append(s)

        # 4. Attribution query if organization/authority is present
        for attr_q in q_pkg.get("attribution_queries", [])[:2]:
            res_attr = search_tavily(attr_q, max_results=2)
            if res_attr.get("success"):
                found_auth = False
                for s in res_attr.get("sources", []):
                    if s.get("authoritative"):
                        found_auth = True
                    if s["url"] and s["url"] not in seen_urls:
                        seen_urls.add(s["url"])
                        collected_sources.append(s)
                if not found_auth and res_attr.get("sources"):
                    attribution_findings.append({
                        "query": attr_q,
                        "status": "NO_AUTHORITATIVE_RECORDS",
                        "detail": f"Search for {attr_q} returned no official or authoritative institutional domains."
                    })

    # Deduplicate syndicated stories
    deduped_sources = deduplicate_syndicated_sources(collected_sources)

    # Inspect all collected sources for stances against extracted claims
    primary_claim = claims[0] if claims else (full_text or "")
    for s in deduped_sources:
        stance = "neutral"
        matched_claim = primary_claim

        for c in claims[:3]:
            st = inspect_source_stance(s, claim=c)
            if st == "contradicts":
                stance = "contradicts"
                matched_claim = c
                break
            elif st == "supports" and stance != "contradicts":
                stance = "supports"
                matched_claim = c

        s["stance"] = stance
        
        if stance == "contradicts":
            contradicting.append({
                "source": s.get("publisher", "Web Source"),
                "claim_addressed": matched_claim,
                "detail": f"{s.get('title')}: {s.get('description', '')[:250]}",
                "url": s.get("url"),
                "source_type": "web_fact_check",
                "reliability": s.get("reliability", "High")
            })
        elif stance == "supports" and s.get("authoritative"):
            supporting.append({
                "source": s.get("publisher", "Authoritative Source"),
                "claim_addressed": matched_claim,
                "detail": f"{s.get('title')}: {s.get('description', '')[:250]}",
                "url": s.get("url"),
                "source_type": "authoritative_news",
                "reliability": "High"
            })

    # Sort sources: authoritative first, then by relevance score
    deduped_sources.sort(key=lambda x: (1 if x.get("authoritative") else 0, x.get("relevance_score", 0)), reverse=True)

    return {
        "configured": True,
        "sources": deduped_sources[:8],
        "contradicting": contradicting,
        "supporting": supporting,
        "answers": answers,
        "attribution_findings": attribution_findings
    }
