import logging
import httpx
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from app.config import TAVILY_API_KEY

logger = logging.getLogger("truthlens.tavily")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_TIMEOUT_SECONDS = 9.0

# Authoritative primary news, scientific, and institutional domains
AUTHORITATIVE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "who.int", "cdc.gov",
    "nih.gov", "nasa.gov", "nature.com", "thelancet.com", "sciencemag.org",
    "un.org", "pib.gov.in", "thehindu.com", "indianexpress.com", "nytimes.com",
    "washingtonpost.com", "wsj.com", "bloomberg.com", "ft.com", "economist.com",
    "snopes.com", "politifact.com", "factcheck.org", "afp.com", "aljazeera.com"
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
        "include_raw_content": False,
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
            
            clean_sources.append({
                "title": item.get("title") or "Web Search Reference",
                "publisher": domain.capitalize() if domain else "Web Source",
                "domain": domain,
                "url": raw_url,
                "description": item.get("content") or "",
                "published_date": item.get("published_date") or None,
                "relevance_score": round(float(item.get("score", 0.0)), 3),
                "authoritative": is_auth,
                "source_type": "web_search",
                "reliability": "High" if is_auth else "Moderate"
            })

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

def search_claims_evidence(claims: List[str]) -> Dict[str, Any]:
    """
    Orchestrates targeted Tavily searches across top extracted claims.
    Conducts dual-track queries:
    1. Direct factual search for corroboration.
    2. Debunk / fact-check search for contradictions.
    """
    if not is_tavily_configured() or not claims:
        return {
            "configured": is_tavily_configured(),
            "sources": [],
            "answers": []
        }

    collected_sources: List[Dict[str, Any]] = []
    seen_urls = set()
    answers = []

    # Search top 2 most significant claims
    for claim in claims[:2]:
        claim_clean = claim.strip()
        if len(claim_clean) < 10:
            continue

        # 1. Corroboration query
        res_direct = search_tavily(claim_clean, max_results=3)
        if res_direct.get("success"):
            if res_direct.get("answer"):
                answers.append(res_direct["answer"])
            for s in res_direct.get("sources", []):
                if s["url"] and s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    collected_sources.append(s)

        # 2. Debunk / fact-check targeted search
        debunk_query = f"{claim_clean} fact check hoax debunk"
        res_debunk = search_tavily(debunk_query, max_results=2)
        if res_debunk.get("success"):
            for s in res_debunk.get("sources", []):
                if s["url"] and s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    collected_sources.append(s)

    # Sort sources: authoritative first, then by relevance score
    collected_sources.sort(key=lambda x: (1 if x.get("authoritative") else 0, x.get("relevance_score", 0)), reverse=True)

    return {
        "configured": True,
        "sources": collected_sources[:8],
        "answers": answers
    }
