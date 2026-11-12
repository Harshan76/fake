import re
import socket
import logging
import urllib.parse
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from app.config import USER_AGENT
from app.services.url_scraper import is_safe_public_url

logger = logging.getLogger("truthlens.evidence_search")

# Reputable general news, scientific, and institutional domains
TRUSTED_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "factcheck.org",
    "politifact.com", "snopes.com", "pib.gov.in", "who.int", "cdc.gov",
    "thehindu.com", "indianexpress.com", "nature.com", "thelancet.com",
    "nasa.gov", "un.org", "en.wikipedia.org"
]

def sanitize_untrusted_page_text(raw_text: str) -> str:
    """
    Sanitizes retrieved web content to protect against prompt injection or malicious instructions.
    Strips explicit instruction headers and removes script / markup remnants.
    """
    if not raw_text:
        return ""
    # Strip potential prompt injection headers
    clean = re.sub(r'(?i)(ignore (?:all )?(?:previous|prior) instructions|system prompt|you are now|new directive).*', '', raw_text)
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

def search_wikipedia_knowledge(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Universal encyclopedia retrieval: searches Wikipedia official REST API
    for established factual articles, entities, historical events, and scientific concepts.
    Requires no API key.
    """
    sources = []
    clean_query = query.strip()
    if not clean_query:
        return sources

    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": clean_query,
            "format": "json",
            "srlimit": limit,
            "utf8": 1
        }
        headers = {"User-Agent": USER_AGENT}
        res = requests.get(url, params=params, headers=headers, timeout=3.5)
        if res.status_code == 200:
            data = res.json()
            search_results = data.get("query", {}).get("search", [])
            for item in search_results:
                title = item.get("title", "")
                snippet_html = item.get("snippet", "")
                snippet = BeautifulSoup(snippet_html, "html.parser").get_text()
                clean_title = urllib.parse.quote(title.replace(" ", "_"))
                article_url = f"https://en.wikipedia.org/wiki/{clean_title}"
                
                sources.append({
                    "title": f"Wikipedia: {title}",
                    "publisher": "Wikipedia Foundation",
                    "url": article_url,
                    "description": sanitize_untrusted_page_text(snippet)[:300] + ("..." if len(snippet) > 300 else ""),
                    "reliability": "High",
                    "domain": "en.wikipedia.org",
                    "source_type": "encyclopedia"
                })
    except Exception as e:
        logger.debug(f"Wikipedia search exception for '{clean_query}': {e}")
        
    return sources

def search_google_news_rss(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Universal live news retrieval: queries Google News RSS for published journalistic reporting.
    Requires no API key.
    """
    sources = []
    clean_query = query.strip()
    if not clean_query:
        return sources

    try:
        encoded_query = urllib.parse.quote(clean_query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": USER_AGENT}
        res = requests.get(rss_url, headers=headers, timeout=3.5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, "xml")
            items = soup.find_all("item")[:limit]
            for item in items:
                title = item.find("title").get_text(strip=True) if item.find("title") else "News Report"
                link_tag = item.find("link")
                link = link_tag.get_text(strip=True) if link_tag else ""
                source_tag = item.find("source")
                publisher = source_tag.get_text(strip=True) if source_tag else "Verified News Publisher"
                pub_date_tag = item.find("pubDate")
                pub_date = pub_date_tag.get_text(strip=True) if pub_date_tag else None
                
                if link:
                    sources.append({
                        "title": title,
                        "publisher": publisher,
                        "url": link,
                        "description": f"Published coverage and reporting by {publisher} ({pub_date[:16] if pub_date else 'Recent'}).",
                        "published_date": pub_date,
                        "reliability": "High",
                        "domain": "news.google.com",
                        "source_type": "live_news_rss"
                    })
    except Exception as e:
        logger.debug(f"Google News RSS search exception for '{clean_query}': {e}")
        
    return sources

def search_duckduckgo_lite(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Universal public web search: queries DuckDuckGo HTML Lite for live web coverage.
    Requires no API key.
    """
    sources = []
    clean_query = query.strip()
    if not clean_query:
        return sources

    try:
        encoded = urllib.parse.quote(clean_query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=4.0)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            results = soup.find_all("div", class_="result")[:limit]
            for r in results:
                title_tag = r.find("a", class_="result__a")
                snippet_tag = r.find("a", class_="result__snippet")
                if title_tag:
                    raw_title = title_tag.get_text(strip=True)
                    raw_href = title_tag.get("href", "")
                    raw_snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                    
                    # Unescape DuckDuckGo redirect
                    if "/l/?uddg=" in raw_href:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_href).query)
                        actual_url = parsed.get("uddg", [raw_href])[0]
                    else:
                        actual_url = raw_href

                    if actual_url and actual_url.startswith("http"):
                        domain = urllib.parse.urlparse(actual_url).netloc.lower()
                        sources.append({
                            "title": raw_title,
                            "publisher": domain.replace("www.", "").capitalize(),
                            "url": actual_url,
                            "description": sanitize_untrusted_page_text(raw_snippet)[:300],
                            "reliability": "Moderate",
                            "domain": domain,
                            "source_type": "web_search"
                        })
    except Exception as e:
        logger.debug(f"DuckDuckGo search exception for '{clean_query}': {e}")
        
    return sources

def fetch_source_page_text(url: str, max_chars: int = 2500) -> Optional[str]:
    """
    Retrieves and inspects the actual web page content of a retrieved source.
    Includes SSRF protection, strict timeout, and text sanitization.
    """
    safe, msg = is_safe_public_url(url)
    if not safe:
        logger.debug(f"Skipping page fetch for unsafe URL '{url}': {msg}")
        return None

    try:
        headers = {"User-Agent": USER_AGENT}
        res = requests.get(url, headers=headers, timeout=4.0)
        if res.status_code == 200 and "text/html" in res.headers.get("Content-Type", ""):
            soup = BeautifulSoup(res.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
                tag.decompose()
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
            body_text = " ".join(paragraphs[:6])
            return sanitize_untrusted_page_text(body_text[:max_chars])
    except Exception as e:
        logger.debug(f"Failed fetching page content for '{url}': {e}")
    return None

def deduplicate_syndicated_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identifies and collapses duplicate syndicated copies of wire service reports.
    Prevents counting identical copies of the same report as independent corroborations.
    """
    if not sources:
        return []

    unique_sources = []
    seen_titles = []

    for s in sources:
        title = s.get("title", "").strip().lower()
        title_tokens = set(re.findall(r'\b[a-z]{4,}\b', title))
        
        is_duplicate = False
        for seen in seen_titles:
            if title_tokens and seen:
                overlap = len(title_tokens.intersection(seen)) / max(len(title_tokens), len(seen))
                if overlap > 0.75:
                    is_duplicate = True
                    break
        if not is_duplicate:
            seen_titles.append(title_tokens)
            unique_sources.append(s)

    return unique_sources

def find_credible_evidence(claims: List[str], full_text: str) -> Dict[str, Any]:
    """
    Dynamic Universal Fallback Retriever:
    Queries Wikipedia, Google News RSS, and DuckDuckGo without hardcoded hoaxes or keywords.
    Deduplicates syndicated sources and inspects source stance dynamically.
    """
    from app.services.tavily_search import inspect_source_stance

    all_sources = []
    supporting_evidence = []
    contradicting_evidence = []

    # Dynamic query formulation
    search_queries = []
    for claim in claims[:3]:
        clean_words = re.findall(r'\b[A-Za-z0-9]{3,}\b', claim)
        if len(clean_words) >= 4:
            search_queries.append(" ".join(clean_words[:7]))

    if not search_queries and full_text:
        words = re.findall(r'\b[A-Za-z0-9]{3,}\b', full_text)
        search_queries.append(" ".join(words[:6]))

    # Retrieve sources across open providers
    for q in search_queries[:2]:
        # 1. Wikipedia Knowledge
        for w in search_wikipedia_knowledge(q, limit=2):
            if not any(s["url"] == w["url"] for s in all_sources):
                all_sources.append(w)

        # 2. Google News RSS
        for n in search_google_news_rss(q, limit=2):
            if not any(s["url"] == n["url"] for s in all_sources):
                all_sources.append(n)

        # 3. DuckDuckGo Web
        for d in search_duckduckgo_lite(q, limit=2):
            if not any(s["url"] == d["url"] for s in all_sources):
                all_sources.append(d)

    # Deduplicate syndicated stories
    deduped_sources = deduplicate_syndicated_sources(all_sources)

    # Inspect stance for each deduplicated source
    primary_claim = claims[0] if claims else full_text[:120]
    for src in deduped_sources:
        stance = inspect_source_stance(src, claim=primary_claim)
        src["stance"] = stance
        if stance == "supports":
            supporting_evidence.append({
                "source": src.get("publisher", "Authoritative Source"),
                "claim_addressed": src.get("title", primary_claim),
                "detail": src.get("description", "")[:240],
                "url": src.get("url")
            })
        elif stance == "contradicts":
            contradicting_evidence.append({
                "source": src.get("publisher", "Accredited Source"),
                "claim_addressed": src.get("title", primary_claim),
                "detail": src.get("description", "")[:240],
                "url": src.get("url")
            })

    # Calculate evidence strength
    if contradicting_evidence:
        stance_summary = "contradicts"
        evidence_score = 88.0 + min(len(contradicting_evidence) * 3, 6.0)
    elif len(supporting_evidence) >= 1:
        stance_summary = "supports"
        evidence_score = 78.0 + min(len(supporting_evidence) * 4, 14.0)
    elif len(deduped_sources) > 0:
        stance_summary = "neutral"
        evidence_score = 55.0
    else:
        stance_summary = "insufficient"
        evidence_score = 40.0

    return {
        "sources": deduped_sources[:6],
        "supporting_evidence": supporting_evidence[:5],
        "contradicting_evidence": contradicting_evidence[:5],
        "evidence_score": round(evidence_score, 1),
        "stance": stance_summary
    }
