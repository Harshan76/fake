import logging
import httpx
from typing import List, Dict, Any, Optional
from app.config import GOOGLE_FACT_CHECK_API_KEY

logger = logging.getLogger("truthlens.factcheck")

GOOGLE_FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
DEFAULT_TIMEOUT_SECONDS = 8.0

# Negative verdict keywords from accredited IFCN fact-checkers
CONTRADICTING_RATINGS = {
    "false", "pants on fire", "mostly false", "fake", "incorrect", "debunked",
    "misleading", "hoax", "unproven", "fabricated", "scam", "satire", "distorted",
    "partly false", "altered", "manipulated", "lacks context", "two pinocchios",
    "three pinocchios", "four pinocchios"
}

# Positive verdict keywords
SUPPORTING_RATINGS = {
    "true", "mostly true", "correct", "accurate", "verified", "confirmed",
    "geppetto checkmark"
}

def is_fact_check_configured() -> bool:
    """Checks whether the Google Fact Check Tools API key is set."""
    return bool(GOOGLE_FACT_CHECK_API_KEY and GOOGLE_FACT_CHECK_API_KEY.strip())

def parse_rating_stance(rating: Optional[str]) -> str:
    """Classifies a fact-checker textual rating into 'contradicts', 'supports', or 'unclear'."""
    if not rating:
        return "unclear"
    r = rating.strip().lower()
    for neg in CONTRADICTING_RATINGS:
        if neg in r:
            return "contradicts"
    for pos in SUPPORTING_RATINGS:
        if pos in r:
            return "supports"
    return "unclear"

def search_google_fact_check(query: str, language_code: str = "en") -> Dict[str, Any]:
    """
    Queries Google Fact Check Tools API for prior peer-reviewed fact checks.
    Treats absent results strictly as 'no matching fact-check found', NEVER as proof.
    """
    if not is_fact_check_configured():
        return {
            "success": False,
            "configured": False,
            "query": query,
            "reviews": [],
            "error": "GOOGLE_FACT_CHECK_API_KEY environment variable is not configured."
        }

    clean_query = query.strip()
    if not clean_query:
        return {"success": False, "configured": True, "reviews": [], "error": "Query cannot be empty."}

    params = {
        "key": GOOGLE_FACT_CHECK_API_KEY.strip(),
        "query": clean_query,
        "languageCode": language_code
    }

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            response = client.get(GOOGLE_FACT_CHECK_URL, params=params)

            if response.status_code == 400:
                logger.warning(f"Google Fact Check API query invalid or rejected: {response.text}")
                return {"success": False, "configured": True, "reviews": [], "error": "Invalid fact check query."}
            elif response.status_code == 403:
                logger.error("Google Fact Check API Key invalid or API not enabled in Google Cloud Console.")
                return {"success": False, "configured": True, "reviews": [], "error": "Google Fact Check API key unauthorized or disabled."}
            elif response.status_code == 429:
                logger.warning("Google Fact Check API quota exceeded (429).")
                return {"success": False, "configured": True, "reviews": [], "error": "Fact check quota exceeded."}
            
            response.raise_for_status()
            data = response.json()

        claims = data.get("claims", [])
        reviews_found: List[Dict[str, Any]] = []

        for item in claims:
            claim_text = item.get("text", "")
            claimant = item.get("claimant")
            claim_date = item.get("claimDate")

            for review in item.get("claimReview", []):
                pub = review.get("publisher", {})
                pub_name = pub.get("name", "Accredited Fact Checker")
                pub_site = pub.get("site", "")
                rating = review.get("textualRating", "")
                stance = parse_rating_stance(rating)

                reviews_found.append({
                    "title": review.get("title") or f"Fact Check: {claim_text[:80]}",
                    "publisher": pub_name,
                    "publisher_site": pub_site,
                    "url": review.get("url", "#"),
                    "review_date": review.get("reviewDate"),
                    "claim_addressed": claim_text,
                    "claimant": claimant,
                    "claim_date": claim_date,
                    "textual_rating": rating,
                    "stance": stance,
                    "description": f"{pub_name} evaluated this claim as '{rating}'.",
                    "source_type": "fact_check_review",
                    "reliability": "Very High"
                })

        return {
            "success": True,
            "configured": True,
            "query": clean_query,
            "count": len(reviews_found),
            "reviews": reviews_found,
            "error": None
        }

    except httpx.TimeoutException:
        logger.warning(f"Google Fact Check API request timed out after {DEFAULT_TIMEOUT_SECONDS}s.")
        return {
            "success": False,
            "configured": True,
            "reviews": [],
            "error": "Google Fact Check request timed out."
        }
    except Exception as e:
        logger.error(f"Google Fact Check unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "configured": True,
            "reviews": [],
            "error": f"Fact Check API failure: {str(e)}"
        }

def search_claims_fact_checks(claims: List[str]) -> Dict[str, Any]:
    """
    Searches Google Fact Check Tools across extracted claims.
    Collects established reviews and partitions into contradicting and supporting items.
    """
    if not is_fact_check_configured() or not claims:
        return {
            "configured": is_fact_check_configured(),
            "reviews": [],
            "contradicting": [],
            "supporting": []
        }

    all_reviews: List[Dict[str, Any]] = []
    seen_urls = set()

    for claim in claims[:3]:
        res = search_google_fact_check(claim)
        if res.get("success"):
            for r in res.get("reviews", []):
                if r["url"] and r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_reviews.append(r)

    contradicting = [r for r in all_reviews if r.get("stance") == "contradicts"]
    supporting = [r for r in all_reviews if r.get("stance") == "supports"]

    return {
        "configured": True,
        "reviews": all_reviews,
        "contradicting": contradicting,
        "supporting": supporting
    }
