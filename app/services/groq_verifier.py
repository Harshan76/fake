import json
import logging
import httpx
from typing import List, Dict, Any, Optional
from app.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger("truthlens.groq")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 15.0

def is_groq_configured() -> bool:
    """Checks whether the Groq API key is present."""
    return bool(GROQ_API_KEY and GROQ_API_KEY.strip())

def extract_claims_with_groq(text: str) -> List[str]:
    """
    Uses Groq to extract 1 to 5 concise, falsifiable factual propositions from news text.
    Falls back gracefully to rule-based claim extraction if Groq is unavailable.
    """
    from app.services.claim_extractor import extract_key_claims

    if not is_groq_configured() or len(text.strip()) < 30:
        return extract_key_claims(text)

    prompt = (
        "Extract between 1 and 4 specific, falsifiable factual claims from the following news text. "
        "Each claim must be a standalone proposition that can be verified as true or false against external evidence. "
        "Do not include opinions, emotional commentary, or questions.\n\n"
        f"Text:\n{text[:4000]}\n\n"
        "Return ONLY a JSON array of claim strings. Example format:\n"
        '["Claim statement 1", "Claim statement 2"]'
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a factual claim extraction system. Return strictly a JSON array of claim strings."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "response_format": {"type": "json_object"}
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(GROQ_CHAT_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()]
            elif isinstance(parsed, dict):
                # Search for claims key
                for k in ["claims", "extracted_claims", "propositions", "items"]:
                    if k in parsed and isinstance(parsed[k], list):
                        return [str(c).strip() for c in parsed[k] if str(c).strip()]
                # If values are list
                for val in parsed.values():
                    if isinstance(val, list):
                        return [str(c).strip() for c in val if str(c).strip()]

    except Exception as e:
        logger.warning(f"Groq claim extraction failed ({e}), using rule-based extraction.")

    return extract_key_claims(text)

def synthesize_evidence_with_groq(
    article_text: str,
    claims: List[str],
    fact_check_reviews: List[Dict[str, Any]],
    tavily_sources: List[Dict[str, Any]],
    ml_stylometrics: Dict[str, Any],
    attribution_findings: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Uses Groq to cross-reference extracted claims against retrieved Fact Checks & Tavily web evidence.
    Produces an evidence-grounded final verdict (REAL, FAKE, or INSUFFICIENT EVIDENCE).
    """
    if not is_groq_configured():
        return deterministic_evidence_synthesis(
            article_text, claims, fact_check_reviews, tavily_sources, ml_stylometrics, attribution_findings
        )

    # Format evidence summaries for the prompt
    fact_check_summary = []
    for r in fact_check_reviews:
        fact_check_summary.append({
            "fact_checker": r.get("publisher"),
            "claim_reviewed": r.get("claim_addressed"),
            "rating": r.get("textual_rating"),
            "url": r.get("url"),
            "date": r.get("review_date")
        })

    web_evidence_summary = []
    for s in tavily_sources:
        web_evidence_summary.append({
            "title": s.get("title"),
            "publisher": s.get("publisher"),
            "url": s.get("url"),
            "date": s.get("published_date"),
            "authoritative": s.get("authoritative", False),
            "stance": s.get("stance", "neutral"),
            "snippet": (s.get("description") or "")[:350]
        })

    system_instruction = (
        "You are the TruthLens Forensic Evidence Engine. Your duty is to evaluate news authenticity "
        "based STRICTLY on provided external evidence.\n\n"
        "STRICT VERDICT RULES:\n"
        "1. FAKE: Return FAKE when credible evidence directly contradicts a material factual claim or a reliable fact-check explicitly rates it false.\n"
        "2. REAL: Return REAL when material claims have sufficient credible supporting evidence from authoritative sources, with no unresolved contradictions.\n"
        "3. INSUFFICIENT EVIDENCE: Return INSUFFICIENT EVIDENCE when neither conclusion is supported, or when evidence is absent, conflicting, or inconclusive.\n\n"
        "ATTRIBUTION & FICTIONAL ORGANISATIONS:\n"
        "- When a claim is attributed to an organization, authority, government agency, or committee:\n"
        "  * Investigate the attribution using authoritative sources.\n"
        "  * If authoritative sources explicitly confirm the organisation is fictitious, non-existent, or never made the announcement, rate the claim as CONTRADICTED / FAKE.\n"
        "  * IMPORTANT: Absence of an organisation from search results alone MUST NOT automatically establish that a claim is false (it may be unindexed or local). In such cases, if no direct contradiction exists, classify as INSUFFICIENT EVIDENCE.\n\n"
        "CONFLICTING EVIDENCE:\n"
        "- If reputable sources offer conflicting or contradictory accounts on core claims without consensus, return INSUFFICIENT EVIDENCE and note the conflict.\n\n"
        "NEGATIVE CONSTRAINTS:\n"
        "- NEVER classify as REAL merely because writing sounds professional, calm, or grammatical.\n"
        "- NEVER classify as FAKE merely because text contains sensational words or exclamations, unless claims are contradicted by evidence.\n"
        "- NEVER fabricate sources, quotes, URLs, or dates not present in the supplied evidence.\n\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "verdict": "REAL" | "FAKE" | "INSUFFICIENT EVIDENCE",\n'
        '  "confidence_score": <number between 45.0 and 94.0>,\n'
        '  "summary": "<1-2 sentence executive summary>",\n'
        '  "explanation": "<detailed rationale citing only the provided evidence>",\n'
        '  "claim_analyses": [\n'
        '    {\n'
        '      "claim": "<claim text>",\n'
        '      "status": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT EVIDENCE",\n'
        '      "finding": "<summary of what evidence shows regarding this specific claim>",\n'
        '      "sources_used": ["<source titles>"]\n'
        '    }\n'
        '  ],\n'
        '  "supporting_evidence": [\n'
        '    {"source": "<publisher>", "claim_addressed": "<claim>", "detail": "<specific corroboration>"}\n'
        '  ],\n'
        '  "contradicting_evidence": [\n'
        '    {"source": "<publisher>", "claim_addressed": "<claim>", "detail": "<specific contradiction>"}\n'
        '  ],\n'
        '  "limitations": ["<verification limitation statement>"]\n'
        "}"
    )

    user_payload = {
        "article_text": article_text[:2500],
        "extracted_claims": claims,
        "google_fact_check_reviews": fact_check_summary,
        "tavily_web_sources": web_evidence_summary,
        "attribution_findings": attribution_findings or [],
        "ml_stylometric_indicator": {
            "linguistic_label": ml_stylometrics.get("predicted_label"),
            "fake_probability": ml_stylometrics.get("fake_probability"),
            "real_probability": ml_stylometrics.get("real_probability"),
            "sensational_score": ml_stylometrics.get("linguistic_markers", {}).get("sensational_score", 0)
        }
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": json.dumps(user_payload, indent=2)}
        ],
        "temperature": 0.1,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"}
    }

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            resp = client.post(GROQ_CHAT_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)

            verdict = str(result.get("verdict", "INSUFFICIENT EVIDENCE")).upper().strip()
            if verdict not in ["REAL", "FAKE", "INSUFFICIENT EVIDENCE"]:
                verdict = "INSUFFICIENT EVIDENCE"

            conf = float(result.get("confidence_score", 50.0))
            # Bound confidence ethically between 45.0 and 94.0
            conf = min(max(conf, 45.0), 94.0)

            return {
                "verdict": verdict,
                "confidence_score": round(conf, 1),
                "summary": result.get("summary") or "Summary of evidence analysis.",
                "explanation": result.get("explanation") or "Evidence analysis completed.",
                "claim_analyses": result.get("claim_analyses") or [],
                "supporting_evidence": result.get("supporting_evidence") or [],
                "contradicting_evidence": result.get("contradicting_evidence") or [],
                "limitations": result.get("limitations") or ["Verified against retrieved external evidence."],
                "reasoning_engine": f"Groq LLM ({GROQ_MODEL})"
            }

    except Exception as e:
        logger.error(f"Groq synthesis failed: {e}. Falling back to deterministic analysis.", exc_info=True)
        return deterministic_evidence_synthesis(
            article_text, claims, fact_check_reviews, tavily_sources, ml_stylometrics, attribution_findings
        )

def deterministic_evidence_synthesis(
    article_text: str,
    claims: List[str],
    fact_check_reviews: List[Dict[str, Any]],
    tavily_sources: List[Dict[str, Any]],
    ml_stylometrics: Dict[str, Any],
    attribution_findings: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Deterministic fallback rule engine when Groq API is unconfigured or unreachable.
    Evaluates evidence strictly without hallucination:
    - Return FAKE when credible evidence directly contradicts a material factual claim or a reliable fact-check explicitly rates it false.
    - Return REAL when material claims have sufficient credible supporting evidence.
    - Return INSUFFICIENT EVIDENCE when neither conclusion is supported or evidence is conflicting.
    - For fictional organisations or demonstrably fabricated attributions: investigate attribution using authoritative sources. Absence of an organisation from search results alone must not automatically establish that a claim is false.
    """
    from app.services.tavily_search import inspect_source_stance

    contradicting = []
    supporting = []
    claim_analyses = []
    primary_claim = claims[0] if claims else "Primary Claim"

    # 1. Inspect Fact-Check Reviews
    for r in fact_check_reviews:
        stance = r.get("stance", "unclear")
        item = {
            "source": r.get("publisher", "Accredited Fact Checker"),
            "claim_addressed": r.get("claim_addressed", primary_claim),
            "detail": f"Rated as '{r.get('textual_rating')}' by {r.get('publisher')}."
        }
        if stance == "contradicts":
            contradicting.append(item)
        elif stance == "supports":
            supporting.append(item)

    # 2. Inspect Tavily Web Sources
    for s in tavily_sources:
        stance = s.get("stance") or inspect_source_stance(s)
        item = {
            "source": s.get("publisher", "Web Source"),
            "claim_addressed": primary_claim,
            "detail": f"{s.get('title')}: {s.get('description', '')[:220]}"
        }
        if stance == "contradicts":
            if not any(c.get("detail") == item["detail"] for c in contradicting):
                contradicting.append(item)
        elif stance == "supports" and s.get("authoritative"):
            if not any(sup.get("detail") == item["detail"] for sup in supporting):
                supporting.append(item)

    # 3. Check for conflicting evidence
    is_conflicting = bool(contradicting and supporting and len(supporting) >= 2 and len(contradicting) >= 1)

    # 4. Determine Verdict
    if is_conflicting:
        verdict = "INSUFFICIENT EVIDENCE"
        confidence = 50.0
        summary = "Conflicting evidence identified across external authoritative sources."
        explanation = (
            "External sources provide conflicting accounts regarding these claims. While some sources corroborate the statements, "
            "independent investigations or fact checks raise substantial contradictions. Under strict verification standards, "
            "the claim is classified as INSUFFICIENT EVIDENCE pending decisive consensus."
        )
    elif contradicting:
        verdict = "FAKE"
        confidence = 88.0 + min(len(contradicting) * 3, 5.0)
        source_name = contradicting[0].get("source", "Accredited Source")
        summary = f"Claim evaluated as FAKE based on documented fact checks and refutations from {source_name}."
        explanation = (
            f"Independent fact-checking and published investigations directly refute the core claims in this text. "
            f"Evidence from {source_name} confirms the assertions are fabricated, disproven, or misleading."
        )
    elif len(supporting) >= 2:
        verdict = "REAL"
        confidence = 78.0 + min(len(supporting) * 4, 12.0)
        source_name = supporting[0].get("source", "Authoritative Source")
        summary = f"Factual claims are corroborated by published reports from authoritative sources including {source_name}."
        explanation = (
            "The primary factual assertions are corroborated by multiple independent, authoritative news and institutional sources. "
            "No active debunking alerts or credible contradictions were identified."
        )
    else:
        verdict = "INSUFFICIENT EVIDENCE"
        confidence = 50.0
        summary = "Insufficient external evidence to definitively verify or refute these claims."
        explanation = (
            "TruthLens could not locate decisive corroborating primary sources or established fact-check reviews for this text. "
            "In accordance with responsible verification standards, uncorroborated news is classified as INSUFFICIENT EVIDENCE rather than assumed true or false."
        )

    # Build claim analyses
    for c in claims:
        c_status = "INSUFFICIENT EVIDENCE"
        if is_conflicting:
            c_status = "INSUFFICIENT EVIDENCE"
        elif contradicting:
            c_status = "CONTRADICTED"
        elif verdict == "REAL":
            c_status = "SUPPORTED"

        claim_analyses.append({
            "claim": c,
            "status": c_status,
            "finding": f"Evaluated against {len(fact_check_reviews)} fact-check reviews and {len(tavily_sources)} web sources.",
            "sources_used": [s.get("title", "") for s in (fact_check_reviews + tavily_sources)[:2]]
        })

    limitations = [
        "Automated cross-referencing against indexed Google Fact Check reviews and Tavily Search results.",
        "Absence of an organisation from search results alone does not automatically establish that a claim is false without direct refutation.",
        "Newly developing breaking events or offline records may require additional manual confirmation."
    ]

    return {
        "verdict": verdict,
        "confidence_score": round(confidence, 1),
        "summary": summary,
        "explanation": explanation,
        "claim_analyses": claim_analyses,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "limitations": limitations,
        "reasoning_engine": "Deterministic Evidence Rules (Fallback)"
    }
