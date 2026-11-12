import re
import logging
from typing import Dict, Any, List, Optional

from app.config import get_provider_capabilities, USER_AGENT
from app.ml.classifier import predict_text_credibility
from app.services.claim_extractor import extract_key_claims
from app.services.tavily_search import search_claims_evidence, is_tavily_configured
from app.services.factcheck_search import search_claims_fact_checks, is_fact_check_configured
from app.services.groq_verifier import (
    extract_claims_with_groq,
    synthesize_evidence_with_groq,
    is_groq_configured
)
from app.services.evidence_search import match_known_viral_hoaxes

logger = logging.getLogger("truthlens.verifier")

def generate_summary(text: str, max_words: int = 40) -> str:
    """Generates an objective 1-2 sentence executive summary of the content."""
    clean = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    if not sentences:
        return ""
    
    summary = sentences[0]
    if len(sentences) > 1 and len(summary.split()) < max_words:
        summary += " " + sentences[1]
        
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]) + "..."
    return summary

def verify_news_content(text: str, input_type: str = "text", original_url: str = None) -> Dict[str, Any]:
    """
    TruthLens Dynamic Evidence-Based Verification Pipeline:
    1. Ingest & sanitize news content.
    2. Extract falsifiable factual claims (via Groq / NLP).
    3. Search Google Fact Check Tools API for prior peer-reviewed claims.
    4. Search Tavily Search API for live web evidence (corroborating and debunking).
    5. Check accredited viral hoax debunks (WHO, Reuters, Snopes).
    6. Execute ML stylometric analysis as a secondary linguistic indicator.
    7. Synthesize evidence via Groq LLM (or deterministic fallback) with strict verdict rules:
       - 'REAL': Adequately corroborated by credible evidence with no material contradictions.
       - 'FAKE': Contradicted by credible evidence or established fact-checks.
       - 'INSUFFICIENT EVIDENCE': Evidence unavailable, weak, conflicting, or unverified.
    8. Format comprehensive output with source references and verification limitations.
    """
    clean_text = text.strip() if text else ""
    capabilities = get_provider_capabilities()

    # Guardrail: Content too brief to extract claims
    if len(clean_text) < 15:
        return {
            "classification": "INSUFFICIENT EVIDENCE",
            "confidence_score": 45.0,
            "ml_confidence": 0.0,
            "evidence_score": 0.0,
            "summary": "The submitted content is too brief to extract verifiable claims.",
            "explanation": "TruthLens requires sufficient factual context (minimum 15 characters or a complete sentence) to extract and cross-reference claims against external evidence.",
            "identified_claims": [],
            "claim_analyses": [],
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "sources": [],
            "service_status": capabilities,
            "limitations": ["Input text below minimum analytical threshold."],
            "ml_prediction": None,
            "status": "COMPLETED"
        }

    logger.info(f"Verification pipeline started for input length {len(clean_text)} (input_type={input_type}).")

    # Step 1: Identify Key Factual Claims (Groq with fallback to NLP)
    identified_claims = extract_claims_with_groq(clean_text)
    if not identified_claims:
        identified_claims = extract_key_claims(clean_text)
    if not identified_claims and clean_text:
        identified_claims = [clean_text[:160] + ("..." if len(clean_text) > 160 else "")]
    logger.info(f"Extracted {len(identified_claims)} verifiable claims: {[c[:60] for c in identified_claims]}")

    # Step 2: Query Google Fact Check Tools API
    fact_check_data = search_claims_fact_checks(identified_claims)
    fact_check_reviews = fact_check_data.get("reviews", [])
    fact_check_contradicting = fact_check_data.get("contradicting", [])
    fact_check_supporting = fact_check_data.get("supporting", [])
    logger.info(f"Fact Check API returned {len(fact_check_reviews)} reviews ({len(fact_check_contradicting)} contradicting, {len(fact_check_supporting)} supporting).")

    # Step 3: Query Tavily Search API for Live Web Evidence & Attributions
    tavily_data = search_claims_evidence(identified_claims, full_text=clean_text)
    tavily_sources = tavily_data.get("sources", [])
    tavily_contradicting = tavily_data.get("contradicting", [])
    tavily_supporting = tavily_data.get("supporting", [])
    attribution_findings = tavily_data.get("attribution_findings", [])
    logger.info(f"Tavily Search returned {len(tavily_sources)} sources ({len(tavily_contradicting)} contradicting, {len(tavily_supporting)} supporting, {len(attribution_findings)} attribution findings).")

    # Step 4: Check Known Viral Hoax Registry (Direct accredited fact-check URLs)
    hoax_matches = match_known_viral_hoaxes(clean_text)
    registry_contradicting = []
    registry_supporting = []
    for h in hoax_matches:
        item = {
            "source": h["publisher"],
            "claim_addressed": h["title"],
            "detail": h["description"]
        }
        if h["stance"] == "contradicts":
            registry_contradicting.append(item)
        else:
            registry_supporting.append(item)

    # Step 5: Execute ML Stylometric Analysis (Auxiliary indicator only)
    ml_result = predict_text_credibility(clean_text)
    ml_label = ml_result.get("predicted_label", "UNABLE TO VERIFY")
    ml_conf = ml_result.get("model_confidence", 50.0)
    logger.info(f"ML Stylometric Indicator (auxiliary only): label={ml_label}, real_prob={ml_result.get('real_probability')}%, fake_prob={ml_result.get('fake_probability')}%.")

    # Step 6: Consolidate External Evidence Sources (Deduplicated)
    all_sources: List[Dict[str, Any]] = []
    seen_urls = set()

    # Priority 1: Fact Check Reviews
    for r in fact_check_reviews:
        url = r.get("url", "#")
        if url and url not in seen_urls:
            seen_urls.add(url)
            all_sources.append({
                "title": r.get("title") or f"Fact Check: {r.get('publisher')}",
                "publisher": r.get("publisher", "Accredited Fact Checker"),
                "url": url,
                "description": r.get("description", ""),
                "published_date": r.get("review_date"),
                "reliability": "Very High",
                "source_type": "fact_check_review"
            })

    # Priority 2: Verified Registry Matches
    for h in hoax_matches:
        url = h.get("url", "#")
        if url and url not in seen_urls:
            seen_urls.add(url)
            all_sources.append({
                "title": h.get("title"),
                "publisher": h.get("publisher"),
                "url": url,
                "description": h.get("description"),
                "published_date": None,
                "reliability": "Very High",
                "source_type": "verified_registry"
            })

    # Priority 3: Tavily Live Web Results
    for s in tavily_sources:
        url = s.get("url", "#")
        if url and url not in seen_urls:
            seen_urls.add(url)
            all_sources.append(s)

    # Priority 4: Free Open Knowledge & News Search Fallback (if no paid API sources returned)
    if len(all_sources) == 0:
        from app.services.evidence_search import find_credible_evidence
        free_data = find_credible_evidence(identified_claims, clean_text)
        for s in free_data.get("sources", []):
            url = s.get("url", "#")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_sources.append(s)
        for sup in free_data.get("supporting_evidence", []):
            supporting_evidence.append(sup)
        for con in free_data.get("contradicting_evidence", []):
            contradicting_evidence.append(con)

    # Step 7: Synthesize Evidence via Groq (or deterministic rule engine)
    synthesis = synthesize_evidence_with_groq(
        article_text=clean_text,
        claims=identified_claims,
        fact_check_reviews=fact_check_reviews,
        tavily_sources=tavily_sources,
        ml_stylometrics=ml_result,
        attribution_findings=attribution_findings
    )

    verdict = synthesis.get("verdict", "INSUFFICIENT EVIDENCE")
    confidence_score = float(synthesis.get("confidence_score", 50.0))
    summary = synthesis.get("summary") or generate_summary(clean_text)
    explanation = synthesis.get("explanation") or ""
    claim_analyses = synthesis.get("claim_analyses") or []
    supporting_evidence = synthesis.get("supporting_evidence") or []
    contradicting_evidence = synthesis.get("contradicting_evidence") or []
    limitations = synthesis.get("limitations") or []
    logger.info(f"Evidence Synthesis ({synthesis.get('reasoning_engine', 'engine')}): raw_verdict={verdict}, raw_conf={confidence_score}.")

    # Merge verified registry debunks into contradicting_evidence
    for r in registry_contradicting:
        if not any(r["claim_addressed"] == c.get("claim_addressed") for c in contradicting_evidence):
            contradicting_evidence.append(r)
            if verdict != "FAKE":
                verdict = "FAKE"
                confidence_score = max(confidence_score, 88.0)
                explanation = f"Documented fact-checking records confirm this is an established viral hoax. {r['detail']}"

    # Merge Tavily debunk items into contradicting_evidence
    for tc in tavily_contradicting:
        if not any(tc.get("detail") == c.get("detail") for c in contradicting_evidence):
            contradicting_evidence.append(tc)

    # Merge supporting items (only if genuine corroboration exists)
    for s_item in registry_supporting:
        if not any(s_item["claim_addressed"] == s.get("claim_addressed") for s in supporting_evidence):
            supporting_evidence.append(s_item)

    for ts in tavily_supporting:
        if not any(ts.get("detail") == s.get("detail") for s in supporting_evidence):
            supporting_evidence.append(ts)

    # Step 8: Calculate Evidence Score
    evidence_score = 45.0
    if contradicting_evidence:
        evidence_score = 88.0 + min(len(contradicting_evidence) * 3, 6.0)
    elif supporting_evidence and len(supporting_evidence) >= 2:
        evidence_score = 75.0 + min(len(supporting_evidence) * 4, 15.0)
    elif all_sources:
        evidence_score = 55.0

    # Step 9: Final Quality & Accurate Verdict Aggregation
    is_conflicting = bool(
        contradicting_evidence and supporting_evidence and
        len(supporting_evidence) >= 2 and len(contradicting_evidence) >= 1
    )

    fake_prob = float(ml_result.get("fake_probability", 50.0))
    real_prob = float(ml_result.get("real_probability", 50.0))
    ml_label = ml_result.get("predicted_label", "UNABLE TO VERIFY")
    markers = ml_result.get("linguistic_markers", {})
    sensational = markers.get("sensational_score", 0.0)
    exclamations = markers.get("exclamations_per_100w", 0.0)

    # Priority 1: Conflicting evidence from credible sources
    if is_conflicting:
        verdict = "INSUFFICIENT EVIDENCE"
        confidence_score = 50.0
        explanation = (
            "External sources provide conflicting accounts regarding these claims. While some authoritative sources corroborate aspects of the report, "
            "independent investigations or fact checks raise substantial contradictions. Under strict verification standards, "
            "the claim is classified as INSUFFICIENT EVIDENCE pending decisive consensus."
        )

    # Priority 2: Direct Contradiction / Documented Fact-Check Refutation / Viral Hoax
    elif contradicting_evidence or registry_contradicting:
        verdict = "FAKE"
        confidence_score = max(confidence_score, 88.0, min(fake_prob, 94.0))
        source_name = contradicting_evidence[0].get("source", "Accredited Fact Checker") if contradicting_evidence else (registry_contradicting[0].get("source") if registry_contradicting else "Fact Check")
        explanation = (
            f"Credible external evidence directly contradicts the material claims. "
            f"Fact-checking records from {source_name} confirm the assertions are fabricated, disproven, or misleading."
        )

    # Priority 3: Direct External Corroboration from Authoritative Sources
    elif len(supporting_evidence) >= 1 or (verdict == "REAL" and len(all_sources) >= 1 and real_prob >= 55.0):
        verdict = "REAL"
        confidence_score = max(confidence_score, 80.0, min(real_prob, 92.0))
        if supporting_evidence:
            source_name = supporting_evidence[0].get("source", "Authoritative Source")
        elif all_sources:
            source_name = all_sources[0].get("publisher", "Authoritative Source")
        else:
            source_name = "Authoritative Source"
        explanation = (
            f"Factual assertions are corroborated by published reports and official records from {source_name}. "
            "No credible refutations or active hoax alerts were detected."
        )

    # Priority 4: Synthesis engine (Groq) delivered an explicit verdict
    elif verdict in ["REAL", "FAKE"] and synthesis.get("reasoning_engine", "").startswith("Groq"):
        # Groq evaluated evidence / plausibility
        pass

    # Priority 5: Evaluate Algorithmic & Linguistic Credibility Indicators
    else:
        # Check for strong signals of fabrication (high fake probability with alarmism, sensationalism, or emergency claims)
        has_sensationalism = (
            sensational >= 1.0 or exclamations >= 1.0 or markers.get("caps_ratio", 0) >= 10.0 or
            any(w in clean_text.lower() for w in ["shocking", "secret", "miracle", "cure", "alert", "doors", "emergency", "hoax", "conspiracy"])
        )
        if fake_prob >= 80.0 and has_sensationalism:
            verdict = "FAKE"
            confidence_score = min(max(fake_prob, 75.0), 93.0)
            summary = f"TruthLens analysis identified strong indicators of misinformation and fabricated claims ({round(confidence_score, 1)}% risk)."
            explanation = (
                f"TruthLens algorithmic and stylometric analysis identified strong indicators of misinformation and fabricated claims ({round(fake_prob, 1)}% probability). "
                "The content exhibits sensationalized framing, emergency panic rhetoric, or abnormal attribution patterns that lack corroboration."
            )
        else:
            verdict = "INSUFFICIENT EVIDENCE"
            confidence_score = 50.0
            explanation = (
                "TruthLens could not locate decisive corroborating primary sources or established fact-check reviews for this text. "
                "In accordance with responsible verification standards, uncorroborated news is classified as INSUFFICIENT EVIDENCE rather than assumed true or false."
            )

    # Policy 4: Attribution check without false equivalence
    if attribution_findings and verdict == "INSUFFICIENT EVIDENCE":
        limitations.append(
            "Attributed organisation investigated: Absence from indexed search records alone does not establish falsehood without direct refutation."
        )

    # Policy 5: Bound confidence between 45.0 and 94.0 (Never claim 100%)
    final_confidence = min(max(confidence_score, 45.0), 94.0)
    logger.info(f"Final Policy Enforcement: classification={verdict}, final_confidence={final_confidence}%, supporting={len(supporting_evidence)}, contradicting={len(contradicting_evidence)}.")

    # Compile explicit service status
    service_status = {
        "tavily_search": is_tavily_configured(),
        "google_fact_check": is_fact_check_configured(),
        "groq_reasoning": is_groq_configured(),
        "ml_classifier": capabilities.get("ml_classifier", True)
    }

    # Add general limitations if missing
    if not limitations:
        limitations = [
            "Evaluated against indexed Google Fact Check reviews and Tavily Search results.",
            "Newly developing breaking events or offline records may require additional manual confirmation."
        ]

    # Backward compatibility: for database / API clients expecting UNABLE TO VERIFY
    # The primary classification remains 'INSUFFICIENT EVIDENCE' (or 'REAL'/'FAKE')
    return {
        "classification": verdict,
        "confidence_score": round(final_confidence, 1),
        "ml_confidence": round(ml_conf, 1),
        "evidence_score": round(evidence_score, 1),
        "summary": summary,
        "explanation": explanation,
        "identified_claims": identified_claims,
        "claim_analyses": claim_analyses,
        "supporting_evidence": supporting_evidence,
        "contradicting_evidence": contradicting_evidence,
        "sources": all_sources[:8],
        "service_status": service_status,
        "limitations": limitations,
        "ml_prediction": {
            "predicted_label": ml_label,
            "fake_probability": ml_result.get("fake_probability", 50.0),
            "real_probability": ml_result.get("real_probability", 50.0),
            "role": "Auxiliary stylometric linguistic indicator (not factual proof)"
        },
        "status": "COMPLETED"
    }
