import re
from typing import Dict, Any, List
from app.ml.classifier import predict_text_credibility
from app.services.claim_extractor import extract_key_claims
from app.services.evidence_search import find_credible_evidence

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
    Core TruthLens Verification Pipeline:
    Step 1: Extract & clean news content.
    Step 2: Identify core factual claims.
    Step 3: Query credible knowledge sources & fact-check repositories.
    Step 4: Analyze linguistic markers & execute ML classification.
    Step 5: Triangulate external evidence stance with model predictions.
    Step 6: Classify as REAL, FAKE, or UNABLE TO VERIFY.
    Step 7: Compute calibrated confidence score and build comprehensive breakdown.
    """
    clean_text = text.strip() if text else ""
    if len(clean_text) < 15:
        return {
            "classification": "UNABLE TO VERIFY",
            "confidence_score": 40.0,
            "ml_confidence": 0.0,
            "evidence_score": 0.0,
            "summary": "The submitted content is too brief to extract verifiable claims.",
            "explanation": "TruthLens requires sufficient factual context (minimum 15 characters or a complete sentence) to analyze claims against credible databases.",
            "identified_claims": [],
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "sources": []
        }

    # Step 2: Extract claims
    identified_claims = extract_key_claims(clean_text)

    # Step 3 & 4: Search for external evidence
    evidence_data = find_credible_evidence(identified_claims, clean_text)
    sources = evidence_data["sources"]
    supporting_evidence = evidence_data["supporting_evidence"]
    contradicting_evidence = evidence_data["contradicting_evidence"]
    evidence_score = evidence_data["evidence_score"]
    evidence_stance = evidence_data["stance"]

    # Step 5: ML Stylometric & NLP inference
    ml_result = predict_text_credibility(clean_text)
    ml_label = ml_result["predicted_label"]
    ml_conf = ml_result["model_confidence"]
    fake_prob = ml_result["fake_probability"]
    real_prob = ml_result["real_probability"]
    markers = ml_result["linguistic_markers"]

    # Step 6: Triangulation & Decision Logic
    # Case A: Debunking / Contradicting evidence found
    if evidence_stance == "contradicts" or len(contradicting_evidence) > 0:
        classification = "FAKE"
        # Weight evidence heavily over text style
        confidence_score = (evidence_score * 0.70) + (fake_prob * 0.30)
        explanation = (
            "This item was classified as FAKE because documented fact-checking evidence and official records "
            "directly contradict the claims made. External investigations have previously debunked similar assertions, "
            f"and linguistic analysis indicates {fake_prob:.1f}% stylistic alignment with misleading or sensationalized content."
        )

    # Case B: Verified supporting evidence found from credible institutions
    elif evidence_stance == "supports" and len(supporting_evidence) >= 2:
        if fake_prob > 80.0 and markers.get("sensational_score", 0) > 4:
            # Stylistically sensationalized but has supporting facts -> nuance
            classification = "UNABLE TO VERIFY"
            confidence_score = 58.0
            explanation = (
                "While some factual entities or events in this submission match published news reports, "
                "the text employs intense sensationalist framing and unverified exaggerations. "
                "Because verified facts are mixed with sensational rhetoric, TruthLens advises reader discretion."
            )
        else:
            classification = "REAL"
            confidence_score = (evidence_score * 0.65) + (real_prob * 0.35)
            explanation = (
                "This report was classified as REAL because the core factual claims correspond with established records, "
                "credible encyclopedic documentation, or reputable news coverage. The linguistic tone adheres to standard "
                "journalistic attribution without artificial viral triggers."
            )

    # Case C: Stylistically high fake probability with sensationalist markers, but no external evidence
    elif fake_prob >= 75.0 and markers.get("sensational_score", 0) >= 2:
        classification = "FAKE"
        confidence_score = min(fake_prob * 0.85, 84.0)  # Bound to avoid false certainty
        explanation = (
            "This content exhibits strong structural and linguistic patterns typical of disinformation campaigns and clickbait hoaxes: "
            f"heightened sensationalism (score: {markers.get('sensational_score', 0)}/10), urgent sharing directives, "
            "and a total absence of corroboration in standard news archives or scientific databases."
        )

    # Case D: High journalistic style with clear attribution
    elif real_prob >= 75.0 and markers.get("journalistic_attribution", 0) >= 1.5:
        classification = "REAL"
        confidence_score = min(real_prob * 0.85, 82.0)
        explanation = (
            "The content demonstrates authentic journalistic structure, balanced factual tone, and verifiable institutional attribution. "
            "No known debunking reports or disinformation flags were associated with this text."
        )

    # Case E: Inconclusive / Insufficient evidence
    else:
        classification = "UNABLE TO VERIFY"
        confidence_score = 50.0
        explanation = (
            "TruthLens found insufficient external evidence to definitively classify this content as either authentic or fabricated. "
            "The text lacks clear contradiction by established fact-checkers, yet also lacks decisive corroboration from independent primary sources. "
            "We strongly encourage manually verifying the original sources linked below."
        )

    # Cap confidence score realistically (Never claim 100%)
    final_confidence = min(max(confidence_score, 45.0), 94.0)

    # Executive summary
    summary = generate_summary(clean_text)

    return {
        "classification": classification,
        "confidence_score": round(final_confidence, 1),
        "ml_confidence": round(ml_conf, 1),
        "evidence_score": round(evidence_score, 1),
        "summary": summary,
        "explanation": explanation,
        "identified_claims": identified_claims,
        "supporting_evidence": supporting_evidence,
        "contradicting_evidence": contradicting_evidence,
        "sources": sources,
        "linguistic_markers": markers
    }
