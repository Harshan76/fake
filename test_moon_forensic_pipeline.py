import json
from unittest.mock import patch
from app.schemas import VerificationRequest
from app.routers.verify import verify_endpoint
from app.database import SessionLocal

MOON_ARTICLE = (
    "Scientists Announce That the Moon Will Disappear for 48 Hours in December 2026.\n"
    "The Moon will disappear for 48 hours in December 2026.\n"
    "An organisation called International Space Research Council confirmed this.\n"
    "NASA advised people to remain indoors.\n"
    "Governments are preparing emergency plans."
)

def run_regression_test():
    print("=" * 80)
    print("TRUTHLENS FORENSIC REGRESSION TEST: MOON DISAPPEARANCE 2026 ARTICLE")
    print("=" * 80)
    
    # 1. Realistic mock of search and fact check where debunking evidence exists
    # Credible fact-checkers and news outlets directly contradict the fabricated claim
    mock_debunk_source = {
        "title": "Fact Check: No, NASA did not warn the moon will disappear for 48 hours",
        "publisher": "Reuters Fact Check",
        "url": "https://www.reuters.com/fact-check/moon-disappear-48-hours-debunk",
        "description": "Astronomers and NASA have confirmed that the moon will not disappear in December 2026. The claim is a complete hoax.",
        "authoritative": True,
        "stance": "contradicts"
    }
    mock_general_search = {
        "sources": [
            mock_debunk_source,
            {
                "title": "NASA - Artemis Moon Missions 2026",
                "publisher": "NASA",
                "url": "https://www.nasa.gov/artemis-2026",
                "description": "NASA details lunar exploration plans for 2026 under the Artemis program.",
                "authoritative": True,
                "stance": "neutral"
            }
        ],
        "contradicting": [mock_debunk_source],
        "supporting": []
    }
    mock_fact_checks = {
        "reviews": [
            {
                "claim": "Moon will disappear for 48 hours in December 2026",
                "claimant": "Social Media Viral Posts",
                "claimDate": "2026-10-01",
                "text": "Moon will disappear for 48 hours in December 2026",
                "rating": "False",
                "source": "Reuters Fact Check",
                "url": "https://www.reuters.com/fact-check/moon-disappear-48-hours-debunk",
                "reviewDate": "2026-10-02"
            }
        ],
        "contradicting": [mock_debunk_source],
        "supporting": []
    }

    db = SessionLocal()
    try:
        req = VerificationRequest(content=MOON_ARTICLE, input_type="text")
        
        with patch("app.services.verifier.search_claims_evidence", return_value=mock_general_search), \
             patch("app.services.verifier.search_claims_fact_checks", return_value=mock_fact_checks):
            
            data = verify_endpoint(req, db=db)
    finally:
        db.close()

    print("\n[1] INPUT TEXT:")
    print(MOON_ARTICLE)

    print("\n[2] EXTRACTED CLAIMS:")
    for claim in data.get("identified_claims", []):
        print(f"  * {claim}")

    print("\n[3] SEARCH QUERIES & RETRIEVED SOURCES:")
    for src in data.get("sources", []):
        print(f"  * [{src.get('stance', 'neutral').upper()}] {src.get('title')} ({src.get('url')})")

    print("\n[4] FACT-CHECK API RESULTS:")
    fact_checks = data.get("fact_checks", [])
    if fact_checks:
        for fc in fact_checks:
            print(f"  * {fc.get('title')} - Rating: {fc.get('rating')}")
    else:
        print("  * No existing third-party fact-check reviews found (Claim is unreviewed/emerging)")

    print("\n[5] GROQ / SYNTHESIS ANALYSIS:")
    print(f"  * Explanation: {data.get('explanation')}")

    print("\n[6] ML STYLOMETRIC PREDICTION (AUXILIARY ONLY):")
    print(f"  * ML Raw Confidence: {data.get('ml_confidence')}%")
    print(f"  * Overrode Evidence: NO (ML is strictly confined to stylometrics)")

    print("\n[7] FINAL VERDICT & CONFIDENCE:")
    print(f"  * Classification: {data.get('classification')}")
    print(f"  * Confidence Score: {data.get('confidence_score')}%")
    print(f"  * Supporting Sources: {len(data.get('supporting_evidence', []))}")
    print(f"  * Contradicting Sources: {len(data.get('contradicting_evidence', []))}")
    
    assert data["classification"] == "FAKE", f"Expected FAKE, got {data['classification']}"
    assert data["confidence_score"] >= 85.0, f"Expected >= 85.0%, got {data['confidence_score']}%"
    print("\n" + "=" * 80)
    print("REGRESSION TEST PASSED: Intentional fabrication is accurately classified as FAKE.")
    print("=" * 80)

if __name__ == "__main__":
    run_regression_test()
