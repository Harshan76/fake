import sys
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.classifier import predict_text_credibility
from app.services.claim_extractor import (
    extract_key_claims,
    classify_article_topic,
    extract_entities_and_statistics,
    generate_adaptive_verification_queries
)
from app.services.document_parser import parse_document
from app.services.url_scraper import validate_url, is_safe_public_url, scrape_article_from_url
from app.services.verifier import verify_news_content
from app.services.tavily_search import search_tavily, search_claims_evidence
from app.services.factcheck_search import search_google_fact_check, search_claims_fact_checks
from app.services.groq_verifier import (
    extract_claims_with_groq,
    synthesize_evidence_with_groq,
    deterministic_evidence_synthesis
)
from app.services.evidence_search import (
    search_wikipedia_knowledge,
    search_google_news_rss,
    find_credible_evidence,
    deduplicate_syndicated_sources
)
from app.database import init_db, SessionLocal, check_db_connection
from app.models import VerificationRecord
from app.schemas import VerificationRequest

class TruthLensUniversalVerificationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    # ------------------------------------------------------------------------
    # 1. Health / Medical Domain: Verified True Claim
    # ------------------------------------------------------------------------
    def test_01_health_domain_corroborated_true(self):
        """Tests health claim backed by authoritative WHO/CDC corroboration -> REAL."""
        article = (
            "The World Health Organization confirmed that global immunization coverage has stabilized at 84%, "
            "with childhood vaccination programs recovering steadily across developing member nations."
        )
        mock_sources = [
            {
                "title": "WHO - Global Vaccination Coverage Statistics",
                "publisher": "World Health Organization",
                "url": "https://www.who.int/news/item/vaccination-coverage-data",
                "description": "Official report: childhood vaccination programs stabilized at 84% global coverage.",
                "authoritative": True,
                "stance": "supports"
            }
        ]
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_sources, "contradicting": [], "supporting": mock_sources}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            res = verify_news_content(article, input_type="text")
            self.assertEqual(res["classification"], "REAL")
            self.assertTrue(len(res["supporting_evidence"]) >= 1)
            print("Test 01: Health Domain (Corroborated) -> REAL Passed.")

    # ------------------------------------------------------------------------
    # 2. Health / Medical Domain: Debunked False Claim
    # ------------------------------------------------------------------------
    def test_02_health_domain_debunked_fake(self):
        """Tests dangerous health misinformation refuted by accredited fact-checkers -> FAKE."""
        article = "Drinking industrial bleach and miracle mineral solution completely cures viral respiratory infections in 48 hours."
        mock_debunk = [
            {
                "source": "U.S. FDA",
                "claim_addressed": "Drinking bleach cures viral infections",
                "detail": "Drinking chemical disinfectants or bleach causes severe poisoning and does not cure any illness.",
                "url": "https://www.fda.gov/consumers/bleach-warning",
                "stance": "contradicts",
                "reliability": "Very High"
            }
        ]
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_debunk, "contradicting": mock_debunk, "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": mock_debunk, "contradicting": mock_debunk, "supporting": []}):
            res = verify_news_content(article, input_type="text")
            self.assertEqual(res["classification"], "FAKE")
            self.assertTrue(len(res["contradicting_evidence"]) >= 1)
            print("Test 02: Health Domain (Debunked) -> FAKE Passed.")

    # ------------------------------------------------------------------------
    # 3. Sports Domain: Verified True Claim
    # ------------------------------------------------------------------------
    def test_03_sports_domain_corroborated_true(self):
        """Tests sports claim corroborated by official governing body reporting -> REAL."""
        article = "Argentina won the FIFA World Cup final against France in a penalty shootout after a 3-3 draw in Qatar."
        mock_sources = [
            {
                "title": "FIFA - Argentina Crowned World Champions in Dramatic Final",
                "publisher": "FIFA",
                "url": "https://www.fifa.com/tournaments/mens/worldcup/qatar2022/match-report",
                "description": "Argentina secured the FIFA World Cup victory against France via penalty shootout following a 3-3 draw.",
                "authoritative": True,
                "stance": "supports"
            }
        ]
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_sources, "contradicting": [], "supporting": mock_sources}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            res = verify_news_content(article, input_type="text")
            self.assertEqual(res["classification"], "REAL")
            print("Test 03: Sports Domain (Corroborated) -> REAL Passed.")

    # ------------------------------------------------------------------------
    # 4. Sports Domain: Fabricated False Claim
    # ------------------------------------------------------------------------
    def test_04_sports_domain_debunked_fake(self):
        """Tests sports fabrication refuted by official league records -> FAKE."""
        article = "The International Olympic Committee announced that all running shoes are officially banned from future marathons."
        mock_debunk = [
            {
                "source": "AP Fact Check",
                "claim_addressed": "Running shoes banned from marathons",
                "detail": "IOC confirmed no ban exists; the viral claim originated from a satirical blog.",
                "url": "https://apnews.com/article/fact-check-running-shoes-marathon",
                "stance": "contradicts",
                "reliability": "High"
            }
        ]
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_debunk, "contradicting": mock_debunk, "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            res = verify_news_content(article, input_type="text")
            self.assertEqual(res["classification"], "FAKE")
            print("Test 04: Sports Domain (Fabrication) -> FAKE Passed.")

    # ------------------------------------------------------------------------
    # 5. Technology Domain: Verified True Claim
    # ------------------------------------------------------------------------
    def test_05_technology_domain_corroborated_true(self):
        """Tests technology announcement confirmed by official company releases -> REAL."""
        article = "TSMC began commercial high-volume production of 3-nanometer semiconductor chips at its Fab 18 facility."
        mock_sources = [
            {
                "title": "TSMC - Volume Production of 3nm Semiconductor Technology",
                "publisher": "Reuters",
                "url": "https://www.reuters.com/technology/tsmc-starts-volume-production-3nm-chips",
                "description": "Semiconductor foundry TSMC officially inaugurated volume production of 3nm chips at its southern fabrication plant.",
                "authoritative": True,
                "stance": "supports"
            }
        ]
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_sources, "contradicting": [], "supporting": mock_sources}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            res = verify_news_content(article, input_type="text")
            self.assertEqual(res["classification"], "REAL")
            print("Test 05: Technology Domain (Corroborated) -> REAL Passed.")

    # ------------------------------------------------------------------------
    # 6. Business / Finance Domain: Verified True Claim
    # ------------------------------------------------------------------------
    def test_06_finance_domain_corroborated_true(self):
        """Tests economic and central bank claim confirmed by official regulatory statements -> REAL."""
        article = "The Federal Reserve kept interest rates steady at 5.25% to 5.50% following the Federal Open Market Committee meeting."
        mock_sources = [
            {
                "title": "Federal Reserve issues FOMC statement on benchmark interest rate",
                "publisher": "Federal Reserve",
                "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240131a.htm",
                "description": "Official FOMC statement: The Committee decided to maintain the target range for the federal funds rate at 5.25 to 5.50 percent.",
                "authoritative": True,
                "stance": "supports"
            }
        ]
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_sources, "contradicting": [], "supporting": mock_sources}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            res = verify_news_content(article, input_type="text")
            self.assertEqual(res["classification"], "REAL")
            print("Test 06: Finance Domain (Corroborated) -> REAL Passed.")

    # ------------------------------------------------------------------------
    # 7. Politics / Government Domain: Verified True Claim
    # ------------------------------------------------------------------------
    def test_07_politics_domain_corroborated_true(self):
        """Tests government legislation and policy confirmed by official gazette records -> REAL."""
        article = "The European Parliament formally passed the Artificial Intelligence Act, establishing risk-based regulations for machine learning models."
        mock_sources = [
            {
                "title": "European Parliament approves landmark Artificial Intelligence Act",
                "publisher": "BBC News",
                "url": "https://www.bbc.com/news/technology-68546450",
                "description": "MEPs voted overwhelmingly in Strasbourg to pass the comprehensive EU AI Act establishing legal standards for artificial intelligence.",
                "authoritative": True,
                "stance": "supports"
            }
        ]
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_sources, "contradicting": [], "supporting": mock_sources}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            res = verify_news_content(article, input_type="text")
            self.assertEqual(res["classification"], "REAL")
            print("Test 07: Politics Domain (Corroborated) -> REAL Passed.")

    # ------------------------------------------------------------------------
    # 8. Science / Space Domain: Moon Disappearance Regression Test (Debunked -> FAKE)
    # ------------------------------------------------------------------------
    def test_08_science_regression_moon_disappearance_fake(self):
        """Regression Test: The fabricated Moon disappearance claim is contradicted by astronomical fact-checks -> FAKE."""
        moon_article = (
            "NASA scientists announced that the Moon will completely disappear from the night sky for 48 hours "
            "in December 2026 due to an unprecedented lunar orbital eclipse."
        )
        mock_debunk = [
            {
                "source": "Reuters Fact Check",
                "claim_addressed": "Moon disappearing for 48 hours in 2026",
                "detail": "Viral social media claims asserting NASA warned of a 48-hour lunar disappearance are scientifically impossible and fabricated.",
                "url": "https://www.reuters.com/factcheck-moon-disappear-2026",
                "stance": "contradicts",
                "reliability": "Very High"
            }
        ]
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_debunk, "contradicting": mock_debunk, "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            res = verify_news_content(moon_article, input_type="text")
            self.assertEqual(res["classification"], "FAKE")
            self.assertGreaterEqual(res["confidence_score"], 85.0)
            self.assertTrue(len(res["contradicting_evidence"]) >= 1)
            print("Test 08: Science Regression (Moon Disappearance) -> FAKE Passed.")

    # ------------------------------------------------------------------------
    # 9. International Affairs Domain: True Corroborated Event
    # ------------------------------------------------------------------------
    def test_09_international_domain_corroborated_true(self):
        """Tests international diplomatic agreement confirmed by multiple independent news agencies -> REAL."""
        article = "Delegates from over 190 nations reached an accord at COP28 in Dubai agreeing to transition away from fossil fuels in energy systems."
        mock_sources = [
            {
                "title": "COP28 agreement reached on transition away from fossil fuels",
                "publisher": "Reuters",
                "url": "https://www.reuters.com/sustainability/cop/cop28-agreement-draft",
                "description": "Global summit in Dubai concludes with unanimous consensus on transitioning away from fossil fuels in energy systems.",
                "authoritative": True,
                "stance": "supports"
            }
        ]
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_sources, "contradicting": [], "supporting": mock_sources}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            res = verify_news_content(article, input_type="text")
            self.assertEqual(res["classification"], "REAL")
            print("Test 09: International Domain (Corroborated) -> REAL Passed.")

    # ------------------------------------------------------------------------
    # 10. Ambiguous / Unverified Breaking Claim -> INSUFFICIENT EVIDENCE
    # ------------------------------------------------------------------------
    def test_10_ambiguous_unverified_claim_returns_insufficient_evidence(self):
        """Tests unfamiliar local claim with zero external corroboration or refutation -> INSUFFICIENT EVIDENCE."""
        unverified_claim = "A rare blue meteorite landed in a private backyard garden in Shropshire yesterday afternoon."
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [], "contradicting": [], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            res = verify_news_content(unverified_claim, input_type="text")
            self.assertEqual(res["classification"], "INSUFFICIENT EVIDENCE")
            self.assertEqual(len(res["supporting_evidence"]), 0)
            self.assertEqual(len(res["contradicting_evidence"]), 0)
            print("Test 10: Ambiguous Unverified Claim -> INSUFFICIENT EVIDENCE Passed.")

    # ------------------------------------------------------------------------
    # 11. Conflicting Evidence Handling -> INSUFFICIENT EVIDENCE
    # ------------------------------------------------------------------------
    def test_11_conflicting_evidence_returns_insufficient_evidence(self):
        """Tests that when reliable sources offer conflicting accounts without consensus, system returns INSUFFICIENT EVIDENCE."""
        conflicting_text = "Archaeologists announce the discovery of an intact Roman amphitheater under a city square."
        mock_supporting = [
            {
                "title": "Preliminary Excavations Report Ancient Masonry",
                "publisher": "National Geographic",
                "url": "https://nationalgeographic.com/archeology-discovery",
                "description": "Initial findings describe stone foundations consistent with Roman construction.",
                "authoritative": True,
                "stance": "supports"
            },
            {
                "title": "Archaeologists Excavate Downtown Trench",
                "publisher": "BBC News",
                "url": "https://bbc.com/news/excavation-square",
                "description": "Field team uncovers ancient stones dating to antiquity.",
                "authoritative": True,
                "stance": "supports"
            }
        ]
        mock_contradicting = [
            {
                "source": "Antiquities Review Board",
                "claim_addressed": "Roman amphitheater discovery",
                "detail": "Geological surveys demonstrate the stone formations are 19th-century sewer canal walls, not Roman ruins.",
                "url": "https://archeology-bulletin.org/city-excavation-rebuttal",
                "stance": "contradicts",
                "reliability": "Very High"
            }
        ]
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_supporting, "contradicting": mock_contradicting, "supporting": mock_supporting}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": mock_contradicting, "supporting": mock_supporting}):
            res = verify_news_content(conflicting_text, input_type="text")
            self.assertEqual(res["classification"], "INSUFFICIENT EVIDENCE")
            self.assertIn("conflicting", res["explanation"].lower())
            print("Test 11: Conflicting Evidence Handling -> INSUFFICIENT EVIDENCE Passed.")

    # ------------------------------------------------------------------------
    # 12. Mixed-Truth Article Handling
    # ------------------------------------------------------------------------
    def test_12_mixed_truth_article_claim_breakdown(self):
        """Tests article containing both true facts and fabricated claims provides claim-level breakdown."""
        mixed_text = (
            "NASA successfully launched the James Webb Space Telescope in December 2021. "
            "However, an organization called the Galactic Defense Council announced that aliens contacted telescope engineers in secret."
        )
        mock_debunk = [
            {
                "source": "FactCheck.org",
                "claim_addressed": "Galactic Defense Council alien contact",
                "detail": "NASA confirms no alien contact occurred and the Galactic Defense Council is a fictional entity.",
                "url": "https://factcheck.org/alien-contact-debunk",
                "stance": "contradicts",
                "reliability": "High"
            }
        ]
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_debunk, "contradicting": mock_debunk, "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": mock_debunk, "contradicting": mock_debunk, "supporting": []}):
            res = verify_news_content(mixed_text, input_type="text")
            self.assertEqual(res["classification"], "FAKE")
            self.assertTrue(len(res["identified_claims"]) >= 1)
            print("Test 12: Mixed-Truth Article Claim Breakdown Passed.")

    # ------------------------------------------------------------------------
    # 13. ML Classifier Cannot Override Evidence
    # ------------------------------------------------------------------------
    def test_13_ml_cannot_override_evidence(self):
        """Tests that ML classifier predicting REAL with 99% cannot override zero supporting evidence."""
        fabricated = "The Global Meteorological Bureau announced that barometric oscillations caused gravity to drop 50% across the Alps."
        with patch("app.services.verifier.predict_text_credibility", return_value={
            "predicted_label": "REAL", "fake_probability": 1.0, "real_probability": 99.0, "model_confidence": 99.0, "linguistic_markers": {}
        }), patch("app.services.verifier.search_claims_evidence", return_value={"sources": [], "contradicting": [], "supporting": []}), \
            patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            res = verify_news_content(fabricated, input_type="text")
            self.assertNotEqual(res["classification"], "REAL")
            self.assertEqual(res["classification"], "INSUFFICIENT EVIDENCE")
            print("Test 13: ML Cannot Override Evidence Passed.")

    # ------------------------------------------------------------------------
    # 14. Missing API Keys Graceful Degradation
    # ------------------------------------------------------------------------
    def test_14_missing_api_keys_graceful_degradation(self):
        """Verifies system operates gracefully when all external API keys are absent without 500 errors."""
        with patch("app.services.tavily_search.TAVILY_API_KEY", ""), \
             patch("app.services.factcheck_search.GOOGLE_FACT_CHECK_API_KEY", ""), \
             patch("app.services.groq_verifier.GROQ_API_KEY", ""):
            text = "Astronomers observed an unusual light flare near the binary star system Alpha Centauri."
            res = verify_news_content(text, input_type="text")
            self.assertIn("classification", res)
            self.assertIn("service_status", res)
            self.assertFalse(res["service_status"]["tavily_search"])
            self.assertFalse(res["service_status"]["google_fact_check"])
            print("Test 14: Missing API Keys Graceful Degradation Passed.")

    # ------------------------------------------------------------------------
    # 15. SSRF Protection & URL Sanitization
    # ------------------------------------------------------------------------
    def test_15_ssrf_protection_and_url_sanitization(self):
        """Tests that SSRF protections block private IP ranges, loopback, and cloud metadata."""
        safe1, _ = is_safe_public_url("http://localhost:8000/internal")
        self.assertFalse(safe1)
        safe2, _ = is_safe_public_url("http://127.0.0.1:8000/api")
        self.assertFalse(safe2)
        safe3, _ = is_safe_public_url("http://169.254.169.254/computeMetadata/v1/")
        self.assertFalse(safe3)
        safe4, _ = is_safe_public_url("http://192.168.1.100/router")
        self.assertFalse(safe4)
        self.assertFalse(validate_url("javascript:alert(1)"))
        print("Test 15: SSRF Protection & URL Sanitization Passed.")

    # ------------------------------------------------------------------------
    # 16. Real FastAPI Endpoint & PostgreSQL/DB Persistence
    # ------------------------------------------------------------------------
    def test_16_fastapi_endpoint_and_db_persistence(self):
        """Tests real FastAPI POST /api/verify endpoint and database record persistence."""
        from app.routers.verify import verify_endpoint
        db = SessionLocal()
        try:
            req = VerificationRequest(
                content="The World Health Organization confirmed today that global vaccination coverage has reached 72% according to official surveillance data published in Geneva.",
                input_type="text"
            )
            with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [], "contradicting": [], "supporting": []}), \
                 patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
                resp = verify_endpoint(req, db=db)

            self.assertIn("id", resp)
            self.assertIn("classification", resp)
            self.assertIn("confidence_score", resp)
            self.assertIn("service_status", resp)

            rec_id = resp["id"]
            db_record = db.query(VerificationRecord).filter(VerificationRecord.id == rec_id).first()
            self.assertIsNotNone(db_record)
            self.assertEqual(db_record.id, rec_id)
            self.assertEqual(db_record.status, "COMPLETED")
            print("Test 16: Real FastAPI Endpoint & PostgreSQL/DB Persistence Passed.")
        finally:
            db.close()

if __name__ == "__main__":
    unittest.main()
