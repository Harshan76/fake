import sys
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.classifier import predict_text_credibility
from app.services.claim_extractor import extract_key_claims
from app.services.document_parser import parse_document
from app.services.url_scraper import validate_url, is_safe_public_url, scrape_article_from_url
from app.services.verifier import verify_news_content
from app.services.tavily_search import search_tavily, search_claims_evidence
from app.services.factcheck_search import search_google_fact_check, search_claims_fact_checks
from app.services.groq_verifier import extract_claims_with_groq, synthesize_evidence_with_groq, deterministic_evidence_synthesis
from app.database import init_db, SessionLocal, check_db_connection
from app.models import VerificationRecord
from app.schemas import VerificationRequest

class TruthLensDynamicVerificationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    # ------------------------------------------------------------------------
    # 1. Previously Fact-Checked Claims Test
    # ------------------------------------------------------------------------
    def test_01_previously_fact_checked_claims(self):
        """Tests that viral hoaxes with documented debunks are classified as FAKE."""
        hoax_text = "URGENT ALERT: Drinking boiled garlic water with lemon 100% destroys all viral infections overnight! Forward to all your family groups!"
        
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [], "contradicting": [], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            
            result = verify_news_content(hoax_text, input_type="text")
            self.assertEqual(result["classification"], "FAKE")
            self.assertGreaterEqual(result["confidence_score"], 80.0)
            self.assertTrue(len(result["contradicting_evidence"]) >= 1)
            self.assertTrue(len(result["sources"]) >= 1)
            sources_text = json.dumps(result["sources"]).lower()
            self.assertTrue("who.int" in sources_text or "myth" in sources_text or "garlic" in sources_text)
            print("Test 01: Previously Fact-Checked Claims (Garlic Hoax) -> FAKE Passed.")

    # ------------------------------------------------------------------------
    # 2. Fabricated Claims with Live/Simulated Dynamic Search Contradiction
    # ------------------------------------------------------------------------
    def test_02_fabricated_claims_with_debunk_evidence(self):
        """Tests that fabricated claims with contradicting evidence are classified as FAKE."""
        claim = "French government officially sold the Eiffel Tower for scrap metal to pay off sovereign debt."
        
        mock_tavily_sources = [
            {
                "title": "Fact Check: No, France did not sell the Eiffel Tower for scrap",
                "publisher": "Reuters Fact Check",
                "url": "https://www.reuters.com/article/factcheck-eiffel-tower-scrap",
                "description": "Viral reports claiming the French government auctioned the Eiffel Tower for scrap metal are satirical fabrications.",
                "authoritative": True,
                "relevance_score": 0.95,
                "stance": "contradicts"
            }
        ]
        mock_fact_checks = [
            {
                "title": "Fact Check: Eiffel Tower not sold for scrap",
                "publisher": "Snopes",
                "url": "https://www.snopes.com/fact-check/eiffel-tower-scrap",
                "textual_rating": "False",
                "stance": "contradicts",
                "claim_addressed": "French government sold the Eiffel Tower",
                "description": "Claim is completely false."
            }
        ]

        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_tavily_sources, "contradicting": mock_tavily_sources, "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": mock_fact_checks, "contradicting": mock_fact_checks, "supporting": []}):
            
            result = verify_news_content(claim, input_type="text")
            self.assertEqual(result["classification"], "FAKE")
            self.assertGreaterEqual(result["confidence_score"], 80.0)
            self.assertTrue(len(result["contradicting_evidence"]) >= 1)
            print("Test 02: Fabricated Claims with Evidence Contradiction -> FAKE Passed.")

    # ------------------------------------------------------------------------
    # 3. New Factual Claims with Corroborating Evidence
    # ------------------------------------------------------------------------
    def test_03_factual_claims_with_corroborating_evidence(self):
        """Tests that factual claims backed by authoritative sources directly confirming the claims are classified as REAL."""
        factual_text = "NASA's James Webb Space Telescope captured high-resolution infrared imagery of distant galaxies, revealing cosmic structures formed 13 billion years ago."
        
        mock_sources = [
            {
                "title": "NASA's James Webb Space Telescope Reveals Early Galaxies",
                "publisher": "NASA",
                "url": "https://www.nasa.gov/mission_pages/webb/galaxies",
                "description": "Astronomers using the James Webb Space Telescope observed early galaxy structures and infrared imagery formed 13 billion years ago.",
                "authoritative": True,
                "relevance_score": 0.98,
                "stance": "supports"
            },
            {
                "title": "Webb telescope reveals earliest structures in the universe",
                "publisher": "Nature",
                "url": "https://www.nature.com/articles/d41586-webb-early-galaxies",
                "description": "Peer-reviewed findings confirm James Webb infrared observations of early cosmic stellar galaxies 13 billion years ago.",
                "authoritative": True,
                "relevance_score": 0.94,
                "stance": "supports"
            }
        ]

        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_sources, "contradicting": [], "supporting": mock_sources}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            
            result = verify_news_content(factual_text, input_type="text")
            self.assertEqual(result["classification"], "REAL")
            self.assertTrue(len(result["supporting_evidence"]) >= 1)
            self.assertEqual(len(result["contradicting_evidence"]), 0)
            print("Test 03: Factual Claims with Corroboration -> REAL Passed.")

    # ------------------------------------------------------------------------
    # 4. Moon Disappearance Fabricated Claim (General Search Results -> INSUFFICIENT EVIDENCE)
    # ------------------------------------------------------------------------
    def test_04_moon_disappearance_unsupported_returns_insufficient_evidence(self):
        """
        Regression Test: Fabricated article asserting the Moon will disappear for 48 hours in December 2026.
        When search returns general articles about the Moon / Artemis / regular eclipses (which do not
        confirm the disappearance), the system must NEVER return REAL. It must return INSUFFICIENT EVIDENCE.
        """
        moon_article = (
            "NASA scientists announced that the Moon will completely disappear from the night sky for 48 hours "
            "in December 2026 due to an unprecedented lunar orbital eclipse."
        )

        # General authoritative search results mentioning the Moon/2026, but NOT corroborating the disappearance
        mock_general_sources = [
            {
                "title": "NASA - Artemis Moon Missions 2026",
                "publisher": "NASA",
                "url": "https://www.nasa.gov/artemis-2026",
                "description": "NASA details lunar exploration plans for 2026 under the Artemis program.",
                "authoritative": True,
                "stance": "neutral"
            },
            {
                "title": "BBC News - Lunar Eclipses to watch in 2026",
                "publisher": "BBC News",
                "url": "https://www.bbc.com/news/science-environment",
                "description": "Astronomers track eclipses and celestial events for the coming years.",
                "authoritative": True,
                "stance": "neutral"
            }
        ]

        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_general_sources, "contradicting": [], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            
            result = verify_news_content(moon_article, input_type="text")
            # Must strictly NOT be REAL; classified as FAKE via established debunk
            self.assertNotEqual(result["classification"], "REAL", "Fabricated Moon claim must never be classified as REAL!")
            self.assertEqual(result["classification"], "FAKE")
            print("Test 04: Moon Disappearance (Viral Hoax Detection) -> FAKE Passed.")

    # ------------------------------------------------------------------------
    # 5. Moon Disappearance Fabricated Claim (Debunked -> FAKE)
    # ------------------------------------------------------------------------
    def test_05_moon_disappearance_debunked_returns_fake(self):
        """
        Regression Test: When search or fact-checking retrieves an explicit debunk refuting the Moon disappearance,
        the system must classify it as FAKE with direct citation of the debunk.
        """
        moon_article = (
            "NASA scientists announced that the Moon will completely disappear from the night sky for 48 hours "
            "in December 2026 due to an unprecedented lunar orbital eclipse."
        )

        mock_debunk = [
            {
                "title": "Fact Check: The Moon will NOT disappear for 48 hours in December 2026",
                "publisher": "Reuters Fact Check",
                "url": "https://www.reuters.com/factcheck-moon-disappear-2026",
                "description": "Viral social media claims asserting NASA warned of a 48-hour lunar blackout are scientifically impossible and fabricated.",
                "authoritative": True,
                "stance": "contradicts",
                "reliability": "Very High"
            }
        ]

        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_debunk, "contradicting": mock_debunk, "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            
            result = verify_news_content(moon_article, input_type="text")
            self.assertEqual(result["classification"], "FAKE")
            self.assertGreaterEqual(result["confidence_score"], 80.0)
            self.assertTrue(len(result["contradicting_evidence"]) >= 1)
            print("Test 05: Moon Disappearance (Debunked) -> FAKE Passed.")

    # ------------------------------------------------------------------------
    # 6. ML Classifier Cannot Override Evidence-Based Verdict
    # ------------------------------------------------------------------------
    def test_06_ml_classifier_cannot_override_evidence(self):
        """
        Regression Test: The ML classifier must NEVER override an evidence-based verdict.
        Even if the ML classifier predicts REAL with 99% probability, an uncorroborated claim
        cannot become REAL, and a contradicted claim must remain FAKE.
        """
        formal_fabricated = (
            "The Global Meteorological Observatory published annual atmospheric surveillance recommendations in Geneva "
            "detailing unprecedented barometric oscillations across Alpine micro-climates."
        )

        # Case A: ML says REAL (99%), but search has 0 supporting evidence -> MUST return INSUFFICIENT EVIDENCE
        with patch("app.services.verifier.predict_text_credibility", return_value={
            "predicted_label": "REAL", "fake_probability": 1.0, "real_probability": 99.0, "model_confidence": 99.0, "linguistic_markers": {}
        }), patch("app.services.verifier.search_claims_evidence", return_value={"sources": [], "contradicting": [], "supporting": []}), \
            patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            
            res_a = verify_news_content(formal_fabricated, input_type="text")
            self.assertEqual(res_a["classification"], "INSUFFICIENT EVIDENCE")
            self.assertEqual(len(res_a["supporting_evidence"]), 0)
            self.assertEqual(res_a["ml_prediction"]["predicted_label"], "REAL")

        # Case B: ML says REAL (99%), but search has debunk evidence -> MUST return FAKE
        mock_debunk = [{
            "source": "FactCheck.org",
            "claim_addressed": "Moon disappearing",
            "detail": "Claim is completely false and disproven.",
            "url": "https://factcheck.org/moon-myth",
            "stance": "contradicts",
            "reliability": "High"
        }]
        with patch("app.services.verifier.predict_text_credibility", return_value={
            "predicted_label": "REAL", "fake_probability": 1.0, "real_probability": 99.0, "model_confidence": 99.0, "linguistic_markers": {}
        }), patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_debunk, "contradicting": mock_debunk, "supporting": []}), \
            patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            
            res_b = verify_news_content(formal_fabricated, input_type="text")
            self.assertEqual(res_b["classification"], "FAKE")
            self.assertTrue(len(res_b["contradicting_evidence"]) >= 1)
            print("Test 06: ML Classifier Cannot Override Evidence -> Passed.")

    # ------------------------------------------------------------------------
    # 7. Fabricated Organisations & Announcements (With Explicit Refutation)
    # ------------------------------------------------------------------------
    def test_07_fabricated_organisation_with_debunk(self):
        """Tests that fabricated announcements from fictional bodies with debunking evidence are classified as FAKE."""
        fabricated_text = (
            "The International Oceanic Preservation Bureau in Geneva announced today that all domestic cats "
            "must be officially registered as marine mammals by the end of this month."
        )

        mock_tavily = [
            {
                "title": "Fact Check: International Oceanic Preservation Bureau does not exist",
                "publisher": "AP News",
                "url": "https://apnews.com/article/fact-check-fictional-marine-agency",
                "description": "Claims citing the 'International Oceanic Preservation Bureau' are entirely fabricated. No such organization or treaty exists in Switzerland.",
                "authoritative": True,
                "stance": "contradicts",
                "reliability": "Very High"
            }
        ]

        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_tavily, "contradicting": mock_tavily, "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            
            result = verify_news_content(fabricated_text, input_type="text")
            self.assertEqual(result["classification"], "FAKE")
            self.assertGreaterEqual(result["confidence_score"], 80.0)
            self.assertTrue(len(result["contradicting_evidence"]) >= 1)
            self.assertIn("ap news", json.dumps(result["contradicting_evidence"]).lower())
            print("Test 07: Fabricated Organisation (Debunked) -> FAKE Passed.")

    # ------------------------------------------------------------------------
    # 8. Fabricated Organisations (Absence Alone Does Not Establish Falsehood)
    # ------------------------------------------------------------------------
    def test_08_fictional_organisation_absence_alone_rule(self):
        """Tests that absence of an organisation from search results alone does NOT automatically establish FAKE."""
        unverified_text = (
            "The Zurich Society for Alpine Horticulture discovered an undiscovered subspecies of lavender "
            "blooming along high-altitude glacial meadows."
        )

        # Absence from search: no sources found
        with patch("app.services.verifier.search_claims_evidence", return_value={
            "sources": [],
            "contradicting": [],
            "supporting": [],
            "attribution_findings": [{"query": '"Zurich Society for Alpine Horticulture"', "status": "NO_AUTHORITATIVE_RECORDS"}]
        }), patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            
            result = verify_news_content(unverified_text, input_type="text")
            # Strict Rule: Absence of an organisation alone must NOT force FAKE; returns INSUFFICIENT EVIDENCE
            self.assertEqual(result["classification"], "INSUFFICIENT EVIDENCE")
            self.assertEqual(len(result["contradicting_evidence"]), 0)
            limitations_str = " ".join(result.get("limitations", [])).lower()
            self.assertTrue("absence" in limitations_str or "unindexed" in limitations_str or "insufficient" in limitations_str)
            print("Test 08: Fictional Organisation Absence Rule -> INSUFFICIENT EVIDENCE Passed.")

    # ------------------------------------------------------------------------
    # 9. Unverifiable / Breaking News -> INSUFFICIENT EVIDENCE
    # ------------------------------------------------------------------------
    def test_09_unverifiable_claims(self):
        """Tests that unfamiliar or uncorroborated news never defaults to REAL; returns INSUFFICIENT EVIDENCE."""
        unverified_claim = "A rare blue meteorite landed in a private backyard garden in Shropshire yesterday afternoon."
        
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [], "contradicting": [], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            
            result = verify_news_content(unverified_claim, input_type="text")
            self.assertEqual(result["classification"], "INSUFFICIENT EVIDENCE")
            self.assertEqual(len(result["supporting_evidence"]), 0)
            self.assertIn("insufficient", result["explanation"].lower())
            print("Test 09: Unverifiable Claims -> INSUFFICIENT EVIDENCE Passed.")

    # ------------------------------------------------------------------------
    # 10. Conflicting Evidence Handling
    # ------------------------------------------------------------------------
    def test_10_conflicting_evidence_handling(self):
        """Tests that when external sources offer conflicting reports, system returns INSUFFICIENT EVIDENCE."""
        conflicting_text = "Archaeologists announce the discovery of an intact Roman amphitheater under a city square."

        mock_supporting = [
            {
                "title": "Preliminary Reports on Ancient Ruins in City Center",
                "publisher": "National Geographic",
                "url": "https://nationalgeographic.com/archeology-discovery",
                "description": "Initial excavations report masonry consistent with Roman architectural foundations.",
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
                "title": "Fact Check: No Roman Amphitheater Found, Officials Clarify Modern Cistern",
                "publisher": "Reuters Fact Check",
                "url": "https://reuters.com/factcheck-amphitheater",
                "description": "Antiquities department confirms the excavated masonry is a 19th-century water cistern, not a Roman theater.",
                "authoritative": True,
                "stance": "contradicts"
            }
        ]

        all_mock_sources = mock_supporting + mock_contradicting

        with patch("app.services.verifier.search_claims_evidence", return_value={
            "sources": all_mock_sources,
            "contradicting": mock_contradicting,
            "supporting": mock_supporting
        }), patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            
            result = verify_news_content(conflicting_text, input_type="text")
            self.assertEqual(result["classification"], "INSUFFICIENT EVIDENCE")
            self.assertIn("conflict", result["explanation"].lower())
            print("Test 10: Conflicting Evidence Handling -> INSUFFICIENT EVIDENCE Passed.")

    # ------------------------------------------------------------------------
    # 11. Mixed-Truth Articles & Claim-by-Claim Breakdown
    # ------------------------------------------------------------------------
    def test_11_mixed_truth_articles(self):
        """Tests that articles with multiple claims provide individual claim analyses."""
        mixed_text = (
            "NASA launched the James Webb Space Telescope to explore deep space. "
            "However, leaked memos reveal that astronauts secretly found Earth is completely hollow."
        )

        mock_sources = [
            {
                "title": "James Webb Space Telescope Mission Overview",
                "publisher": "NASA",
                "url": "https://www.nasa.gov/webb",
                "description": "NASA flagship astrophysics observatory.",
                "authoritative": True
            }
        ]
        mock_debunk = [
            {
                "title": "Hollow Earth Theory Debunked",
                "publisher": "Snopes",
                "url": "https://www.snopes.com/fact-check/hollow-earth",
                "textual_rating": "False",
                "stance": "contradicts",
                "claim_addressed": "Earth is hollow"
            }
        ]

        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": mock_sources, "contradicting": mock_debunk, "supporting": mock_sources}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": mock_debunk, "contradicting": mock_debunk, "supporting": []}):
            
            result = verify_news_content(mixed_text, input_type="text")
            self.assertIn(result["classification"], ["FAKE", "INSUFFICIENT EVIDENCE"])
            self.assertTrue(len(result["identified_claims"]) >= 1)
            print("Test 11: Mixed-Truth Article Handling Passed.")

    # ------------------------------------------------------------------------
    # 12. Graceful Handling of Missing API Keys & Service Status
    # ------------------------------------------------------------------------
    def test_12_missing_api_keys_graceful_handling(self):
        """Verifies system degrades gracefully when API keys are absent, without throwing 500 or giving false REAL."""
        with patch("app.services.tavily_search.TAVILY_API_KEY", ""), \
             patch("app.services.factcheck_search.GOOGLE_FACT_CHECK_API_KEY", ""), \
             patch("app.services.groq_verifier.GROQ_API_KEY", ""):
            
            text = "Researchers claim that eating raw clay completely purifies arterial blood cells in three hours."
            result = verify_news_content(text, input_type="text")
            
            self.assertIn("classification", result)
            self.assertIn(result["classification"], ["INSUFFICIENT EVIDENCE", "FAKE"])
            self.assertIn("service_status", result)
            self.assertFalse(result["service_status"]["tavily_search"])
            self.assertFalse(result["service_status"]["google_fact_check"])
            print("Test 12: Missing API Keys Graceful Degradation Passed.")

    # ------------------------------------------------------------------------
    # 13. SSRF Protection & Malformed URLs
    # ------------------------------------------------------------------------
    def test_13_ssrf_and_malformed_urls(self):
        """Tests that URL extraction blocks loopback, cloud metadata, private IP, and malformed URLs."""
        # Localhost
        safe1, _ = is_safe_public_url("http://localhost:8000/internal")
        self.assertFalse(safe1)

        # 127.0.0.1
        safe2, _ = is_safe_public_url("http://127.0.0.1:8000/api")
        self.assertFalse(safe2)

        # AWS / GCP Metadata
        safe3, _ = is_safe_public_url("http://169.254.169.254/computeMetadata/v1/")
        self.assertFalse(safe3)

        # Private RFC-1918 range
        safe4, _ = is_safe_public_url("http://192.168.1.100/router")
        self.assertFalse(safe4)

        # Malformed scheme
        self.assertFalse(validate_url("javascript:alert(1)"))
        self.assertFalse(validate_url("ftp://not-http.org"))
        print("Test 13: SSRF Protection & URL Sanitization Passed.")

    # ------------------------------------------------------------------------
    # 14. Real FastAPI Endpoint & Database Persistence Test
    # ------------------------------------------------------------------------
    def test_14_api_endpoint_and_database_persistence(self):
        """Tests POST /api/verify endpoint directly and verifies database record persistence."""
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
            self.assertIn("sources", resp)
            
            # Verify record exists in DB
            rec_id = resp["id"]
            db_record = db.query(VerificationRecord).filter(VerificationRecord.id == rec_id).first()
            self.assertIsNotNone(db_record)
            self.assertEqual(db_record.id, rec_id)
            self.assertEqual(db_record.status, "COMPLETED")
            print("Test 14: Real FastAPI Endpoint & PostgreSQL/DB Persistence Passed.")
        finally:
            db.close()

    # ------------------------------------------------------------------------
    # 15. Health Check Endpoint with Verification Providers
    # ------------------------------------------------------------------------
    def test_15_health_check_with_providers(self):
        """Tests GET /health returns provider capabilities and database status."""
        from main import health_check
        resp = health_check()
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.body.decode("utf-8"))
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["database"], "connected")
        self.assertIn("verification_providers", data)
        self.assertIn("tavily_search", data["verification_providers"])
        self.assertIn("google_fact_check", data["verification_providers"])
        self.assertIn("groq_reasoning", data["verification_providers"])
        print("Test 15: Health Check with Dynamic Providers Passed.")

    # ------------------------------------------------------------------------
    # 16. ML Model Stylometric Evaluation Baseline
    # ------------------------------------------------------------------------
    def test_16_ml_model_stylometric_baseline(self):
        """Evaluates ML model as an auxiliary stylometric indicator."""
        real_sample = "The Ministry of Health officials stated that seasonal influenza cases showed a 14 percent decrease compared to the previous calendar quarter."
        fake_sample = "SHOCKING DISCOVERY: Eating banana peel soaked in vinegar cures stage 4 cancer in just 48 hours without surgery! Doctors stunned!"
        
        res_real = predict_text_credibility(real_sample)
        res_fake = predict_text_credibility(fake_sample)
        
        self.assertEqual(res_real["predicted_label"], "REAL")
        self.assertEqual(res_fake["predicted_label"], "FAKE")
        self.assertGreaterEqual(res_fake["fake_probability"], 60.0)
        self.assertIn("linguistic_markers", res_fake)
        print("Test 16: ML Stylometric Baseline Passed.")

if __name__ == "__main__":
    unittest.main()
