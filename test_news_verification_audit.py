"""
TruthLens News Verification Comprehensive Audit
Tests the 12 required verification scenarios:
- 4 genuine news articles from credible publishers
- 4 fabricated news examples matching existing dataset
- 2 unfamiliar articles with insufficient evidence
- 2 screenshot extraction examples
"""
import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.database import init_db, SessionLocal
from app.schemas import VerificationRequest
from app.routers.verify import verify_endpoint
from app.services.verifier import verify_news_content

class TruthLensVerificationAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    # =========================================================================
    # Group A: 4 Genuine News Articles from Credible Publishers
    # =========================================================================

    def test_A1_who_immunization_stabilized(self):
        """Genuine news: WHO childhood immunization stabilized at 84%."""
        content = (
            "The World Health Organization confirmed that global immunization coverage has stabilized at 84%, "
            "with childhood vaccination programs recovering steadily across developing member nations."
        )
        mock_source = {
            "title": "WHO - Global Vaccination Coverage Statistics",
            "publisher": "World Health Organization",
            "url": "https://www.who.int/news/item/vaccination-coverage-data",
            "description": "Official report: childhood vaccination programs stabilized at 84% global coverage.",
            "authoritative": True,
            "stance": "supports"
        }
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_source], "contradicting": [], "supporting": [mock_source]}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "REAL")
        self.assertGreaterEqual(resp["confidence_score"], 80.0)
        self.assertGreaterEqual(len(resp["supporting_evidence"]), 1)
        self.assertEqual(len(resp["contradicting_evidence"]), 0)
        print(f"\n[Test A1] WHO Immunization -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, ML Conf: {resp['ml_confidence']}%, Sources: {len(resp['sources'])}")

    def test_A2_esa_euclid_mission_launch(self):
        """Genuine news: ESA Euclid space telescope launch."""
        content = (
            "The European Space Agency launched the Euclid space telescope from Cape Canaveral "
            "on a mission to map the geometry of the dark universe and study dark energy."
        )
        mock_source = {
            "title": "ESA - Euclid space mission launched to study dark universe",
            "publisher": "European Space Agency",
            "url": "https://www.esa.int/Science_Exploration/Space_Science/Euclid",
            "description": "Euclid was launched to create the most accurate 3D map of the universe and study dark energy.",
            "authoritative": True,
            "stance": "supports"
        }
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_source], "contradicting": [], "supporting": [mock_source]}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "REAL")
        self.assertGreaterEqual(resp["confidence_score"], 80.0)
        self.assertGreaterEqual(len(resp["supporting_evidence"]), 1)
        print(f"[Test A2] ESA Euclid -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, ML Conf: {resp['ml_confidence']}%, Sources: {len(resp['sources'])}")

    def test_A3_nasa_james_webb_telescope(self):
        """Genuine news: NASA James Webb Space Telescope launch."""
        content = (
            "NASA successfully launched the James Webb Space Telescope in 2021 "
            "to observe the earliest stars and galaxies formed after the Big Bang."
        )
        mock_source = {
            "title": "NASA - James Webb Space Telescope Mission Overview",
            "publisher": "NASA",
            "url": "https://www.nasa.gov/mission_pages/webb/main/index.html",
            "description": "NASA's James Webb Space Telescope was successfully launched in 2021 to explore the early universe.",
            "authoritative": True,
            "stance": "supports"
        }
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_source], "contradicting": [], "supporting": [mock_source]}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "REAL")
        self.assertGreaterEqual(resp["confidence_score"], 80.0)
        self.assertGreaterEqual(len(resp["supporting_evidence"]), 1)
        print(f"[Test A3] NASA JWST -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, ML Conf: {resp['ml_confidence']}%, Sources: {len(resp['sources'])}")

    def test_A4_cern_higgs_boson_discovery(self):
        """Genuine news: CERN Higgs boson observation."""
        content = (
            "Scientists at CERN announced the historic discovery and observation of the Higgs boson particle "
            "using the Large Hadron Collider in Geneva."
        )
        mock_source = {
            "title": "CERN - New results indicate that new particle is a Higgs boson",
            "publisher": "CERN",
            "url": "https://home.cern/news/press-release/cern/cern-experiments-observe-particle-consistent-long-sought-higgs-boson",
            "description": "Experiments at the Large Hadron Collider observe particle consistent with long-sought Higgs boson.",
            "authoritative": True,
            "stance": "supports"
        }
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_source], "contradicting": [], "supporting": [mock_source]}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "REAL")
        self.assertGreaterEqual(resp["confidence_score"], 80.0)
        self.assertGreaterEqual(len(resp["supporting_evidence"]), 1)
        print(f"[Test A4] CERN Higgs Boson -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, ML Conf: {resp['ml_confidence']}%, Sources: {len(resp['sources'])}")

    # =========================================================================
    # Group B: 4 Fabricated News Examples Matching Existing Dataset
    # =========================================================================

    def test_B1_moon_disappearing_hoax(self):
        """Fabricated news: Moon disappearing for 48 hours."""
        content = "Scientists Announce That the Moon Will Disappear for 48 Hours in December 2026."
        mock_debunk = {
            "title": "Fact Check: NASA did not warn the moon will disappear in December 2026",
            "publisher": "Reuters Fact Check",
            "url": "https://www.reuters.com/fact-check/moon-disappear-48-hours-debunk",
            "description": "Astronomers and NASA have confirmed that the moon will not disappear in December 2026. The claim is a complete hoax.",
            "authoritative": True,
            "stance": "contradicts"
        }
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_debunk], "contradicting": [mock_debunk], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [mock_debunk], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "FAKE")
        self.assertGreaterEqual(resp["confidence_score"], 85.0)
        self.assertGreaterEqual(len(resp["contradicting_evidence"]), 1)
        print(f"[Test B1] Moon Disappearing -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, Contradictions: {len(resp['contradicting_evidence'])}")

    def test_B2_lemon_water_cancer_hoax(self):
        """Fabricated news: Lemon water and baking soda eliminates cancer."""
        content = "Drinking hot lemon water with baking soda immediately eliminates all cancer cells within 48 hours according to Johns Hopkins Medicine."
        mock_debunk = {
            "title": "Fact Check: Lemon water and baking soda does not cure cancer",
            "publisher": "FactCheck.org",
            "url": "https://www.factcheck.org/lemon-baking-soda-cancer-hoax",
            "description": "Oncologists confirm there is no scientific basis that lemon water eliminates cancer cells. The viral claim is false.",
            "authoritative": True,
            "stance": "contradicts"
        }
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_debunk], "contradicting": [mock_debunk], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [mock_debunk], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "FAKE")
        self.assertGreaterEqual(resp["confidence_score"], 85.0)
        self.assertGreaterEqual(len(resp["contradicting_evidence"]), 1)
        print(f"[Test B2] Lemon Cancer Cure -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, Contradictions: {len(resp['contradicting_evidence'])}")

    def test_B3_canadian_bitcoin_currency_announcement(self):
        """Fabricated news: Canadian currency replaced by Bitcoin."""
        content = "Prime Minister of Canada declared the national currency will be replaced by Bitcoin next month."
        mock_unrelated_bio = {
            "title": "Wikipedia: Mark Carney",
            "publisher": "Wikipedia Foundation",
            "url": "https://en.wikipedia.org/wiki/Mark_Carney",
            "description": "Canadian politician and economist who served as governor of the Bank of Canada.",
            "authoritative": True,
            "stance": "neutral"
        }
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_unrelated_bio], "contradicting": [], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertNotEqual(resp["classification"], "REAL")
        self.assertEqual(resp["classification"], "INSUFFICIENT EVIDENCE")
        self.assertEqual(len(resp["supporting_evidence"]), 0)
        print(f"[Test B3] Canadian Bitcoin Replacement -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, Uncorroborated safely.")

    def test_B4_alien_spacecraft_docked_with_jwst(self):
        """Fabricated news: Alien spacecraft docked with JWST."""
        content = "NASA and international observers confirmed that an alien spacecraft secretly docked with the James Webb Space Telescope."
        mock_debunk = {
            "title": "Fact Check: No alien spacecraft docked with James Webb Telescope",
            "publisher": "Lead Stories",
            "url": "https://leadstories.com/alien-dock-jwst-hoax",
            "description": "Viral social media posts claiming NASA documented alien spacecraft near JWST are completely fabricated.",
            "authoritative": True,
            "stance": "contradicts"
        }
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_debunk], "contradicting": [mock_debunk], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [mock_debunk], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "FAKE")
        self.assertGreaterEqual(len(resp["contradicting_evidence"]), 1)
        print(f"[Test B4] Alien Docking JWST -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, Contradictions: {len(resp['contradicting_evidence'])}")

    # =========================================================================
    # Group C: 2 Unfamiliar Articles with Insufficient Evidence
    # =========================================================================

    def test_C1_secret_zurich_quantum_lab(self):
        """Unfamiliar claim: Secret quantum lab room-temperature superconductor."""
        content = "A stealth quantum computing laboratory in Zurich achieved Room-Temperature Superconductivity using graphene bilayers."
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [], "contradicting": [], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "INSUFFICIENT EVIDENCE")
        self.assertEqual(resp["confidence_score"], 50.0)
        self.assertEqual(len(resp["supporting_evidence"]), 0)
        self.assertEqual(len(resp["contradicting_evidence"]), 0)
        print(f"[Test C1] Unfamiliar Quantum Lab -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, Safely unverified.")

    def test_C2_small_town_parking_rule(self):
        """Unfamiliar claim: Local municipal town council weekend parking rule."""
        content = "The municipal town council of Oakhaven voted to eliminate weekend metered parking fees for hybrid vehicles starting next Tuesday."
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [], "contradicting": [], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "INSUFFICIENT EVIDENCE")
        self.assertEqual(resp["confidence_score"], 50.0)
        self.assertEqual(len(resp["supporting_evidence"]), 0)
        self.assertEqual(len(resp["contradicting_evidence"]), 0)
        print(f"[Test C2] Small Town Parking -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, Safely unverified.")

    # =========================================================================
    # Group D: 2 Screenshot Extraction Verification Examples
    # =========================================================================

    def test_D1_clean_screenshot_extraction(self):
        """Screenshot verification: Readable headline and article body."""
        extracted_text = (
            "European Union regulators approved a comprehensive artificial intelligence act "
            "establishing binding transparency and risk rules for machine learning across all member nations."
        )
        mock_source = {
            "title": "EU Artificial Intelligence Act approved by European Parliament",
            "publisher": "Reuters",
            "url": "https://www.reuters.com/technology/eu-ai-act-passed",
            "description": "European lawmakers voted to adopt landmark rules on artificial intelligence governance.",
            "authoritative": True,
            "stance": "supports"
        }
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_source], "contradicting": [], "supporting": [mock_source]}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(
                content=extracted_text,
                input_type="image",
                extracted_text=extracted_text
            )
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["input_type"], "image")
        self.assertEqual(resp["extracted_text"], extracted_text)
        self.assertEqual(resp["classification"], "REAL")
        self.assertGreaterEqual(resp["confidence_score"], 80.0)
        print(f"[Test D1] Clean Screenshot Extraction -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, Extracted text preserved.")

    def test_D2_incomplete_unreadable_screenshot(self):
        """Screenshot verification: Incomplete or unreadable image text."""
        blurry_ocr_text = "12:45 PM | 5G"
        req = VerificationRequest(
            content=blurry_ocr_text,
            input_type="image",
            extracted_text=blurry_ocr_text
        )
        resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "INSUFFICIENT EVIDENCE")
        self.assertEqual(resp["confidence_score"], 50.0)
        self.assertIn("screenshot", resp["explanation"].lower())
        self.assertEqual(len(resp["supporting_evidence"]), 0)
        print(f"[Test D2] Unreadable Screenshot -> Verdict: {resp['classification']}, Conf: {resp['confidence_score']}%, Handled safely with explanation.")

if __name__ == "__main__":
    unittest.main()
