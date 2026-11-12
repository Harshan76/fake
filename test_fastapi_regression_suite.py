import unittest
from unittest.mock import patch
from app.database import init_db, SessionLocal
from app.models import VerificationRecord
from app.schemas import VerificationRequest
from app.routers.verify import verify_endpoint

class TruthLensFastAPIRegressionSuite(unittest.TestCase):
    """
    Step 5 Comprehensive Regression Test Suite:
    Executes all 10 required test categories directly through the actual FastAPI verification endpoint.
    Verifies that unverified, fabricated, or ambiguous claims NEVER default to REAL.
    """

    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    # 1. Clearly verified genuine news -> REAL
    def test_01_clearly_verified_genuine_news(self):
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

    # 2. Clearly debunked fake news -> FAKE
    def test_02_clearly_debunked_fake_news(self):
        content = "Scientists Announce That the Moon Will Disappear for 48 Hours in December 2026."
        mock_debunk = {
            "title": "Fact Check: No, NASA did not warn the moon will disappear for 48 hours",
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

    # 3. Fabricated government announcement -> INSUFFICIENT EVIDENCE (Never REAL)
    def test_03_fabricated_government_announcement(self):
        content = "Prime Minister of Canada declared the national currency will be replaced by Bitcoin next month."
        mock_unrelated_bio = {
            "title": "Wikipedia: Mark Carney",
            "publisher": "Wikipedia Foundation",
            "url": "https://en.wikipedia.org/wiki/Mark_Carney",
            "description": "Canadian politician and economist who is the 24th prime minister of Canada and leader of the Liberal Party.",
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

    # 4. False medical claim -> FAKE (Fact check refutation)
    def test_04_false_medical_claim(self):
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
        self.assertGreaterEqual(len(resp["contradicting_evidence"]), 1)

    # 5. False sports announcement -> INSUFFICIENT EVIDENCE (Never REAL)
    def test_05_false_sports_announcement(self):
        content = "Lionel Messi announced his immediate retirement from international football to join the Indian Super League."
        mock_bio = {
            "title": "Wikipedia: Lionel Messi",
            "publisher": "Wikipedia Foundation",
            "url": "https://en.wikipedia.org/wiki/Lionel_Messi",
            "description": "Lionel Andres Messi is an Argentine professional footballer who plays as a forward and captains both Inter Miami and Argentina.",
            "authoritative": True,
            "stance": "neutral"
        }
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_bio], "contradicting": [], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertNotEqual(resp["classification"], "REAL")
        self.assertEqual(resp["classification"], "INSUFFICIENT EVIDENCE")
        self.assertEqual(len(resp["supporting_evidence"]), 0)

    # 6. Mixed true and false article -> FAKE / Contradicted
    def test_06_mixed_true_and_false_article(self):
        content = (
            "NASA launched the James Webb Space Telescope in 2021. "
            "However, an agency confirmed that alien spacecraft secretly docked with the telescope in 2026."
        )
        mock_debunk = {
            "title": "Fact Check: No aliens docked with James Webb Telescope",
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

    # 7. Breaking news without sufficient evidence -> INSUFFICIENT EVIDENCE
    def test_07_breaking_news_without_sufficient_evidence(self):
        content = "A stealth quantum computing lab in Zurich achievedRoom-Temperature Superconductivity using graphene bilayers."
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [], "contradicting": [], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "INSUFFICIENT EVIDENCE")
        self.assertEqual(resp["confidence_score"], 50.0)

    # 8. Search API failure -> INSUFFICIENT EVIDENCE (Never crash, never REAL)
    def test_08_search_api_failure(self):
        content = "Global stock exchanges experienced sudden 15-minute trading halt due to transatlantic cable failure."
        with patch("app.services.verifier.search_claims_evidence", side_effect=Exception("Search API Network Failure")), \
             patch("app.services.verifier.search_claims_fact_checks", side_effect=Exception("Fact Check API Timeout")):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "INSUFFICIENT EVIDENCE")
        self.assertEqual(resp["confidence_score"], 50.0)

    # 9. Groq API failure -> Falls back cleanly to deterministic synthesis
    def test_09_groq_api_failure(self):
        content = "The European Space Agency launched the Euclid space telescope from Cape Canaveral to study dark energy."
        mock_source = {
            "title": "ESA - Euclid space mission launched",
            "publisher": "European Space Agency",
            "url": "https://www.esa.int/Science_Exploration/Space_Science/Euclid",
            "description": "The Euclid space mission was successfully launched to study the geometry of the dark universe.",
            "authoritative": True,
            "stance": "supports"
        }
        with patch("app.services.verifier.synthesize_evidence_with_groq", side_effect=Exception("Groq 500 Internal Error")), \
             patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_source], "contradicting": [], "supporting": [mock_source]}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertEqual(resp["classification"], "REAL")
        self.assertGreaterEqual(resp["confidence_score"], 80.0)

    # 10. Unrelated search results -> INSUFFICIENT EVIDENCE (Never REAL)
    def test_10_unrelated_search_results(self):
        content = "Astronomers announce that Mars will develop a glowing green gravitational ring visible in October."
        mock_unrelated = {
            "title": "Wikipedia: Mars Exploration Rover",
            "publisher": "Wikipedia Foundation",
            "url": "https://en.wikipedia.org/wiki/Mars_Exploration_Rover",
            "description": "NASA's Mars Exploration Rover mission was a robotic space mission involving two rovers, Spirit and Opportunity.",
            "authoritative": True,
            "stance": "neutral"
        }
        with patch("app.services.verifier.search_claims_evidence", return_value={"sources": [mock_unrelated], "contradicting": [], "supporting": []}), \
             patch("app.services.verifier.search_claims_fact_checks", return_value={"reviews": [], "contradicting": [], "supporting": []}):
            req = VerificationRequest(content=content, input_type="text")
            resp = verify_endpoint(req, db=self.db)

        self.assertNotEqual(resp["classification"], "REAL")
        self.assertEqual(resp["classification"], "INSUFFICIENT EVIDENCE")
        self.assertEqual(len(resp["supporting_evidence"]), 0)

if __name__ == "__main__":
    unittest.main()
