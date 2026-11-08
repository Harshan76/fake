import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ml.classifier import predict_text_credibility
from app.services.claim_extractor import extract_key_claims
from app.services.document_parser import parse_document
from app.services.url_scraper import validate_url, is_safe_public_url, scrape_article_from_url
from app.services.verifier import verify_news_content
from app.database import init_db, SessionLocal, check_db_connection
from app.models import VerificationRecord

class TruthLensIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_01_ml_prediction_real_and_fake(self):
        real_text = "The World Health Organization confirmed today that global vaccination coverage has reached 72% according to official surveillance data published in Geneva."
        fake_text = "URGENT ALERT: Drinking boiled garlic water with lemon 100% destroys all viral infections overnight! Big Pharma doctors are desperately trying to ban this secret cure! SHARE BEFORE DELETED!!!"

        res_real = predict_text_credibility(real_text)
        res_fake = predict_text_credibility(fake_text)

        self.assertIn(res_real["predicted_label"], ["REAL", "FAKE"])
        self.assertEqual(res_fake["predicted_label"], "FAKE")
        self.assertGreater(res_fake["fake_probability"], 60.0)
        print("ML Prediction Test Passed.")

    def test_02_claim_extraction(self):
        text = "NASA announced yesterday that the James Webb Telescope discovered water vapor on exoplanet K2-18b. Researchers estimate atmospheric levels at 34%."
        claims = extract_key_claims(text)
        self.assertTrue(len(claims) >= 1)
        print(f"Extracted claims: {claims}")

    def test_03_txt_document_parsing(self):
        sample_txt = b"Breaking news regarding transportation infrastructure upgrades in 2026."
        res = parse_document(sample_txt, "news.txt")
        self.assertTrue(res["success"])
        self.assertEqual(res["file_type"], "TXT")
        self.assertIn("transportation", res["extracted_text"])
        print("Document Parsing Test Passed.")

    def test_04_url_validation(self):
        self.assertTrue(validate_url("https://www.reuters.com/world/"))
        self.assertFalse(validate_url("not-a-valid-url"))
        print("URL Validation Test Passed.")

    def test_05_full_verification_pipeline(self):
        # Test debunked hoax
        fake_news = "URGENT: Boiling garlic and lemon cures 100% of all viral infections in 24 hours. Doctors are hiding this miracle cure!"
        result = verify_news_content(fake_news, input_type="text")

        self.assertEqual(result["classification"], "FAKE")
        self.assertGreater(result["confidence_score"], 60.0)
        self.assertTrue(len(result["sources"]) > 0)
        self.assertTrue(any("who.int" in s["url"] or "myth" in s["url"] or "garlic" in s["title"].lower() for s in result["sources"]))
        print("Verification Pipeline (Fake News) Test Passed.")

        # Test unable to verify short or vague statement
        vague_news = "Someone was seen walking on the sidewalk this morning wearing a blue hat."
        result_vague = verify_news_content(vague_news, input_type="text")
        self.assertIn(result_vague["classification"], ["UNABLE TO VERIFY", "REAL"])
        print("Verification Pipeline (Vague News) Test Passed.")

    def test_06_database_persistence(self):
        db = SessionLocal()
        record = VerificationRecord(
            input_type="text",
            submitted_content="Test verification entry for database verification.",
            classification="REAL",
            confidence_score=82.0,
            explanation="Test explanation for database persistence.",
            status="COMPLETED"
        )
        db.add(record)
        db.commit()
        record_id = record.id

        # Query back
        retrieved = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.classification, "REAL")
        self.assertEqual(retrieved.confidence_score, 82.0)
        db.delete(retrieved)
        db.commit()
        db.close()
        print("Database Persistence Test Passed.")

    def test_07_fastapi_router_endpoints(self):
        from app.routers.verify import verify_endpoint
        from app.routers.history import get_history
        from app.routers.stats import get_platform_statistics
        from app.schemas import VerificationRequest
        from main import serve_home
        
        db = SessionLocal()
        try:
            # 1. Test serve_home
            home_resp = serve_home()
            self.assertEqual(home_resp.status_code, 200)
            self.assertIn("TruthLens", home_resp.body.decode("utf-8"))
            
            # 2. Test get_platform_statistics
            stats_resp = get_platform_statistics(db=db)
            self.assertTrue(stats_resp.total_verifications >= 0)
            
            # 3. Test verify_endpoint with real news
            req = VerificationRequest(
                content="The World Health Organization confirmed today that global vaccination coverage has reached 72% according to official surveillance data published in Geneva.",
                input_type="text"
            )
            verify_resp = verify_endpoint(req, db=db)
            self.assertIn(verify_resp["classification"], ["REAL", "FAKE", "UNABLE TO VERIFY"])
            self.assertTrue(verify_resp["confidence_score"] > 0)
            self.assertTrue(len(verify_resp["sources"]) >= 0)
            
            # 4. Test get_history
            history_resp = get_history(db=db)
            self.assertTrue(len(history_resp) >= 1)
            print("FastAPI Router Endpoints Test Passed.")
        finally:
            db.close()

    def test_08_health_check_endpoint(self):
        from main import health_check
        health_resp = health_check()
        self.assertEqual(health_resp.status_code, 200)
        import json
        data = json.loads(health_resp.body.decode("utf-8"))
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["database"], "connected")
        self.assertIn(data["database_type"], ["postgresql", "sqlite"])
        self.assertTrue(data["model_loaded"])
        print("Production Health Check Test Passed.")

    def test_09_ssrf_protection(self):
        # Block localhost
        safe, _ = is_safe_public_url("http://localhost:8000/admin")
        self.assertFalse(safe)

        # Block 127.0.0.1
        safe, _ = is_safe_public_url("http://127.0.0.1:8000/api")
        self.assertFalse(safe)

        # Block cloud metadata
        safe, _ = is_safe_public_url("http://169.254.169.254/latest/meta-data")
        self.assertFalse(safe)

        # Block private IP
        safe, _ = is_safe_public_url("http://10.0.0.1/secret")
        self.assertFalse(safe)
        print("SSRF Security Protection Test Passed.")

    def test_10_postgresql_psycopg_driver_and_normalization(self):
        import psycopg
        from app.config import normalize_database_url
        from sqlalchemy import create_engine

        # 1. Verify psycopg version 3 is loaded and meets >= 3.2 requirement
        self.assertTrue(hasattr(psycopg, "__version__"))
        major_minor = tuple(int(p) for p in psycopg.__version__.split(".")[:2])
        self.assertGreaterEqual(major_minor, (3, 2))
        self.assertLess(major_minor[0], 4)

        # 2. Test URL normalization across common Render / Cloud variations
        urls = {
            "postgres://user:pass@host:5432/db": "postgresql+psycopg://user:pass@host:5432/db",
            "postgresql://user:pass@host:5432/db": "postgresql+psycopg://user:pass@host:5432/db",
            "postgresql+psycopg2://user:pass@host:5432/db": "postgresql+psycopg://user:pass@host:5432/db",
            "postgresql+psycopg://user:pass@host:5432/db": "postgresql+psycopg://user:pass@host:5432/db",
            "sqlite:///truthlens.db": "sqlite:///truthlens.db",
        }
        for raw, expected in urls.items():
            normalized = normalize_database_url(raw)
            self.assertEqual(normalized, expected, f"Failed normalizing {raw}")

        # 3. Verify SQLAlchemy loads psycopg (v3) dialect without ModuleNotFoundError
        test_engine = create_engine(normalize_database_url("postgres://u:p@localhost:5432/truthlens"))
        self.assertEqual(test_engine.dialect.name, "postgresql")
        self.assertEqual(test_engine.dialect.driver, "psycopg")

        # 4. Verify live database connection probe and non-destructive initialization
        self.assertTrue(check_db_connection(), "Database connectivity check failed")
        self.assertTrue(init_db(), "Database initialization failed")

        # 5. Verify records are preserved
        db = SessionLocal()
        try:
            count = db.query(VerificationRecord).count()
            self.assertGreaterEqual(count, 1, "Existing database records were lost")
        finally:
            db.close()

        print(f"PostgreSQL psycopg Driver & Normalization Test Passed (psycopg v{psycopg.__version__}).")

    def test_11_frontend_scripts_and_verify_button(self):
        # 1. Verify index.html contains the actual Verify News button with type="button"
        index_html_path = Path(__file__).resolve().parent / "templates" / "index.html"
        self.assertTrue(index_html_path.exists())
        with open(index_html_path, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn('id="btnVerifyText"', html)
        self.assertIn('type="button"', html)
        self.assertIn('Verify News', html)
        self.assertIn('id="resEvidenceContainer"', html)
        self.assertIn('id="resContradictingSection"', html)
        self.assertIn('id="resSupportingSection"', html)

        # 2. Verify static JS files are valid classic scripts and DO NOT use import.meta outside a module
        js_files = ["app.js", "ocr.js", "voice.js"]
        for jf in js_files:
            js_path = Path(__file__).resolve().parent / "static" / "js" / jf
            self.assertTrue(js_path.exists(), f"Missing static/js/{jf}")
            with open(js_path, "r", encoding="utf-8") as f:
                code = f.read()
            self.assertNotIn("import.meta", code, f"Forbidden import.meta found in classic script {jf}")
            self.assertNotIn("export ", code, f"Forbidden export found in classic script {jf}")

        print("Frontend Script Parsing & Verify Button Integrity Test Passed.")

if __name__ == "__main__":
    unittest.main()
