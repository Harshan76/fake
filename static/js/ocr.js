/**
 * TruthLens OCR Module
 * Extracts text from images and screenshots (WhatsApp, Instagram, Twitter/X, Newsprint)
 * using client-side Tesseract.js with editable confirmation before verification.
 */

class TruthLensOCR {
  constructor() {
    this.dropzone = document.getElementById("imageDropzone");
    this.fileInput = document.getElementById("imageFileInput");
    this.previewContainer = document.getElementById("imagePreviewContainer");
    this.imageElement = document.getElementById("ocrPreviewImg");
    this.extractedTextArea = document.getElementById("imageExtractedText");
    this.progressBar = document.getElementById("ocrProgressBar");
    this.progressStatus = document.getElementById("ocrProgressStatus");
    this.statusWrapper = document.getElementById("ocrStatusWrapper");
    this.verifyBtn = document.getElementById("btnVerifyImageContent");
    
    this.currentExtractedText = "";
    this.initEvents();
  }

  initEvents() {
    if (!this.dropzone || !this.fileInput) return;

    // Click to browse
    this.dropzone.addEventListener("click", () => this.fileInput.click());

    // File selected
    this.fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) this.processImageFile(file);
    });

    // Drag & Drop
    ["dragenter", "dragover"].forEach(evt => {
      this.dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(evt => {
      this.dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.classList.remove("dragover");
      });
    });

    this.dropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      const file = dt.files[0];
      if (file && file.type.startsWith("image/")) {
        this.processImageFile(file);
      } else {
        alert("Please upload a valid image file (PNG, JPG, JPEG, WEBP).");
      }
    });

    // Verify button
    if (this.verifyBtn) {
      this.verifyBtn.addEventListener("click", (e) => {
        if (e) e.preventDefault();
        const textToVerify = this.extractedTextArea ? this.extractedTextArea.value.trim() : "";
        const words = textToVerify.split(/\s+/).filter(Boolean);
        if (!textToVerify || textToVerify.length < 20 || words.length < 4) {
          alert("The screenshot text is incomplete or unreadable. Please type or paste the headline and article text in the box above to verify.");
          if (this.extractedTextArea) this.extractedTextArea.focus();
          return;
        }
        if (textToVerify.startsWith("Could not extract") || textToVerify.startsWith("No clear readable text")) {
          alert("Please type or paste the actual news text from the screenshot before verifying.");
          if (this.extractedTextArea) this.extractedTextArea.focus();
          return;
        }
        if (window.TruthLensApp && typeof window.TruthLensApp.executeVerification === "function") {
          window.TruthLensApp.executeVerification(textToVerify, "image", null, textToVerify);
        } else {
          alert("TruthLens verification engine is initializing. Please try again in a moment.");
        }
      });
    }
  }

  cleanOcrText(rawText) {
    if (!rawText) return "";
    const lines = rawText.split(/\r?\n/);
    const filtered = [];
    const noisePatterns = [
      /^(?:\d{1,2}:\d{2}(?:\s*[ap]m)?|\d{1,3}%|5g|4g|lte|wi-?fi)$/i,
      /^(?:like|comment|share|follow|following|sponsored|subscribe|saved?|reply)$/i,
      /^(?:swipe (?:left|up|down)|tap to view|view insights|promoted)$/i,
      /^[|~«»©™•—_\-+=]{1,3}$/
    ];
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      if (noisePatterns.some(p => p.test(trimmed))) continue;
      filtered.push(trimmed);
    }
    return filtered.join("\n").trim();
  }

  processImageFile(file) {
    // Show image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      this.imageElement.src = e.target.result;
      this.previewContainer.classList.add("active");
      this.performOCR(e.target.result, file.name);
    };
    reader.readAsDataURL(file);
  }

  async performOCR(imageDataUrl, filename) {
    this.statusWrapper.style.display = "block";
    this.progressStatus.textContent = "Initializing OCR optical engine...";
    this.progressBar.style.width = "15%";
    this.extractedTextArea.value = "Reading and analyzing text in screenshot...";

    // Ensure Tesseract library is loaded
    if (typeof Tesseract === "undefined") {
      this.progressStatus.textContent = "Loading optical scanner library...";
      try {
        await new Promise((resolve, reject) => {
          if (typeof Tesseract !== "undefined") return resolve();
          const s = document.createElement("script");
          s.src = "https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js";
          s.onload = () => resolve();
          s.onerror = () => reject(new Error("CDN unavailable"));
          document.head.appendChild(s);
          setTimeout(() => reject(new Error("Timeout loading OCR engine")), 6000);
        });
      } catch (loadErr) {
        console.warn("Client Tesseract engine unavailable, utilizing optical parser fallback:", loadErr);
        this.fallbackExtraction(filename);
        return;
      }
    }

    if (typeof Tesseract !== "undefined") {
      try {
        const worker = await Tesseract.createWorker("eng");
        
        this.progressStatus.textContent = "Scanning screenshot characters and claims...";
        this.progressBar.style.width = "50%";
        
        const ret = await worker.recognize(imageDataUrl);
        const rawExtracted = ret.data.text.trim();
        await worker.terminate();

        const cleaned = this.cleanOcrText(rawExtracted);
        const words = cleaned.split(/\s+/).filter(Boolean);

        this.progressBar.style.width = "100%";
        
        if (cleaned && cleaned.length >= 20 && words.length >= 4) {
          this.progressStatus.textContent = "OCR extraction completed successfully! Please review the text below.";
          this.currentExtractedText = cleaned;
          this.extractedTextArea.value = cleaned;
        } else {
          this.progressStatus.textContent = "Screenshot text appears incomplete or blurry. Please edit or complete the text below.";
          this.extractedTextArea.value = cleaned;
          this.extractedTextArea.placeholder = "Could not clearly detect news text. Please type or paste the headline and article text here...";
        }
      } catch (err) {
        console.warn("Client Tesseract error, using fallback parser:", err);
        this.fallbackExtraction(filename);
      }
    } else {
      this.fallbackExtraction(filename);
    }
  }

  fallbackExtraction(filename) {
    this.progressBar.style.width = "100%";
    this.progressStatus.textContent = "OCR scanner unavailable. Please type or paste the text from your screenshot below.";
    if (!this.extractedTextArea.value || this.extractedTextArea.value.includes("Reading and analyzing")) {
      this.extractedTextArea.value = "";
      this.extractedTextArea.placeholder = "Optical scanner could not read this image automatically. Please type or paste the headline and article text here...";
    }
  }
}

function initTruthLensOCR() {
  if (!window.TruthLensOCR) {
    window.TruthLensOCR = new TruthLensOCR();
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initTruthLensOCR);
} else {
  initTruthLensOCR();
}
