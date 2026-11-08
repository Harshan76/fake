/**
 * TruthLens Main Application Controller (Production Vercel & Render Ready)
 * Handles tabs, multi-input verification pipeline, results rendering, and persistent history archive.
 */

import './ocr.js';
import './voice.js';

// Dynamic API Base URL resolution for Vercel -> Render cross-origin communication
function getApiBaseUrl() {
  try {
    if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL) {
      const viteUrl = import.meta.env.VITE_API_BASE_URL.trim().replace(/\/+$/, '');
      if (viteUrl) return viteUrl;
    }
  } catch (e) {}

  if (typeof window !== 'undefined' && window.__TRUTHLENS_CONFIG__?.API_BASE_URL) {
    return window.__TRUTHLENS_CONFIG__.API_BASE_URL.trim().replace(/\/+$/, '');
  }

  return '';
}

const API_BASE = getApiBaseUrl();

class TruthLensApp {
  constructor() {
    this.currentInputType = "text";
    this.apiBase = API_BASE;
    this.initTabs();
    this.initTextTab();
    this.initUrlTab();
    this.initDocTab();
    this.initHistory();
  }

  // ------------------------------------------------------------------------
  // Tabs Navigation
  // ------------------------------------------------------------------------
  initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const panels = document.querySelectorAll(".tab-content-panel");

    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        this.currentInputType = targetTab;

        tabBtns.forEach(b => b.classList.remove("active"));
        panels.forEach(p => p.classList.remove("active"));

        btn.classList.add("active");
        const activePanel = document.getElementById(`panel-${targetTab}`);
        if (activePanel) activePanel.classList.add("active");
      });
    });
  }

  // ------------------------------------------------------------------------
  // Text Input Tab
  // ------------------------------------------------------------------------
  initTextTab() {
    const textArea = document.getElementById("textInputArea");
    const verifyBtn = document.getElementById("btnVerifyText");
    const wordCountSpan = document.getElementById("textWordCount");
    const sampleBtns = document.querySelectorAll(".sample-btn");

    if (textArea && wordCountSpan) {
      textArea.addEventListener("input", () => {
        const words = textArea.value.trim().split(/\s+/).filter(Boolean);
        wordCountSpan.textContent = `${words.length} words`;
      });
    }

    if (sampleBtns) {
      sampleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
          const sampleText = btn.getAttribute("data-sample");
          if (textArea && sampleText) {
            textArea.value = sampleText;
            textArea.dispatchEvent(new Event("input"));
          }
        });
      });
    }

    if (verifyBtn && textArea) {
      verifyBtn.addEventListener("click", () => {
        const text = textArea.value.trim();
        if (!text) {
          alert("Please enter a news article, headline, or claim to verify.");
          return;
        }
        this.executeVerification(text, "text");
      });
    }
  }

  // ------------------------------------------------------------------------
  // URL Input Tab
  // ------------------------------------------------------------------------
  initUrlTab() {
    const urlInput = document.getElementById("urlInputField");
    const fetchBtn = document.getElementById("btnFetchUrl");
    const previewBox = document.getElementById("urlPreviewBox");
    const headlineElem = document.getElementById("urlPreviewTitle");
    const metaElem = document.getElementById("urlPreviewMeta");
    const bodyElem = document.getElementById("urlPreviewBody");
    const verifyBtn = document.getElementById("btnVerifyUrl");
    const errorBox = document.getElementById("urlErrorAlert");

    let currentUrlData = null;

    if (fetchBtn && urlInput) {
      fetchBtn.addEventListener("click", async () => {
        const url = urlInput.value.trim();
        if (!url) {
          alert("Please enter a complete article URL (e.g., https://...)");
          return;
        }

        if (errorBox) errorBox.style.display = "none";
        fetchBtn.disabled = true;
        fetchBtn.innerHTML = `<span>Fetching...</span>`;

        try {
          const res = await fetch(`${this.apiBase}/api/extract-url`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url })
          });
          const data = await res.json();

          if (data.success) {
            currentUrlData = data;
            headlineElem.textContent = data.title || "News Article";
            metaElem.textContent = `Publisher: ${data.publisher || 'Unknown'} | Author: ${data.author || 'Staff'} | Words: ${data.word_count}`;
            bodyElem.value = data.extracted_text;
            previewBox.classList.add("active");
          } else {
            if (errorBox) {
              errorBox.style.display = "block";
              errorBox.textContent = data.error || "Unable to retrieve content from this URL.";
            }
          }
        } catch (err) {
          if (errorBox) {
            errorBox.style.display = "block";
            errorBox.textContent = "Network error connecting to verification backend service.";
          }
        } finally {
          fetchBtn.disabled = false;
          fetchBtn.innerHTML = `<span>Fetch & Analyze</span>`;
        }
      });
    }

    if (verifyBtn && bodyElem) {
      verifyBtn.addEventListener("click", () => {
        const textToVerify = bodyElem.value.trim();
        if (!textToVerify) {
          alert("No article content available to verify.");
          return;
        }
        const originalUrl = currentUrlData ? currentUrlData.url : urlInput.value.trim();
        this.executeVerification(textToVerify, "url", originalUrl, textToVerify);
      });
    }
  }

  // ------------------------------------------------------------------------
  // Document Upload Tab
  // ------------------------------------------------------------------------
  initDocTab() {
    const dropzone = document.getElementById("docDropzone");
    const fileInput = document.getElementById("docFileInput");
    const previewBox = document.getElementById("docPreviewBox");
    const verifyBtn = document.getElementById("btnVerifyDocContent");

    if (dropzone && fileInput) {
      dropzone.addEventListener("click", () => fileInput.click());

      fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) this.uploadDocument(file);
      });

      ["dragenter", "dragover"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
          e.preventDefault();
          dropzone.classList.add("dragover");
        });
      });

      ["dragleave", "drop"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
          e.preventDefault();
          dropzone.classList.remove("dragover");
        });
      });

      dropzone.addEventListener("drop", (e) => {
        const file = e.dataTransfer.files[0];
        if (file) this.uploadDocument(file);
      });
    }

    if (verifyBtn) {
      const docBodyElem = document.getElementById("docExtractedText");
      verifyBtn.addEventListener("click", () => {
        const text = docBodyElem ? docBodyElem.value.trim() : "";
        if (!text) {
          alert("No document text to verify.");
          return;
        }
        this.executeVerification(text, "document", null, text);
      });
    }
  }

  async uploadDocument(file) {
    const errorBox = document.getElementById("docErrorAlert");
    const previewBox = document.getElementById("docPreviewBox");
    const docMetaElem = document.getElementById("docPreviewMeta");
    const docBodyElem = document.getElementById("docExtractedText");

    if (errorBox) errorBox.style.display = "none";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${this.apiBase}/api/extract-doc`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();

      if (data.success) {
        docMetaElem.textContent = `File: ${data.filename} (${data.file_type}) • ${data.word_count} words`;
        docBodyElem.value = data.extracted_text;
        previewBox.classList.add("active");
      } else {
        if (errorBox) {
          errorBox.style.display = "block";
          errorBox.textContent = data.error || "Failed to parse document text.";
        }
      }
    } catch (e) {
      if (errorBox) {
        errorBox.style.display = "block";
        errorBox.textContent = "Network error while uploading document to backend.";
      }
    }
  }

  // ------------------------------------------------------------------------
  // Core Verification Execution
  // ------------------------------------------------------------------------
  async executeVerification(content, inputType = "text", originalUrl = null, extractedText = null) {
    const loadingElem = document.getElementById("verificationLoading");
    const resultsContainer = document.getElementById("resultsDashboard");
    const stepMsg = document.getElementById("loadingStepMessage");

    // Show loading UI
    if (resultsContainer) resultsContainer.classList.remove("active");
    if (loadingElem) {
      loadingElem.classList.add("active");
      loadingElem.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    // Step progression visualizer
    const stepTimer1 = setTimeout(() => {
      if (stepMsg) stepMsg.textContent = "Step 2/4: Decomposing factual assertions and key claims...";
    }, 400);

    const stepTimer2 = setTimeout(() => {
      if (stepMsg) stepMsg.textContent = "Step 3/4: Querying credible fact-checkers & institutional databases...";
    }, 900);

    const stepTimer3 = setTimeout(() => {
      if (stepMsg) stepMsg.textContent = "Step 4/4: Combining ML stylometric model with external evidence...";
    }, 1400);

    try {
      const res = await fetch(`${this.apiBase}/api/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: content,
          input_type: inputType,
          original_url: originalUrl,
          extracted_text: extractedText
        })
      });

      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);
      clearTimeout(stepTimer3);

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Backend returned status ${res.status}`);
      }

      const result = await res.json();
      this.renderVerificationResult(result);
      this.loadHistory(); // refresh history list
    } catch (err) {
      console.error("Verification failed:", err);
      alert(`Verification failed: ${err.message}. Please check connection or try again.`);
    } finally {
      if (loadingElem) loadingElem.classList.remove("active");
    }
  }

  // ------------------------------------------------------------------------
  // Render Result Dashboard
  // ------------------------------------------------------------------------
  renderVerificationResult(result) {
    const resultsContainer = document.getElementById("resultsDashboard");
    if (!resultsContainer) return;

    // 1. Classification Tag
    const tagElem = document.getElementById("resClassificationTag");
    tagElem.className = "classification-tag";
    if (result.classification === "REAL") {
      tagElem.classList.add("tag-real");
      tagElem.innerHTML = `<span>✓</span> REAL`;
    } else if (result.classification === "FAKE") {
      tagElem.classList.add("tag-fake");
      tagElem.innerHTML = `<span>⚠</span> FAKE`;
    } else {
      tagElem.classList.add("tag-unverified");
      tagElem.innerHTML = `<span>?</span> UNABLE TO VERIFY`;
    }

    // 2. Confidence Score Gauge
    const confVal = document.getElementById("resConfidenceVal");
    confVal.textContent = `${result.confidence_score}%`;

    // 3. Sub-meters (ML Stylometric vs External Evidence)
    const mlMeterFill = document.getElementById("resMlMeterFill");
    const mlMeterVal = document.getElementById("resMlMeterVal");
    const mlScore = result.ml_confidence || result.confidence_score;
    if (mlMeterFill && mlMeterVal) {
      mlMeterFill.style.width = `${mlScore}%`;
      mlMeterVal.textContent = `${mlScore}%`;
    }

    const evMeterFill = document.getElementById("resEvMeterFill");
    const evMeterVal = document.getElementById("resEvMeterVal");
    const evScore = result.evidence_score || 50;
    if (evMeterFill && evMeterVal) {
      evMeterFill.style.width = `${evScore}%`;
      evMeterVal.textContent = `${evScore}%`;
    }

    // 4. Summary & Explanation
    const summaryBox = document.getElementById("resSummaryBox");
    summaryBox.textContent = result.summary || "Summary of analyzed claims and factual context.";

    const explanationElem = document.getElementById("resExplanationText");
    explanationElem.textContent = result.explanation;

    // 5. Extracted Claims List
    const claimsContainer = document.getElementById("resClaimsList");
    claimsContainer.innerHTML = "";
    if (result.identified_claims && result.identified_claims.length > 0) {
      result.identified_claims.forEach((claim, idx) => {
        const item = document.createElement("div");
        item.className = "claim-item-card";
        item.innerHTML = `
          <div class="claim-index-bullet">${idx + 1}</div>
          <div class="claim-text-content">${this.escapeHtml(claim)}</div>
        `;
        claimsContainer.appendChild(item);
      });
    } else {
      claimsContainer.innerHTML = `<div style="color: var(--text-muted); font-size: 0.9rem;">No distinct atomic claims identified.</div>`;
    }

    // 6. Mandatory Section: "Explore More – Verify This News Yourself"
    const sourcesGrid = document.getElementById("resSourcesGrid");
    const noSourcesMsg = document.getElementById("resNoSourcesAlert");
    sourcesGrid.innerHTML = "";

    if (result.sources && result.sources.length > 0) {
      noSourcesMsg.style.display = "none";
      result.sources.forEach(src => {
        const card = document.createElement("a");
        card.className = "source-hyperlink-card";
        card.href = src.url;
        card.target = "_blank";
        card.rel = "noopener noreferrer";
        card.innerHTML = `
          <div class="source-pub-tag">
            <span>●</span> ${this.escapeHtml(src.publisher || 'Verified Source')}
          </div>
          <h4 class="source-card-title">${this.escapeHtml(src.title)}</h4>
          <p class="source-card-desc">${this.escapeHtml(src.description || 'Verified evidence documentation and news reporting.')}</p>
          <div class="source-card-footer">
            <span>Open Original Report ↗</span>
          </div>
        `;
        sourcesGrid.appendChild(card);
      });
    } else {
      noSourcesMsg.style.display = "block";
      noSourcesMsg.textContent = "No verified external sources were found for this specific query or claim.";
    }

    // 7. Timestamp & ID
    const timeElem = document.getElementById("resTimestamp");
    if (timeElem) {
      const d = result.timestamp ? new Date(result.timestamp) : new Date();
      timeElem.textContent = `Verified: ${d.toLocaleString()} • ID: ${result.id.slice(0, 8)}`;
    }

    // Show result and scroll
    resultsContainer.classList.add("active");
    resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ------------------------------------------------------------------------
  // Verification History & Archive
  // ------------------------------------------------------------------------
  initHistory() {
    this.currentHistoryFilter = "ALL";
    const filterBtns = document.querySelectorAll(".filter-pill-btn");
    filterBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        filterBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        this.currentHistoryFilter = btn.getAttribute("data-filter");
        this.loadHistory();
      });
    });

    const searchInput = document.getElementById("historySearchInput");
    if (searchInput) {
      searchInput.addEventListener("input", () => {
        this.loadHistory(searchInput.value.trim());
      });
    }

    this.loadHistory();
  }

  async loadHistory(searchQuery = "") {
    const container = document.getElementById("historyCardsList");
    if (!container) return;

    try {
      let endpoint = `/api/history?filter=${encodeURIComponent(this.currentHistoryFilter)}`;
      if (searchQuery) endpoint += `&search=${encodeURIComponent(searchQuery)}`;

      const res = await fetch(`${this.apiBase}${endpoint}`);
      const records = await res.json();

      container.innerHTML = "";
      if (!records || records.length === 0) {
        container.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">No verification records found.</div>`;
        return;
      }

      records.forEach(rec => {
        const row = document.createElement("div");
        row.className = "history-item-row";

        let badgeClass = "tag-real";
        if (rec.classification === "FAKE") badgeClass = "tag-fake";
        if (rec.classification === "UNABLE TO VERIFY") badgeClass = "tag-unverified";

        const dateStr = rec.timestamp ? new Date(rec.timestamp).toLocaleDateString() : "Recent";
        const contentSnippet = rec.submitted_content.slice(0, 110) + (rec.submitted_content.length > 110 ? "..." : "");

        row.innerHTML = `
          <div class="history-text-col">
            <h4 class="history-headline">${this.escapeHtml(contentSnippet)}</h4>
            <div class="history-meta-sub">
              <span>Input: <strong>${rec.input_type.toUpperCase()}</strong></span>
              <span>•</span>
              <span>Date: ${dateStr}</span>
              <span>•</span>
              <span>Confidence: <strong>${rec.confidence_score}%</strong></span>
            </div>
          </div>
          <div class="history-action-col">
            <span class="classification-tag ${badgeClass}" style="font-size: 0.85rem; padding: 0.35rem 0.85rem;">
              ${rec.classification}
            </span>
          </div>
        `;

        // Click row to view full report
        row.addEventListener("click", () => {
          this.renderVerificationResult(rec);
        });

        container.appendChild(row);
      });
    } catch (e) {
      console.warn("Error loading history from backend:", e);
    }
  }

  escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#039;");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.TruthLensApp = new TruthLensApp();
});
