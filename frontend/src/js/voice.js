/**
 * TruthLens Voice Studio Module
 * Real-time Speech-to-Text transcription with microphone permissions,
 * waveform animations, and transcript verification.
 */

class TruthLensVoice {
  constructor() {
    this.micBtn = document.getElementById("micToggleBtn");
    this.micStatusLabel = document.getElementById("micStatusLabel");
    this.waveform = document.getElementById("voiceWaveform");
    this.previewBox = document.getElementById("voicePreviewBox");
    this.transcriptArea = document.getElementById("voiceTranscriptArea");
    this.verifyBtn = document.getElementById("btnVerifyVoiceContent");
    this.permissionErrorBox = document.getElementById("voicePermissionError");

    this.recognition = null;
    this.isRecording = false;
    this.finalTranscript = "";

    this.initSpeechEngine();
    this.initEvents();
  }

  initSpeechEngine() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      if (this.permissionErrorBox) {
        this.permissionErrorBox.style.display = "block";
        this.permissionErrorBox.innerHTML = "<strong>Notice:</strong> Your browser does not natively support the Web Speech API. Chrome, Edge, or Safari are recommended for live voice verification.";
      }
      return;
    }

    this.recognition = new SpeechRecognition();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = "en-US";

    this.recognition.onstart = () => {
      this.isRecording = true;
      this.micBtn.classList.add("recording");
      this.waveform.classList.add("active");
      this.micStatusLabel.textContent = "Listening... Speak clearly into your microphone";
      if (this.previewBox) this.previewBox.classList.add("active");
    };

    this.recognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          this.finalTranscript += event.results[i][0].transcript + " ";
        } else {
          interim += event.results[i][0].transcript;
        }
      }
      this.transcriptArea.value = (this.finalTranscript + interim).trim();
    };

    this.recognition.onerror = (event) => {
      console.warn("Speech recognition error:", event.error);
      this.stopRecording();
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        if (this.permissionErrorBox) {
          this.permissionErrorBox.style.display = "block";
          this.permissionErrorBox.innerHTML = "<strong>Microphone Permission Denied:</strong> Please click the camera/microphone icon in your browser address bar and grant access to TruthLens.";
        }
      }
    };

    this.recognition.onend = () => {
      this.stopRecording();
    };
  }

  initEvents() {
    if (this.micBtn) {
      this.micBtn.addEventListener("click", () => {
        if (this.isRecording) {
          this.stopRecording();
        } else {
          this.startRecording();
        }
      });
    }

    if (this.verifyBtn) {
      this.verifyBtn.addEventListener("click", () => {
        const text = this.transcriptArea.value.trim();
        if (!text) {
          alert("Please record or enter a statement to verify.");
          return;
        }
        window.TruthLensApp.executeVerification(text, "voice", null, text);
      });
    }
  }

  startRecording() {
    if (this.permissionErrorBox) this.permissionErrorBox.style.display = "none";
    if (!this.recognition) {
      alert("Speech recognition is not available in this browser environment.");
      return;
    }
    try {
      this.finalTranscript = "";
      this.transcriptArea.value = "";
      this.recognition.start();
    } catch (e) {
      console.error("Speech recognition start failed:", e);
    }
  }

  stopRecording() {
    this.isRecording = false;
    if (this.micBtn) this.micBtn.classList.remove("recording");
    if (this.waveform) this.waveform.classList.remove("active");
    if (this.micStatusLabel) this.micStatusLabel.textContent = "Click microphone to record a news claim";
    if (this.recognition) {
      try { this.recognition.stop(); } catch (e) {}
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.TruthLensVoice = new TruthLensVoice();
});
