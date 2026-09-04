/**
 * NeuroScan AI - Enterprise Clinical Client Controller
 * Handles acquisition, neural inference, interactive Grad-CAM vision studio,
 * split comparison sliders, report synthesis, and clinical copilot.
 */

let selectedFile = null;
let currentPrediction = null;
let currentRawImageUrl = null;
let currentHeatmapOverlayUrl = null;
let currentPureHeatmapUrl = null;
let currentColorMap = "turbo";
let currentOpacity = 0.55;
let currentViewMode = "split";
let clinicalReport = "";
let patientReport = "";
let currentReportType = "clinical";

// ==========================================
// 1. Scan Acquisition & File Handling
// ==========================================

function handleFileSelect(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    loadFileInput(file);
    event.target.value = ""; // Reset so selecting the same file triggers change
}

function loadFileInput(file) {
    if (!file) return;
    selectedFile = file;

    // Instant zero-latency preview via ObjectURL, with FileReader fallback
    try {
        const objectUrl = URL.createObjectURL(file);
        setRawImage(objectUrl, file.name || "mri_scan.jpg");
    } catch (err) {
        const reader = new FileReader();
        reader.onload = (e) => {
            setRawImage(e.target.result, file.name || "mri_scan.jpg");
        };
        reader.readAsDataURL(file);
    }
}

function setRawImage(dataUrl, fileName = "scan.jpg") {
    currentRawImageUrl = dataUrl;
    
    // 1. Hide Upload Prompt view & Show Image Preview
    const uploadPrompt = document.getElementById("upload-prompt");
    if (uploadPrompt) {
        uploadPrompt.classList.add("hidden");
        uploadPrompt.style.display = "none";
    }

    const previewContainer = document.getElementById("preview-container");
    if (previewContainer) {
        previewContainer.classList.remove("hidden");
        previewContainer.style.display = "block";
    }

    const imgPreview = document.getElementById("image-preview");
    if (imgPreview) {
        imgPreview.src = dataUrl;
        imgPreview.style.display = "block";
    }

    // 2. Show Active File Info Badge
    const fileBadge = document.getElementById("file-info-badge");
    if (fileBadge) {
        fileBadge.classList.remove("hidden");
        fileBadge.style.display = "flex";
    }
    const fileNameEl = document.getElementById("file-name");
    if (fileNameEl) fileNameEl.innerText = fileName;

    // 3. Enable Diagnostic Button
    const analyzeBtn = document.getElementById("analyze-btn");
    if (analyzeBtn) analyzeBtn.disabled = false;

    // 4. Update Studio Views
    const placeholder = document.getElementById("canvas-placeholder");
    if (placeholder) {
        placeholder.classList.add("hidden");
        placeholder.style.display = "none";
    }

    const splitRaw = document.getElementById("split-img-raw");
    if (splitRaw) splitRaw.src = dataUrl;
    const overlayRaw = document.getElementById("overlay-img-raw");
    if (overlayRaw) overlayRaw.src = dataUrl;
    const sideRaw = document.getElementById("side-img-raw");
    if (sideRaw) sideRaw.src = dataUrl;

    // Default split view bottom layer shows raw scan until heatmap is generated
    if (!currentHeatmapOverlayUrl) {
        const splitHm = document.getElementById("split-img-heatmap");
        if (splitHm) splitHm.src = dataUrl;
    }

    // 5. Display Active Vision View
    updateViewModeDisplay();
}

function resetAcquisition() {
    selectedFile = null;
    currentPrediction = null;
    currentRawImageUrl = null;
    currentHeatmapOverlayUrl = null;
    currentPureHeatmapUrl = null;
    clinicalReport = "";
    patientReport = "";

    if (window.speechSynthesis && (window.speechSynthesis.speaking || isSpeakingReport)) {
        window.speechSynthesis.cancel();
        isSpeakingReport = false;
        const speakIcon = document.getElementById("speak-icon");
        if (speakIcon) {
            speakIcon.className = "fa-solid fa-volume-high";
            speakIcon.classList.remove("text-cyan-400", "animate-pulse");
        }
    }

    const mriInput = document.getElementById("mri-input");
    if (mriInput) mriInput.value = "";

    const uploadPrompt = document.getElementById("upload-prompt");
    if (uploadPrompt) {
        uploadPrompt.classList.remove("hidden");
        uploadPrompt.style.display = "block";
    }

    const previewContainer = document.getElementById("preview-container");
    if (previewContainer) {
        previewContainer.classList.add("hidden");
        previewContainer.style.display = "none";
    }

    const imgPreview = document.getElementById("image-preview");
    if (imgPreview) {
        imgPreview.src = "";
        imgPreview.style.display = "none";
    }

    const fileBadge = document.getElementById("file-info-badge");
    if (fileBadge) {
        fileBadge.classList.add("hidden");
        fileBadge.style.display = "none";
    }

    const analyzeBtn = document.getElementById("analyze-btn");
    if (analyzeBtn) analyzeBtn.disabled = true;

    const placeholder = document.getElementById("canvas-placeholder");
    if (placeholder) {
        placeholder.classList.remove("hidden");
        placeholder.style.display = "block";
    }

    // Hide views
    document.getElementById("view-container-split").classList.add("hidden");
    document.getElementById("view-container-overlay").classList.add("hidden");
    document.getElementById("view-container-side").classList.add("hidden");
    document.getElementById("view-container-pure").classList.add("hidden");

    // Reset results
    document.getElementById("results-placeholder").classList.remove("hidden");
    document.getElementById("results-content").classList.add("hidden");
    const statusPill = document.getElementById("status-pill");
    if (statusPill) {
        statusPill.innerText = "AWAITING SCAN";
        statusPill.className = "text-[10px] font-mono px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700";
    }
}

// 1-Click Sample Scan Loader
async function loadSampleScan(sampleType, btnElement = null) {
    let originalText = "";
    if (btnElement) {
        originalText = btnElement.innerHTML;
        btnElement.innerHTML = '<i class="fa-solid fa-spinner animate-spin"></i> Loading...';
        btnElement.disabled = true;
    }
    try {
        const res = await fetch(`/samples/${sampleType}.jpg`);
        if (!res.ok) throw new Error("Could not load sample scan");
        const blob = await res.blob();
        const file = new File([blob], `${sampleType}_mri_sample.jpg`, { type: "image/jpeg" });
        loadFileInput(file);
        
        // Auto-run diagnosis for quick seamless demo
        setTimeout(() => {
            analyzeScan();
        }, 150);
    } catch (err) {
        alert("Failed to load sample: " + err.message);
    } finally {
        if (btnElement) {
            btnElement.innerHTML = originalText;
            btnElement.disabled = false;
        }
    }
}

// Drag and Drop & Event Listeners
document.addEventListener("DOMContentLoaded", () => {
    const dropArea = document.getElementById("drop-area");
    const fileInput = document.getElementById("mri-input");
    const removeBtn = document.getElementById("btn-remove-scan");

    if (fileInput) {
        fileInput.addEventListener("change", handleFileSelect);
    }

    if (dropArea && fileInput) {
        dropArea.addEventListener("click", (e) => {
            if (e.target.closest("#btn-remove-scan")) return;
            fileInput.click();
        });

        // Prevent browser opening dropped files outside drop area
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
            window.addEventListener(evt, (e) => {
                if (e.target !== dropArea && !dropArea.contains(e.target)) {
                    e.preventDefault();
                }
            });
        });

        let dragCounter = 0;

        dropArea.addEventListener('dragenter', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dragCounter++;
            dropArea.classList.add("border-cyan-400", "bg-cyan-950/40");
        });

        dropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
        });

        dropArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dragCounter--;
            if (dragCounter <= 0) {
                dragCounter = 0;
                dropArea.classList.remove("border-cyan-400", "bg-cyan-950/40");
            }
        });

        dropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dragCounter = 0;
            dropArea.classList.remove("border-cyan-400", "bg-cyan-950/40");
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                loadFileInput(dt.files[0]);
            }
        });
    }

    if (removeBtn) {
        removeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            resetAcquisition();
        });
    }

    // Keep split slider images in sync on resize
    window.addEventListener("resize", syncSplitImageDimensions);
});

// ==========================================
// 2. Neural Diagnostic Inference
// ==========================================

async function analyzeScan() {
    if (!selectedFile) {
        alert("Please acquire or upload an MRI scan first.");
        return;
    }

    const btn = document.getElementById("analyze-btn");
    const btnText = document.getElementById("analyze-btn-text");
    btn.disabled = true;
    btnText.innerText = "Analyzing Neural Features...";
    btn.classList.add("animate-pulse");

    const statusPill = document.getElementById("status-pill");
    statusPill.innerText = "INFERENCE IN PROGRESS";
    statusPill.className = "text-[10px] font-mono px-2.5 py-1 rounded-full bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 animate-pulse";

    const formData = new FormData();
    formData.append("file", selectedFile, selectedFile.name || "scan.jpg");

    try {
        const res = await fetch("/predict?include_heatmap=true", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || ("Diagnosis failed with status " + res.status));

        currentPrediction = data;
        renderDiagnosticResults(data);

        // Load Grad-CAM heatmap into studio
        if (data.heatmap_base64) {
            currentHeatmapOverlayUrl = "data:image/png;base64," + data.heatmap_base64;
            currentPureHeatmapUrl = data.pure_heatmap_base64 ? ("data:image/png;base64," + data.pure_heatmap_base64) : currentHeatmapOverlayUrl;
            setHeatmapImages(currentHeatmapOverlayUrl, currentPureHeatmapUrl);
        }

        statusPill.innerText = "ANALYSIS COMPLETE";
        statusPill.className = "text-[10px] font-mono px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";

    } catch (err) {
        console.error("Clinical diagnostic error:", err);
        alert("Clinical diagnostic error: " + err.message);
        statusPill.innerText = "INFERENCE FAILED";
        statusPill.className = "text-[10px] font-mono px-2.5 py-1 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/40";
    } finally {
        btn.disabled = false;
        btnText.innerText = "Execute Neural Diagnostic";
        btn.classList.remove("animate-pulse");
    }
}

function renderDiagnosticResults(data) {
    document.getElementById("results-placeholder").classList.add("hidden");
    document.getElementById("results-content").classList.remove("hidden");

    // Primary Hero Card
    const diagnosisEl = document.getElementById("diagnosis-text");
    const confidenceEl = document.getElementById("confidence-text");
    const whoBadge = document.getElementById("who-grade-badge");
    const riskBadge = document.getElementById("risk-level-badge");
    const heroBorder = document.getElementById("diagnosis-hero-border");

    const meta = data.tumor_info || {};
    diagnosisEl.innerText = meta.title || data.diagnosis;
    confidenceEl.innerText = (data.confidence_score * 100).toFixed(1) + "%";
    whoBadge.innerText = meta.who_grade || "WHO Unclassified";
    riskBadge.innerText = meta.risk_level || "Clinical Review";

    // Dynamic Risk Palette Styling
    const badgeColor = meta.badge_color || "slate";
    if (badgeColor === "rose") {
        riskBadge.className = "px-2.5 py-1 rounded-lg text-xs font-mono font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/40";
        heroBorder.className = "absolute inset-x-0 top-0 h-[3px] bg-gradient-to-r from-rose-500 via-orange-500 to-amber-500";
    } else if (badgeColor === "amber") {
        riskBadge.className = "px-2.5 py-1 rounded-lg text-xs font-mono font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/40";
        heroBorder.className = "absolute inset-x-0 top-0 h-[3px] bg-gradient-to-r from-amber-500 via-yellow-400 to-emerald-500";
    } else if (badgeColor === "indigo") {
        riskBadge.className = "px-2.5 py-1 rounded-lg text-xs font-mono font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40";
        heroBorder.className = "absolute inset-x-0 top-0 h-[3px] bg-gradient-to-r from-indigo-500 via-cyan-400 to-purple-500";
    } else {
        riskBadge.className = "px-2.5 py-1 rounded-lg text-xs font-mono font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
        heroBorder.className = "absolute inset-x-0 top-0 h-[3px] bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-500";
    }

    // Warnings & Differentials
    const lowConfWarning = document.getElementById("low-confidence-warning");
    lowConfWarning.classList.toggle("hidden", !data.low_confidence_flag);

    const diffAlert = document.getElementById("differential-alert");
    const diffText = document.getElementById("differential-text");
    if (data.differential && data.differential.is_close_call) {
        diffAlert.classList.remove("hidden");
        diffText.innerHTML = `<strong>Differential Alert:</strong> Secondary class <span class="uppercase font-bold">${data.differential.secondary_class}</span> (${(data.differential.secondary_probability*100).toFixed(1)}%) is within ${ (data.differential.probability_gap*100).toFixed(1) }% margin. Consider secondary confirmation.`;
    } else {
        diffAlert.classList.add("hidden");
    }

    // 4-Class Probability Spectrum Bars
    const barsContainer = document.getElementById("probability-bars");
    barsContainer.innerHTML = "";
    
    // Order: glioma, meningioma, pituitary, notumor
    const sortedEntries = Object.entries(data.class_probabilities).sort((a, b) => b[1] - a[1]);
    
    sortedEntries.forEach(([cls, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const isTop = cls === data.diagnosis;
        
        let barColor = "bg-slate-700";
        let textColor = "text-slate-400";
        if (isTop) {
            if (cls === "glioma") barColor = "bg-gradient-to-r from-rose-500 to-orange-500";
            else if (cls === "meningioma") barColor = "bg-gradient-to-r from-amber-500 to-yellow-400";
            else if (cls === "pituitary") barColor = "bg-gradient-to-r from-indigo-500 to-cyan-400";
            else barColor = "bg-gradient-to-r from-emerald-500 to-teal-400";
            textColor = "text-white font-bold";
        }

        const row = document.createElement("div");
        row.className = "space-y-1";
        row.innerHTML = `
            <div class="flex justify-between items-center text-xs">
                <span class="${textColor} uppercase tracking-wider flex items-center gap-1.5">
                    ${isTop ? '<i class="fa-solid fa-caret-right text-cyan-400"></i>' : ''}
                    <span>${cls}</span>
                </span>
                <span class="font-mono ${isTop ? 'text-cyan-400 font-bold' : 'text-slate-400'}">${pct}%</span>
            </div>
            <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                <div class="h-full rounded-full transition-all duration-700 ${barColor}" style="width: ${pct}%"></div>
            </div>`;
        barsContainer.appendChild(row);
    });

    // Reset report box for new scan
    document.getElementById("report-container").classList.add("hidden");
    document.getElementById("report-mode-toggle").classList.add("hidden");
    clinicalReport = "";
    patientReport = "";
}

// ==========================================
// 3. Explainability Vision Studio & Slider
// ==========================================

function setHeatmapImages(overlayUrl, pureUrl = null) {
    if (!pureUrl) pureUrl = overlayUrl;
    currentHeatmapOverlayUrl = overlayUrl;
    currentPureHeatmapUrl = pureUrl;

    const splitHm = document.getElementById("split-img-heatmap");
    if (splitHm) splitHm.src = overlayUrl;

    const overlayPure = document.getElementById("overlay-img-pure");
    if (overlayPure) overlayPure.src = pureUrl;

    const sideOverlay = document.getElementById("side-img-overlay");
    if (sideOverlay) sideOverlay.src = overlayUrl;

    const pureView = document.getElementById("pure-img-view");
    if (pureView) pureView.src = pureUrl;

    updateViewModeDisplay();
}

function syncSplitImageDimensions() {
    const wrapper = document.getElementById("view-container-split");
    const rawImg = document.getElementById("split-img-raw");
    if (wrapper && rawImg && wrapper.clientWidth > 0) {
        rawImg.style.width = wrapper.clientWidth + "px";
        rawImg.style.height = wrapper.clientHeight + "px";
    }
}

function setViewMode(mode) {
    currentViewMode = mode;
    ["split", "overlay", "side", "pure"].forEach(m => {
        const tab = document.getElementById(`tab-${m}`);
        if (!tab) return;
        if (m === mode) {
            tab.className = "px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-semibold transition";
        } else {
            tab.className = "px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition";
        }
    });
    updateViewModeDisplay();
}

function updateViewModeDisplay() {
    if (!currentRawImageUrl) return;

    document.getElementById("view-container-split").classList.toggle("hidden", currentViewMode !== "split");
    document.getElementById("view-container-overlay").classList.toggle("hidden", currentViewMode !== "overlay");
    document.getElementById("view-container-side").classList.toggle("hidden", currentViewMode !== "side");
    document.getElementById("view-container-pure").classList.toggle("hidden", currentViewMode !== "pure");

    if (currentViewMode === "split") {
        initSplitSlider();
    }
}

// Interactive Before/After Split Comparison Slider
function initSplitSlider() {
    const wrapper = document.getElementById("view-container-split");
    const handle = document.getElementById("slider-handle");
    const clipWrapper = document.getElementById("split-clip-wrapper");
    if (!wrapper || !handle || !clipWrapper) return;

    syncSplitImageDimensions();

    let isDragging = false;

    function setPosition(x) {
        const rect = wrapper.getBoundingClientRect();
        let posX = x - rect.left;
        posX = Math.max(0, Math.min(posX, rect.width));
        const percent = (posX / rect.width) * 100;
        
        handle.style.left = `${percent}%`;
        clipWrapper.style.width = `${percent}%`;
    }

    wrapper.onmousedown = (e) => {
        isDragging = true;
        setPosition(e.clientX);
    };

    window.onmousemove = (e) => {
        if (!isDragging) return;
        setPosition(e.clientX);
    };

    window.onmouseup = () => {
        isDragging = false;
    };

    // Touch support for mobile/tablets
    wrapper.ontouchstart = (e) => {
        isDragging = true;
        setPosition(e.touches[0].clientX);
    };

    window.ontouchmove = (e) => {
        if (!isDragging) return;
        setPosition(e.touches[0].clientX);
    };

    window.ontouchend = () => {
        isDragging = false;
    };
}

// Change Colormap dynamically
async function changeColormap(cmap) {
    currentColorMap = cmap;
    ["turbo", "jet", "inferno", "viridis"].forEach(c => {
        const btn = document.getElementById(`cmap-${c}`);
        if (!btn) return;
        if (c === cmap) {
            btn.className = "py-1.5 rounded-lg bg-indigo-600 text-white font-semibold transition";
        } else {
            btn.className = "py-1.5 rounded-lg text-slate-400 hover:text-white transition";
        }
    });

    if (!selectedFile) return;

    try {
        const formData = new FormData();
        formData.append("file", selectedFile, selectedFile.name || "scan.jpg");
        const res = await fetch(`/gradcam?colormap=${cmap}&alpha=${currentOpacity}`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.heatmap_base64) {
            currentHeatmapOverlayUrl = "data:image/png;base64," + data.heatmap_base64;
            currentPureHeatmapUrl = data.pure_heatmap_base64 ? ("data:image/png;base64," + data.pure_heatmap_base64) : currentHeatmapOverlayUrl;
            setHeatmapImages(currentHeatmapOverlayUrl, currentPureHeatmapUrl);
        }
    } catch (e) {
        console.error("Colormap switch failed:", e);
    }
}

// Live Opacity Slider
function updateOpacity(val) {
    currentOpacity = val / 100;
    document.getElementById("opacity-label").innerText = val + "%";
    
    // In overlay mode, update opacity instantly via CSS
    const overlayPure = document.getElementById("overlay-img-pure");
    if (overlayPure) {
        overlayPure.style.opacity = currentOpacity;
    }
}

// ==========================================
// 4. AI Clinical Report Synthesis (RAG) & Speech
// ==========================================

let isSpeakingReport = false;

/**
 * Lightweight, dependency-free in-place markdown parser.
 * Converts section headers, bold, italics, lists, and disclaimers to semantic HTML.
 * Ensures asterisks never display to the user.
 */
function renderMarkdownToHtml(markdownText) {
    if (!markdownText) return "";

    // 1. Basic HTML sanitization
    let t = markdownText
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // 2. Markdown Headings (#, ##, ###)
    t = t.replace(/^###\s+(.*)$/gm, '<h4 class="text-cyan-300 font-bold text-xs uppercase tracking-wider mt-3 mb-1.5 flex items-center gap-1.5"><i class="fa-solid fa-angle-right text-[10px] text-cyan-400"></i>$1</h4>');
    t = t.replace(/^##\s+(.*)$/gm, '<h3 class="text-cyan-300 font-bold text-xs uppercase tracking-wider mt-3 mb-1.5 flex items-center gap-1.5"><i class="fa-solid fa-angle-right text-[10px] text-cyan-400"></i>$1</h3>');
    t = t.replace(/^#\s+(.*)$/gm, '<h2 class="text-white font-extrabold text-sm uppercase tracking-wider mt-3 mb-2">$1</h2>');

    // 3. Bold standalone section titles (e.g., **Findings:** or **Interpretation**)
    t = t.replace(/^\*\*([^*]+)\*\*$/gm, '<h4 class="text-cyan-300 font-bold text-xs uppercase tracking-wider mt-3 mb-1.5 flex items-center gap-1.5"><i class="fa-solid fa-caret-right text-cyan-400 text-xs"></i>$1</h4>');

    // 4. Inline Bold & Italic Formatting
    t = t.replace(/\*\*([^*]+)\*\*/g, '<strong class="text-white font-semibold">$1</strong>');
    t = t.replace(/__([^_]+)__/g, '<strong class="text-white font-semibold">$1</strong>');
    t = t.replace(/\*([^*\n]+)\*/g, '<em class="text-slate-200 italic">$1</em>');
    t = t.replace(/_([^_\n]+)_/g, '<em class="text-slate-200 italic">$1</em>');

    // 5. Line-by-line processing for lists, paragraphs, and disclaimers
    const lines = t.split("\n");
    let out = [];
    let inUl = false;
    let inOl = false;

    for (let i = 0; i < lines.length; i++) {
        const s = lines[i].trim();
        const bulletMatch = s.match(/^[-*•]\s+(.*)$/);
        const numberMatch = s.match(/^(\d+)\.\s+(.*)$/);

        if (bulletMatch) {
            if (inOl) { out.push("</ol>"); inOl = false; }
            if (!inUl) { out.push('<ul class="space-y-1.5 my-2">'); inUl = true; }
            out.push(`<li class="flex items-start gap-2 text-slate-300 pl-1"><span class="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 shrink-0"></span><span>${bulletMatch[1]}</span></li>`);
        } else if (numberMatch) {
            if (inUl) { out.push("</ul>"); inUl = false; }
            if (!inOl) { out.push('<ol class="space-y-1.5 my-2">'); inOl = true; }
            out.push(`<li class="flex items-start gap-2 text-slate-300 pl-1"><span class="font-mono text-cyan-400 text-xs font-bold mt-0.5 shrink-0">${numberMatch[1]}.</span><span>${numberMatch[2]}</span></li>`);
        } else {
            if (inUl) { out.push("</ul>"); inUl = false; }
            if (inOl) { out.push("</ol>"); inOl = false; }

            if (s.length > 0) {
                if (s.startsWith("<h2") || s.startsWith("<h3") || s.startsWith("<h4")) {
                    out.push(s);
                } else if (s.toLowerCase().includes("screening assistance only") || s.toLowerCase().includes("screening assistance")) {
                    out.push(`<div class="mt-3.5 pt-2 border-t border-slate-800 text-[11px] font-mono text-amber-300/90 flex items-center gap-1.5"><i class="fa-solid fa-shield-halved text-amber-400 shrink-0"></i><span>${s}</span></div>`);
                } else {
                    out.push(`<p class="mb-2 leading-relaxed text-slate-300">${s}</p>`);
                }
            } else {
                out.push('<div class="h-1"></div>');
            }
        }
    }

    if (inUl) out.push("</ul>");
    if (inOl) out.push("</ol>");

    let html = out.join("\n");
    // Strip any remaining stray asterisks
    html = html.replace(/\*/g, "");
    return html;
}

/**
 * Strips all markdown symbols (*, #, _, -, bullets) for smooth Text-to-Speech playback.
 */
function cleanTextForSpeech(text) {
    if (!text) return "";
    return text
        .replace(/#{1,6}\s+/g, "")        // Remove markdown headers
        .replace(/\*\*([^*]+)\*\*/g, "$1") // Remove bold asterisks
        .replace(/\*([^*]+)\*/g, "$1")     // Remove italic asterisks
        .replace(/__([^_]+)__/g, "$1")     // Remove bold underscores
        .replace(/_([^_]+)_/g, "$1")       // Remove italic underscores
        .replace(/^[\s*•-]+\s+/gm, "")     // Remove bullet symbols
        .replace(/^\d+\.\s+/gm, "")        // Remove list numbers
        .replace(/[*_`~>]/g, "")           // Remove remaining markdown characters
        .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // Clean links
        .replace(/\s+/g, " ")              // Normalize spaces
        .trim();
}

async function generateReport() {
    if (!currentPrediction) return;

    const btn = document.getElementById("report-btn");
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner animate-spin mr-2"></i> Synthesizing WHO CNS Grounded Report...';

    try {
        const res = await fetch("/generate-report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                diagnosis: currentPrediction.diagnosis,
                confidence_score: currentPrediction.confidence_score,
                class_probabilities: currentPrediction.class_probabilities,
            }),
        });

        const data = await res.json();
        clinicalReport = data.report;
        patientReport = "";

        document.getElementById("report-container").classList.remove("hidden");
        document.getElementById("report-mode-toggle").classList.remove("hidden");
        setReportMode("clinical");

    } catch (err) {
        alert("Report synthesis error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles mr-2"></i> Regenerate Medical Report';
    }
}

async function setReportMode(mode) {
    currentReportType = mode;
    const clinicalBtn = document.getElementById("mode-clinical-btn");
    const patientBtn = document.getElementById("mode-patient-btn");
    const indicator = document.getElementById("report-type-indicator");
    const reportText = document.getElementById("report-text");

    if (mode === "clinical") {
        clinicalBtn.className = "px-2 py-0.5 rounded bg-emerald-600 text-white font-medium";
        patientBtn.className = "px-2 py-0.5 rounded text-slate-400 hover:text-white";
        indicator.innerText = "Clinical Specialist Report (WHO CNS v5)";
        reportText.innerHTML = renderMarkdownToHtml(clinicalReport);
    } else {
        patientBtn.className = "px-2 py-0.5 rounded bg-emerald-600 text-white font-medium";
        clinicalBtn.className = "px-2 py-0.5 rounded text-slate-400 hover:text-white";
        indicator.innerText = "Patient-Friendly Summary (Plain Language)";

        if (patientReport) {
            reportText.innerHTML = renderMarkdownToHtml(patientReport);
        } else {
            reportText.innerHTML = '<div class="py-4 text-center text-slate-400 italic flex items-center justify-center gap-2"><i class="fa-solid fa-spinner animate-spin text-emerald-400"></i><span>Simplifying clinical terminology into plain patient language...</span></div>';
            try {
                const res = await fetch("/simplify-report", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ report: clinicalReport }),
                });
                const data = await res.json();
                patientReport = data.simplified_report;
                reportText.innerHTML = renderMarkdownToHtml(patientReport);
            } catch {
                reportText.innerHTML = renderMarkdownToHtml(clinicalReport);
            }
        }
    }
}

function copyReport() {
    const rawText = currentReportType === "clinical" ? clinicalReport : (patientReport || clinicalReport);
    const cleanText = cleanTextForSpeech(rawText);
    if (!cleanText) return;

    navigator.clipboard.writeText(cleanText).then(() => {
        const icon = document.getElementById("copy-icon");
        if (icon) {
            icon.className = "fa-solid fa-check text-emerald-400";
            setTimeout(() => {
                icon.className = "fa-regular fa-copy";
            }, 2000);
        }
    });
}

function printReport() {
    if (!currentPrediction) return;
    const rawText = currentReportType === "clinical" ? clinicalReport : (patientReport || clinicalReport);
    const formattedHtml = renderMarkdownToHtml(rawText);
    if (!formattedHtml) return;

    const printWin = window.open("", "_blank");
    printWin.document.write(`
        <html>
        <head>
            <title>NeuroScan AI - Diagnostic Report</title>
            <style>
                body { font-family: 'Segoe UI', Arial, sans-serif; padding: 40px; color: #111; line-height: 1.6; }
                .header { border-bottom: 2px solid #0891b2; padding-bottom: 12px; margin-bottom: 24px; }
                h1 { color: #0891b2; margin: 0; font-size: 24px; }
                .meta { font-size: 13px; color: #555; margin-top: 4px; }
                .card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
                .diagnosis { font-size: 20px; font-weight: bold; color: #1e293b; text-transform: uppercase; }
                .confidence { font-size: 14px; font-weight: bold; color: #0891b2; }
                .report-body { font-size: 13px; line-height: 1.6; background: #fff; padding: 16px; border: 1px solid #cbd5e1; border-radius: 6px; }
                .report-body h4 { font-size: 13px; font-weight: bold; color: #0891b2; margin-top: 14px; margin-bottom: 6px; text-transform: uppercase; }
                .report-body p { margin-bottom: 8px; color: #334155; }
                .report-body ul, .report-body ol { margin: 8px 0; padding-left: 20px; color: #334155; }
                .report-body li { margin-bottom: 4px; }
                .disclaimer { margin-top: 30px; font-size: 11px; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 12px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>NEUROSCAN AI — CLINICAL DECISION SUPPORT REPORT</h1>
                <div class="meta">Date: ${new Date().toLocaleString()} | Classification Model: EfficientNetB0</div>
            </div>
            <div class="card">
                <div class="diagnosis">Primary Finding: ${currentPrediction.diagnosis}</div>
                <div class="confidence">Confidence: ${(currentPrediction.confidence_score*100).toFixed(1)}%</div>
            </div>
            <h3>Structured Diagnostic Summary:</h3>
            <div class="report-body">${formattedHtml}</div>
            <div class="disclaimer">
                DISCLAIMER: Screening assistance only. Final evaluation must be confirmed by a board-certified neuro-radiologist and correlation with full DICOM sequences.
            </div>
            <script>window.print();</script>
        </body>
        </html>
    `);
    printWin.document.close();
}

/**
 * Text-to-Speech handler:
 * - Checks window.speechSynthesis availability
 * - Toggles between speak / stop states
 * - Cleans all markdown symbols before speaking
 * - Cancels previous speech queue immediately before speaking to prevent browser lockups
 */
function speakReport() {
    if (!('speechSynthesis' in window)) {
        alert("Text-to-Speech is not supported by your browser.");
        return;
    }

    const speakIcon = document.getElementById("speak-icon");

    // 1. Toggle: if currently speaking, cancel immediately and reset icon
    if (window.speechSynthesis.speaking || isSpeakingReport) {
        window.speechSynthesis.cancel();
        isSpeakingReport = false;
        if (speakIcon) {
            speakIcon.className = "fa-solid fa-volume-high";
            speakIcon.classList.remove("text-cyan-400", "animate-pulse");
        }
        return;
    }

    // 2. Extract and clean text based on current active view
    const rawReport = currentReportType === "clinical" ? clinicalReport : (patientReport || clinicalReport);
    const speechText = cleanTextForSpeech(rawReport);
    if (!speechText) return;

    // 3. Flush any pending/frozen utterances in browser queue
    window.speechSynthesis.cancel();

    // 4. Create clean speech utterance
    const utterance = new SpeechSynthesisUtterance(speechText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
        isSpeakingReport = true;
        if (speakIcon) {
            speakIcon.className = "fa-solid fa-circle-stop text-cyan-400 animate-pulse";
        }
    };

    utterance.onend = () => {
        isSpeakingReport = false;
        if (speakIcon) {
            speakIcon.className = "fa-solid fa-volume-high";
            speakIcon.classList.remove("text-cyan-400", "animate-pulse");
        }
    };

    utterance.onerror = (e) => {
        console.warn("Speech synthesis playback interrupted or stopped:", e);
        isSpeakingReport = false;
        if (speakIcon) {
            speakIcon.className = "fa-solid fa-volume-high";
            speakIcon.classList.remove("text-cyan-400", "animate-pulse");
        }
    };

    window.speechSynthesis.speak(utterance);
}

// ==========================================
// 5. Decision Support AI Copilot Q&A
// ==========================================

function askPrompt(promptText) {
    document.getElementById("chat-input").value = promptText;
    askQuestion();
}

async function askQuestion() {
    const input = document.getElementById("chat-input");
    const question = input.value.trim();
    if (!question) return;
    input.value = "";

    const thread = document.getElementById("chat-thread");
    thread.innerHTML += `
        <div class="bg-indigo-950/40 border border-indigo-500/20 p-2.5 rounded-xl space-y-1">
            <div class="text-[10px] font-mono text-cyan-400 font-semibold flex items-center gap-1">
                <i class="fa-solid fa-user-doctor"></i> Attending Physician
            </div>
            <p class="text-slate-200">${escapeHtml(question)}</p>
        </div>
    `;
    thread.scrollTop = thread.scrollHeight;

    // Loading bubble
    const loadingId = "loading-" + Date.now();
    thread.innerHTML += `
        <div id="${loadingId}" class="bg-slate-900 border border-slate-800 p-2.5 rounded-xl space-y-1 text-slate-400 italic flex items-center gap-2">
            <i class="fa-solid fa-spinner animate-spin text-cyan-400"></i>
            <span>Consulting WHO CNS reference knowledge base...</span>
        </div>
    `;
    thread.scrollTop = thread.scrollHeight;

    try {
        const res = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        const data = await res.json();
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();

        thread.innerHTML += `
            <div class="bg-slate-900/90 border border-cyan-500/30 p-2.5 rounded-xl space-y-1 shadow-lg">
                <div class="text-[10px] font-mono text-cyan-400 font-semibold flex items-center justify-between">
                    <span class="flex items-center gap-1">
                        <i class="fa-solid fa-brain text-cyan-400"></i> NeuroScan Decision Support
                    </span>
                    <span class="text-slate-500 text-[9px]">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                </div>
                <p class="text-slate-200 leading-relaxed">${escapeHtml(data.answer)}</p>
            </div>
        `;
    } catch {
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();
        thread.innerHTML += `<p class="text-xs text-rose-400">Decision support assistant is currently unavailable. Please verify network or API keys.</p>`;
    }
    thread.scrollTop = thread.scrollHeight;
}

function escapeHtml(text) {
    return text.replace(/[&<>"']/g, function(m) {
        return {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        }[m];
    });
}