let selectedFile = null;
let lastPrediction = null;
let originalReport = "";
let simplifiedReport = "";
let gradcamLoaded = false;

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    selectedFile = file;
    gradcamLoaded = false;
    document.getElementById("gradcam-section").classList.add("hidden");

    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById("image-preview").src = e.target.result;
        document.getElementById("upload-prompt").classList.add("hidden");
        document.getElementById("preview-container").classList.remove("hidden");
        document.getElementById("file-name").innerText = file.name;
        document.getElementById("analyze-btn").disabled = false;
    };
    reader.readAsDataURL(file);
}

async function analyzeScan() {
    if (!selectedFile) return;
    const btn = document.getElementById("analyze-btn");
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner animate-spin"></i> <span>Analyzing...</span>';

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const res = await fetch("/predict", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Analysis failed");
        lastPrediction = data;
        displayResults(data);
        document.getElementById("gradcam-section").classList.remove("hidden");
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-microscope"></i> <span>Run AI Diagnosis</span>';
    }
}

function displayResults(data) {
    document.getElementById("results-placeholder").classList.add("hidden");
    document.getElementById("results-content").classList.remove("hidden");
    document.getElementById("diagnosis-text").innerText = data.diagnosis;
    document.getElementById("confidence-text").innerText = (data.confidence_score * 100).toFixed(1) + "%";
    document.getElementById("low-confidence-warning").classList.toggle("hidden", !data.low_confidence_flag);
    document.getElementById("report-section").classList.add("hidden");
    document.getElementById("chat-thread").innerHTML = "";

    const barsContainer = document.getElementById("probability-bars");
    barsContainer.innerHTML = "";
    Object.entries(data.class_probabilities).forEach(([cls, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const isTop = cls === data.diagnosis;
        const row = document.createElement("div");
        row.className = "space-y-1";
        row.innerHTML = `
            <div class="flex justify-between text-xs font-medium">
                <span class="${isTop ? "text-indigo-300 font-bold" : "text-slate-400"} uppercase">${cls}</span>
                <span class="${isTop ? "text-emerald-400 font-bold" : "text-slate-400"}">${pct}%</span>
            </div>
            <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-700/50">
                <div class="h-full rounded-full ${isTop ? "bg-indigo-500" : "bg-slate-700"}" style="width: ${pct}%"></div>
            </div>`;
        barsContainer.appendChild(row);
    });
}

async function toggleGradcam() {
    const img = document.getElementById("gradcam-image");
    const btn = document.getElementById("gradcam-toggle-btn");

    if (gradcamLoaded) {
        img.classList.toggle("hidden");
        btn.innerHTML = img.classList.contains("hidden")
            ? '<i class="fa-solid fa-fire text-orange-400"></i> Show Heatmap'
            : '<i class="fa-solid fa-image text-orange-400"></i> Hide Heatmap';
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner animate-spin"></i> Loading...';
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const res = await fetch("/gradcam", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Heatmap generation failed");
        img.src = "data:image/png;base64," + data.heatmap_base64;
        img.classList.remove("hidden");
        gradcamLoaded = true;
        btn.innerHTML = '<i class="fa-solid fa-image text-orange-400"></i> Hide Heatmap';
    } catch (err) {
        alert("Error: " + err.message);
        btn.innerHTML = '<i class="fa-solid fa-fire text-orange-400"></i> Show Heatmap';
    } finally {
        btn.disabled = false;
    }
}

async function generateReport() {
    if (!lastPrediction) return;
    const btn = document.getElementById("report-btn");
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner animate-spin"></i> Generating...';

    try {
        const res = await fetch("/generate-report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                diagnosis: lastPrediction.diagnosis,
                confidence_score: lastPrediction.confidence_score,
                class_probabilities: lastPrediction.class_probabilities,
            }),
        });
        const data = await res.json();
        originalReport = data.report;
        simplifiedReport = "";
        document.getElementById("simplify-toggle").checked = false;
        document.getElementById("report-text").innerText = originalReport;
        document.getElementById("report-section").classList.remove("hidden");
    } catch (err) {
        alert("Report generation failed: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-file-waveform"></i> Generate Clinical Report';
    }
}

async function toggleSimplify() {
    const checked = document.getElementById("simplify-toggle").checked;
    const reportText = document.getElementById("report-text");

    if (!checked) {
        reportText.innerText = originalReport;
        return;
    }
    if (simplifiedReport) {
        reportText.innerText = simplifiedReport;
        return;
    }
    reportText.innerText = "Simplifying...";
    try {
        const res = await fetch("/simplify-report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ report: originalReport }),
        });
        const data = await res.json();
        simplifiedReport = data.simplified_report;
        reportText.innerText = simplifiedReport;
    } catch {
        reportText.innerText = originalReport;
    }
}

async function askQuestion() {
    const input = document.getElementById("chat-input");
    const question = input.value.trim();
    if (!question) return;
    input.value = "";

    const thread = document.getElementById("chat-thread");
    thread.innerHTML += `<p class="text-xs text-indigo-300"><strong>You:</strong> ${question}</p>`;

    try {
        const res = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });
        const data = await res.json();
        thread.innerHTML += `<p class="text-xs text-slate-300"><strong>NeuroScan:</strong> ${data.answer}</p>`;
    } catch {
        thread.innerHTML += `<p class="text-xs text-red-400">Unable to answer right now.</p>`;
    }
    thread.scrollTop = thread.scrollHeight;
}