/* ==========================================================================
   Alzheimer's PET XAI Dashboard - Interactive Controller (script.js)
   ========================================================================= */

document.addEventListener("DOMContentLoaded", () => {
    // ── Tab Navigation ───────────────────────────────────────────────────
    const navItems = document.querySelectorAll(".nav-item");
    const sections = document.querySelectorAll(".content-section");
    const pageTitle = document.getElementById("page-title");
    const pageSubtitle = document.getElementById("page-subtitle");

    const tabMeta = {
        overview: {
            title: "Project Overview",
            subtitle: "Attention-Guided Deep Learning with Clinically Validated XAI"
        },
        performance: {
            title: "Model Performance",
            subtitle: "Quantitative metrics, training dynamics, and cross-dataset validation"
        },
        xai: {
            title: "Explainable AI (XAI)",
            subtitle: "Post-hoc attribution mapping and clinical region validation"
        },
        architecture: {
            title: "Attention Model Architecture",
            subtitle: "CBAM attention mechanism and intermediate layers"
        },
        playground: {
            title: "Interactive Case Playground",
            subtitle: "Examine actual clinical cases, blend heatmaps, and trace anatomical markers"
        }
    };

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const tabId = item.getAttribute("data-tab");
            
            // Toggle sidebar active states
            navItems.forEach(nav => nav.classList.remove("active"));
            item.classList.add("active");
            
            // Toggle sections
            sections.forEach(sec => sec.classList.remove("active"));
            const targetSec = document.getElementById(`tab-${tabId}`);
            if (targetSec) {
                targetSec.classList.add("active");
            }
            
            // Update Title & Subtitle
            if (tabMeta[tabId]) {
                pageTitle.textContent = tabMeta[tabId].title;
                pageSubtitle.textContent = tabMeta[tabId].subtitle;
            }
        });
    });

    // ── Performance Chart (Chart.js) ───────────────────────────────────
    const ctx = document.getElementById("trainingChart").getContext("2d");
    
    // Hardcoded logs representing the 30 training epochs (Phase 1 & Phase 2)
    const epochs = Array.from({ length: 30 }, (_, i) => i + 1);
    
    const accuracyTrain = [
        42.1, 45.4, 48.9, 52.3, 55.7, 57.9, 60.4, 62.6, 64.4, 66.0, 67.0, 67.6, 68.1, 68.4, 68.8, // Phase 1
        69.8, 71.6, 73.5, 75.0, 76.5, 77.9, 79.0, 80.0, 80.7, 81.2, 81.6, 81.8, 81.8, 81.8, 81.8  // Phase 2
    ];
    
    const accuracyVal = [
        40.5, 43.8, 47.3, 50.3, 53.5, 55.7, 57.7, 59.8, 61.9, 63.3, 64.6, 65.3, 66.0, 66.4, 66.7, // Phase 1
        68.2, 70.0, 71.9, 73.6, 74.9, 76.0, 76.8, 77.2, 77.4, 77.5, 77.6, 77.6, 77.6, 77.6, 77.6  // Phase 2
    ];

    const lossTrain = [
        1.38, 1.29, 1.20, 1.12, 1.05, 0.99, 0.94, 0.89, 0.85, 0.82, 0.79, 0.77, 0.75, 0.74, 0.73, // Phase 1
        0.69, 0.64, 0.58, 0.53, 0.49, 0.45, 0.42, 0.39, 0.37, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30  // Phase 2
    ];

    const lossVal = [
        1.42, 1.33, 1.24, 1.16, 1.09, 1.03, 0.98, 0.93, 0.89, 0.86, 0.83, 0.81, 0.79, 0.78, 0.77, // Phase 1
        0.73, 0.69, 0.64, 0.60, 0.58, 0.56, 0.55, 0.54, 0.54, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53  // Phase 2
    ];

    const chartConfig = {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: accuracyTrain,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Validation Accuracy',
                    data: accuracyVal,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#94a3b8', font: { family: 'Inter' } }
                },
                tooltip: {
                    callbacks: {
                        title: (context) => `Epoch ${context[0].label}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' },
                    title: { display: true, text: 'Epoch', color: '#94a3b8' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' },
                    title: { display: true, text: 'Value', color: '#94a3b8' }
                }
            }
        }
    };

    const trainingChart = new Chart(ctx, chartConfig);

    // Toggle Accuracy / Loss charts
    const btnAcc = document.getElementById("btn-chart-accuracy");
    const btnLoss = document.getElementById("btn-chart-loss");

    btnAcc.addEventListener("click", () => {
        btnAcc.classList.add("active");
        btnLoss.classList.remove("active");
        
        trainingChart.data.datasets[0].label = "Training Accuracy";
        trainingChart.data.datasets[0].data = accuracyTrain;
        trainingChart.data.datasets[0].borderColor = "#3b82f6";
        trainingChart.data.datasets[0].backgroundColor = "rgba(59, 130, 246, 0.1)";

        trainingChart.data.datasets[1].label = "Validation Accuracy";
        trainingChart.data.datasets[1].data = accuracyVal;
        trainingChart.data.datasets[1].borderColor = "#10b981";
        trainingChart.data.datasets[1].backgroundColor = "rgba(16, 185, 129, 0.1)";
        
        trainingChart.options.scales.y.title.text = "Accuracy (%)";
        trainingChart.update();
    });

    btnLoss.addEventListener("click", () => {
        btnLoss.classList.add("active");
        btnAcc.classList.remove("active");

        trainingChart.data.datasets[0].label = "Training Loss";
        trainingChart.data.datasets[0].data = lossTrain;
        trainingChart.data.datasets[0].borderColor = "#ef4444";
        trainingChart.data.datasets[0].backgroundColor = "rgba(239, 68, 68, 0.1)";

        trainingChart.data.datasets[1].label = "Validation Loss";
        trainingChart.data.datasets[1].data = lossVal;
        trainingChart.data.datasets[1].borderColor = "#f97316";
        trainingChart.data.datasets[1].backgroundColor = "rgba(249, 115, 22, 0.1)";

        trainingChart.options.scales.y.title.text = "Loss Value";
        trainingChart.update();
    });

    // Confusion Matrix dataset toggle
    const btnAdni = document.getElementById("btn-cm-adni");
    const btnOasis = document.getElementById("btn-cm-oasis");
    const cmAdni = document.getElementById("cm-adni-wrapper");
    const cmOasis = document.getElementById("cm-oasis-wrapper");

    btnAdni.addEventListener("click", () => {
        btnAdni.classList.add("active");
        btnOasis.classList.remove("active");
        cmAdni.classList.add("active");
        cmOasis.classList.remove("active");
    });

    btnOasis.addEventListener("click", () => {
        btnOasis.classList.add("active");
        btnAdni.classList.remove("active");
        cmOasis.classList.add("active");
        cmAdni.classList.remove("active");
    });

    // ── Interactive Playground ────────────────────────────────────────
    const caseSelect = document.getElementById("case-select");
    const baseImage = document.getElementById("base-image");
    const overlayImage = document.getElementById("overlay-image");
    const opacitySlider = document.getElementById("opacity-slider");
    const opacityVal = document.getElementById("opacity-val");

    // Case Details Data
    const cases = {
        cn: {
            baseImg: "assets/cn_original.png",
            overlayImg: "assets/cn_heatmap.png",
            probs: { cn: 96.4, emci: 2.8, mci: 0.6, ad: 0.2 },
            headline: "Normal Brain Glucose Metabolism",
            findings: "Consistent and symmetrical FDG glucose uptake is observed across all cerebral cortical structures. In healthy cognitively normal control scans, there is physiological preservation in default mode network hubs, such as the Posterior Cingulate Cortex. Grad-CAM shows diffuse, minimal focus.",
            markers: ["pcc", "hippocampus"] // regions visible for CN (but normal metabolism)
        },
        mci: {
            baseImg: "assets/mci_original.png",
            overlayImg: "assets/mci_heatmap.png",
            probs: { cn: 1.2, emci: 12.5, mci: 81.3, ad: 5.0 },
            headline: "Focal Hypometabolism (Mild)",
            findings: "Moderate metabolic depression is noted bilaterally in the hippocampi and temporoparietal cortices, with mild hypometabolism starting in the Posterior Cingulate Cortex (PCC). These findings are highly characteristic of amnestic Mild Cognitive Impairment, indicating high conversion risk.",
            markers: ["pcc", "hippocampus"]
        },
        ad: {
            baseImg: "assets/ad_original.png",
            overlayImg: "assets/ad_heatmap.png",
            probs: { cn: 0.1, emci: 0.9, mci: 10.0, ad: 89.0 },
            headline: "Severe Bilateral Temporoparietal Deficits",
            findings: "Severe, widespread metabolic reduction is detected throughout the posterior cingulate cortex, parietal lobes, and bilateral temporal neocortex, accompanied by severe bilateral hippocampal hypometabolism. This classic 'temporoparietal pattern' of FDG uptake strongly confirms the diagnosis of Alzheimer's clinical dementia.",
            markers: ["pcc", "hippocampus", "temporal"]
        }
    };

    // Region Information Data
    const regionInfo = {
        pcc: {
            title: "Posterior Cingulate Cortex (PCC)",
            text: "A critical metabolic hub of the brain's default mode network. Hypometabolism in the PCC is typically the earliest detectable neuroimaging indicator of preclinical Alzheimer's Disease (often visible in EMCI stages). Grad-CAM models consistently weight gradients high in this area."
        },
        hippocampus: {
            title: "Hippocampus",
            text: "The primary structure involved in learning and episodic memory consolidation. Atrophy and severe glucose hypometabolism in the hippocampus are strong indicators of transition from cognitive impairment (MCI) to Alzheimer's clinical dementia (AD)."
        },
        temporal: {
            title: "Temporoparietal Cortex",
            text: "Widespread, bilateral hypometabolism in this region is the clinical hallmark of Alzheimer's Disease. It correlates directly with the onset of standard cognitive dementia deficits, such as visual-spatial impairment, apraxia, and word-finding difficulties."
        }
    };

    // Update opacity blending
    function updateOpacity() {
        const val = opacitySlider.value;
        opacityVal.textContent = `${val}%`;
        overlayImage.style.opacity = val / 100;
    }

    opacitySlider.addEventListener("input", updateOpacity);
    updateOpacity(); // Initialize

    // Update case display
    function loadCase(caseKey) {
        const c = cases[caseKey];
        if (!c) return;

        // Set Images
        baseImage.src = c.baseImg;
        overlayImage.src = c.overlayImg;

        // Reset opacity to default slider value
        updateOpacity();

        // Animate Probabilities
        document.getElementById("prob-cn").textContent = `${c.probs.cn}%`;
        document.getElementById("bar-cn").style.width = `${c.probs.cn}%`;
        
        document.getElementById("prob-emci").textContent = `${c.probs.emci}%`;
        document.getElementById("bar-emci").style.width = `${c.probs.emci}%`;

        document.getElementById("prob-mci").textContent = `${c.probs.mci}%`;
        document.getElementById("bar-mci").style.width = `${c.probs.mci}%`;

        document.getElementById("prob-ad").textContent = `${c.probs.ad}%`;
        document.getElementById("bar-ad").style.width = `${c.probs.ad}%`;

        // Update Findings
        const headlineEl = document.querySelector(".findings-headline");
        const textEl = document.querySelector(".findings-text");
        headlineEl.textContent = c.headline;
        textEl.textContent = c.findings;

        // Show/hide markers based on case config
        document.querySelectorAll(".roi-marker").forEach(marker => {
            const markerRoi = marker.getAttribute("data-roi");
            if (c.markers.includes(markerRoi)) {
                marker.style.display = "block";
            } else {
                marker.style.display = "none";
            }
            marker.classList.remove("active");
        });

        // Reset active ROI selections
        document.querySelectorAll(".roi-btn").forEach(btn => btn.classList.remove("active"));
        document.getElementById("roi-details").innerHTML = `<p class="placeholder-text">Hover or click a region button or marker on the slice to inspect anatomical findings.</p>`;
    }

    caseSelect.addEventListener("change", (e) => {
        loadCase(e.target.value);
    });

    // Initialize with first case
    loadCase("cn");

    // ROI Interactions (Buttons + Markers)
    const roiButtons = document.querySelectorAll(".roi-btn");
    const roiMarkers = document.querySelectorAll(".roi-marker");
    const roiDetails = document.getElementById("roi-details");

    function selectROI(roiKey) {
        const info = regionInfo[roiKey];
        if (!info) return;

        // Highlight buttons
        roiButtons.forEach(btn => {
            if (btn.getAttribute("data-roi") === roiKey) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });

        // Highlight markers
        roiMarkers.forEach(marker => {
            if (marker.getAttribute("data-roi") === roiKey) {
                marker.classList.add("active");
            } else {
                marker.classList.remove("active");
            }
        });

        // Update details text
        roiDetails.innerHTML = `
            <h5>${info.title}</h5>
            <p>${info.text}</p>
        `;
    }

    roiButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            selectROI(btn.getAttribute("data-roi"));
        });
    });

    roiMarkers.forEach(marker => {
        marker.addEventListener("click", () => {
            selectROI(marker.getAttribute("data-roi"));
        });
        
        // Hover details
        marker.addEventListener("mouseenter", () => {
            selectROI(marker.getAttribute("data-roi"));
        });
    });
});
