document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const userInput = document.getElementById('user-input');
    const charCount = document.getElementById('char-count');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnClear = document.getElementById('btn-clear');
    const sampleButtons = document.querySelectorAll('.sample-btn');

    const emptyState = document.getElementById('empty-state');
    const resultsContainer = document.getElementById('results-container');
    const resultBanner = document.getElementById('result-banner');
    const bannerIcon = document.getElementById('banner-icon');
    const resultClassName = document.getElementById('result-class-name');
    const resultDescription = document.getElementById('result-description');
    const resultConfidence = document.getElementById('result-confidence');

    const probHateVal = document.getElementById('prob-hate-val');
    const probHateBar = document.getElementById('prob-hate-bar');
    const probOffenseVal = document.getElementById('prob-offense-val');
    const probOffenseBar = document.getElementById('prob-offense-bar');
    const probNeitherVal = document.getElementById('prob-neither-val');
    const probNeitherBar = document.getElementById('prob-neither-bar');

    const tokensList = document.getElementById('tokens-list');
    const preprocessingReportText = document.getElementById('preprocessing-report-text');
    const btnCopyReport = document.getElementById('btn-copy-report');

    const batchInput = document.getElementById('batch-input');
    const btnBatchAnalyze = document.getElementById('btn-batch-analyze');
    const batchResultsTable = document.getElementById('batch-results-table');
    const batchTableBody = document.getElementById('batch-table-body');

    // 1. Navigation Tab Switching
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');

            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(tabId).classList.add('active');

            if (tabId === 'tab-preprocessing' && preprocessingReportText.textContent.includes("Loading")) {
                fetchPreprocessingReport();
            }
        });
    });

    // 2. Character Count Tracker
    userInput.addEventListener('input', () => {
        charCount.textContent = `${userInput.value.length} chars`;
    });

    // 3. Sample Button Clicks
    sampleButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const text = btn.getAttribute('data-sample');
            userInput.value = text;
            charCount.textContent = `${text.length} chars`;
            analyzeText(text);
        });
    });

    // 4. Analyze Button Click
    btnAnalyze.addEventListener('click', () => {
        const text = userInput.value.trim();
        if (text) {
            analyzeText(text);
        }
    });

    btnClear.addEventListener('click', () => {
        userInput.value = '';
        charCount.textContent = '0 chars';
        emptyState.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
    });

    // 5. Single Inference API Request
    async function analyzeText(text) {
        btnAnalyze.disabled = true;
        btnAnalyze.querySelector('.btn-text').textContent = 'Analyzing...';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) throw new Error('API request failed');

            const data = await response.json();
            displayResults(data);
        } catch (err) {
            console.error('Error analyzing text:', err);
            alert('Error running inference. Please ensure the Python server is running.');
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.querySelector('.btn-text').textContent = 'Analyze Content';
        }
    }

    // 6. Display Analysis Results in UI
    function displayResults(data) {
        emptyState.classList.add('hidden');
        resultsContainer.classList.remove('hidden');

        const classId = data.class_id;
        const className = data.class_name;
        const confPct = (data.confidence * 100).toFixed(1);

        resultClassName.textContent = className;
        resultDescription.textContent = data.description;
        resultConfidence.textContent = `${confPct}%`;

        // Update Banner Style
        resultBanner.className = 'result-banner';
        if (classId === 0) {
            resultBanner.classList.add('banner-hate');
            bannerIcon.textContent = '🚫';
        } else if (classId === 1) {
            resultBanner.classList.add('banner-offensive');
            bannerIcon.textContent = '⚠️';
        } else {
            resultBanner.classList.add('banner-neither');
            bannerIcon.textContent = '✅';
        }

        // Update Probabilities
        const probs = data.probabilities;
        const p0 = ((probs['Hate Speech'] || 0) * 100).toFixed(1);
        const p1 = ((probs['Offensive Language'] || 0) * 100).toFixed(1);
        const p2 = ((probs['Neither'] || 0) * 100).toFixed(1);

        probHateVal.textContent = `${p0}%`;
        probHateBar.style.width = `${p0}%`;

        probOffenseVal.textContent = `${p1}%`;
        probOffenseBar.style.width = `${p1}%`;

        probNeitherVal.textContent = `${p2}%`;
        probNeitherBar.style.width = `${p2}%`;

        // Render Tokens List
        tokensList.innerHTML = '';
        if (data.cleaned_text) {
            const tokens = data.cleaned_text.split(' ');
            tokens.forEach(t => {
                const chip = document.createElement('span');
                chip.className = 'token-chip';
                chip.textContent = t;
                tokensList.appendChild(chip);
            });
        } else {
            tokensList.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">No tokens remaining after preprocessing</span>';
        }
    }

    // 7. Fetch Preprocessing Conclusion Report
    async function fetchPreprocessingReport() {
        try {
            const response = await fetch('/api/preprocessing_summary');
            const data = await response.json();
            preprocessingReportText.textContent = data.report;
        } catch (err) {
            preprocessingReportText.textContent = 'Error loading preprocessing report.';
        }
    }

    if (btnCopyReport) {
        btnCopyReport.addEventListener('click', () => {
            navigator.clipboard.writeText(preprocessingReportText.textContent);
            btnCopyReport.textContent = 'Copied!';
            setTimeout(() => btnCopyReport.textContent = 'Copy Report', 2000);
        });
    }

    // 8. Batch Text Analyzer
    btnBatchAnalyze.addEventListener('click', async () => {
        const rawText = batchInput.value.trim();
        if (!rawText) return;

        const lines = rawText.split('\n').filter(l => l.trim() !== '');
        btnBatchAnalyze.disabled = true;
        btnBatchAnalyze.textContent = 'Processing Batch...';

        try {
            const response = await fetch('/api/batch_predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ texts: lines })
            });

            const data = await response.json();
            renderBatchTable(data.results);
        } catch (err) {
            alert('Batch processing failed.');
        } finally {
            btnBatchAnalyze.disabled = false;
            btnBatchAnalyze.textContent = 'Process Batch';
        }
    });

    function renderBatchTable(results) {
        batchResultsTable.classList.remove('hidden');
        batchTableBody.innerHTML = '';

        results.forEach(res => {
            const tr = document.createElement('tr');
            const confPct = (res.confidence * 100).toFixed(1);
            tr.innerHTML = `
                <td>${res.raw_text}</td>
                <td><strong>${res.class_name}</strong></td>
                <td>${confPct}%</td>
            `;
            batchTableBody.appendChild(tr);
        });
    }

    // Load metrics on startup
    fetch('/api/stats')
        .then(r => r.json())
        .then(data => {
            if (data.best_model) {
                document.getElementById('badge-model-name').textContent = data.best_model;
                if (data.all_models && data.all_models[data.best_model]) {
                    const bestMetrics = data.all_models[data.best_model];
                    document.getElementById('stat-accuracy').textContent = `${(bestMetrics.accuracy * 100).toFixed(1)}%`;
                    document.getElementById('stat-f1').textContent = bestMetrics.macro_f1.toFixed(3);
                }
            }
        }).catch(e => console.log('Stats load ignored:', e));
});
