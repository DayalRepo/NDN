document.getElementById('csvFile').addEventListener('change', function() {
    const fileName = this.files[0] ? this.files[0].name : "No file chosen";
    document.getElementById('fileName').textContent = fileName;
});

// Chart instances
let interestChartInst = null;
let pitChartInst = null;
let satisfactionChartInst = null;
let timeoutChartInst = null;
let nackChartInst = null;
let loadChartInst = null;

document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('csvFile');
    const formData = new FormData();
    
    if(fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
    }

    // UI Updates
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('alertBanner').classList.add('hidden');
    document.getElementById('dashboardGrid').classList.add('hidden');
    document.getElementById('metricsGrid').classList.add('hidden');

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').classList.add('hidden');
        
        if(data.error) {
            alert("Error: " + data.error);
            return;
        }

        updateDashboard(data);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('loading').classList.add('hidden');
        alert("An error occurred while connecting to the server.");
    });
});

function updateDashboard(data) {
    // Show sections
    document.getElementById('metricsGrid').classList.remove('hidden');
    document.getElementById('dashboardGrid').classList.remove('hidden');
    
    const banner = document.getElementById('alertBanner');
    banner.classList.remove('hidden');

    // Update Banner & Status
    const isAttack = data.status !== 'Normal Traffic';
    banner.className = isAttack ? 'banner attack' : 'banner normal';
    banner.innerHTML = isAttack 
        ? `⚠️ Threat Detected: ${data.status} (Severity: ${data.severity_level})`
        : `✅ Network Operating Optimally`;

    // Update Metrics Dashboard
    document.getElementById('mInterest').textContent = data.metrics.interest_rate;
    document.getElementById('mPIT').textContent = data.metrics.pit_occupancy;
    document.getElementById('mSatisfaction').textContent = data.metrics.satisfaction_ratio;
    document.getElementById('mLoad').textContent = data.metrics.network_load;
    document.getElementById('mScore').textContent = data.metrics.attack_score;
    document.getElementById('mAccuracy').textContent = data.metrics.accuracy;

    // Update Status Card
    const statusEl = document.getElementById('resStatus');
    statusEl.textContent = data.status;
    statusEl.style.color = isAttack ? 'var(--danger)' : 'var(--text-primary)';
    
    document.getElementById('resSeverity').textContent = data.severity_level;
    document.getElementById('resSeverity').style.color = data.severity_level === 'High' ? 'var(--danger)' : (data.severity_level === 'Medium' ? 'var(--warning)' : 'var(--text-muted)');
    document.getElementById('resTime').textContent = data.alert_timestamp;

    // Update Summary Report (Counts and Breakdown)
    document.getElementById('sTotalAttacks').textContent = data.summary_report.total_attacks;
    document.getElementById('sAttackPerc').textContent = data.summary_report.attack_percentage + "%";
    
    // Type Breakdown
    const counts = data.class_counts || { 'IFA':0, 'Slow_IFA':0, 'Cache_Pollution':0, 'Distributed_IFA':0, 'Pulsing_IFA':0 };
    document.getElementById('c_ifa').textContent = counts['IFA'] || 0;
    document.getElementById('c_slow').textContent = counts['Slow_IFA'] || 0;
    document.getElementById('c_cache').textContent = counts['Cache_Pollution'] || 0;
    document.getElementById('c_dist').textContent = counts['Distributed_IFA'] || 0;
    document.getElementById('c_pulse').textContent = counts['Pulsing_IFA'] || 0;

    // Update Charts
    renderCharts(data.graph_data);
}

function renderCharts(graphData) {
    const labels = graphData.labels;

    // Minimalistic Chart Configuration
    const getChartConfig = (data, color, yMin = null, yMax = null) => ({
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                borderColor: color,
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                fill: false,
                tension: 0.1,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { 
                    beginAtZero: true, 
                    min: yMin, 
                    max: yMax,
                    grid: { display:false, drawBorder: false }, 
                    ticks: { color: '#94a3b8', font: { size: 10, family: "'Outfit', sans-serif" }, padding: 10 } 
                },
                x: { 
                    grid: { display: false, drawBorder: false }, 
                    ticks: { display: false } 
                }
            },
            layout: { padding: { left: -10, bottom: -10 }}
        }
    });

    // Interest Chart
    if (interestChartInst) interestChartInst.destroy();
    interestChartInst = new Chart(document.getElementById('interestChart'), 
        getChartConfig(graphData.interest_rate, '#0ea5e9'));

    // PIT Chart
    if (pitChartInst) pitChartInst.destroy();
    pitChartInst = new Chart(document.getElementById('pitChart'), 
        getChartConfig(graphData.pit_occupancy, '#ef4444'));

    // Satisfaction Chart
    if (satisfactionChartInst) satisfactionChartInst.destroy();
    satisfactionChartInst = new Chart(document.getElementById('satisfactionChart'), 
        getChartConfig(graphData.satisfaction_ratio, '#10b981', 0, 100));

    // Timeout Ratio Chart
    if (timeoutChartInst) timeoutChartInst.destroy();
    timeoutChartInst = new Chart(document.getElementById('timeoutChart'), 
        getChartConfig(graphData.timeout_ratio, '#f59e0b'));

    // NACK Ratio Chart
    if (nackChartInst) nackChartInst.destroy();
    nackChartInst = new Chart(document.getElementById('nackChart'), 
        getChartConfig(graphData.nack_ratio, '#8b5cf6'));

    // Load Chart
    if (loadChartInst) loadChartInst.destroy();
    loadChartInst = new Chart(document.getElementById('loadChart'), 
        getChartConfig(graphData.network_load, '#ec4899'));
}

