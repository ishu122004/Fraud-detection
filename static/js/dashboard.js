document.addEventListener("DOMContentLoaded", function () {
    if (!dashboardData) {
        console.error("No dashboard data available.");
        return;
    }

    // ==========================
    // Fraud Trend Line Chart
    // ==========================
    const fraudTrendCtx = document.getElementById("fraudTrendChart").getContext("2d");
    new Chart(fraudTrendCtx, {
        type: "line",
        data: {
            labels: dashboardData.daily_fraud_rate.map(d => d.date),
            datasets: [
                {
                    label: "Fraud Cases",
                    data: dashboardData.daily_fraud_rate.map(d => d.fraud),
                    borderColor: "red",
                    backgroundColor: "rgba(255,0,0,0.2)",
                    fill: true,
                    tension: 0.3
                },
                {
                    label: "Total Transactions",
                    data: dashboardData.daily_fraud_rate.map(d => d.total),
                    borderColor: "blue",
                    backgroundColor: "rgba(0,0,255,0.1)",
                    fill: false,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: true, position: "bottom" }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });

    // ==========================
    // Fraud Reasons Pie Chart
    // ==========================
    // ==========================
// Fraud Reasons Pie Chart
// ==========================
const fraudReasonsCtx = document.getElementById("fraudReasonsChart").getContext("2d");

// Filter out unwanted reasons
const filteredReasons = dashboardData.fraud_reasons.filter(r =>
    !["card4", "Unusual card network usage", "C14: 0"].some(unwanted =>
        r.reason.toLowerCase().includes(unwanted.toLowerCase())
    )
);

new Chart(fraudReasonsCtx, {
    type: "pie",
    data: {
        labels: filteredReasons.map(r => r.reason),
        datasets: [{
            data: filteredReasons.map(r => r.count),
            backgroundColor: [
                "#ff6384",
                "#36a2eb",
                "#ffce56",
                "#4bc0c0",
                "#9966ff",
                "#ff9f40",
                "#8dd17e",
                "#c45850"
            ]
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { position: "bottom" }
        }
    }
});

        
       

    // ==========================
    // Recent Transactions Table
    // ==========================
    const tbody = document.querySelector("#transactionsTable tbody");
    if (tbody) {
        tbody.innerHTML = "";
        dashboardData.recent_transactions.forEach(tx => {
            const row = `
                <tr>
                    <td>${tx.id}</td>
                    <td>${tx.name} (${tx.email})</td>
                    <td>$${tx.amount.toFixed(2)}</td>
                    <td>${tx.fraud_score}</td>
                    <td>${tx.decision}</td>
                    <td>${tx.created_at}</td>
                    <td>${tx.reason}</td>
                </tr>`;
            tbody.insertAdjacentHTML("beforeend", row);
        });
    }
});

// ==========================
// Refresh Button
// ==========================
function refreshDashboard() {
    location.reload();
}
