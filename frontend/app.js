const RESULTS_LATEST = "../backend/results/latest.json";
const RESULTS_HISTORY = "../backend/results/history.json";

function fmt(n, decimals = 4) {
  if (n === null || n === undefined) return "-";
  if (typeof n === "number") return n.toFixed(decimals);
  return String(n);
}

async function loadJson(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed ${url}: ${r.status}`);
  return await r.json();
}

function renderHoldings(holdings) {
  if (!holdings || Object.keys(holdings).length === 0) {
    return "<p>No holdings (100% cash)</p>";
  }
  const rows = Object.entries(holdings)
    .filter(([_, qty]) => qty > 0.0000001)
    .map(([coin, qty]) => `<tr><td>${coin.toUpperCase()}</td><td>${fmt(qty, 8)}</td></tr>`)
    .join("");
  return `<table><thead><tr><th>Coin</th><th>Quantity</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderActions(actions) {
  if (!actions || actions.length === 0) {
    return "<p>No trades this run</p>";
  }
  const rows = actions.map(a => `
    <tr class="${a.action.toLowerCase()}">
      <td>${a.action}</td>
      <td>${a.coin.toUpperCase()}</td>
      <td>${fmt(a.qty, 6)}</td>
      <td>$${fmt(a.price, 2)}</td>
      <td>$${fmt(a.notional, 2)}</td>
      <td>${a.reason}</td>
      <td>${fmt(a.score, 3)}</td>
    </tr>
  `).join("");
  return `
    <table>
      <thead><tr><th>Action</th><th>Coin</th><th>Qty</th><th>Price</th><th>Notional</th><th>Reason</th><th>Score</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderScores(scores) {
  if (!scores || scores.length === 0) return "<p>No scores</p>";
  const rows = scores.slice(0, 10).map(s => `
    <tr>
      <td>${s.coin.toUpperCase()}</td>
      <td class="${s.score >= 0 ? 'positive' : 'negative'}">${fmt(s.score, 4)}</td>
    </tr>
  `).join("");
  return `
    <table>
      <thead><tr><th>Coin</th><th>Score</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderTargetAllocation(alloc) {
  if (!alloc || Object.keys(alloc).length === 0) {
    return "<p>Target: 100% Cash (no good opportunities)</p>";
  }
  const items = Object.entries(alloc)
    .sort((a, b) => b[1] - a[1])
    .map(([coin, weight]) => `${coin.toUpperCase()}: ${(weight * 100).toFixed(1)}%`)
    .join(", ");
  return `<p><strong>Target Allocation:</strong> ${items}</p>`;
}

let chart;

function renderChart(equityHistory) {
  if (!equityHistory || equityHistory.length === 0) {
    return;
  }
  
  const labels = equityHistory.map(p => {
    const d = new Date(p.ts);
    return d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  });
  const values = equityHistory.map(p => p.value);

  const ctx = document.getElementById("equityChart");
  if (chart) chart.destroy();

  // Determine color based on performance
  const startVal = values[0] || 1000;
  const endVal = values[values.length - 1] || 1000;
  const color = endVal >= startVal ? "#22c55e" : "#ef4444";

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Portfolio Value (USDT)",
        data: values,
        borderColor: color,
        backgroundColor: color + "20",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      plugins: { 
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `$${ctx.parsed.y.toFixed(2)}`
          }
        }
      },
      scales: {
        x: { 
          display: true,
          ticks: { maxTicksLimit: 10, color: '#888' },
          grid: { color: '#333' }
        },
        y: {
          ticks: { color: '#888' },
          grid: { color: '#333' }
        }
      }
    }
  });
}

async function main() {
  try {
    const data = await loadJson(RESULTS_LATEST);
    
    // Summary card
    document.getElementById("summary").innerHTML = `
      <div class="stats-grid">
        <div class="stat">
          <span class="stat-label">Portfolio Value</span>
          <span class="stat-value">$${fmt(data.portfolio_value, 2)}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Cash</span>
          <span class="stat-value">$${fmt(data.cash, 2)}</span>
        </div>
        <div class="stat">
          <span class="stat-label">P&L</span>
          <span class="stat-value ${data.pnl_pct >= 0 ? 'positive' : 'negative'}">
            $${fmt(data.pnl_total, 2)} (${fmt(data.pnl_pct, 2)}%)
          </span>
        </div>
        <div class="stat">
          <span class="stat-label">High Water Mark</span>
          <span class="stat-value">$${fmt(data.high_water_mark, 2)}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Max Drawdown</span>
          <span class="stat-value negative">${fmt(data.max_drawdown_pct, 2)}%</span>
        </div>
        <div class="stat">
          <span class="stat-label">Total Trades</span>
          <span class="stat-value">${data.total_trades}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Tax Paid</span>
          <span class="stat-value">$${fmt(data.total_tax_paid, 2)}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Last Update</span>
          <span class="stat-value small">${new Date(data.generated_at).toLocaleString()}</span>
        </div>
      </div>
    `;
    
    // Chart
    renderChart(data.equity_history || []);
    
    // Holdings
    document.getElementById("holdings").innerHTML = renderHoldings(data.holdings);
    
    // Target allocation
    document.getElementById("allocation").innerHTML = renderTargetAllocation(data.target_allocations);
    
    // Scores
    document.getElementById("scores").innerHTML = renderScores(data.scores);
    
    // Actions this run
    document.getElementById("actions").innerHTML = renderActions(data.actions_this_run);
    
  } catch (e) {
    console.error(e);
    document.getElementById("summary").innerHTML = `<p class="error">Error: ${e.message}</p>`;
  }
}

// Refresh every 2 minutes
main();
setInterval(main, 120000);
