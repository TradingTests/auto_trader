const RESULTS_LATEST = "../backend/results/latest.json";
const RESULTS_INDEX  = "../backend/results/index.json";

function fmt(n) {
  if (n === null || n === undefined) return "";
  if (typeof n === "number") return n.toFixed(4);
  return String(n);
}

async function loadJson(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed ${url}: ${r.status}`);
  return await r.json();
}

function renderTrades(trades) {
  const rows = trades.map(t => `
    <tr>
      <td>${t.ts}</td>
      <td>${t.type}</td>
      <td>${t.coin}</td>
      <td>${fmt(t.qty)}</td>
      <td>${fmt(t.price)}</td>
      <td>${fmt(t.notional)}</td>
      <td>${fmt(t.tax)}</td>
      <td class="small">${t.reason || ""}</td>
    </tr>
  `).join("");

  return `
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Type</th><th>Coin</th><th>Qty</th><th>Price</th><th>Notional</th><th>Tax</th><th>Reason</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderScores(scores) {
  const top = scores.slice(0, 10);
  const rows = top.map(s => `
    <tr>
      <td>${s.coin}</td>
      <td>${fmt(s.score)}</td>
    </tr>
  `).join("");
  return `
    <table>
      <thead><tr><th>Coin</th><th>Score</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="small">Scores are aggregated weighted signals from all configured strategies.</div>
  `;
}

function renderRunsIndex(idx) {
  const links = (idx.runs || []).slice().reverse().slice(0, 30).map(fn => {
    return `<li><a href="../backend/results/${fn}" target="_blank" rel="noreferrer">${fn}</a></li>`;
  }).join("");
  return `<div class="small">Latest: <a href="../backend/results/${idx.latest}" target="_blank">${idx.latest}</a></div><ul>${links}</ul>`;
}

let chart;

function renderChart(equityCurve) {
  const labels = equityCurve.map(p => p.ts);
  const values = equityCurve.map(p => p.value);

  const ctx = document.getElementById("equityChart");
  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Equity (USDT)",
        data: values,
        borderColor: "#7aa2ff",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.15
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true } },
      scales: {
        x: { ticks: { display: false } }
      }
    }
  });
}

async function main() {
  try {
    const latest = await loadJson(RESULTS_LATEST);
    document.getElementById("summary").textContent = JSON.stringify(latest.summary, null, 2);
    document.getElementById("allocations").textContent = JSON.stringify(latest.latest_allocations, null, 2);

    renderChart(latest.equity_curve || []);
    document.getElementById("trades").innerHTML = renderTrades(latest.trades || []);
    document.getElementById("scores").innerHTML = renderScores(latest.latest_scores || []);

    const idx = await loadJson(RESULTS_INDEX);
    document.getElementById("runs").innerHTML = renderRunsIndex(idx);
  } catch (e) {
    document.getElementById("summary").textContent = String(e);
  }
}

main();
