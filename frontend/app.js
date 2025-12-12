const LATEST = "../backend/results/latest.json";

async function loadJson(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed ${url}: ${r.status}`);
  return await r.json();
}

function fmt(n, d=4) {
  if (n === null || n === undefined) return "-";
  if (typeof n === "number") return n.toFixed(d);
  return String(n);
}

function renderMain(main) {
  if (!main) return "No data";
  const cls = (main.pnl_pct || 0) >= 0 ? "pos" : "neg";
  const holdings = Object.entries(main.holdings || {})
    .filter(([_,q]) => q > 0)
    .map(([c,q]) => `${c.toUpperCase()}: ${fmt(q, 8)}`)
    .join(", ") || "CASH";

  return `
    <div><b>Value:</b> ${fmt(main.value, 2)} USDT</div>
    <div><b>Cash:</b> ${fmt(main.cash, 2)} USDT</div>
    <div><b>PnL:</b> <span class="${cls}">${fmt(main.pnl, 2)} (${fmt(main.pnl_pct, 2)}%)</span></div>
    <div><b>Trades:</b> ${main.total_trades} | <b>Tax paid:</b> ${fmt(main.tax_paid, 4)}</div>
    <div class="small"><b>Holdings:</b> ${holdings}</div>
  `;
}

function renderLeaderboard(rows) {
  if (!rows || rows.length === 0) return "No leaderboard yet";
  const htmlRows = rows.map(r => {
    const cls = (r.return_lookback_pct || 0) >= 0 ? "pos" : "neg";
    return `
      <tr>
        <td>${r.algorithm}</td>
        <td class="${cls}">${fmt(r.return_lookback_pct, 3)}%</td>
        <td>${fmt(r.value, 2)}</td>
        <td class="small">${(r.holdings || []).join(", ")}</td>
      </tr>
    `;
  }).join("");

  return `
    <table>
      <thead>
        <tr><th>Algorithm</th><th>Return (lookback)</th><th>Value</th><th>Holdings</th></tr>
      </thead>
      <tbody>${htmlRows}</tbody>
    </table>
  `;
}

function renderActions(actions) {
  if (!actions || actions.length === 0) return "<div>No trades this run.</div>";
  const rows = actions.map(a => `
    <tr>
      <td>${a.ts}</td>
      <td>${a.type}</td>
      <td>${a.coin.toUpperCase()}</td>
      <td>${fmt(a.qty, 8)}</td>
      <td>${fmt(a.price, 2)}</td>
      <td>${fmt(a.notional, 2)}</td>
      <td>${fmt(a.tax, 4)}</td>
      <td class="small">${a.reason || ""}</td>
    </tr>
  `).join("");

  return `
    <table>
      <thead>
        <tr><th>Time</th><th>Type</th><th>Coin</th><th>Qty</th><th>Price</th><th>Notional</th><th>Tax</th><th>Reason</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

let chart;
function renderChart(equity) {
  if (!equity || equity.length < 2) return;
  const labels = equity.map(p => new Date(p.ts).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}));
  const values = equity.map(p => p.value);

  const ctx = document.getElementById("equityChart");
  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Main Equity (USDT)",
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
      scales: { x: { ticks: { display: false } } }
    }
  });
}

async function main() {
  const data = await loadJson(LATEST);

  document.getElementById("mainSummary").innerHTML = renderMain(data.main);

  document.getElementById("selectedAlgo").innerHTML =
    `<b>${data.selected_algorithm || "-"}</b>`;

  const alloc = data.winner_target_allocations || {};
  const allocText = Object.keys(alloc).length
    ? Object.entries(alloc).map(([c,w]) => `${c.toUpperCase()}: ${(w*100).toFixed(1)}%`).join(", ")
    : "100% CASH";
  document.getElementById("winnerAlloc").textContent = `Winner target: ${allocText}`;

  document.getElementById("leaderboard").innerHTML = renderLeaderboard(data.algorithm_leaderboard || []);
  document.getElementById("actions").innerHTML = renderActions(data.actions_this_run || []);

  renderChart(data.equity_history || []);
}

main().catch(e => {
  document.getElementById("mainSummary").textContent = String(e);
});
