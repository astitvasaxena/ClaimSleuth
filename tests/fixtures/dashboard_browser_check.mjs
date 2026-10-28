// TEST FIXTURE: drives the ClaimSleuth dashboard in headless Chrome over the DevTools protocol
// (Node 22 built-in WebSocket, no extra packages) and prints one JSON object with what the page
// showed. Used by tests/test_dashboard_dossier.py.
//   node dashboard_browser_check.mjs <base url> <chrome executable>
// Only the "legacy" and "adk_local" engines are used: no MCP subprocess, no CrewAI, no model call.
// Delays/failures are injected by wrapping window.fetch inside the page (test-only).
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import net from "node:net";

const [BASE, CHROME] = process.argv.slice(2);
const port = await new Promise((res) => { const s = net.createServer(); s.listen(0, () => { const p = s.address().port; s.close(() => res(p)); }); });
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "cs-dossier-"))}`,
  "--no-first-run", "--no-default-browser-check", "--disable-gpu", "--window-size=1366,900", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 80 && !target; i++) {
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page"); } catch { await sleep(250); }
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0; const pending = new Map(); const errors = []; const dialogs = [];
const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
  if (m.method === "Runtime.exceptionThrown") errors.push((m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text).slice(0, 300));
  if (m.method === "Page.javascriptDialogOpening") { dialogs.push(m.params.message.slice(0, 200)); send("Page.handleJavaScriptDialog", { accept: true }); }
});
const ev = async (expression) => {
  const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error("page evaluate failed: " + JSON.stringify(r.result.exceptionDetails).slice(0, 500));
  return r.result?.result?.value;
};
const width = (w, h = 900) => send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: 1, mobile: false });

// What the dossier and review panel currently show
const SNAPSHOT = `(() => ({
  claimId: document.getElementById('metaClaimId').innerText.trim(),
  pill: document.getElementById('systemClaimStatus').innerText.trim(),
  tabs: [...document.querySelectorAll('#docTabs .doc-tab')].map(t => t.textContent),
  activeTab: (document.querySelector('#docTabs .doc-tab.active') || {}).textContent || null,
  docText: document.getElementById('docContent').textContent,
  evidenceMessage: (document.querySelector('#evidenceList .dossier-empty') || {}).textContent || null,
  evidenceIsError: !!document.querySelector('#evidenceList .dossier-empty.error'),
  facts: [...document.querySelectorAll('#evidenceList .evidence-card')].map(c => ({
    field: c.querySelector('.ev-field').textContent, value: c.querySelector('.ev-value').textContent,
    quote: (c.querySelector('.ev-quote') || {}).textContent || null, meta: c.querySelector('.ev-meta').textContent })),
  outcome: document.getElementById('hitlOutcome').innerText.trim(),
  visibleDecisions: ['APPROVED', 'REJECTED', 'INFO_REQUESTED'].filter(d => !document.getElementById('btnDecision' + d).hidden),
  notesVisible: !document.getElementById('humanNotesInput').hidden,
  runButtonEnabled: !document.getElementById('runInvestigationBtn').disabled,
  payout: document.getElementById('rPayout').innerText.trim(),
  resultsBannerVisible: document.getElementById('resultsBanner').classList.contains('visible'),
}))()`;
const run = (scenario, engine) => ev(`(async () => { await selectScenario(${JSON.stringify(scenario)});
  document.getElementById('engineSelect').value = ${JSON.stringify(engine)}; await runInvestigation(); return 1; })()`);
const INJECT = `(() => { if (window.__realFetch) return 1; window.__realFetch = window.fetch; window.__fetchRules = [];
  window.fetch = async (url, opts) => { for (const r of window.__fetchRules) { if (String(url).includes(r.match)) {
      if (r.fail) return new Response(JSON.stringify({ detail: 'injected test failure' }), { status: r.fail, headers: { 'Content-Type': 'application/json' } });
      await new Promise(res => setTimeout(res, r.delay)); } }
    return window.__realFetch(url, opts); }; return 1; })()`;

const out = { errors, dialogs };
try {
  await send("Page.enable"); await send("Runtime.enable"); await width(1366);
  await send("Page.navigate", { url: BASE + "/" });
  for (let i = 0; i < 80; i++) { if (await ev("document.readyState") === "complete") break; await sleep(250); }
  for (let i = 0; i < 80; i++) { if (await ev("document.querySelectorAll('#docTabs .doc-tab').length") > 0) break; await sleep(250); }
  await ev(INJECT);

  // 1. Before any investigation
  out.initial = await ev(SNAPSHOT);
  // 2. Document switching shows the selected file's full text
  out.docSwitch = await ev(`(async () => { const r = {}; for (const t of document.querySelectorAll('#docTabs .doc-tab')) {
      t.click(); r[t.textContent] = document.getElementById('docContent').textContent; } return r; })()`);
  // 3. Loading state while an investigation is in flight
  await ev(`window.__fetchRules = [{ match: '/api/adk/investigate', delay: 1200 }]; 1`);
  await ev(`(async () => { await selectScenario('scenario_1_valid'); document.getElementById('engineSelect').value = 'adk_local';
      window.__pendingRun = runInvestigation(); return 1; })()`);
  await sleep(250);
  out.loading = await ev(SNAPSHOT);
  await ev(`window.__pendingRun.then(() => 1)`);
  await ev(`window.__fetchRules = []; 1`);
  // 4. Scenario 1 (ADK, local tools) result, then a human approval
  out.s1 = await ev(SNAPSHOT);
  dialogs.length = 0;
  await ev(`(async () => { await adjudicate('APPROVED'); return 1; })()`);
  out.s1AfterApprove = { ...(await ev(SNAPSHOT)), dialogs: [...dialogs] };
  // 5. Scenario 3 (original loop): approval is not offered, and forcing it is refused by the server
  await run("scenario_3_conflict", "legacy");
  out.s3 = await ev(SNAPSHOT);
  dialogs.length = 0;
  await ev(`(async () => { await adjudicate('APPROVED'); return 1; })()`);
  out.s3ForcedApprove = { ...(await ev(SNAPSHOT)), dialogs: [...dialogs] };
  // 6. Scenario 2 (ADK): rejected deterministically
  await run("scenario_2_expired", "adk_local");
  out.s2 = await ev(SNAPSHOT);
  // 7. Stale investigation: a slow run for scenario 3 must not overwrite scenario 1 after switching
  await ev(`window.__fetchRules = [{ match: '/api/investigate', delay: 1500 }]; 1`);
  await ev(`(async () => { await selectScenario('scenario_3_conflict'); document.getElementById('engineSelect').value = 'legacy';
      window.__staleRun = runInvestigation(); await selectScenario('scenario_1_valid'); return 1; })()`);
  await ev(`window.__staleRun.then(() => 1)`);
  await sleep(300);
  out.staleRun = await ev(SNAPSHOT);
  // 8. Stale documents: a slow docs response for scenario 2 must not replace scenario 1's tabs
  await ev(`window.__fetchRules = [{ match: '/api/scenario/scenario_2_expired/docs', delay: 1200 }]; 1`);
  await ev(`(async () => { const slow = selectScenario('scenario_2_expired'); await selectScenario('scenario_1_valid'); await slow; return 1; })()`);
  await sleep(300);
  out.staleDocs = await ev(SNAPSHOT);
  // 9. A failed investigation shows an error, not stale facts or controls
  await run("scenario_1_valid", "adk_local");
  dialogs.length = 0;
  await ev(`window.__fetchRules = [{ match: '/api/adk/investigate', fail: 500 }]; 1`);
  await ev(`(async () => { document.getElementById('engineSelect').value = 'adk_local'; await runInvestigation(); return 1; })()`);
  out.failed = { ...(await ev(SNAPSHOT)), dialogs: [...dialogs] };
  await ev(`window.__fetchRules = []; 1`);

  // 10. Layout: long values in the summary row and long document text, at several widths
  await run("scenario_3_conflict", "adk_local");
  await ev(`(() => {
    document.getElementById('metaPolicyId').textContent = 'POL-APPLIANCE-PLUS-03-EXTENDED-PREMIUM-COVERAGE-IDENTIFIER-2024-XYZ';
    document.getElementById('metaClaimant').textContent = 'A Very Long Claimant Display Name For Layout Testing Purposes';
    document.getElementById('metaEngine').textContent = 'Google ADK workflow · deterministic · tools: mcp_stdio · CrewAI Crew + LLM (nvidia/nemotron-3.5-super-vl:free): 1 finding(s), 1 model call(s) · RAG: 6 policy citations · Deterministic fallback (no Gemini configured for this process) · extra-long-unbroken-token-' + 'x'.repeat(120);
    const doc = document.getElementById('docContent');
    doc.textContent = doc.textContent + '\\nVERY-LONG-UNBROKEN-LINE-' + 'Z'.repeat(400);
    return 1; })()`);
  const MEASURE = `(() => {
    const vw = document.documentElement.clientWidth;
    const past = (container) => { const box = container.getBoundingClientRect(); let n = 0, max = 0;
      container.querySelectorAll('*').forEach(el => { const r = el.getBoundingClientRect(); if (!r.width) return;
        const over = Math.round(r.right - box.right); if (over > 1) { n++; max = Math.max(max, over); } }); return { n, max }; };
    let beyondViewport = 0;
    document.querySelectorAll('.case-dossier-card *, .main-grid *').forEach(el => { const r = el.getBoundingClientRect(); if (r.width && r.right > vw + 1) beyondViewport++; });
    const inv = document.getElementById('metaEngine'); const doc = document.getElementById('docContent');
    const chips = [...document.querySelectorAll('.meta-chip')].map(c => c.getBoundingClientRect());
    let overlaps = 0;
    for (let i = 0; i < chips.length; i++) for (let j = i + 1; j < chips.length; j++) {
      const a = chips[i], b = chips[j]; if (a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1) overlaps++; }
    return { viewport: vw, summaryCard: past(document.querySelector('.case-dossier-card')),
             dossierPanel: past(document.getElementById('hitlBox').closest('.panel')), beyondViewport, chipOverlaps: overlaps,
             summaryColumns: getComputedStyle(document.querySelector('.meta-chips-grid')).gridTemplateColumns.split(' ').length,
             investigationClipped: inv.scrollWidth > inv.clientWidth + 1, investigationEllipsis: getComputedStyle(inv).textOverflow,
             investigationText: inv.textContent.length, docHorizontalOverflow: doc.scrollWidth > doc.clientWidth + 1,
             evidenceHorizontalOverflow: (() => { const e = document.getElementById('evidenceList'); return e.scrollWidth > e.clientWidth + 1; })() };
  })()`;
  out.layout = {};
  for (const w of [1920, 1440, 1366, 1280, 1024, 768, 420]) { await width(w); await sleep(250); out.layout[w] = await ev(MEASURE); }
} catch (err) {
  out.fatal = String(err).slice(0, 800);
}
console.log(JSON.stringify(out));
ws.close(); chrome.kill(); process.exit(0);
