/**
 * AgoraConsole — the fleet console.
 *
 * Three things this shows that no dashboard in this category shows:
 *
 *   1. Evidence intervals, not filled bars. A bar at 92% is three completely
 *      different facts depending on the evidence behind it, and a percentage
 *      cannot express the difference. Width is what you do not know.
 *   2. Assurance grade per capability, including the ones that are Grade D.
 *      Stating what a capability cannot do is what makes the rest credible.
 *   3. Live factory verdicts, including the blocked ones and why.
 *
 * Self-contained: no imports beyond React, no CSS framework, no chart library.
 * Feed it real data by passing `fleet`, `ledger` and `verdicts` props; it falls
 * back to the sample payload below so the component renders on its own.
 */

import React, { useMemo, useState } from "react";

/* ------------------------------------------------------------------ */
/* Palette                                                             */
/* ------------------------------------------------------------------ */

const C = {
  bg: "#0E0E11",
  panel: "#15151A",
  panelAlt: "#1B1B20",
  line: "#2A2A31",
  text: "#E8E6E1",
  dim: "#8E8B85",
  gold: "#E5C687",
  goldDim: "#8C7A4E",
  green: "#7BB661",
  amber: "#D4A34A",
  red: "#C4553D",
  blue: "#6E8FB8",
};

const GRADE = {
  A: { color: C.green, label: "Verified", may: "autonomous within budget, sampled verification" },
  B: { color: C.blue, label: "Checked", may: "autonomous, every output verified by a second path" },
  C: { color: C.amber, label: "Guarded", may: "drafts only, human approves before anything leaves" },
  D: { color: C.red, label: "Observed", may: "drafts only, full human review" },
};

const EVIDENCE = {
  "production-measured": { color: C.green, note: "routable unsupervised" },
  "harness-verified": { color: C.amber, note: "verification mandatory" },
  declared: { color: C.dim, note: "not routable — this is a guess" },
  deprecated: { color: C.red, note: "below floor, excluded from routing" },
};

/* ------------------------------------------------------------------ */
/* Sample payload                                                      */
/* ------------------------------------------------------------------ */

const SAMPLE = {
  fleet: [
    {
      id: "agent://eng/implementation",
      name: "Implementation",
      department: "engineering",
      klass: "persistent",
      grade: "A",
      phase: 2,
      probation: "graduated",
      skills: [
        { name: "Migration writer", tier: "T2", stage: "canary", evals: "evals/eng/migration_v1.yaml" },
        { name: "Test scaffolder", tier: "T1", stage: "promoted", evals: "evals/eng/scaffold_v2.yaml" },
      ],
    },
    {
      id: "agent://eng/code-reviewer",
      name: "Code Reviewer",
      department: "engineering",
      klass: "persistent",
      grade: "A",
      phase: 2,
      probation: "graduated",
      skills: [],
    },
    {
      id: "agent://customer/reply-drafter",
      name: "Reply Drafter",
      department: "customer",
      klass: "persistent",
      grade: "C",
      phase: 4,
      probation: "probation",
      skills: [{ name: "Refund policy lookup", tier: "T0", stage: "shadow", evals: "evals/customer/refund_v1.yaml" }],
    },
    {
      id: "agent://control/archivist",
      name: "Archivist",
      department: "control",
      klass: "control-plane",
      grade: "A",
      phase: 5,
      probation: "graduated",
      skills: [],
    },
  ],
  ledger: [
    { agent: "agent://eng/implementation", capability: "code.implement", bucket: "python · small · familiar", mean: 0.92, lo: 0.85, hi: 0.96, n: 67, evidence: "production-measured" },
    { agent: "agent://eng/implementation", capability: "code.implement", bucket: "swift · small · familiar", mean: 0.86, lo: 0.68, hi: 0.97, n: 12, evidence: "harness-verified" },
    { agent: "agent://eng/implementation", capability: "code.implement", bucket: "terraform · novel", mean: 0.55, lo: 0.32, hi: 0.77, n: 67, evidence: "harness-verified", borrowed: "pooled root, distance 2" },
    { agent: "agent://eng/implementation", capability: "repo.refactor", bucket: "rust · large · novel", mean: 0.43, lo: 0.31, hi: 0.55, n: 40, evidence: "deprecated" },
    { agent: "agent://customer/reply-drafter", capability: "support.draft", bucket: "billing", mean: 0.71, lo: 0.44, hi: 0.91, n: 9, evidence: "harness-verified" },
  ],
  verdicts: [
    { charter: "agent://revenue/deal-desk", rung: "P3-persistent", approved: true, skeptic: "P3-persistent is the smallest thing that solves it", blockers: [] },
    { charter: "agent://eng/test-writer", rung: "P1-skill", approved: false, skeptic: "the proposal asks for P3-persistent but the evidence supports P1-skill", blockers: ["capability overlap 0.91 with agent://eng/implementation exceeds 0.85 — extend that agent instead"] },
    { charter: "agent://marketing/campaign-runner", rung: "P2-ephemeral", approved: false, skeptic: "P2-ephemeral is the smallest thing that solves it", blockers: ["no retirement criteria", "no eval suite"] },
  ],
  gaps: [
    { capability: "legal.redline", count: 14, reason: "no agent declares this capability" },
    { capability: "data.migrate", count: 6, reason: "policy excluded every candidate" },
  ],
};

/* ------------------------------------------------------------------ */
/* Primitives                                                          */
/* ------------------------------------------------------------------ */

function Panel({ title, subtitle, children, right }) {
  return (
    <section style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 10, padding: 20, marginBottom: 18 }}>
      <header style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: subtitle ? 4 : 14 }}>
        <h2 style={{ margin: 0, fontSize: 13, letterSpacing: "0.14em", textTransform: "uppercase", color: C.gold, fontWeight: 600 }}>{title}</h2>
        {right}
      </header>
      {subtitle && <p style={{ margin: "0 0 16px", fontSize: 12.5, color: C.dim, lineHeight: 1.55, maxWidth: "68ch" }}>{subtitle}</p>}
      {children}
    </section>
  );
}

function Pill({ children, color = C.dim, filled = false }) {
  return (
    <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: 999, fontSize: 10.5, letterSpacing: "0.06em", textTransform: "uppercase", color: filled ? C.bg : color, background: filled ? color : "transparent", border: `1px solid ${color}`, whiteSpace: "nowrap" }}>
      {children}
    </span>
  );
}

/**
 * The evidence interval. This is the component the whole console exists for.
 * The bar is the 90% credible interval; the tick is the mean. Two capabilities
 * with the same mean and different widths are not comparable, and drawing them
 * this way is the only honest way to say so.
 */
function IntervalBar({ lo, hi, mean, color }) {
  const pct = (v) => `${Math.max(0, Math.min(1, v)) * 100}%`;
  return (
    <div style={{ position: "relative", height: 22, background: C.panelAlt, borderRadius: 4, border: `1px solid ${C.line}` }}>
      {[0.25, 0.5, 0.75].map((g) => (
        <div key={g} style={{ position: "absolute", left: pct(g), top: 0, bottom: 0, width: 1, background: C.line }} />
      ))}
      <div
        title={`90% credible interval ${lo.toFixed(2)}–${hi.toFixed(2)} (width ${(hi - lo).toFixed(2)})`}
        style={{ position: "absolute", left: pct(lo), width: pct(hi - lo), top: 4, bottom: 4, background: color, opacity: 0.28, borderRadius: 3 }}
      />
      <div style={{ position: "absolute", left: pct(mean), top: 1, bottom: 1, width: 2, background: color }} />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Sections                                                            */
/* ------------------------------------------------------------------ */

function LedgerRow({ row }) {
  const ev = EVIDENCE[row.evidence] || EVIDENCE.declared;
  const width = row.hi - row.lo;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(180px, 1.4fr) minmax(220px, 2fr) 62px", gap: 14, alignItems: "center", padding: "12px 0", borderTop: `1px solid ${C.line}` }}>
      <div>
        <div style={{ fontSize: 13, color: C.text }}>{row.capability}</div>
        <div style={{ fontSize: 11.5, color: C.dim, marginTop: 2 }}>{row.bucket}</div>
        {row.borrowed && (
          <div style={{ fontSize: 11, color: C.amber, marginTop: 4 }}>borrowed — {row.borrowed}</div>
        )}
      </div>
      <div>
        <IntervalBar lo={row.lo} hi={row.hi} mean={row.mean} color={ev.color} />
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 5, fontSize: 11, color: C.dim }}>
          <span>n={row.n} · {row.evidence}</span>
          <span style={{ color: ev.color }}>{ev.note}</span>
        </div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div style={{ fontSize: 19, color: C.text, fontVariantNumeric: "tabular-nums" }}>{row.mean.toFixed(2)}</div>
        <div style={{ fontSize: 10.5, color: C.dim }}>±{(width / 2).toFixed(2)}</div>
      </div>
    </div>
  );
}

function AgentCard({ agent }) {
  const [open, setOpen] = useState(false);
  const g = GRADE[agent.grade] || GRADE.D;
  return (
    <div style={{ background: C.panelAlt, border: `1px solid ${C.line}`, borderRadius: 8, padding: 14, marginBottom: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <button onClick={() => setOpen(!open)} aria-expanded={open}
          style={{ background: "none", border: "none", color: C.gold, cursor: "pointer", fontSize: 12, padding: 0, width: 14 }}>
          {agent.skills.length ? (open ? "▾" : "▸") : "·"}
        </button>
        <strong style={{ fontSize: 14, color: C.text }}>{agent.name}</strong>
        <Pill color={g.color} filled>{`grade ${agent.grade}`}</Pill>
        <Pill color={C.goldDim}>{agent.klass}</Pill>
        <Pill color={agent.probation === "graduated" ? C.green : C.amber}>{agent.probation}</Pill>
        <span style={{ marginLeft: "auto", fontSize: 11, color: C.dim }}>phase {agent.phase} · {agent.department}</span>
      </div>
      <div style={{ fontSize: 11.5, color: C.dim, marginTop: 8, paddingLeft: 24 }}>{g.may}</div>
      {open && agent.skills.length > 0 && (
        <div style={{ marginTop: 12, paddingLeft: 24, borderLeft: `1px solid ${C.line}`, marginLeft: 6 }}>
          {agent.skills.map((s) => (
            <div key={s.name} style={{ display: "flex", gap: 10, alignItems: "center", padding: "6px 0 6px 12px", flexWrap: "wrap" }}>
              <span style={{ fontSize: 12.5, color: C.text }}>{s.name}</span>
              <Pill color={s.tier === "T2" ? C.amber : C.goldDim}>{s.tier}</Pill>
              <Pill color={s.stage === "promoted" ? C.green : C.amber}>{s.stage}</Pill>
              <span style={{ fontSize: 11, color: C.dim, marginLeft: "auto" }}>{s.evals}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function VerdictRow({ v }) {
  return (
    <div style={{ padding: "12px 0", borderTop: `1px solid ${C.line}` }}>
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <Pill color={v.approved ? C.green : C.red} filled>{v.approved ? "approved" : "blocked"}</Pill>
        <code style={{ fontSize: 12.5, color: C.text }}>{v.charter}</code>
        <Pill color={C.goldDim}>{v.rung}</Pill>
      </div>
      <div style={{ fontSize: 11.5, color: C.dim, marginTop: 7, fontStyle: "italic" }}>skeptic: {v.skeptic}</div>
      {v.blockers.map((b) => (
        <div key={b} style={{ fontSize: 11.5, color: C.red, marginTop: 5 }}>· {b}</div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Console                                                             */
/* ------------------------------------------------------------------ */

export default function AgoraConsole({ fleet, ledger, verdicts, gaps }) {
  const data = {
    fleet: fleet || SAMPLE.fleet,
    ledger: ledger || SAMPLE.ledger,
    verdicts: verdicts || SAMPLE.verdicts,
    gaps: gaps || SAMPLE.gaps,
  };
  const [onlyRoutable, setOnlyRoutable] = useState(false);

  const rows = useMemo(
    () => data.ledger.filter((r) => !onlyRoutable || (r.evidence !== "deprecated" && r.evidence !== "declared")),
    [data.ledger, onlyRoutable]
  );

  const counts = useMemo(() => {
    const acc = { A: 0, B: 0, C: 0, D: 0 };
    data.fleet.forEach((a) => { acc[a.grade] = (acc[a.grade] || 0) + 1; });
    return acc;
  }, [data.fleet]);

  return (
    <div style={{ background: C.bg, color: C.text, minHeight: "100vh", padding: "28px 24px", fontFamily: "ui-sans-serif, -apple-system, 'Segoe UI', system-ui, sans-serif" }}>
      <div style={{ maxWidth: 1080, margin: "0 auto" }}>
        <header style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.22em", textTransform: "uppercase", color: C.goldDim }}>Control plane</div>
          <h1 style={{ margin: "6px 0 8px", fontSize: 30, fontWeight: 600, letterSpacing: "-0.01em" }}>AGORA</h1>
          <p style={{ margin: 0, color: C.dim, fontSize: 13.5, maxWidth: "72ch", lineHeight: 1.6 }}>
            Add any agent. Run any task. Know which ones you can trust unsupervised.
          </p>
          <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
            {Object.entries(counts).map(([g, n]) => n > 0 && (
              <Pill key={g} color={GRADE[g].color}>{`${n} × grade ${g} — ${GRADE[g].label}`}</Pill>
            ))}
          </div>
        </header>

        <Panel
          title="Capability Ledger"
          subtitle="Every bar is a 90% credible interval over independently verified outcomes. The tick is the mean. Width is what you do not know — and width, not the mean, is what buys autonomy."
          right={
            <label style={{ fontSize: 11.5, color: C.dim, display: "flex", gap: 7, alignItems: "center", cursor: "pointer" }}>
              <input type="checkbox" checked={onlyRoutable} onChange={(e) => setOnlyRoutable(e.target.checked)} />
              routable only
            </label>
          }
        >
          {rows.map((r, i) => <LedgerRow key={`${r.agent}-${r.capability}-${i}`} row={r} />)}
          {rows.length === 0 && <p style={{ color: C.dim, fontSize: 12.5 }}>Nothing routable yet. That is a normal cold start, not a fault.</p>}
        </Panel>

        <Panel title="Fleet" subtitle="Skills nest under the agent that owns them. An agent's grade is derived from the checks its domain pack can supply — it is never asserted by hand.">
          {data.fleet.map((a) => <AgentCard key={a.id} agent={a} />)}
        </Panel>

        <Panel title="Factory verdicts" subtitle="Charter review is mechanical first: overlap detection, retirement criteria, eval suite. The skeptic argues for the smallest thing that solves the problem.">
          {data.verdicts.map((v) => <VerdictRow key={v.charter} v={v} />)}
        </Panel>

        <Panel title="Gap log" subtitle="Every unroutable task is a capability-gap candidate. Nothing gets built because it seemed like a good idea — only because demand for it was measured here.">
          {data.gaps.map((g) => (
            <div key={g.capability} style={{ display: "flex", gap: 12, alignItems: "center", padding: "10px 0", borderTop: `1px solid ${C.line}` }}>
              <code style={{ fontSize: 12.5, color: C.text, minWidth: 160 }}>{g.capability}</code>
              <Pill color={g.count >= 10 ? C.amber : C.dim}>{`${g.count} in 14 days`}</Pill>
              <span style={{ fontSize: 11.5, color: C.dim }}>{g.reason}</span>
              {g.count >= 10 && <span style={{ marginLeft: "auto", fontSize: 11.5, color: C.amber }}>eligible for skill synthesis — eval first</span>}
            </div>
          ))}
        </Panel>

        <footer style={{ color: C.dim, fontSize: 11.5, textAlign: "center", padding: "8px 0 24px", lineHeight: 1.7 }}>
          Capability is a claim; proficiency is a measurement.<br />
          Self-reported success never enters this ledger.
        </footer>
      </div>
    </div>
  );
}
