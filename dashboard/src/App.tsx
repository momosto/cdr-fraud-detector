import { useEffect, useMemo, useState } from 'react';
import HourGrid from './HourGrid';
import { Benchmark, FEATURE_LABEL, FraudType, PERCENT_FEATURES, Summary, Suspect, Timelines, TYPE_LABEL } from './types';

const REPO = 'https://github.com/momosto/cdr-fraud-detector';
const METHODS = [
  { key: 'rules', name: 'Rules only', note: 'Known fraud patterns, written as thresholds' },
  { key: 'model', name: 'Model only', note: 'Isolation Forest, no labels, top 1.5% flagged' },
  { key: 'combined', name: 'Rules + model', note: 'Flag if either fires; rules rank first' },
] as const;
const VARIANTS = [
  ['wangiri', 'obvious'], ['wangiri', 'stealthy'],
  ['irsf', 'obvious'], ['irsf', 'stealthy'],
  ['simbox', 'obvious'], ['simbox', 'stealthy'],
] as const;

function load<T>(name: string): Promise<T> {
  return fetch(`/data/${name}.json`).then((r) => {
    if (!r.ok) throw new Error(`Couldn't load ${name}.json (${r.status}). Run the pipeline to generate it.`);
    return r.json();
  });
}

const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;

function TypeChip({ type }: { type: FraudType }) {
  return <span className={`chip t-${type}`}><i aria-hidden="true" />{TYPE_LABEL[type]}</span>;
}

function Truth({ truth, long = false }: { truth: string; long?: boolean }) {
  const fraud = !truth.startsWith('not fraud');
  // "irsf (stealthy)" -> "IRSF, stealthy"; "not fraud: diaspora caller" -> "Not fraud: diaspora caller"
  const match = truth.match(/^(\w+) \((\w+)\)$/);
  const text = fraud && match ? `${TYPE_LABEL[match[1] as FraudType] ?? match[1]}, ${match[2]}` : truth.replace('not fraud: ', '');
  return (
    <span className={fraud ? 'truth yes' : 'truth no'}>
      <span aria-hidden="true">{fraud ? '✓' : '✕'}</span> {fraud ? (long ? `Planted fraud: ${text}` : text) : `Not fraud: ${text}`}
    </span>
  );
}

export default function App() {
  const [summary, setSummary] = useState<Summary>();
  const [bench, setBench] = useState<Benchmark>();
  const [suspects, setSuspects] = useState<Suspect[]>([]);
  const [timelines, setTimelines] = useState<Timelines>();
  const [error, setError] = useState<string>();
  const [filter, setFilter] = useState<FraudType | 'all'>('all');
  const [flaggedOnly, setFlaggedOnly] = useState(true);
  const [selected, setSelected] = useState<string>();

  useEffect(() => {
    Promise.all([load<Summary>('summary'), load<Suspect[]>('suspects'), load<Timelines>('timelines'), load<Benchmark>('benchmark').catch(() => undefined)])
      .then(([s, sus, t, b]) => {
        setSummary(s);
        setSuspects(sus);
        setTimelines(t);
        setBench(b);
        setSelected(sus[0]?.number);
      })
      .catch((e) => setError(e.message));
  }, []);

  const rows = useMemo(
    () => suspects.filter((s) => (filter === 'all' || s.fraud_type === filter) && (!flaggedOnly || s.flagged)),
    [suspects, filter, flaggedOnly],
  );
  const current = suspects.find((s) => s.number === selected);

  if (error) return <main className="page"><p className="error">{error}</p></main>;
  if (!summary) return <main className="page"><p className="muted">Loading results…</p></main>;

  const fpCohorts = Object.entries(summary.methods.combined.false_positives_by_cohort).sort((a, b) => b[1] - a[1]);
  const series = current && timelines?.series[current.number];

  return (
    <main className="page">
      <header className="head">
        <p className="eyebrow">Synthetic network · {summary.days} days from {summary.period_start}</p>
        <h1>Telecom fraud in call records</h1>
        <p className="lede">
          Wangiri, IRSF and SIM box fraud were planted in {summary.cdrs.toLocaleString()} invented call records.
          Fraud rules and an unsupervised anomaly model try to find them; the scores below are measured against what was planted.
        </p>
        <p className="links"><a href={REPO}>Source code and method</a></p>
      </header>

      <section className="tiles" aria-label="Run summary">
        <div className="tile"><span>Call records</span><strong>{summary.cdrs.toLocaleString()}</strong></div>
        <div className="tile"><span>Numbers scored</span><strong>{summary.numbers_scored.toLocaleString()}</strong></div>
        <div className="tile"><span>Fraud numbers planted</span><strong>{summary.fraud_planted}</strong></div>
        <div className="tile"><span>Flagged by rules + model</span><strong>{summary.methods.combined.flagged}</strong></div>
      </section>

      <section className="panel">
        <h2>How well each method does</h2>
        <div className="methods">
          {METHODS.map((m) => {
            const r = summary.methods[m.key];
            return (
              <article key={m.key} className="method">
                <h3>{m.name}</h3>
                <p className="muted small">{m.note}</p>
                {(['precision', 'recall'] as const).map((k) => (
                  <div key={k} className="meter">
                    <div className="meter-label"><span>{k === 'precision' ? 'Precision' : 'Recall'}</span><strong>{pct(r[k])}</strong></div>
                    <div className="meter-track"><div className={`meter-fill ${k}`} style={{ width: pct(r[k]) }} /></div>
                  </div>
                ))}
                <p className="small">{r.true_positives} of {r.flagged} flags were fraud; {r.true_positives} of {summary.fraud_planted} fraud numbers found.</p>
              </article>
            );
          })}
        </div>
        <p className="small muted">
          Precision: share of flags that were really fraud. Recall: share of planted fraud that was flagged.
          {bench && <> Over {bench.seeds.length} differently seeded networks the combined ranking scores an average precision of {bench.combined_average_precision.mean.toFixed(2)} (range {bench.combined_average_precision.min.toFixed(2)}–{bench.combined_average_precision.max.toFixed(2)}).</>}
        </p>

        <div className="split">
          <div>
            <h3>Fraud caught, by variant</h3>
            <table className="matrix">
              <thead><tr><th scope="col">Planted fraud</th>{METHODS.map((m) => <th key={m.key} scope="col">{m.name}</th>)}</tr></thead>
              <tbody>
                {VARIANTS.map(([type, variant]) => (
                  <tr key={type + variant}>
                    <th scope="row"><TypeChip type={type} /> <span className="muted">{variant}</span></th>
                    {METHODS.map((m) => {
                      const v = summary.methods[m.key].by_variant.find((x) => x.fraud_type === type && x.variant === variant)!;
                      const share = v.caught / v.planted;
                      return <td key={m.key} className={share === 1 ? 'all' : share === 0 ? 'none' : 'some'}>{v.caught} / {v.planted}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="small muted">Stealthy variants were built to stay under the rule thresholds. The model catches the stealthy Wangiri and IRSF numbers; no method catches the low-volume SIM box.</p>
          </div>
          <div>
            <h3>Who the false alarms were</h3>
            <ul className="fp">
              {fpCohorts.map(([cohort, n]) => (
                <li key={cohort}><span>{cohort[0].toUpperCase() + cohort.slice(1)}</span><strong>{n}</strong></li>
              ))}
            </ul>
            <p className="small muted">All from the model; the rules raised none. Wangiri victims who rang back a premium number look like IRSF, and are worth a call anyway.</p>
          </div>
        </div>
      </section>

      <section className="panel suspects-panel">
        <div className="suspects-head">
          <h2>Suspects</h2>
          <div className="filters" role="group" aria-label="Filter suspects">
            {(['all', 'wangiri', 'irsf', 'simbox', 'unusual'] as const).map((t) => (
              <button key={t} className={filter === t ? 'on' : ''} aria-pressed={filter === t} onClick={() => setFilter(t)}>
                {t === 'all' ? 'All' : TYPE_LABEL[t]}
              </button>
            ))}
            <label className="toggle"><input type="checkbox" checked={flaggedOnly} onChange={(e) => setFlaggedOnly(e.target.checked)} /> Flagged only</label>
          </div>
        </div>

        <div className="workspace">
          <div className="table-wrap">
            <table className="suspects">
              <thead><tr><th scope="col">Risk</th><th scope="col">Number</th><th scope="col">Pattern</th><th scope="col">Found by</th><th scope="col">What it really is</th></tr></thead>
              <tbody>
                {rows.map((s) => (
                  <tr key={s.number} className={s.number === selected ? 'sel' : ''} onClick={() => setSelected(s.number)}>
                    <td className="num"><span className="risk"><span style={{ width: pct(s.risk) }} /></span>{s.risk.toFixed(2)}</td>
                    <td><button className="link" onClick={() => setSelected(s.number)} aria-label={`Show details for ${s.number}`}>{s.number}</button></td>
                    <td><TypeChip type={s.fraud_type} /></td>
                    <td className="by">{[s.rule_hit && 'Rule', s.model_flag && 'Model'].filter(Boolean).join(' + ') || '—'}</td>
                    <td><Truth truth={s.truth} /></td>
                  </tr>
                ))}
                {rows.length === 0 && <tr><td colSpan={5} className="muted">No suspects match these filters. Choose another pattern or include unflagged numbers.</td></tr>}
              </tbody>
            </table>
          </div>

          {current && (
            <aside className="detail" aria-live="polite">
              <p className="eyebrow">Selected number</p>
              <h3 className="detail-number">{current.number}</h3>
              <p><TypeChip type={current.fraud_type} /> <span className="muted">risk {current.risk.toFixed(2)}</span></p>
              <p><Truth truth={current.truth} long /></p>
              <h4>Why it was flagged</h4>
              <ul className="reasons">{current.reasons.length ? current.reasons.map((r) => <li key={r}>{r}</li>) : <li className="muted">Not flagged: below the model threshold and no rule fired.</li>}</ul>
              {series && (
                <>
                  <HourGrid label="Calls made, by hour" values={series.made} verb="made" />
                  <HourGrid label="Calls received, by hour" values={series.received} verb="received" />
                </>
              )}
              <h4>Behaviour over the week</h4>
              <dl className="features">
                {Object.entries(current.features).map(([k, v]) => (
                  <div key={k}><dt>{FEATURE_LABEL[k] ?? k}</dt><dd>{PERCENT_FEATURES.has(k) ? pct(v) : v.toLocaleString(undefined, { maximumFractionDigits: 1 })}</dd></div>
                ))}
              </dl>
            </aside>
          )}
        </div>
      </section>

      <footer className="foot muted small">
        Every number, call and handset here is invented by a seeded generator. <a href={REPO}>Rules, model, evaluation and API on GitHub</a>.
      </footer>
    </main>
  );
}
