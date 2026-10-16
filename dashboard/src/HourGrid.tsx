import { useState } from 'react';

const DAYS = ['Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon']; // the synthetic week starts Tue 1 Sep 2026
const STEPS = 7; // sequential ramp steps, defined in CSS as --seq-1..7

interface Props {
  label: string;
  values: number[]; // 168 hourly counts
  verb: string; // "made" | "received"
}

/** A 7-day by 24-hour grid of call counts: the shape of a number's week at a glance. */
export default function HourGrid({ label, values, verb }: Props) {
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const max = Math.max(1, ...values);
  const total = values.reduce((a, b) => a + b, 0);

  const step = (v: number) => (v === 0 ? 0 : Math.max(1, Math.ceil((Math.sqrt(v) / Math.sqrt(max)) * STEPS)));

  return (
    <figure className="hourgrid">
      <figcaption>
        <span>{label}</span>
        <span className="muted">{total.toLocaleString()} calls · busiest hour {max.toLocaleString()}</span>
      </figcaption>
      <div className="grid-wrap" onMouseLeave={() => setHover(null)}>
        <div className="grid-hours" aria-hidden="true">
          {[0, 6, 12, 18].map((h) => (
            <span key={h} style={{ gridColumn: h + 2 }}>{String(h).padStart(2, '0')}</span>
          ))}
        </div>
        <div className="grid" role="img" aria-label={`${label}: ${total} calls over 7 days, at most ${max} in one hour`}>
          {DAYS.map((d, day) => (
            <div className="grid-row" key={d}>
              <span className="grid-day">{d}</span>
              {values.slice(day * 24, day * 24 + 24).map((v, h) => {
                const i = day * 24 + h;
                return (
                  <span
                    key={h}
                    className={`cell s${step(v)}`}
                    onMouseEnter={(e) => {
                      const r = (e.target as HTMLElement).getBoundingClientRect();
                      const box = (e.currentTarget.closest('.grid-wrap') as HTMLElement).getBoundingClientRect();
                      setHover({ i, x: r.left - box.left + r.width / 2, y: r.top - box.top });
                    }}
                  />
                );
              })}
            </div>
          ))}
        </div>
        {hover && (
          <div className="tip" style={{ left: hover.x, top: hover.y }} role="status">
            <strong>{values[hover.i].toLocaleString()} calls {verb}</strong>
            <span>{DAYS[Math.floor(hover.i / 24)]} {String(hover.i % 24).padStart(2, '0')}:00–{String((hover.i % 24) + 1).padStart(2, '0')}:00</span>
          </div>
        )}
      </div>
    </figure>
  );
}
