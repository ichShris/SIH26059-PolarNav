import { LEVEL_COLORS } from '../colors'
import { fmtLat, fmtLon } from '../geo'
import { activeOption, routeOptions } from '../routes'
import type { Meta, RouteEval, RouteForm, RoutePlan } from '../types'

interface Props {
  meta: Meta
  form: RouteForm
  setForm: (f: RouteForm) => void
  plan: RoutePlan | null
  busy: boolean
  stale: boolean
  onCompute: () => void
  pickMode: 'origin' | 'dest' | null
  setPickMode: (m: 'origin' | 'dest' | null) => void
  onSeek: (tH: number) => void
  activeRoute: string
  setActiveRoute: (k: string) => void
}

const fmtH = (h: number) => (h < 48 ? `${h.toFixed(0)} h` : `${(h / 24).toFixed(1)} d`)

export default function VoyagePanel(p: Props) {
  const { meta, form, setForm, plan } = p
  const places = Object.entries(meta.places)
  const placeSelect = (which: 'origin' | 'dest') => {
    const v = form[which]
    const ll = which === 'origin' ? form.originLL : form.destLL
    return (
      <div className="field">
        <label htmlFor={`sel-${which}`}>{which === 'origin' ? 'Departure' : 'Destination'}</label>
        <div className="row">
          <select id={`sel-${which}`} value={v} onChange={(e) => setForm({ ...form, [which]: e.target.value })}>
            {places.map(([k, pl]) => (
              <option key={k} value={k}>
                {pl.name}
              </option>
            ))}
            {ll && (
              <option value="custom">
                Map point {fmtLat(ll.lat)} {fmtLon(ll.lon)}
              </option>
            )}
          </select>
          <button
            className={`icon-btn ${p.pickMode === which ? 'active' : ''}`}
            title="Pick on map"
            aria-label={`Pick ${which} on map`}
            onClick={() => p.setPickMode(p.pickMode === which ? null : which)}
          >
            ⌖
          </button>
        </div>
      </div>
    )
  }
  const slider = (id: keyof RouteForm, label: string, value: string, min: number, max: number, step: number, title: string) => (
    <div className="field">
      <label htmlFor={id} title={title}>
        {label} <span className="mono">{value}</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={form[id] as number}
        onChange={(e) => setForm({ ...form, [id]: +e.target.value })}
      />
    </div>
  )

  const opts = routeOptions(plan)
  const active = activeOption(plan, p.activeRoute)
  // compare the selected route (or the fuel-optimal one when the conventional route is selected) with the conventional route
  const subject = active && active.key !== 'baseline' ? active : opts[0]
  const o = subject?.ev.summary
  const b = plan?.baseline?.summary
  const pct = o && b ? (100 * (b.fuel_t - o.fuel_t)) / b.fuel_t : null
  const notes = plan ? whyUnchanged(plan, form) : []

  return (
    <aside className="panel left">
      <section>
        <h2>Voyage plan</h2>
        {placeSelect('origin')}
        {placeSelect('dest')}
        {slider('cruise_kn', 'Cruise speed', `${form.cruise_kn} kn`, 6, 16, 0.5, 'Open-water speed; in ice the ship is limited by attainable speed')}
        {slider('w_time', 'Time value', `${form.w_time.toFixed(1)} t/h`, 0, 3, 0.1, 'Fuel-equivalent value of one hour of ship time (charter + science days)')}
        {slider('w_risk', 'Risk aversion', form.w_risk.toFixed(2), 0, 0.3, 0.01, 'Penalty for sailing in ice with a POLARIS RIO below 10')}
        {slider('w_ice', 'Ice-exposure aversion', form.w_ice.toFixed(2), 0, 1, 0.05, 'Penalty per km weighted by sea-ice concentration')}
        {slider('berg_margin_nm', 'Iceberg clearance', `${form.berg_margin_nm} nm`, 2, 20, 1, 'Minimum distance kept beyond each berg and its 2-sigma drift uncertainty')}
        <label className="check">
          <input type="checkbox" checked={form.avoid_bergs} onChange={(e) => setForm({ ...form, avoid_bergs: e.target.checked })} />
          Avoid forecast iceberg zones (72 h ensemble)
        </label>
        <div className="row compute-row">
          <button className="primary" onClick={p.onCompute} disabled={p.busy}>
            {p.busy ? 'Optimising…' : 'Recompute routes'}
          </button>
        </div>
        <p className="muted small status-line">
          {p.busy ? 'Updating routes for the new settings…' : p.stale ? 'Settings changed — updating…' : 'Routes update automatically when you change a setting.'}
        </p>
      </section>

      {plan && (
        <section>
          <h2>Route options</h2>
          {plan.warnings.map((w) => (
            <p key={w} className="warn">
              ⚠ {w}
            </p>
          ))}
          <ul className="route-opts">
            {opts.map((r) => (
              <li key={r.key}>
                <button className={`route-opt ${active?.key === r.key ? 'on' : ''}`} onClick={() => p.setActiveRoute(r.key)} aria-pressed={active?.key === r.key}>
                  <i className={`sw ${r.dashed ? 'dash' : ''}`} style={r.dashed ? undefined : { background: r.color }} />
                  <span className="ro-name">{r.label}</span>
                  <span className="ro-stats mono">
                    {r.ev.summary.fuel_t.toFixed(0)} t · {fmtH(r.ev.summary.hours)} · {r.ev.summary.distance_nm.toFixed(0)} nm
                  </span>
                  {r.sameAsOptimal && <span className="ro-note">same track as recommended</span>}
                </button>
              </li>
            ))}
          </ul>

          {o && b && pct !== null && (
            <div className="hero">
              <div>
                <div className="hero-num" style={{ color: pct >= 0 ? 'var(--good)' : 'var(--warn)' }}>
                  {pct > 0 ? '−' : '+'}
                  {Math.abs(pct).toFixed(1)}%
                </div>
                <div className="muted">{subject?.label.toLowerCase()} fuel vs conventional</div>
              </div>
              <div className="hero-side">
                <div>
                  <b>{Math.abs(b.fuel_t - o.fuel_t).toFixed(1)} t</b> MGO {b.fuel_t >= o.fuel_t ? 'saved' : 'extra'}
                </div>
                <div>
                  <b>{Math.abs(b.co2_t - o.co2_t).toFixed(1)} t</b> CO₂ {b.co2_t >= o.co2_t ? 'avoided' : 'extra'}
                </div>
                <div>
                  <b>
                    {Math.abs(o.hours - b.hours).toFixed(1)} h {o.hours <= b.hours ? 'faster' : 'longer'}
                  </b>
                </div>
              </div>
            </div>
          )}
          <table className="cmp">
            <thead>
              <tr>
                <th />
                <th>
                  <i className="sw" style={{ background: subject?.color }} />
                  {subject?.label}
                </th>
                <th>
                  <i className="sw dash" />
                  Conventional
                </th>
              </tr>
            </thead>
            <tbody>
              {row('Distance', o, b, (s) => `${s.distance_nm.toFixed(0)} nm`)}
              {row('Passage', o, b, (s) => fmtH(s.hours))}
              {row('Fuel (MGO)', o, b, (s) => `${s.fuel_t.toFixed(1)} t`)}
              {row('CO₂', o, b, (s) => `${s.co2_t.toFixed(0)} t`)}
              {row('Ice passage', o, b, (s) => `${s.ice_nm.toFixed(0)} nm`)}
              {row('Elevated-risk ice', o, b, (s) => `${s.elevated_nm.toFixed(0)} nm`)}
              {row('Min RIO', o, b, (s) => s.min_rio.toFixed(1))}
              {row('Closest iceberg (fcst)', o, b, (s) => (s.min_berg_cpa_km == null ? '—' : `${(s.min_berg_cpa_km / 1.852).toFixed(1)} nm`))}
              {row('Bergs < 10 nm', o, b, (s) => String(s.bergs_within_10nm))}
            </tbody>
          </table>
          {notes.length > 0 && (
            <details className="why" open>
              <summary>Why some settings don't move the track today</summary>
              <ul>
                {notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </details>
          )}
          <p className="muted small">
            Solved in {plan.compute_s}s · {plan.request.ice_class as string} · worst level: {o?.worst_level}
            {o?.beyond_forecast_horizon ? ' · later legs use the +72 h ice field' : ''}
          </p>
        </section>
      )}

      {active && (
        <section>
          <h2>Waypoints · {active.label}</h2>
          <WaypointTable route={active.ev} onSeek={p.onSeek} />
        </section>
      )}
    </aside>
  )
}

/** Plain-language reasons a control has no effect on today's track, derived from the plan itself. */
function whyUnchanged(plan: RoutePlan, form: RouteForm): string[] {
  const o = plan.optimal?.summary
  if (!o) return []
  const out: string[] = []
  if (o.min_rio >= 10) out.push(`Risk aversion acts on ice with RIO below 10; the lowest RIO on this route is ${o.min_rio.toFixed(1)}, so it has little or no effect here.`)
  const cpaNm = o.min_berg_cpa_km == null ? null : o.min_berg_cpa_km / 1.852
  if (form.avoid_bergs && cpaNm !== null && cpaNm > form.berg_margin_nm + 30)
    out.push(`No forecast iceberg comes near this corridor (closest ${cpaNm.toFixed(0)} nm), so the iceberg settings don't change the track.`)
  const iceSpeeds = plan.optimal!.points.filter((q) => q.sic > 0.15).map((q) => q.speed_kn)
  if (iceSpeeds.length) {
    const mean = iceSpeeds.reduce((a, c) => a + c, 0) / iceSpeeds.length
    if (mean < form.cruise_kn - 0.5)
      out.push(`Cruise speed sets open-water speed. In the pack the ship averages ${mean.toFixed(1)} kn (limited by the ice), so speed changes time and fuel, not the track.`)
  }
  if (plan.alternatives.find((a) => a.key === 'fastest')?.same_track_as_optimal)
    out.push('Fastest and recommended coincide today: in pack ice both are driven by ice resistance, so the least-resistance track wins on both counts.')
  return out
}

function row(label: string, o: RouteEval['summary'] | undefined, b: RouteEval['summary'] | undefined, f: (s: RouteEval['summary']) => string) {
  return (
    <tr>
      <td>{label}</td>
      <td className="mono">{o ? f(o) : '—'}</td>
      <td className="mono">{b ? f(b) : '—'}</td>
    </tr>
  )
}

function WaypointTable({ route, onSeek }: { route: RouteEval; onSeek: (t: number) => void }) {
  return (
    <div className="wp-scroll">
      <table className="wp">
        <thead>
          <tr>
            <th>WP</th>
            <th>Position</th>
            <th>ETA</th>
            <th>Crs</th>
            <th>SIC</th>
            <th>kn</th>
          </tr>
        </thead>
        <tbody>
          {route.waypoints.map((w) => {
            const lvl = w.rio >= 0 ? 0 : 1
            return (
              <tr key={w.name} onClick={() => onSeek(w.t_h)} title="Jump timeline to this waypoint">
                <td>{w.name}</td>
                <td className="mono small">
                  {fmtLat(w.lat)}
                  <br />
                  {fmtLon(w.lon)}
                </td>
                <td className="mono">+{w.t_h.toFixed(0)}h</td>
                <td className="mono">{w.course == null ? '—' : `${w.course.toFixed(0).padStart(3, '0')}°`}</td>
                <td className="mono">
                  <i className="dot" style={{ background: LEVEL_COLORS[lvl] }} />
                  {Math.round(w.sic * 100)}%
                </td>
                <td className="mono">{w.speed_kn.toFixed(1)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
