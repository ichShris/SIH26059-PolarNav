import { useEffect, useRef, useState } from 'react'
import { api, decodeU8 } from '../api'
import type { Meta, RoutePlan, SarResult, Scenario } from '../types'
import GroupedBars from './GroupedBars'

// ------------------------------------------------------------------ hazards
export function HazardsPanel({ scenario, plan, selected, onSelect }: {
  scenario: Scenario | null
  plan: RoutePlan | null
  selected: string | null
  onSelect: (id: string) => void
}) {
  if (!scenario) return <p className="muted">Loading…</p>
  const clearance = new Map((plan?.optimal?.berg_clearance ?? []).map((c) => [c.id, c]))
  const bergs = [...scenario.bergs.bergs].sort((a, b) => {
    const ca = clearance.get(a.id)?.cpa_km ?? 1e9
    const cb = clearance.get(b.id)?.cpa_km ?? 1e9
    return ca !== cb ? ca - cb : b.length_m - a.length_m
  })
  const s = scenario.bergs.summary.mean_err_72h_km
  const live = scenario.mode === 'live'
  const rep = live ? scenario.sources?.icebergs.report : null
  return (
    <div>
      {live ? (
        <p className="muted small">
          {bergs.length} real icebergs from the US National Ice Center (report {rep}). Each is carried forward{' '}
          {Math.round(scenario.bergs.dead_reckoned_h ?? 0)} h to the sea-ice analysis time on ECMWF winds and live currents
          (dead reckoning), then forecast 72 h with a {scenario.bergs.members}-member hybrid ensemble. Ellipses are 2σ. No
          verifying truth exists yet.
        </p>
      ) : (
        <p className="muted small">
          {bergs.length} icebergs tracked from SAR. 72 h forecasts: {scenario.bergs.members}-member ensemble (forecast-wind members,
          drag perturbations, SAR geolocation error) with hybrid physics-AI drift. Ellipses are 2σ.
        </p>
      )}
      {s.hybrid !== null && s.physics !== null && (
        <div className="stat-row">
          <div className="stat">
            <div className="stat-num">{s.hybrid.toFixed(1)} km</div>
            <div className="muted small">hybrid 72 h error today</div>
          </div>
          <div className="stat">
            <div className="stat-num">{s.physics.toFixed(1)} km</div>
            <div className="muted small">physics-only</div>
          </div>
        </div>
      )}
      <ul className="berg-list">
        {bergs.map((b) => {
          const c = clearance.get(b.id)
          const f = b.forecast
          const km = Math.hypot(f[f.length - 1][0] - f[0][0], f[f.length - 1][1] - f[0][1])
          const kn = km / 72 / 1.852
          const close = c && c.cpa_km < 37
          return (
            <li key={b.id} className={b.id === selected ? 'sel' : ''} onClick={() => onSelect(b.id)}>
              <div className="berg-head">
                <span className="berg-id">▲ {b.id}</span>
                {c && (
                  <span className={`badge ${close ? 'bad' : ''}`}>
                    CPA {(c.cpa_km / 1.852).toFixed(1)} nm @ +{c.t_h.toFixed(0)}h
                  </span>
                )}
              </div>
              <div className="muted small">
                {(b.length_m / 1000).toFixed(1)} × {(b.width_m / 1000).toFixed(1)} km · draft {b.draft_m.toFixed(0)} m · drift {kn.toFixed(2)} kn
                {b.grounded ? ' · grounded' : ''}
              </div>
              {b.id === selected && b.reported && (
                <div className="small muted">
                  Reported {b.reported.date} at {b.reported.lat.toFixed(2)}°, {b.reported.lon.toFixed(2)}° · dead-reckoned{' '}
                  {(b.dead_reckoned_km ?? 0).toFixed(1)} km since · keel depth assumed ({b.draft_m.toFixed(0)} m)
                </div>
              )}
              {b.id === selected && b.error_72h_km && (
                <div className="small verify">
                  Verification (truth known in hindcast): 72 h error hybrid <b>{b.error_72h_km.hybrid.toFixed(1)} km</b>, physics-only{' '}
                  <b>{b.error_72h_km.physics.toFixed(1)} km</b>
                  {c && c.cpa_truth_km !== null ? <>; true CPA to route {(c.cpa_truth_km / 1.852).toFixed(1)} nm</> : null}
                </div>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

// ------------------------------------------------------------------ AI models
export function ModelsPanel({ meta, scenario, onRefreshLive, refreshing }: {
  meta: Meta
  scenario: Scenario | null
  onRefreshLive: () => void
  refreshing: boolean
}) {
  const m = meta.metrics.sic
  const d = meta.metrics.drift
  const leads = ['+24h', '+48h', '+72h']
  const src = scenario?.mode === 'live' ? scenario.sources : null
  return (
    <div>
      {src && (
        <div className="sources">
          <h3>Live data sources</h3>
          <ul>
            <li>
              <b>Sea ice</b> {src.sea_ice.name} · valid {src.sea_ice.valid}
            </li>
            <li>
              <b>Wind</b> {src.wind.name} · {src.wind.from.replace('T', ' ')} → {src.wind.to.replace('T', ' ')} UTC
            </li>
            <li>
              <b>Currents</b> {src.currents.name} · {src.currents.points} live points
            </li>
            <li>
              <b>Icebergs</b> {src.icebergs.name} · {src.icebergs.count} bergs, report {src.icebergs.report}
            </li>
          </ul>
          <p className="muted small">Still modelled: {src.modelled.join('; ')}.</p>
          <div className="row gap">
            <button onClick={onRefreshLive} disabled={refreshing}>
              {refreshing ? 'Refreshing…' : '↻ Refresh live data'}
            </button>
            <span className="muted small">fetched {src.fetched_at.replace('T', ' ').slice(0, 16)} UTC · auto-refresh hourly</span>
          </div>
        </div>
      )}
      <h3>Sea-ice concentration · ConvLSTM</h3>
      {m ? (
        <>
          <p className="muted small">
            {m.params.toLocaleString()} parameters · {m.iterations} iterations ({m.train_minutes} min CPU). Held-out test year{' '}
            {m.splits.test[0].slice(0, 4)} ({m.splits.test[0]} → {m.splits.test[1]}); never seen in training.
          </p>
          <GroupedBars
            title="RMSE in the active ice zone"
            unit="% SIC"
            groups={leads}
            series={[
              { name: 'ConvLSTM', values: m.test.model.map((x) => x.rmse) },
              { name: 'Persistence', values: m.test.persistence.map((x) => x.rmse) },
              { name: 'Climatology', values: m.test.climatology.map((x) => x.rmse) },
            ]}
          />
          <GroupedBars
            title="Integrated ice-edge error (15% edge)"
            unit="10³ km²"
            groups={leads}
            series={[
              { name: 'ConvLSTM', values: m.test.model.map((x) => x.iiee) },
              { name: 'Persistence', values: m.test.persistence.map((x) => x.iiee) },
              { name: 'Climatology', values: m.test.climatology.map((x) => x.iiee) },
            ]}
          />
        </>
      ) : (
        <p className="warn">Model metrics not found — run backend/train.py.</p>
      )}
      {scenario && scenario.live_check.per_lead.length > 0 && (
        <>
          <h3>Real-time forecast check</h3>
          <p className="muted small">
            Works live: forecasts issued 1–3 days ago that are valid today, scored against today's satellite analysis. No future data needed.
            {scenario.live_check.note ? ` ${scenario.live_check.note}` : ''}
          </p>
          <table className="cmp">
            <thead>
              <tr>
                <th>Issued → today</th>
                <th>ConvLSTM</th>
                <th>Persistence</th>
              </tr>
            </thead>
            <tbody>
              {scenario.live_check.per_lead.map((r) => (
                <tr key={r.lead_h}>
                  <td>
                    {r.issued.slice(5)} (+{r.lead_h}h)
                  </td>
                  <td className="mono">{r.model.rmse.toFixed(2)}%</td>
                  <td className="mono">{r.persistence.rmse.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {scenario && scenario.skill.length > 0 && (
        <>
        <h3>Hindcast check (truth known only after the fact)</h3>
        <table className="cmp">
          <thead>
            <tr>
              <th>Issued {scenario.date.slice(5)}</th>
              <th>ConvLSTM</th>
              <th>Persistence</th>
            </tr>
          </thead>
          <tbody>
            {scenario.skill.map((s) => (
              <tr key={s.lead_h}>
                <td>RMSE +{s.lead_h}h</td>
                <td className="mono">{s.model.rmse.toFixed(2)}%</td>
                <td className="mono">{s.persistence.rmse.toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
        </>
      )}

      <h3>Iceberg drift · hybrid physics-AI</h3>
      {d ? (
        <>
          <p className="muted small">
            {d.n_tracks} held-out 72 h tracks. Physics: air/water drag, Coriolis, sea-surface slope, pack-ice capture. AI: complex-valued ridge
            correction learned from historical tracks (captures wave drift and drag bias).
          </p>
          <GroupedBars
            title="72 h position error"
            unit="km"
            groups={['Mean', 'Median']}
            series={[
              { name: 'Hybrid', values: [d.hybrid_km.mean, d.hybrid_km.median] },
              { name: 'Persistence', values: [d.persistence_km.mean, d.persistence_km.median] },
              { name: 'Physics only', values: [d.physics_km.mean, d.physics_km.median] },
            ]}
          />
        </>
      ) : (
        <p className="warn">Drift metrics not found — run backend/train.py.</p>
      )}

      <h3>Routing · season benchmark</h3>
      {meta.metrics.routes ? (
        <>
          <p className="muted small">
            {meta.metrics.routes.feasible} ice passages ({meta.metrics.routes.ice_class}) to Maitri and Bharati, every 4th day of the held-out
            2025 sailing windows, PolarNav vs the shortest route on today's ice chart. Same ship model for both.
          </p>
          <div className="stat-row">
            <div className="stat">
              <div className="stat-num">{meta.metrics.routes.fuel_saving_pct.median.toFixed(1)}%</div>
              <div className="muted small">median fuel saving</div>
            </div>
            <div className="stat">
              <div className="stat-num">
                {meta.metrics.routes.fuel_saving_pct.p10.toFixed(0)}–{meta.metrics.routes.fuel_saving_pct.p90.toFixed(0)}%
              </div>
              <div className="muted small">10th–90th percentile</div>
            </div>
          </div>
          <div className="stat-row">
            <div className="stat">
              <div className="stat-num">{meta.metrics.routes.mean_time_saved_h.toFixed(1)} h</div>
              <div className="muted small">mean passage time saved</div>
            </div>
            <div className="stat">
              <div className="stat-num">
                {meta.metrics.routes.voyages_passing_berg_within_10nm.polarnav} vs {meta.metrics.routes.voyages_passing_berg_within_10nm.conventional}
              </div>
              <div className="muted small">voyages passing a berg &lt; 10 nm (verified)</div>
            </div>
          </div>
        </>
      ) : (
        <p className="muted small">Run backend/benchmark_routes.py to populate.</p>
      )}

      <h3>SAR detection · CFAR benchmark</h3>
      <p className="muted small">
        {meta.metrics.sar.scenes} synthetic scenes at P<sub>fa</sub> {meta.metrics.sar.pfa.toExponential(0)}: recall{' '}
        <b>{(meta.metrics.sar.recall * 100).toFixed(1)}%</b>, precision <b>{(meta.metrics.sar.precision * 100).toFixed(1)}%</b>. Synthetic scenes
        are easier than real Sentinel-1 (no sea-state extremes, bergy bits or ships); expect lower scores on real data.
      </p>
    </div>
  )
}

// ------------------------------------------------------------------ SAR
export function SarPanel() {
  const [seed, setSeed] = useState(7)
  const [pfa, setPfa] = useState(1e-5)
  const [res, setRes] = useState<SarResult | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let live = true
    api
      .sar(seed, pfa)
      .then((r) => live && (setRes(r.data), setErr(null)))
      .catch((e) => live && setErr(String(e)))
    return () => {
      live = false
    }
  }, [seed, pfa])

  useEffect(() => {
    if (!res || !ref.current) return
    const n = res.size
    const cv = ref.current
    cv.width = n
    cv.height = n
    const ctx = cv.getContext('2d')!
    const px = decodeU8(res.image_b64)
    const img = ctx.createImageData(n, n)
    for (let i = 0; i < n * n; i++) {
      img.data[i * 4] = px[i]
      img.data[i * 4 + 1] = px[i]
      img.data[i * 4 + 2] = px[i]
      img.data[i * 4 + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
    ctx.lineWidth = 1
    ctx.setLineDash([2, 2])
    ctx.strokeStyle = '#ffd166'
    for (const t of res.truth) ctx.strokeRect(t.x0 - 2.5, t.y0 - 2.5, t.x1 - t.x0 + 5, t.y1 - t.y0 + 5)
    ctx.setLineDash([])
    for (const d of res.detections) {
      ctx.strokeStyle = d.match ? '#3ec28f' : '#e63946'
      ctx.strokeRect(d.x0 - 1.5, d.y0 - 1.5, d.x1 - d.x0 + 3, d.y1 - d.y0 + 3)
    }
  }, [res])

  return (
    <div>
      <p className="muted small">
        Sentinel-1-like scene (40 m pixels, 4-look speckle, wind-modulated sea clutter, pack ice with floes). Cell-averaging CFAR proposes
        iceberg targets; in the full system these chips feed the YOLOv8 classifier.
      </p>
      <div className="row gap">
        <button onClick={() => setSeed((s) => s + 1)}>Next scene</button>
        <label className="small">
          P<sub>fa</sub>{' '}
          <select value={pfa} onChange={(e) => setPfa(+e.target.value)}>
            {[1e-3, 1e-4, 1e-5, 1e-6, 1e-7].map((v) => (
              <option key={v} value={v}>
                {v.toExponential(0)}
              </option>
            ))}
          </select>
        </label>
        <span className="muted small">scene #{seed}</span>
      </div>
      {err && <p className="warn">{err}</p>}
      <canvas ref={ref} className="sar" aria-label="SAR scene with detections" />
      <div className="legend-row small">
        <span><i style={{ background: '#3ec28f' }} />detection (true)</span>
        <span><i style={{ background: '#e63946' }} />false alarm</span>
        <span><i style={{ background: '#ffd166' }} />ground truth</span>
      </div>
      {res && (
        <div className="stat-row">
          <div className="stat">
            <div className="stat-num">{(res.scores.recall * 100).toFixed(0)}%</div>
            <div className="muted small">recall ({res.scores.n_truth} bergs)</div>
          </div>
          <div className="stat">
            <div className="stat-num">{(res.scores.precision * 100).toFixed(0)}%</div>
            <div className="muted small">precision ({res.scores.n_detections} detections)</div>
          </div>
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ edge sync
export function SyncPanel({ plan, offline, savedAt, forceOffline, setForceOffline, onDownload }: {
  plan: RoutePlan | null
  offline: boolean
  savedAt?: string
  forceOffline: boolean
  setForceOffline: (v: boolean) => void
  onDownload: () => void
}) {
  const s = plan?.sync
  const kb = (b: number) => (b / 1024).toFixed(1) + ' KB'
  return (
    <div>
      <p className="muted small">
        Heavy AI runs ashore. The ship receives a compressed bundle — sea-ice analysis + 3 forecast days, POLARIS RIO grids, iceberg forecast
        tracks and the route — and this dashboard keeps working from its local cache when the link drops below 60°S.
      </p>
      <div className={`link-state ${offline ? 'down' : 'up'}`}>
        <b>{offline ? 'LINK DOWN — running on cached bundle' : 'LINK UP — live shore data'}</b>
        {offline && savedAt && <div className="small">bundle synced {new Date(savedAt).toUTCString()}</div>}
      </div>
      <label className="check">
        <input type="checkbox" checked={forceOffline} onChange={(e) => setForceOffline(e.target.checked)} />
        Simulate satellite-link blackout
      </label>
      {s ? (
        <>
          <div className="stat-row">
            <div className="stat">
              <div className="stat-num">{kb(s.packed_bytes)}</div>
              <div className="muted small">packed bundle</div>
            </div>
            <div className="stat">
              <div className="stat-num">{s.ratio}×</div>
              <div className="muted small">smaller than raw ({kb(s.raw_bytes)})</div>
            </div>
          </div>
          <table className="cmp">
            <thead>
              <tr>
                <th>Link</th>
                <th>Raw</th>
                <th>Packed</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(s.transfer_s).map(([k, v]) => (
                <tr key={k}>
                  <td className="small">{k}</td>
                  <td className="mono">{fmtS(v.raw)}</td>
                  <td className="mono">{fmtS(v.packed)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <button onClick={onDownload} disabled={offline}>
            Download ship bundle (.pnb)
          </button>
        </>
      ) : (
        <p className="muted">Compute a route to build a bundle.</p>
      )}
    </div>
  )
}

const fmtS = (s: number) => (s < 90 ? `${s.toFixed(1)} s` : `${(s / 60).toFixed(1)} min`)
