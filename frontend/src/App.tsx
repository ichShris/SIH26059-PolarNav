import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError, setForceOffline as apiSetForceOffline } from './api'
import { LEVEL_COLORS } from './colors'
import GlobeView from './components/GlobeView'
import MapView, { type Overlays, type RasterLayer } from './components/MapView'
import { HazardsPanel, ModelsPanel, SarPanel, SyncPanel } from './components/RightPanels'
import VoyagePanel from './components/VoyagePanel'
import { activeOption, routeOptions } from './routes'
import type { Coastline, DataMode, GlobeLand, Meta, PolarisGrid, RouteForm, RoutePlan, Scenario } from './types'

type Tab = 'hazards' | 'models' | 'sar' | 'sync'

const LAYERS: { id: RasterLayer; label: string }[] = [
  { id: 'sic', label: 'Sea-ice conc.' },
  { id: 'thickness', label: 'Thickness' },
  { id: 'polaris', label: 'POLARIS risk' },
  { id: 'error', label: 'Forecast error' },
  { id: 'none', label: 'Off' },
]

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [coast, setCoast] = useState<Coastline | null>(null)
  const [globeLand, setGlobeLand] = useState<GlobeLand | null>(null)
  const [view, setView] = useState<'chart' | 'globe'>('globe')
  const [date, setDate] = useState<string>('')
  const [mode, setMode] = useState<DataMode>('sim')
  const [refreshing, setRefreshing] = useState(false)
  const [liveTick, setLiveTick] = useState(0)
  const [iceClass, setIceClass] = useState('PC5')
  const [scenario, setScenario] = useState<Scenario | null>(null)
  const [polaris, setPolaris] = useState<PolarisGrid | null>(null)
  const [plan, setPlan] = useState<RoutePlan | null>(null)
  const [form, setForm] = useState<RouteForm>({ origin: 'vessel', dest: 'maitri', cruise_kn: 12, w_time: 0.4, w_risk: 0.02, avoid_bergs: true, w_ice: 0, berg_margin_nm: 5 })
  const [activeRoute, setActiveRoute] = useState('optimal')
  const [planKey, setPlanKey] = useState('')
  const reqSeq = useRef(0)
  const [layer, setLayer] = useState<RasterLayer>('sic')
  const [overlays, setOverlays] = useState<Overlays>({ wind: false, bergs: true, routes: true, verify: false, graticule: true })
  const [tH, setTH] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [selectedBerg, setSelectedBerg] = useState<string | null>(null)
  const [pickMode, setPickMode] = useState<'origin' | 'dest' | null>(null)
  const [focus, setFocus] = useState<{ x: number; y: number; seq: number } | null>(null)
  const [tab, setTab] = useState<Tab>('hazards')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [offline, setOffline] = useState(false)
  const [savedAt, setSavedAt] = useState<string | undefined>()
  const [forceOffline, setForceOffline] = useState(false)
  const [now, setNow] = useState(new Date())

  const note = useCallback((r: { offline: boolean; savedAt?: string }) => {
    setOffline(r.offline)
    if (r.offline) setSavedAt(r.savedAt)
  }, [])

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    Promise.all([api.meta(), api.coastline()])
      .then(([m, c]) => {
        setMeta(m.data)
        setCoast(c.data)
        setDate(m.data.dates.default)
        note(m)
      })
      .catch((e) => setError(`Cannot reach the PolarNav service and no cached bundle is available (${e}).`))
    api.globeLand().then((r) => setGlobeLand(r.data)).catch(() => setGlobeLand(null))
  }, [note])

  useEffect(() => {
    if (!date) return
    let live = true
    setLoading(true)
    api
      .scenario(date, mode)
      .then((r) => {
        if (!live) return
        setScenario(r.data)
        note(r)
        setError(null)
      })
      .catch((e) => live && setError(e instanceof ApiError ? e.message : String(e)))
      .finally(() => live && setLoading(false))
    return () => {
      live = false
    }
  }, [date, mode, liveTick, note])

  useEffect(() => {
    if (!date) return
    api.polaris(date, iceClass, mode).then((r) => setPolaris(r.data)).catch(() => setPolaris(null))
  }, [date, iceClass, mode, liveTick])

  const refreshLive = async () => {
    setRefreshing(true)
    try {
      await api.liveStatus(true)
      setLiveTick((t) => t + 1)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setRefreshing(false)
    }
  }

  const routeBody = useMemo(() => {
    if (!meta) return null
    const ll = (k: string, custom?: { lat: number; lon: number }) =>
      k === 'custom' && custom ? custom : { lat: meta.places[k].lat, lon: meta.places[k].lon }
    return {
      date,
      mode,
      live_tick: liveTick,
      origin: ll(form.origin, form.originLL),
      dest: ll(form.dest, form.destLL),
      ice_class: iceClass,
      cruise_kn: form.cruise_kn,
      w_time: form.w_time,
      w_risk: form.w_risk,
      avoid_bergs: form.avoid_bergs,
      w_ice: form.w_ice,
      berg_margin_nm: form.berg_margin_nm,
    }
  }, [meta, date, mode, liveTick, form, iceClass])
  const bodyKey = useMemo(() => JSON.stringify(routeBody), [routeBody])

  const compute = useCallback(() => {
    if (!routeBody) return
    const seq = ++reqSeq.current
    const key = bodyKey
    setBusy(true)
    api
      .route(routeBody)
      .then((r) => {
        if (seq !== reqSeq.current) return // a newer request superseded this one
        setPlan(r.data)
        setPlanKey(key)
        setActiveRoute((k) => (routeOptions(r.data).some((o) => o.key === k) ? k : 'optimal'))
        note(r)
      })
      .catch((e) => seq === reqSeq.current && setError(e instanceof ApiError ? e.message : String(e)))
      .finally(() => seq === reqSeq.current && setBusy(false))
  }, [routeBody, bodyKey, note])

  // re-plan automatically (debounced) whenever the voyage settings, date or ice class change
  useEffect(() => {
    const ready = scenario && scenario.mode === mode && (mode === 'live' || scenario.date === date)
    if (!routeBody || !ready || bodyKey === planKey) return
    const id = setTimeout(compute, 450)
    return () => clearTimeout(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bodyKey, scenario?.date, scenario?.mode])

  const active = activeOption(plan, activeRoute)
  const maxT = Math.max(72, Math.ceil(active?.ev.summary.hours ?? 0))
  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => setTH((t) => (t + 1 > maxT ? 0 : t + 1)), 60)
    return () => clearInterval(id)
  }, [playing, maxT])

  const slot = Math.min(3, Math.round(tH / 24))

  const toggleOffline = (v: boolean) => {
    apiSetForceOffline(v)
    setForceOffline(v)
    if (!v) setOffline(false)
  }

  const download = async () => {
    if (!routeBody) return
    const blob = await api.bundle(routeBody)
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `polarnav_${date}.pnb`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  if (!meta) {
    return <div className="splash">{error ?? 'Connecting to PolarNav…'}</div>
  }

  const selectBerg = (id: string | null) => {
    setSelectedBerg(id)
    if (id && scenario) {
      const b = scenario.bergs.bergs.find((q) => q.id === id)
      if (b) setFocus({ x: b.x, y: b.y, seq: Date.now() })
    }
  }

  const leadLabel = slot === 0 ? 'Analysis' : `Forecast +${slot * 24} h`
  const vesselAt = active?.ev.points.find((q) => q.t_h >= tH) ?? active?.ev.points.at(-1)

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg viewBox="0 0 64 64" width="26" height="26" aria-hidden>
            <circle cx="32" cy="32" r="29" fill="none" stroke="#5fd4ff" strokeWidth="3" />
            <path d="M32 9 L37 32 L32 55 L27 32 Z" fill="#5fd4ff" />
            <path d="M9 32 L32 28 L55 32 L32 36 Z" fill="#e8f6ff" opacity=".8" />
          </svg>
          <div>
            <div className="brand-name">PolarNav</div>
            <div className="brand-sub">Antarctic sea-ice · iceberg · navigation decision support</div>
          </div>
        </div>
        <div className="top-controls">
          <div className="seg mode-seg" role="radiogroup" aria-label="Data source">
            {(['sim', 'live'] as const).map((m) => (
              <button key={m} role="radio" aria-checked={mode === m} className={mode === m ? 'on' : ''} onClick={() => setMode(m)}>
                {m === 'sim' ? 'Simulation' : '● Live'}
              </button>
            ))}
          </div>
          {mode === 'sim' ? (
            <label>
              Scenario date
              <input type="date" value={date} min={meta.dates.min} max={meta.dates.max} onChange={(e) => e.target.value && setDate(e.target.value)} />
            </label>
          ) : (
            <span className="pill live" title={scenario?.sources ? `Fetched ${scenario.sources.fetched_at}` : ''}>
              Sea ice {scenario?.mode === 'live' ? scenario.date : '…'} · ECMWF winds · USNIC bergs
            </span>
          )}
          <label>
            Ice class
            <select value={iceClass} onChange={(e) => setIceClass(e.target.value)}>
              {meta.ice_classes.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.capability_m} m)
                </option>
              ))}
            </select>
          </label>
          <span className={`pill ${offline ? 'down' : 'up'}`}>{offline ? '● OFFLINE · cached' : '● LINK UP'}</span>
          {mode === 'sim' && scenario?.is_test_period && (
            <span className="pill info" title="Hindcast on the held-out test year: forecasts can be verified against truth">
              Hindcast · verifiable
            </span>
          )}
          {mode === 'sim' && <span className="pill info" title="Data source">Synthetic digital twin</span>}
          <span className="clock mono">{now.toISOString().slice(11, 19)} UTC</span>
        </div>
      </header>

      <VoyagePanel
        meta={meta}
        form={form}
        setForm={setForm}
        plan={plan}
        busy={busy}
        stale={!!plan && planKey !== bodyKey}
        onCompute={compute}
        activeRoute={activeRoute}
        setActiveRoute={setActiveRoute}
        pickMode={pickMode}
        setPickMode={setPickMode}
        onSeek={(t) => {
          setPlaying(false)
          setTH(t)
        }}
      />

      <main className="center">
        {view === 'chart' ? (
          <MapView
            meta={meta}
            coast={coast}
            scenario={scenario}
            polaris={polaris}
            plan={plan}
            activeRoute={activeRoute}
            layer={layer}
            overlays={overlays}
            tH={tH}
            slot={slot}
            selectedBerg={selectedBerg}
            onSelectBerg={selectBerg}
            pickMode={pickMode}
            onPick={(lat, lon) => {
              if (pickMode === 'origin') setForm({ ...form, origin: 'custom', originLL: { lat, lon } })
              else if (pickMode === 'dest') setForm({ ...form, dest: 'custom', destLL: { lat, lon } })
              setPickMode(null)
            }}
            focus={focus}
          />
        ) : (
          <GlobeView
            meta={meta}
            land={globeLand}
            coast={coast}
            scenario={scenario}
            polaris={polaris}
            plan={plan}
            activeRoute={activeRoute}
            layer={layer}
            overlays={overlays}
            tH={tH}
            slot={slot}
            selectedBerg={selectedBerg}
            onSelectBerg={selectBerg}
            pickMode={pickMode}
            onPick={(lat, lon) => {
              if (pickMode === 'origin') setForm({ ...form, origin: 'custom', originLL: { lat, lon } })
              else if (pickMode === 'dest') setForm({ ...form, dest: 'custom', destLL: { lat, lon } })
              setPickMode(null)
            }}
            focus={focus}
          />
        )}
        <div className="layer-bar">
          <div className="seg view-seg" role="radiogroup" aria-label="Map view">
            {(['globe', 'chart'] as const).map((v) => (
              <button key={v} role="radio" aria-checked={view === v} className={view === v ? 'on' : ''} onClick={() => setView(v)}>
                {v === 'globe' ? '🌐 Globe' : 'Polar chart'}
              </button>
            ))}
          </div>
          <div className="seg" role="radiogroup" aria-label="Raster layer">
            {LAYERS.map((l) => (
              <button key={l.id} role="radio" aria-checked={layer === l.id} className={layer === l.id ? 'on' : ''} onClick={() => setLayer(l.id)}>
                {l.label}
              </button>
            ))}
          </div>
          <div className="toggles">
            {(
              [
                ['wind', 'Wind'],
                ['bergs', 'Icebergs'],
                ['routes', 'Routes'],
                ['verify', 'Verify vs truth'],
                ['graticule', 'Grid'],
              ] as [keyof Overlays, string][]
            ).map(([k, lbl]) => (
              <label key={k} className="check small">
                <input type="checkbox" checked={overlays[k]} onChange={(e) => setOverlays({ ...overlays, [k]: e.target.checked })} />
                {lbl}
              </label>
            ))}
          </div>
        </div>
        <Legend layer={layer} verify={overlays.verify} slot={slot} plan={plan} live={mode === 'live'} />
        {loading && <div className="loading">{mode === 'live' ? 'Fetching live satellite, weather and iceberg data…' : `Running models for ${date}…`}</div>}
        {error && (
          <div className="error-toast" role="alert" onClick={() => setError(null)}>
            {error}
          </div>
        )}
        <div className="timeline">
          <button className="icon-btn" onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause' : 'Play'}>
            {playing ? '❚❚' : '▶'}
          </button>
          <div className="tl-main">
            <input type="range" min={0} max={maxT} step={1} value={tH} onChange={(e) => setTH(+e.target.value)} aria-label="Timeline hours" />
            <div className="tl-marks">
              {[0, 24, 48, 72].map((h) => (
                <button key={h} className={slot === h / 24 ? 'on' : ''} style={{ left: `${(h / maxT) * 100}%` }} onClick={() => setTH(h)}>
                  {h === 0 ? 'Now' : `+${h}h`}
                </button>
              ))}
            </div>
          </div>
          <div className="tl-info">
            <div className="mono">T+{tH.toFixed(0)} h</div>
            <div className="muted small">{leadLabel}</div>
            {vesselAt && (
              <div className="small">
                <span className="mono">{vesselAt.speed_kn.toFixed(1)} kn</span> · SIC {Math.round(vesselAt.sic * 100)}% ·{' '}
                <span style={{ color: LEVEL_COLORS[vesselAt.level] }}>RIO {vesselAt.rio}</span>
              </div>
            )}
          </div>
        </div>
      </main>

      <aside className="panel right">
        <nav className="tabs" role="tablist">
          {(
            [
              ['hazards', 'Iceberg hazards'],
              ['models', 'AI models'],
              ['sar', 'SAR detect'],
              ['sync', 'Edge sync'],
            ] as [Tab, string][]
          ).map(([k, l]) => (
            <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>
              {l}
            </button>
          ))}
        </nav>
        <div className="tab-body">
          {tab === 'hazards' && <HazardsPanel scenario={scenario} plan={plan} selected={selectedBerg} onSelect={selectBerg} />}
          {tab === 'models' && <ModelsPanel meta={meta} scenario={scenario} onRefreshLive={refreshLive} refreshing={refreshing} />}
          {tab === 'sar' && <SarPanel />}
          {tab === 'sync' && (
            <SyncPanel plan={plan} offline={offline} savedAt={savedAt} forceOffline={forceOffline} setForceOffline={toggleOffline} onDownload={download} />
          )}
        </div>
      </aside>
    </div>
  )
}

function Legend({ layer, verify, slot, plan, live }: { layer: RasterLayer; verify: boolean; slot: number; plan: RoutePlan | null; live: boolean }) {
  return (
    <div className="legend">
      {layer === 'sic' && (
        <>
          <div className="legend-title">Sea-ice concentration</div>
          <div className="ramp sic" />
          <div className="ramp-labels">
            <span>15%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </>
      )}
      {layer === 'thickness' && (
        <>
          <div className="legend-title">Ice thickness</div>
          <div className="ramp thick" />
          <div className="ramp-labels">
            <span>0</span>
            <span>1.5 m</span>
            <span>4.5 m</span>
          </div>
        </>
      )}
      {layer === 'polaris' && (
        <>
          <div className="legend-title">POLARIS operation level</div>
          {['Normal (RIO ≥ 0)', 'Elevated risk', 'Special consideration'].map((t, i) => (
            <div key={t} className="legend-item">
              <i style={{ background: LEVEL_COLORS[i] }} />
              {t}
            </div>
          ))}
        </>
      )}
      {layer === 'error' && (
        <>
          <div className="legend-title">
            {slot === 0 ? "Yesterday's +24 h forecast − today's analysis" : `Forecast − truth at +${slot * 24} h`}
          </div>
          <div className="legend-note">
            {slot === 0
              ? 'Real-time check: needs no future data.'
              : live
                ? 'Not available live: this time has not happened yet. Use Now for the real-time check.'
                : 'Hindcast only: the truth is not yet observed in live use.'}
          </div>
          <div className="legend-item">
            <i style={{ background: '#f05a5a' }} />
            over-forecast
          </div>
          <div className="legend-item">
            <i style={{ background: '#50a0ff' }} />
            under-forecast
          </div>
        </>
      )}
      <div className="legend-sep" />
      {routeOptions(plan)
        .filter((o) => !o.sameAsOptimal)
        .map((o) => (
          <div key={o.key} className="legend-item">
            <i className={`line ${o.dashed ? 'dash' : ''}`} style={o.dashed ? undefined : { background: o.color }} />
            {o.label}
          </div>
        ))}
      <div className="legend-item">
        <i className="tri" />
        Iceberg · 72 h forecast
      </div>
      {verify && (
        <div className="legend-item">
          <i className="line" style={{ background: '#7dff9a' }} />
          Verifying truth track
        </div>
      )}
    </div>
  )
}

