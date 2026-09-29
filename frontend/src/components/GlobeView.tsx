import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { LEVEL_COLORS, LEVEL_SHORT } from '../colors'
import { alongRoute, fmtLat, fmtLon, lerpTrack, toLatLon, toXY } from '../geo'
import { buildPolarCanvas, buildWorldCanvas, GlobeGL } from '../globeGL'
import { cellColor, decodeScenario, type RasterLayer } from '../raster'
import { activeOption, routeOptions } from '../routes'
import type { Coastline, GlobeLand, Meta, PolarisGrid, RoutePlan, Scenario } from '../types'
import type { Overlays } from './MapView'

interface Props {
  meta: Meta
  land: GlobeLand | null
  coast: Coastline | null
  scenario: Scenario | null
  polaris: PolarisGrid | null
  plan: RoutePlan | null
  activeRoute: string
  layer: RasterLayer
  overlays: Overlays
  tH: number
  slot: number
  selectedBerg: string | null
  onSelectBerg: (id: string | null) => void
  pickMode: 'origin' | 'dest' | null
  onPick: (lat: number, lon: number) => void
  focus: { x: number; y: number; seq: number } | null
}

interface Hover { sx: number; sy: number; lat: number; lon: number; sic?: number; thick?: number; rio?: number; level?: number }

const D2R = Math.PI / 180
const HOME: [number, number] = [-30, 42] // view rotation [-lon, -lat]: centres 30E, 42S (Africa, Indian Ocean, Antarctica)
const MIN_ZOOM = 0.6
const MAX_ZOOM = 40
const IDLE_MS = 160 // full-detail coastline this long after the last movement
const POLAR_HALF_KM = 4500 // extent of the Antarctic land texture

type V3 = [number, number, number]
const unit = (lon: number, lat: number): V3 => {
  const cl = Math.cos(lat * D2R)
  return [cl * Math.cos(lon * D2R), cl * Math.sin(lon * D2R), Math.sin(lat * D2R)]
}

/** Orthographic view: e (screen right), n (screen up), c (towards viewer) for rotation [-lon0, -lat0]. */
function viewBasis(rot: [number, number]) {
  const lon0 = -rot[0] * D2R
  const lat0 = -rot[1] * D2R
  const cl = Math.cos(lat0)
  const sl = Math.sin(lat0)
  const co = Math.cos(lon0)
  const so = Math.sin(lon0)
  return { e: [-so, co, 0] as V3, n: [-sl * co, -sl * so, cl] as V3, c: [cl * co, cl * so, sl] as V3 }
}

/** Flattened unit vectors of many polylines, ready for per-frame rotation. */
interface Lines { xyz: Float32Array; starts: Int32Array }
function packLines(rings: [number, number][][]): Lines {
  const total = rings.reduce((s, r) => s + r.length, 0)
  const xyz = new Float32Array(total * 3)
  const starts = new Int32Array(rings.length + 1)
  let k = 0
  rings.forEach((r, i) => {
    starts[i] = k
    for (const [lon, lat] of r) {
      xyz.set(unit(lon, lat), k * 3)
      k++
    }
  })
  starts[rings.length] = k
  return { xyz, starts }
}

function graticuleLines(): [number, number][][] {
  const out: [number, number][][] = []
  // meridians run to 89 deg so there is still orientation when zoomed onto a pole
  for (let lon = -180; lon < 180; lon += 10) {
    const r: [number, number][] = []
    for (let lat = -89; lat <= 89; lat += lat < -80 || lat >= 80 ? 1 : 2) r.push([lon, lat])
    out.push(r)
  }
  for (const lat of [-85, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 85]) {
    const r: [number, number][] = []
    for (let lon = -180; lon <= 180; lon += 2) r.push([lon, lat])
    out.push(r)
  }
  return out
}
const GRATICULE = packLines(graticuleLines())
const ANT_CIRCLE = packLines([Array.from({ length: 181 }, (_, i) => [-180 + 2 * i, -66.56] as [number, number])])

export default function GlobeView(p: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const glCanvas = useRef<HTMLCanvasElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const glRef = useRef<GlobeGL | null>(null)
  const [glError, setGlError] = useState<string | null>(null)
  const size = useRef({ w: 800, h: 600 })
  const rot = useRef<[number, number]>([...HOME])
  const zoom = useRef(1)
  const zoomTarget = useRef(1)
  const zoomAnchor = useRef<[number, number] | null>(null)
  const vel = useRef<[number, number]>([0, 0])
  const drag = useRef<{ x: number; y: number; moved: boolean; t: number } | null>(null)
  const anim = useRef<number | null>(null)
  const zoomAnim = useRef<number | null>(null)
  const frame = useRef<number | null>(null)
  const idleTimer = useRef<number | null>(null)
  const movingUntil = useRef(0)
  const hoverEvt = useRef<[number, number] | null>(null)
  const hoverFrame = useRef<number | null>(null)
  const [spin, setSpin] = useState(false)
  const [hover, setHover] = useState<Hover | null>(null)
  const N = p.meta.grid.ice_n
  const cell = p.meta.grid.cell_km
  const half = (N * cell) / 2

  const radius = () => Math.min(size.current.w, size.current.h) * 0.44 * zoom.current

  /** Screen position (CSS px) of a lon/lat for the current view, or null on the far side. */
  const project = useCallback((lon: number, lat: number): [number, number] | null => {
    const { e, n, c } = viewBasis(rot.current)
    const [x, y, z] = unit(lon, lat)
    if (x * c[0] + y * c[1] + z * c[2] <= 0.01) return null
    const R = radius()
    return [size.current.w / 2 + R * (x * e[0] + y * e[1]), size.current.h / 2 - R * (x * n[0] + y * n[1] + z * n[2])]
  }, [])

  /** lon/lat under a screen point, or null off the globe. */
  const invert = useCallback((sx: number, sy: number): [number, number] | null => {
    const R = radius()
    const px = (sx - size.current.w / 2) / R
    const py = -(sy - size.current.h / 2) / R
    const r2 = px * px + py * py
    if (r2 > 1) return null
    const z = Math.sqrt(1 - r2)
    const { e, n, c } = viewBasis(rot.current)
    const vx = px * e[0] + py * n[0] + z * c[0]
    const vy = px * e[1] + py * n[1] + z * c[1]
    const vz = px * e[2] + py * n[2] + z * c[2]
    return [Math.atan2(vy, vx) / D2R, Math.asin(Math.max(-1, Math.min(1, vz))) / D2R]
  }, [])

  // ------------------------------------------------------------------ GPU layer setup
  useEffect(() => {
    if (!glCanvas.current) return
    try {
      glRef.current = GlobeGL.create(glCanvas.current)
      if (!glRef.current) setGlError('WebGL is not available in this browser — use the Polar chart view.')
    } catch (err) {
      setGlError(`Globe renderer failed to start (${err}). Use the Polar chart view.`)
    }
    return () => {
      glRef.current?.dispose()
      glRef.current = null
    }
  }, [])

  const coastLines = useMemo(() => {
    if (!p.land) return null
    const isAnt = (poly: [number, number][][]) => poly[0].some(([, lat]) => lat < -65)
    const pack = (polys: [number, number][][][], pred: (x: [number, number][][]) => boolean) => packLines(polys.filter(pred).flat())
    const lvl = (land: [number, number][][][], shelf: [number, number][][][]) => ({
      rest: pack(land, (x) => !isAnt(x)),
      ant: pack(land, isAnt),
      shelf: pack(shelf, () => true),
    })
    return { hi: lvl(p.land.land, p.land.shelf), lo: lvl(p.land.land_lo ?? p.land.land, p.land.shelf_lo ?? p.land.shelf) }
  }, [p.land])

  const frameReq = useRef<() => void>(() => {})

  useEffect(() => {
    const gl = glRef.current
    if (!gl || !p.land) return
    gl.setWorld(buildWorldCanvas(p.land.land, p.land.shelf, gl.maxTex))
    frameReq.current()
  }, [p.land])

  useEffect(() => {
    const gl = glRef.current
    if (!gl || !p.coast) return
    gl.setPolar(buildPolarCanvas(p.coast.land, p.coast.shelf, POLAR_HALF_KM, Math.min(2048, gl.maxTex)))
    frameReq.current()
  }, [p.coast])

  const decoded = useMemo(() => (p.scenario ? decodeScenario(p.scenario, p.polaris) : null), [p.scenario, p.polaris])

  useEffect(() => {
    const gl = glRef.current
    if (!gl) return
    if (!decoded || p.layer === 'none') {
      gl.setIce(null, N)
    } else {
      const rgba = new Uint8Array(N * N * 4)
      for (let i = 0; i < N * N; i++) {
        const c = cellColor(decoded, p.layer, p.slot, i)
        if (c) rgba.set(c, i * 4)
      }
      gl.setIce(rgba, N)
    }
    frameReq.current()
  }, [decoded, p.layer, p.slot, N])

  // ------------------------------------------------------------------ drawing
  const draw = useCallback(() => {
    frame.current = null
    const cv = canvasRef.current
    const gcv = glCanvas.current
    if (!cv || !gcv) return
    const { w, h } = size.current
    const moving = performance.now() < movingUntil.current
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    for (const c of [cv, gcv]) {
      if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
        c.width = Math.round(w * dpr)
        c.height = Math.round(h * dpr)
      }
    }
    const R = radius()
    const tx = w / 2
    const ty = h / 2
    const B = viewBasis(rot.current)

    // filled layers on the GPU
    glRef.current?.render(gcv.width, gcv.height, { cx: gcv.width / 2, cy: gcv.height / 2, R: R * dpr, e: B.e, n: B.n, c: B.c }, POLAR_HALF_KM, half)

    // vector overlays on the 2D canvas
    const ctx = cv.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const P = (lon: number, lat: number): [number, number] | null => {
      const [x, y, z] = unit(lon, lat)
      if (x * B.c[0] + y * B.c[1] + z * B.c[2] <= 0.01) return null
      return [tx + R * (x * B.e[0] + y * B.e[1]), ty - R * (x * B.n[0] + y * B.n[1] + z * B.n[2])]
    }
    // polylines: rotate packed unit vectors, break the line wherever it passes behind the globe or far off-screen
    const pad = 200
    const strokeLines = (L: Lines, color: string, width: number, dash: number[] = []) => {
      const { xyz, starts } = L
      ctx.beginPath()
      for (let r = 0; r < starts.length - 1; r++) {
        let pen = false
        for (let k = starts[r]; k < starts[r + 1]; k++) {
          const o = k * 3
          const x = xyz[o]
          const y = xyz[o + 1]
          const z = xyz[o + 2]
          if (x * B.c[0] + y * B.c[1] + z * B.c[2] <= 0) {
            pen = false
            continue
          }
          const sx = tx + R * (x * B.e[0] + y * B.e[1])
          const sy = ty - R * (x * B.n[0] + y * B.n[1] + z * B.n[2])
          if (sx < -pad || sy < -pad || sx > w + pad || sy > h + pad) {
            pen = false
            continue
          }
          if (pen) ctx.lineTo(sx, sy)
          else ctx.moveTo(sx, sy)
          pen = true
        }
      }
      ctx.setLineDash(dash)
      ctx.strokeStyle = color
      ctx.lineWidth = width
      ctx.stroke()
      ctx.setLineDash([])
    }
    const polyline = (pts: [number, number][]) => {
      ctx.beginPath()
      let pen = false
      for (const [lon, lat] of pts) {
        const s = P(lon, lat)
        if (!s) {
          pen = false
          continue
        }
        if (pen) ctx.lineTo(s[0], s[1])
        else ctx.moveTo(s[0], s[1])
        pen = true
      }
    }

    if (p.overlays.graticule) {
      strokeLines(GRATICULE, 'rgba(140,190,235,0.16)', 0.7)
      strokeLines(ANT_CIRCLE, 'rgba(160,220,255,0.35)', 1, [4, 4])
    }
    if (coastLines) {
      const L = moving ? coastLines.lo : coastLines.hi
      strokeLines(L.rest, 'rgba(170,210,190,0.55)', 0.6)
      strokeLines(L.ant, 'rgba(120,160,190,0.75)', 0.6)
      strokeLines(L.shelf, 'rgba(120,160,190,0.5)', 0.5)
    }

    const z0 = zoom.current

    // South Pole marker
    const sp = P(0, -90)
    if (sp) {
      ctx.strokeStyle = '#ffd166'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(sp[0] - 6, sp[1])
      ctx.lineTo(sp[0] + 6, sp[1])
      ctx.moveTo(sp[0], sp[1] - 6)
      ctx.lineTo(sp[0], sp[1] + 6)
      ctx.stroke()
      if (z0 > 1.5) label(ctx, 'South Pole 90°S', sp[0] + 8, sp[1] - 6, '#ffd166')
    }

    // places
    for (const pl of Object.values(p.meta.places)) {
      const s = P(pl.lon, pl.lat)
      if (!s) continue
      ctx.fillStyle = pl.kind === 'station' ? '#ffd166' : '#9fb3c8'
      ctx.strokeStyle = '#071625'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      if (pl.kind === 'station') {
        ctx.moveTo(s[0], s[1] - 7)
        ctx.lineTo(s[0] + 6, s[1] + 4)
        ctx.lineTo(s[0] - 6, s[1] + 4)
        ctx.closePath()
      } else ctx.rect(s[0] - 4, s[1] - 4, 8, 8)
      ctx.fill()
      ctx.stroke()
      if (z0 > 1.6 || pl.kind === 'station') label(ctx, pl.name, s[0] + 9, s[1] + 4, pl.kind === 'station' ? '#ffd166' : '#b8c7d6')
    }

    // routes: every option thin, the selected one bold with a soft glow
    if (p.overlays.routes) {
      const active = activeOption(p.plan, p.activeRoute)
      const ll = (pts: { lon: number; lat: number }[]) => pts.map((q) => [q.lon, q.lat] as [number, number])
      for (const o of routeOptions(p.plan)) {
        if (o.key === active?.key || o.sameAsOptimal) continue
        polyline(ll(o.ev.points))
        ctx.setLineDash(o.dashed ? [7, 5] : [])
        ctx.strokeStyle = o.color
        ctx.lineWidth = 2
        ctx.stroke()
        ctx.setLineDash([])
      }
      if (active) {
        const pts = active.ev.points
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        polyline(ll(pts))
        ctx.globalAlpha = 0.25
        ctx.strokeStyle = active.color
        ctx.lineWidth = 9
        ctx.stroke()
        ctx.globalAlpha = 1
        ctx.setLineDash(active.dashed ? [8, 5] : [])
        ctx.lineWidth = 3.2
        ctx.stroke()
        ctx.setLineDash([])
        for (let i = 1; i < pts.length; i++) {
          if (pts[i].level === 0) continue
          polyline([[pts[i - 1].lon, pts[i - 1].lat], [pts[i].lon, pts[i].lat]])
          ctx.strokeStyle = LEVEL_COLORS[pts[i].level]
          ctx.stroke()
        }
        for (const wpt of active.ev.waypoints) {
          const s = P(wpt.lon, wpt.lat)
          if (!s) continue
          ctx.beginPath()
          ctx.arc(s[0], s[1], 3.5, 0, Math.PI * 2)
          ctx.fillStyle = '#071625'
          ctx.fill()
          ctx.strokeStyle = active.color
          ctx.lineWidth = 2
          ctx.stroke()
          if (z0 > 3) label(ctx, wpt.name, s[0] + 6, s[1] - 5, '#e6f1fb')
        }
      }
    }

    // icebergs
    if (p.overlays.bergs && p.scenario) {
      const step = p.scenario.bergs.step_h
      const tb = Math.min(p.tH, 72)
      const toLL = (xy: [number, number]) => {
        const [la, lo] = toLatLon(xy[0], xy[1])
        return [lo, la] as [number, number]
      }
      for (const b of p.scenario.bergs.bergs) {
        const sel = b.id === p.selectedBerg
        const [lon, lat] = toLL(lerpTrack(b.forecast, step, tb))
        const s = P(lon, lat)
        if (!s) continue
        if (z0 > 2 || sel) {
          const track = (xs: [number, number][], color: string, width: number, dash: number[] = []) => {
            polyline(xs.map(toLL))
            ctx.setLineDash(dash)
            ctx.strokeStyle = color
            ctx.lineWidth = width
            ctx.stroke()
            ctx.setLineDash([])
          }
          track(b.observed, 'rgba(220,230,240,0.55)', 1, [2, 3])
          track(b.forecast, 'rgba(255,107,107,0.85)', sel ? 2.2 : 1.2, [5, 3])
          if (p.overlays.verify) track(b.truth, '#7dff9a', sel ? 2 : 1)
        }
        const e = b.ellipses.reduce((best, cur) => (Math.abs(cur.t_h - tb) < Math.abs(best.t_h - tb) ? cur : best))
        const rpx = (Math.max(e.a, 1) / 6371) * R
        if (rpx > 2) {
          ctx.beginPath()
          ctx.arc(s[0], s[1], rpx, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(255,90,90,0.15)'
          ctx.fill()
        }
        const sz = sel ? 8 : 5.5
        ctx.beginPath()
        ctx.moveTo(s[0], s[1] - sz)
        ctx.lineTo(s[0] + sz * 0.9, s[1] + sz * 0.6)
        ctx.lineTo(s[0] - sz * 0.9, s[1] + sz * 0.6)
        ctx.closePath()
        ctx.fillStyle = sel ? '#ffffff' : '#ff5a5f'
        ctx.strokeStyle = sel ? '#ff5a5f' : '#2a0b0d'
        ctx.lineWidth = sel ? 2.2 : 1
        ctx.fill()
        ctx.stroke()
        if (z0 > 3 || sel) label(ctx, b.id, s[0] + sz + 3, s[1] + 3, sel ? '#fff' : '#ff9a9e')
      }
    }

    // vessel
    const act = activeOption(p.plan, p.activeRoute)
    if (act) {
      const at = alongRoute(act.ev.points, p.tH)
      if (at) {
        const [la, lo] = toLatLon(at.x, at.y)
        const s = P(lo, la)
        if (s) {
          ctx.beginPath()
          ctx.arc(s[0], s[1], 10, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(255,255,255,0.18)'
          ctx.fill()
          ctx.beginPath()
          ctx.arc(s[0], s[1], 5, 0, Math.PI * 2)
          ctx.fillStyle = '#ffffff'
          ctx.fill()
        }
      }
    }

    if (p.pickMode) {
      ctx.fillStyle = 'rgba(49,225,247,0.12)'
      ctx.fillRect(0, 0, w, 30)
      ctx.fillStyle = '#9ff0ff'
      ctx.font = '600 13px system-ui, Segoe UI, sans-serif'
      ctx.fillText(`Click the globe to set the ${p.pickMode === 'origin' ? 'departure' : 'destination'} point`, 14, 20)
    }
  }, [coastLines, half, p.overlays, p.plan, p.activeRoute, p.scenario, p.tH, p.selectedBerg, p.meta.places, p.pickMode])

  const drawRef = useRef(draw)
  drawRef.current = draw

  /** Schedule one frame; `motion` marks the globe as moving (low-detail outlines) and queues a full-detail frame once it settles. */
  const redraw = useCallback((motion = false) => {
    if (motion) {
      movingUntil.current = performance.now() + IDLE_MS
      if (idleTimer.current) window.clearTimeout(idleTimer.current)
      idleTimer.current = window.setTimeout(() => {
        idleTimer.current = null
        if (frame.current == null) frame.current = requestAnimationFrame(() => drawRef.current())
      }, IDLE_MS + 20)
    }
    if (frame.current == null) frame.current = requestAnimationFrame(() => drawRef.current())
  }, [])
  frameReq.current = redraw

  useEffect(() => {
    redraw()
  }, [draw, redraw])

  // ------------------------------------------------------------------ sizing
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      size.current = { w: el.clientWidth, h: el.clientHeight }
      for (const c of [canvasRef.current, glCanvas.current]) {
        if (!c) continue
        c.style.width = `${el.clientWidth}px`
        c.style.height = `${el.clientHeight}px`
      }
      redraw()
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [redraw])

  // ------------------------------------------------------------------ motion: fly-to, spin, zoom easing
  const stopAnim = () => {
    if (anim.current) cancelAnimationFrame(anim.current)
    anim.current = null
  }

  const flyTo = useCallback(
    (lon: number, lat: number, z?: number) => {
      stopAnim()
      const r0 = [...rot.current] as [number, number]
      const dl = ((((-lon - r0[0]) % 360) + 540) % 360) - 180
      const r1: [number, number] = [r0[0] + dl, -lat]
      const z0 = zoom.current
      const z1 = z ?? z0
      const t0 = performance.now()
      const stepFn = (now: number) => {
        const k = Math.min((now - t0) / 900, 1)
        const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2
        rot.current = [r0[0] + (r1[0] - r0[0]) * e, r0[1] + (r1[1] - r0[1]) * e]
        zoom.current = zoomTarget.current = z0 * Math.pow(z1 / z0, e)
        redraw(true)
        anim.current = k < 1 ? requestAnimationFrame(stepFn) : null
      }
      anim.current = requestAnimationFrame(stepFn)
    },
    [redraw],
  )

  useEffect(() => {
    if (!spin) return
    let last = performance.now()
    let id = 0
    const tick = (now: number) => {
      const dt = Math.min(now - last, 50)
      last = now
      if (!drag.current && !anim.current) {
        rot.current = [rot.current[0] + dt * 0.012, rot.current[1]]
        redraw(true)
      }
      id = requestAnimationFrame(tick)
    }
    id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }, [spin, redraw])

  useEffect(() => {
    if (!p.focus) return
    const [la, lo] = toLatLon(p.focus.x, p.focus.y)
    flyTo(lo, la, Math.max(zoom.current, 4))
  }, [p.focus, flyTo])

  // Bring the route into view when the voyage itself changes (new endpoints), not on every re-plan.
  const framedKey = useRef('')
  useEffect(() => {
    const pts = p.plan?.optimal?.points
    if (!pts?.length) return
    const key = `${pts[0].lat},${pts[0].lon}|${pts[pts.length - 1].lat},${pts[pts.length - 1].lon}`
    if (key === framedKey.current) return
    framedKey.current = key
    const mid = pts[Math.floor(pts.length / 2)]
    flyTo(mid.lon, -42, 1.15)
  }, [p.plan, flyTo])

  /** Ease zoom toward its target, keeping the point under the cursor fixed. */
  const runZoom = useCallback(() => {
    if (zoomAnim.current != null) return
    const tick = () => {
      const zt = zoomTarget.current
      const zc = zoom.current
      const anchor = zoomAnchor.current
      const before = anchor ? invert(anchor[0], anchor[1]) : null
      zoom.current = Math.abs(zt - zc) < zc * 0.002 ? zt : zc * Math.pow(zt / zc, 0.28)
      if (anchor && before) {
        const after = invert(anchor[0], anchor[1])
        if (after) {
          const dl = ((after[0] - before[0] + 540) % 360) - 180
          // near the poles longitude is ill-defined: only correct latitude there
          const lonOk = Math.abs(before[1]) < 80
          rot.current = [rot.current[0] + (lonOk ? dl : 0), Math.max(-90, Math.min(90, rot.current[1] + (after[1] - before[1])))]
        }
      }
      redraw(true)
      zoomAnim.current = zoom.current !== zt ? requestAnimationFrame(tick) : null
    }
    zoomAnim.current = requestAnimationFrame(tick)
  }, [invert, redraw])

  // ------------------------------------------------------------------ interaction
  useEffect(() => {
    const cv = canvasRef.current
    if (!cv) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      stopAnim()
      const r = cv.getBoundingClientRect()
      const pt: [number, number] = [e.clientX - r.left, e.clientY - r.top]
      zoomAnchor.current = invert(pt[0], pt[1]) ? pt : null
      const delta = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY
      zoomTarget.current = Math.min(Math.max(zoomTarget.current * Math.exp(-delta * 0.0015), MIN_ZOOM), MAX_ZOOM)
      runZoom()
    }
    cv.addEventListener('wheel', onWheel, { passive: false })
    return () => cv.removeEventListener('wheel', onWheel)
  }, [invert, runZoom])

  const localXY = (e: React.PointerEvent) => {
    const r = canvasRef.current!.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top] as [number, number]
  }

  const onDown = (e: React.PointerEvent) => {
    stopAnim()
    ;(e.target as Element).setPointerCapture(e.pointerId)
    drag.current = { x: e.clientX, y: e.clientY, moved: false, t: performance.now() }
    vel.current = [0, 0]
  }

  const updateHover = () => {
    hoverFrame.current = null
    const pt = hoverEvt.current
    if (!pt || drag.current?.moved) return
    const ll = invert(pt[0], pt[1])
    if (!ll) return setHover(null)
    const [lon, lat] = ll
    const hv: Hover = { sx: pt[0], sy: pt[1], lat, lon }
    if (decoded) {
      const [x, y] = toXY(lat, lon)
      const i = Math.floor((x + half) / cell)
      const j = Math.floor((y + half) / cell)
      if (i >= 0 && j >= 0 && i < N && j < N) {
        const k = j * N + i
        const s = Math.min(p.slot, decoded.sic.length - 1)
        hv.sic = decoded.sic[s][k]
        hv.thick = decoded.thick[s][k] / 50
        hv.rio = decoded.rio?.[s][k]
        hv.level = decoded.level?.[s][k]
      }
    }
    setHover(hv)
  }

  const onMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (d) {
      const dx = e.clientX - d.x
      const dy = e.clientY - d.y
      if (Math.abs(dx) + Math.abs(dy) > 2) {
        if (!d.moved) setHover(null)
        d.moved = true
      }
      if (d.moved) {
        // degrees per pixel so the surface under the cursor tracks the pointer at any zoom
        const k = 57.3 / radius()
        rot.current = [rot.current[0] + dx * k, Math.max(-90, Math.min(90, rot.current[1] - dy * k))]
        const now = performance.now()
        const dt = Math.max(now - d.t, 1)
        vel.current = [(dx * k) / dt, (-dy * k) / dt]
        d.x = e.clientX
        d.y = e.clientY
        d.t = now
        redraw(true)
        return
      }
    }
    hoverEvt.current = localXY(e)
    if (hoverFrame.current == null) hoverFrame.current = requestAnimationFrame(updateHover)
  }

  const onUp = (e: React.PointerEvent) => {
    const d = drag.current
    drag.current = null
    if (d?.moved) {
      // inertia: keep turning with the release velocity and let it decay
      let [vl, vp] = vel.current
      if (performance.now() - d.t > 80) return
      let last = performance.now()
      const glide = (now: number) => {
        const dt = Math.min(now - last, 50)
        last = now
        rot.current = [rot.current[0] + vl * dt, Math.max(-90, Math.min(90, rot.current[1] + vp * dt))]
        vl *= Math.pow(0.93, dt / 16)
        vp *= Math.pow(0.93, dt / 16)
        redraw(true)
        anim.current = Math.abs(vl) + Math.abs(vp) > 0.0004 ? requestAnimationFrame(glide) : null
      }
      anim.current = requestAnimationFrame(glide)
      return
    }
    const [sx, sy] = localXY(e)
    const ll = invert(sx, sy)
    if (!ll) return
    if (p.pickMode) {
      p.onPick(ll[1], ll[0])
      return
    }
    if (p.scenario && p.overlays.bergs) {
      const tb = Math.min(p.tH, 72)
      let best: string | null = null
      let bd = 14
      for (const b of p.scenario.bergs.bergs) {
        const [x, y] = lerpTrack(b.forecast, p.scenario.bergs.step_h, tb)
        const [la, lo] = toLatLon(x, y)
        const s = project(lo, la)
        if (!s) continue
        const dd = Math.hypot(s[0] - sx, s[1] - sy)
        if (dd < bd) {
          bd = dd
          best = b.id
        }
      }
      p.onSelectBerg(best)
    }
  }

  return (
    <div ref={wrapRef} className="map-wrap globe-wrap">
      <canvas ref={glCanvas} className="globe-gl" aria-hidden />
      <canvas
        ref={canvasRef}
        className="globe-overlay"
        style={{ cursor: p.pickMode ? 'crosshair' : 'grab', touchAction: 'none' }}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={() => {
          hoverEvt.current = null
          setHover(null)
        }}
        role="img"
        aria-label="Rotatable 3-D globe. Drag to rotate, scroll to zoom."
      />
      {glError && <div className="error-toast">{glError}</div>}
      <div className="globe-ctrl">
        <button onClick={() => flyTo(-HOME[0], -HOME[1], 1)} title="Reset view">
          ⟲ Reset
        </button>
        <button onClick={() => flyTo(0, -90, 1.8)} title="Look straight down on the South Pole">
          South Pole
        </button>
        <button
          onClick={() => {
            const pts = activeOption(p.plan, p.activeRoute)?.ev.points
            if (pts?.length) {
              const mid = pts[Math.floor(pts.length / 2)]
              flyTo(mid.lon, mid.lat, 5)
            }
          }}
          disabled={!p.plan?.optimal}
        >
          Route
        </button>
        <button className={spin ? 'on' : ''} onClick={() => setSpin(!spin)} aria-pressed={spin}>
          {spin ? '❚❚ Spin' : '↻ Spin'}
        </button>
      </div>
      <div className="globe-hint small muted">Drag to rotate · scroll to zoom · click an iceberg for details</div>
      {hover && (
        <div className="map-tip" style={{ left: Math.min(hover.sx + 14, size.current.w - 190), top: Math.min(hover.sy + 14, size.current.h - 110) }}>
          <div className="mono">
            {fmtLat(hover.lat)} {fmtLon(hover.lon)}
          </div>
          {hover.sic !== undefined && hover.sic > 2 && (
            <div>
              SIC <b>{hover.sic}%</b>
              {hover.thick !== undefined ? <> · {hover.thick.toFixed(2)} m</> : null}
            </div>
          )}
          {hover.rio !== undefined && hover.sic !== undefined && hover.sic > 2 && (
            <div>
              RIO <b>{hover.rio}</b> · <span style={{ color: LEVEL_COLORS[hover.level ?? 0] }}>{LEVEL_SHORT[hover.level ?? 0]}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  ctx.font = '600 11px system-ui, Segoe UI, sans-serif'
  ctx.lineWidth = 3
  ctx.strokeStyle = 'rgba(4,12,22,0.9)'
  ctx.strokeText(text, x, y)
  ctx.fillStyle = color
  ctx.fillText(text, x, y)
}
