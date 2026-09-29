import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { LEVEL_COLORS, LEVEL_SHORT } from '../colors'
import { alongRoute, fmtLat, fmtLon, lerpTrack, toLatLon, toScreen, toWorld, toXY, type View } from '../geo'
import { cellColor, decodeScenario, type RasterLayer } from '../raster'
import { activeOption, routeOptions } from '../routes'
import type { Coastline, Meta, PolarisGrid, RoutePlan, Scenario } from '../types'

export type { RasterLayer }
export interface Overlays { wind: boolean; bergs: boolean; routes: boolean; verify: boolean; graticule: boolean }

interface Props {
  meta: Meta
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

const ICE_HALF = 4000

export default function MapView(p: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [view, setView] = useState<View>({ cx: 600, cy: 1500, scale: 0.12, w: 800, h: 600 })
  const [hover, setHover] = useState<Hover | null>(null)
  const drag = useRef<{ x: number; y: number; cx: number; cy: number; moved: boolean } | null>(null)
  const fitted = useRef(false)

  const N = p.meta.grid.ice_n
  const decoded = useMemo(() => (p.scenario ? decodeScenario(p.scenario, p.polaris) : null), [p.scenario, p.polaris])

  // ---------------------------------------------------------------- raster image
  const raster = useMemo(() => {
    if (!decoded || p.layer === 'none') return null
    const c = document.createElement('canvas')
    c.width = N
    c.height = N
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(N, N)
    for (let r = 0; r < N; r++) {
      const row = N - 1 - r // data row 0 is the southern edge
      for (let col = 0; col < N; col++) {
        const px = cellColor(decoded, p.layer, p.slot, row * N + col)
        if (!px) continue
        const o = (r * N + col) * 4
        img.data[o] = px[0]
        img.data[o + 1] = px[1]
        img.data[o + 2] = px[2]
        img.data[o + 3] = px[3]
      }
    }
    ctx.putImageData(img, 0, 0)
    return c
  }, [decoded, p.layer, p.slot, N])

  // ---------------------------------------------------------------- sizing
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth
      const h = el.clientHeight
      setView((v) => {
        if (!fitted.current && w > 0 && h > 0) {
          fitted.current = true
          // frame the Indian Ocean sector from Cape Town down to the coast
          return { cx: 1500, cy: 2900, scale: Math.min(w / 6200, h / 5600), w, h }
        }
        return { ...v, w, h }
      })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    if (p.focus) setView((v) => ({ ...v, cx: p.focus!.x, cy: p.focus!.y, scale: Math.max(v.scale, 0.35) }))
  }, [p.focus])

  // frame the route when the voyage changes (new endpoints), not on every re-plan, so moving
  // a slider never yanks the chart away from where you are looking
  const framedKey = useRef('')
  const sized = view.w > 0 && view.h > 0 && fitted.current
  useEffect(() => {
    const opt = p.plan?.optimal?.points
    if (!sized || !p.plan || !opt?.length) return
    const key = `${opt[0].x},${opt[0].y}|${opt[opt.length - 1].x},${opt[opt.length - 1].y}`
    if (framedKey.current === key) return
    framedKey.current = key
    const pts = [...opt, ...(p.plan.baseline?.points ?? [])]
    const xs = pts.map((q) => q.x)
    const ys = pts.map((q) => q.y)
    const x0 = Math.min(...xs)
    const x1 = Math.max(...xs)
    const y0 = Math.min(...ys)
    const y1 = Math.max(...ys)
    setView((v) => {
      const span = Math.max((x1 - x0) / (v.w * 0.55), (y1 - y0) / (v.h * 0.6), 1 / 0.6)
      return { ...v, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 - 0.04 * (y1 - y0), scale: Math.min(1 / span, 0.6) }
    })
  }, [p.plan, sized])

  // ---------------------------------------------------------------- draw
  const draw = useCallback(() => {
    const cv = canvasRef.current
    if (!cv) return
    const dpr = window.devicePixelRatio || 1
    if (cv.width !== Math.round(view.w * dpr) || cv.height !== Math.round(view.h * dpr)) {
      cv.width = Math.round(view.w * dpr)
      cv.height = Math.round(view.h * dpr)
    }
    const ctx = cv.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = '#071625'
    ctx.fillRect(0, 0, view.w, view.h)
    const S = (x: number, y: number) => toScreen(view, x, y)

    if (raster) {
      const [x0, y0] = S(-ICE_HALF, ICE_HALF)
      ctx.imageSmoothingEnabled = true
      ctx.imageSmoothingQuality = 'high'
      ctx.drawImage(raster, x0, y0, 2 * ICE_HALF * view.scale, 2 * ICE_HALF * view.scale)
    }

    if (p.overlays.graticule) drawGraticule(ctx, view)

    if (p.coast) {
      const poly = (rings: [number, number][][], fill: string, stroke: string) => {
        ctx.beginPath()
        for (const ring of rings) {
          ring.forEach(([x, y], i) => {
            const [sx, sy] = S(x, y)
            if (i === 0) ctx.moveTo(sx, sy)
            else ctx.lineTo(sx, sy)
          })
          ctx.closePath()
        }
        ctx.fillStyle = fill
        ctx.fill('evenodd')
        ctx.strokeStyle = stroke
        ctx.lineWidth = 0.8
        ctx.stroke()
      }
      poly(p.coast.land, '#1c2c3d', '#4d6b87')
      poly(p.coast.shelf, '#2a4058', '#6f93b3')
    }

    if (p.overlays.wind && p.scenario) drawWind(ctx, view, p.scenario.wind)

    // places
    for (const pl of Object.values(p.meta.places)) {
      const [sx, sy] = S(...toXY(pl.lat, pl.lon))
      ctx.fillStyle = pl.kind === 'station' ? '#ffd166' : '#9fb3c8'
      ctx.strokeStyle = '#071625'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      if (pl.kind === 'station') {
        ctx.moveTo(sx, sy - 7)
        ctx.lineTo(sx + 6, sy + 4)
        ctx.lineTo(sx - 6, sy + 4)
        ctx.closePath()
      } else ctx.rect(sx - 4, sy - 4, 8, 8)
      ctx.fill()
      ctx.stroke()
      label(ctx, pl.name, sx + 9, sy + 4, pl.kind === 'station' ? '#ffd166' : '#b8c7d6')
    }

    // routes: every option thin, the selected one bold and coloured by POLARIS level
    if (p.overlays.routes) {
      const opts = routeOptions(p.plan)
      const active = activeOption(p.plan, p.activeRoute)
      for (const o of opts) {
        if (o.key === active?.key || o.sameAsOptimal) continue
        ctx.save()
        ctx.setLineDash(o.dashed ? [8, 6] : [])
        ctx.strokeStyle = o.color
        ctx.globalAlpha = 0.85
        ctx.lineWidth = 2
        ctx.beginPath()
        o.ev.points.forEach((q, i) => {
          const [sx, sy] = S(q.x, q.y)
          if (i === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        ctx.restore()
      }
      if (active) {
        const pts = active.ev.points
        ctx.save()
        ctx.lineCap = 'round'
        ctx.shadowColor = active.color
        ctx.shadowBlur = 10
        if (active.dashed) ctx.setLineDash([9, 6])
        for (let i = 1; i < pts.length; i++) {
          const [ax, ay] = S(pts[i - 1].x, pts[i - 1].y)
          const [bx, by] = S(pts[i].x, pts[i].y)
          ctx.strokeStyle = pts[i].level === 0 ? active.color : LEVEL_COLORS[pts[i].level]
          ctx.lineWidth = 3.4
          ctx.beginPath()
          ctx.moveTo(ax, ay)
          ctx.lineTo(bx, by)
          ctx.stroke()
        }
        ctx.restore()
        for (const w of active.ev.waypoints) {
          const [sx, sy] = S(w.x, w.y)
          ctx.fillStyle = '#071625'
          ctx.strokeStyle = active.color
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.arc(sx, sy, 4, 0, Math.PI * 2)
          ctx.fill()
          ctx.stroke()
          if (view.scale > 0.16) label(ctx, w.name, sx + 7, sy - 6, '#e6f1fb')
        }
      }
    }

    // icebergs
    if (p.overlays.bergs && p.scenario) {
      const step = p.scenario.bergs.step_h
      const tb = Math.min(p.tH, 72)
      for (const b of p.scenario.bergs.bergs) {
        const sel = b.id === p.selectedBerg
        const line = (pts: [number, number][], color: string, width: number, dash: number[] = []) => {
          ctx.save()
          ctx.setLineDash(dash)
          ctx.strokeStyle = color
          ctx.lineWidth = width
          ctx.beginPath()
          pts.forEach(([x, y], i) => {
            const [sx, sy] = S(x, y)
            if (i === 0) ctx.moveTo(sx, sy)
            else ctx.lineTo(sx, sy)
          })
          ctx.stroke()
          ctx.restore()
        }
        line(b.observed, 'rgba(220,230,240,0.55)', 1.2, [2, 3])
        line(b.forecast, sel ? '#ff6b6b' : 'rgba(255,107,107,0.75)', sel ? 2.4 : 1.4, [5, 3])
        if (p.overlays.verify) line(b.truth, '#7dff9a', sel ? 2 : 1.2)
        if (sel && p.overlays.verify) line(b.physics_only, '#c38bff', 1.6, [2, 2])
        if (sel) {
          ctx.fillStyle = 'rgba(255,140,140,0.55)'
          for (const [x, y] of b.members_72h) {
            const [sx, sy] = S(x, y)
            ctx.fillRect(sx - 1.2, sy - 1.2, 2.4, 2.4)
          }
        }
        // uncertainty ellipse at current time
        const e = b.ellipses.reduce((best, cur) => (Math.abs(cur.t_h - tb) < Math.abs(best.t_h - tb) ? cur : best))
        const [px, py] = lerpTrack(b.forecast, step, tb)
        const [sx, sy] = S(px, py)
        ctx.save()
        ctx.translate(sx, sy)
        ctx.rotate((-e.angle * Math.PI) / 180)
        ctx.beginPath()
        ctx.ellipse(0, 0, Math.max(e.a * view.scale, 2), Math.max(e.b * view.scale, 2), 0, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(255,90,90,0.12)'
        ctx.strokeStyle = 'rgba(255,120,120,0.7)'
        ctx.lineWidth = 1
        ctx.fill()
        ctx.stroke()
        ctx.restore()
        const size = Math.max(5, Math.min(12, 4 + Math.log2(b.length_m / 300) * 1.6))
        ctx.beginPath()
        ctx.moveTo(sx, sy - size)
        ctx.lineTo(sx + size * 0.9, sy + size * 0.6)
        ctx.lineTo(sx - size * 0.9, sy + size * 0.6)
        ctx.closePath()
        ctx.fillStyle = sel ? '#ffffff' : '#ff5a5f'
        ctx.strokeStyle = sel ? '#ff5a5f' : '#2a0b0d'
        ctx.lineWidth = sel ? 2.5 : 1.2
        ctx.fill()
        ctx.stroke()
        if (view.scale > 0.14 || sel) label(ctx, b.id, sx + size + 3, sy + 3, sel ? '#ffffff' : '#ff9a9e')
      }
    }

    // vessel (follows the selected route)
    const act = activeOption(p.plan, p.activeRoute)
    if (act) {
      const pts = act.ev.points
      const at = alongRoute(pts, p.tH)
      if (at) {
        const nxt = pts[Math.min(at.i + 1, pts.length - 1)]
        const prv = pts[Math.max(at.i - 1, 0)]
        const [sx, sy] = S(at.x, at.y)
        const [nx, ny] = S(nxt.x, nxt.y)
        const [qx, qy] = S(prv.x, prv.y)
        const ang = Math.atan2(ny - qy, nx - qx)
        ctx.save()
        ctx.translate(sx, sy)
        ctx.rotate(ang)
        ctx.shadowColor = '#ffffff'
        ctx.shadowBlur = 12
        ctx.fillStyle = '#ffffff'
        ctx.beginPath()
        ctx.moveTo(11, 0)
        ctx.lineTo(-7, 6)
        ctx.lineTo(-3, 0)
        ctx.lineTo(-7, -6)
        ctx.closePath()
        ctx.fill()
        ctx.restore()
      }
    }

    // pick-mode hint
    if (p.pickMode) {
      ctx.fillStyle = 'rgba(49,225,247,0.12)'
      ctx.fillRect(0, 0, view.w, 30)
      ctx.fillStyle = '#9ff0ff'
      ctx.font = '600 13px system-ui, Segoe UI, sans-serif'
      ctx.fillText(`Click the map to set the ${p.pickMode === 'origin' ? 'departure' : 'destination'} point`, 14, 20)
    }

    drawScaleBar(ctx, view)
  }, [view, raster, p.coast, p.overlays, p.scenario, p.plan, p.activeRoute, p.tH, p.selectedBerg, p.meta.places, p.pickMode])

  useEffect(() => {
    const id = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(id)
  }, [draw])

  // ---------------------------------------------------------------- interaction
  useEffect(() => {
    const cv = canvasRef.current
    if (!cv) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = cv.getBoundingClientRect()
      const sx = e.clientX - rect.left
      const sy = e.clientY - rect.top
      setView((v) => {
        const [wx, wy] = toWorld(v, sx, sy)
        const scale = Math.min(Math.max(v.scale * Math.exp(-e.deltaY * 0.0015), 0.03), 3)
        return { ...v, scale, cx: wx - (sx - v.w / 2) / scale, cy: wy + (sy - v.h / 2) / scale }
      })
    }
    cv.addEventListener('wheel', onWheel, { passive: false })
    return () => cv.removeEventListener('wheel', onWheel)
  }, [])

  const sample = (x: number, y: number) => {
    if (!decoded) return {}
    const i = Math.floor((x + ICE_HALF) / p.meta.grid.cell_km)
    const j = Math.floor((y + ICE_HALF) / p.meta.grid.cell_km)
    if (i < 0 || j < 0 || i >= N || j >= N) return { sic: 0 }
    const k = j * N + i
    const s = Math.min(p.slot, decoded.sic.length - 1)
    return {
      sic: decoded.sic[s][k],
      thick: decoded.thick[s][k] / 50,
      rio: decoded.rio?.[s][k],
      level: decoded.level?.[s][k],
    }
  }

  const onDown = (e: React.MouseEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy, moved: false }
  }
  const onMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top
    const d = drag.current
    if (d) {
      const dx = e.clientX - d.x
      const dy = e.clientY - d.y
      if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true
      if (d.moved) setView((v) => ({ ...v, cx: d.cx - dx / v.scale, cy: d.cy + dy / v.scale }))
    }
    const [wx, wy] = toWorld(view, sx, sy)
    const [lat, lon] = toLatLon(wx, wy)
    setHover({ sx, sy, lat, lon, ...sample(wx, wy) })
  }
  const onUp = (e: React.MouseEvent) => {
    const d = drag.current
    drag.current = null
    if (d && d.moved) return
    const rect = canvasRef.current!.getBoundingClientRect()
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top
    const [wx, wy] = toWorld(view, sx, sy)
    if (p.pickMode) {
      const [lat, lon] = toLatLon(wx, wy)
      p.onPick(lat, lon)
      return
    }
    if (p.scenario && p.overlays.bergs) {
      const tb = Math.min(p.tH, 72)
      let best: string | null = null
      let bd = 14
      for (const b of p.scenario.bergs.bergs) {
        const [bx, by] = toScreen(view, ...lerpTrack(b.forecast, p.scenario.bergs.step_h, tb))
        const dd = Math.hypot(bx - sx, by - sy)
        if (dd < bd) {
          bd = dd
          best = b.id
        }
      }
      p.onSelectBerg(best)
    }
  }

  return (
    <div ref={wrapRef} className="map-wrap">
      <canvas
        ref={canvasRef}
        style={{ width: view.w, height: view.h, cursor: p.pickMode ? 'crosshair' : drag.current ? 'grabbing' : 'grab' }}
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={() => {
          drag.current = null
          setHover(null)
        }}
        aria-label="Polar stereographic navigation chart"
        role="img"
      />
      {hover && (
        <div className="map-tip" style={{ left: Math.min(hover.sx + 14, view.w - 190), top: Math.min(hover.sy + 14, view.h - 110) }}>
          <div className="mono">{fmtLat(hover.lat)} {fmtLon(hover.lon)}</div>
          {hover.sic !== undefined && <div>SIC <b>{hover.sic}%</b>{hover.thick !== undefined && hover.sic > 2 ? <> · {hover.thick.toFixed(2)} m</> : null}</div>}
          {hover.rio !== undefined && hover.sic !== undefined && hover.sic > 2 && (
            <div>
              RIO <b>{hover.rio}</b> ·{' '}
              <span style={{ color: LEVEL_COLORS[hover.level ?? 0] }}>{LEVEL_SHORT[hover.level ?? 0]}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ helpers
function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  ctx.font = '600 11px system-ui, Segoe UI, sans-serif'
  ctx.lineWidth = 3
  ctx.strokeStyle = 'rgba(4,12,22,0.9)'
  ctx.strokeText(text, x, y)
  ctx.fillStyle = color
  ctx.fillText(text, x, y)
}

function drawGraticule(ctx: CanvasRenderingContext2D, v: View) {
  ctx.save()
  ctx.strokeStyle = 'rgba(120,160,200,0.18)'
  ctx.lineWidth = 1
  const [px, py] = toScreen(v, 0, 0)
  for (let lat = -80; lat <= -30; lat += 10) {
    const r = Math.hypot(...toXY(lat, 0)) * v.scale
    ctx.beginPath()
    ctx.arc(px, py, r, 0, Math.PI * 2)
    ctx.stroke()
    const [lx, ly] = toScreen(v, ...toXY(lat + 0.6, 45))
    ctx.fillStyle = 'rgba(150,185,215,0.6)'
    ctx.font = '10px system-ui, sans-serif'
    ctx.fillText(`${-lat}°S`, lx, ly)
  }
  for (let lon = -180; lon < 180; lon += 30) {
    const [x1, y1] = toScreen(v, ...toXY(-88, lon))
    const [x2, y2] = toScreen(v, ...toXY(-25, lon))
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()
    const [lx, ly] = toScreen(v, ...toXY(-41, lon + 1))
    ctx.fillStyle = 'rgba(150,185,215,0.6)'
    ctx.fillText(lon === 0 ? '0°' : lon === -180 ? '180°' : `${Math.abs(lon)}°${lon < 0 ? 'W' : 'E'}`, lx, ly)
  }
  ctx.restore()
}

function drawWind(ctx: CanvasRenderingContext2D, v: View, w: Scenario['wind']) {
  const n = w.n
  const step = w.step_km
  ctx.save()
  ctx.lineWidth = 1.2
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const u = w.u[j * n + i]
      const vv = w.v[j * n + i]
      const spd = Math.hypot(u, vv)
      const x = -ICE_HALF + step * (i + 0.125)
      const y = -ICE_HALF + step * (j + 0.125)
      const [sx, sy] = toScreen(v, x, y)
      if (sx < -20 || sy < -20 || sx > v.w + 20 || sy > v.h + 20) continue
      const len = Math.min(spd * 1.6, 34) * Math.min(1, v.scale / 0.1)
      const ex = sx + (u / (spd + 1e-6)) * len
      const ey = sy - (vv / (spd + 1e-6)) * len
      const t = Math.min(spd / 25, 1)
      ctx.strokeStyle = `rgba(${150 + 105 * t},${220 - 90 * t},${255 - 170 * t},0.75)`
      ctx.beginPath()
      ctx.moveTo(sx, sy)
      ctx.lineTo(ex, ey)
      const a = Math.atan2(ey - sy, ex - sx)
      ctx.lineTo(ex - 4 * Math.cos(a - 0.5), ey - 4 * Math.sin(a - 0.5))
      ctx.moveTo(ex, ey)
      ctx.lineTo(ex - 4 * Math.cos(a + 0.5), ey - 4 * Math.sin(a + 0.5))
      ctx.stroke()
    }
  }
  ctx.restore()
}

function drawScaleBar(ctx: CanvasRenderingContext2D, v: View) {
  const target = 120 / v.scale
  const nice = [50, 100, 200, 250, 500, 1000, 2000].reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a))
  const px = nice * v.scale
  const x = v.w - px - 20
  const y = v.h - 22
  ctx.save()
  ctx.strokeStyle = '#cfe3f5'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(x, y - 5)
  ctx.lineTo(x, y)
  ctx.lineTo(x + px, y)
  ctx.lineTo(x + px, y - 5)
  ctx.stroke()
  label(ctx, `${nice} km @71°S`, x, y - 9, '#cfe3f5')
  ctx.restore()
}
