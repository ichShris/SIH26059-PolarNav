import { useState } from 'react'

// Grouped vertical bars with a legend, direct labels on the first series, and hover tooltips.
// Palette validated for CVD separation and contrast on the dark panel surface.
export const SERIES_COLORS = ['#239fc4', '#c97a36', '#8a70e6']

interface Props {
  groups: string[]
  series: { name: string; values: number[] }[]
  unit: string
  title: string
  lowerIsBetter?: boolean
  height?: number
}

export default function GroupedBars({ groups, series, unit, title, lowerIsBetter = true, height = 150 }: Props) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null)
  const W = 300
  const H = height
  const padL = 34
  const padB = 22
  const padT = 14
  const max = Math.max(...series.flatMap((s) => s.values), 1e-9) * 1.1
  const gw = (W - padL) / groups.length
  const bw = Math.min(18, (gw - 16) / series.length - 2)
  const y = (v: number) => padT + (H - padT - padB) * (1 - v / max)
  const ticks = [0, max / 2, max].map((t) => Math.round(t * 10) / 10)
  return (
    <figure className="chart">
      <figcaption>
        {title} <span className="muted">({unit}{lowerIsBetter ? ', lower is better' : ''})</span>
      </figcaption>
      <div className="legend-row">
        {series.map((s, i) => (
          <span key={s.name}>
            <i style={{ background: SERIES_COLORS[i] }} />
            {s.name}
          </span>
        ))}
      </div>
      <div style={{ position: 'relative' }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={title} onMouseLeave={() => setTip(null)}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={padL} x2={W} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={1} />
              <text x={padL - 5} y={y(t) + 3} textAnchor="end" className="axis">
                {t}
              </text>
            </g>
          ))}
          {groups.map((g, gi) => {
            const x0 = padL + gi * gw + (gw - series.length * (bw + 2)) / 2
            return (
              <g key={g}>
                {series.map((s, si) => {
                  const v = s.values[gi]
                  const x = x0 + si * (bw + 2)
                  const top = y(v)
                  const h = H - padB - top
                  return (
                    <g key={s.name}>
                      <path
                        d={`M${x},${H - padB} V${top + 4} Q${x},${top} ${x + 4},${top} H${x + bw - 4} Q${x + bw},${top} ${x + bw},${top + 4} V${H - padB} Z`}
                        fill={SERIES_COLORS[si]}
                      />
                      {si === 0 && h > 12 && (
                        <text x={x + bw / 2} y={top - 3} textAnchor="middle" className="val">
                          {v.toFixed(1)}
                        </text>
                      )}
                      <rect
                        x={x - 1}
                        y={padT}
                        width={bw + 2}
                        height={H - padT - padB}
                        fill="transparent"
                        onMouseMove={() => setTip({ x: ((x + bw / 2) / W) * 100, y: top, text: `${s.name} · ${g}: ${v.toFixed(2)} ${unit}` })}
                      />
                    </g>
                  )
                })}
                <text x={padL + gi * gw + gw / 2} y={H - 6} textAnchor="middle" className="axis">
                  {g}
                </text>
              </g>
            )
          })}
        </svg>
        {tip && (
          <div className="chart-tip" style={{ left: `${tip.x}%`, top: (tip.y / H) * 100 + '%' }}>
            {tip.text}
          </div>
        )}
      </div>
    </figure>
  )
}
