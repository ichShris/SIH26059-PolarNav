// The route options returned by one planning request, in display order.
import type { RouteEval, RoutePlan } from './types'

export interface RouteOption {
  key: string
  label: string
  color: string
  ev: RouteEval
  dashed?: boolean
  sameAsOptimal?: boolean
}

const ALT_COLORS: Record<string, string> = { fastest: '#e879f9', safest: '#7ee787' }

export function routeOptions(plan: RoutePlan | null): RouteOption[] {
  if (!plan) return []
  const out: RouteOption[] = []
  if (plan.optimal) out.push({ key: 'optimal', label: 'Recommended (your settings)', color: '#31e1f7', ev: plan.optimal })
  for (const a of plan.alternatives ?? []) {
    out.push({ key: a.key, label: a.label, color: ALT_COLORS[a.key] ?? '#cbd5e1', ev: a, sameAsOptimal: a.same_track_as_optimal })
  }
  if (plan.baseline) out.push({ key: 'baseline', label: 'Conventional (shortest)', color: '#f4a259', ev: plan.baseline, dashed: true })
  return out
}

export function activeOption(plan: RoutePlan | null, key: string): RouteOption | null {
  const opts = routeOptions(plan)
  return opts.find((o) => o.key === key) ?? opts[0] ?? null
}
