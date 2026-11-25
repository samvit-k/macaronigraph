// Viewport fitting that leaves room for the legend and stops following once the user pans or zooms.
export const FIT_PADDING = 48
export const MAX_FIT_ZOOM = 1.25
export const GRAPH_MIN_ZOOM = 0.02
export const GRAPH_MAX_ZOOM = 3
const LEGEND_MARGIN = 12
// Cytoscape emits these only for user gestures; programmatic pan/zoom/fit never do.
export const USER_VIEWPORT_EVENTS = 'dragpan scrollzoom pinchzoom'

export const LEGEND_EVENT = 'macaronigraph:legend'

export function legendInsets(size, legend) {
  const none = { left: 0, top: 0, right: 0, bottom: 0 }
  if (!legend || legend.width <= 0 || legend.height <= 0) return none
  const left = Math.min(size.width, legend.right + LEGEND_MARGIN)
  const bottom = Math.min(size.height, size.height - legend.top + LEGEND_MARGIN)
  const areaBeside = Math.max(0, size.width - left) * size.height
  const areaAbove = size.width * Math.max(0, size.height - bottom)
  return areaBeside >= areaAbove ? { ...none, left } : { ...none, bottom }
}

export function fitViewport(bb, size, insets = {}, { padding = FIT_PADDING, maxZoom = MAX_FIT_ZOOM, minZoom = GRAPH_MIN_ZOOM } = {}) {
  const ins = { left: 0, top: 0, right: 0, bottom: 0, ...insets }
  const availW = Math.max(1, size.width - ins.left - ins.right - 2 * padding)
  const availH = Math.max(1, size.height - ins.top - ins.bottom - 2 * padding)
  const needed = Math.min(availW / Math.max(bb.w, 1), availH / Math.max(bb.h, 1))
  const zoom = Math.max(minZoom, Math.min(maxZoom, needed))
  const cx = (bb.x1 + bb.x2) / 2
  const cy = (bb.y1 + bb.y2) / 2
  const centerX = ins.left + (size.width - ins.left - ins.right) / 2
  const centerY = ins.top + (size.height - ins.top - ins.bottom) / 2
  return { zoom, pan: { x: centerX - cx * zoom, y: centerY - cy * zoom }, clipped: needed < minZoom }
}

export function keepCenterPan(pan, oldSize, newSize) {
  return { x: pan.x + (newSize.width - oldSize.width) / 2, y: pan.y + (newSize.height - oldSize.height) / 2 }
}

export function createAutoFit(cy, getLegend = () => null, { animate = true } = {}) {
  let following = true
  let resizing = false // re-entrancy guard: cy.resize() must never re-run this handler
  let lastFit = null
  const measure = () => ({ width: cy.width(), height: cy.height() })
  let size = measure()

  const limits = () => ({ minZoom: cy.minZoom?.() ?? GRAPH_MIN_ZOOM, maxZoom: Math.min(MAX_FIT_ZOOM, cy.maxZoom?.() ?? MAX_FIT_ZOOM) })
  const target = () => {
    const eles = cy.elements(':visible')
    if (!eles.length) return null
    const current = measure()
    return fitViewport(eles.boundingBox(), current, legendInsets(current, getLegend()), limits())
  }
  const stopAnimations = () => cy.stop?.(true, false)
  const apply = (view, duration) => {
    if (!view) return
    lastFit = view
    stopAnimations()
    if (animate && duration) cy.animate({ zoom: view.zoom, pan: view.pan }, { duration, easing: 'ease-in-out-cubic' })
    else { cy.zoom(view.zoom); cy.pan(view.pan) }
  }
  const onUser = () => { following = false }
  cy.on(USER_VIEWPORT_EVENTS, onUser)

  return {
    get following() { return following },

    get lastFit() { return lastFit },

    fit(duration = 0) {
      following = true
      apply(target(), duration)
    },

    userAdjusted() { following = false },

    resized(duration = 0) {
      if (resizing) return
      resizing = true
      try {
        cy.resize()
        const next = measure()
        if (next.width === size.width && next.height === size.height) return
        const prev = size
        size = next
        if (following) apply(target(), duration)
        else {
          stopAnimations()
          cy.pan(keepCenterPan(cy.pan(), prev, next))
        }
      } finally {
        resizing = false
      }
    },

    refitIfFollowing(duration = 0) {
      if (following) apply(target(), duration)
    },
    legendChanged(duration = 0) {
      if (following) apply(target(), duration)
    },
    dispose() { cy.off(USER_VIEWPORT_EVENTS, onUser) },
  }
}