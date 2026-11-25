// Cytoscape stylesheet (structure adapted from SanD Hacks: small nodes with a hairline ink border, labels to the
// right, thin straight edges, wide invisible hit-twins for clickable relationships). Cytoscape ignores rgba() alpha,
// so transparency uses *-opacity properties.
import { CONCEPT_TYPES, EDGE_KINDS, REEL } from './vocab.js'

const FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif'
const DASH = { dashed: [6, 4], dotted: [2, 2.5] }
const MACARONI = '#e3a33b'

export function buildStyle() {
  const style = [
    {
      selector: 'node',
      style: {
        width: 'data(size)', height: 'data(size)', shape: 'ellipse',
        'background-color': '#8a909c', 'background-opacity': 1, 'z-index': 100, 'border-width': 0.85, 'border-color': '#1c2028', 'border-opacity': 0.45,
        label: 'data(displayLabel)', 'font-family': FONT, 'font-size': 12.5, 'font-weight': 450, color: '#1c2028',
        'text-valign': 'center', 'text-halign': 'right', 'text-margin-x': 6, 'text-wrap': 'ellipsis',
        'text-max-width': 180, 'min-zoomed-font-size': 8, 'overlay-opacity': 0,
        'text-opacity': 0, // no labels by default: the number appears only on the hovered or selected node
        'transition-property': 'opacity, text-opacity, border-width', 'transition-duration': 140,
      },
    },
    {
      selector: 'node.reel',
      style: { 'background-color': REEL.color, 'font-weight': 550 },
    },
    { selector: 'node.concept.single', style: { 'background-blacken': -0.45, 'font-size': 11, color: '#4a505c', 'border-opacity': 0.25, 'min-zoomed-font-size': 9 } },
    { selector: 'node.concept.bridge', style: { 'font-weight': 550 } },
    { selector: 'node[?expanded]', style: { 'border-width': 1.8, 'border-style': 'double', 'border-opacity': 0.8 } },
    // hovered, selected, matched and cited nodes always show their label, whatever the zoom
    // the one visible label (hovered / selected node): readable at any zoom, drawn above neighbouring nodes
    { selector: 'node.hover, node:selected', style: { 'min-zoomed-font-size': 0, 'text-opacity': 1, 'z-index': 120,
      // lfs / lpad are set by GraphCanvas from the zoom, so the label keeps a constant on-screen size
      'font-size': 'data(lfs)', 'text-background-padding': 'data(lpad)', 'text-margin-x': 'data(lpad)', 'font-weight': 650,
      color: '#111318', 'text-background-color': '#ffffff', 'text-background-opacity': 0.95,
      'text-background-shape': 'roundrectangle', 'text-border-color': '#d9dbe0', 'text-border-width': 'data(lbw)', 'text-border-opacity': 1 } },
    { selector: 'node.hover', style: { 'border-width': 1.8, 'border-opacity': 1 } },
    { selector: 'node:selected', style: { 'border-width': 2.4, 'border-color': '#111318', 'border-opacity': 1 } },
    { selector: 'node.match', style: { 'font-weight': 700 } },
    // answer highlight: a warm macaroni halo around cited Reels and the concepts of cited passages
    { selector: 'node.cited', style: { 'underlay-color': MACARONI, 'underlay-opacity': 0.45, 'underlay-padding': 6,
      'underlay-shape': 'ellipse', 'border-width': 2, 'border-opacity': 1 } },

    { selector: 'edge', style: { width: 1, 'curve-style': 'straight', 'line-color': '#1c2028', 'line-opacity': 0.25,
      'overlay-opacity': 0, 'z-index': 1, 'transition-property': 'opacity, width', 'transition-duration': 140 } },
    { selector: 'edge.mention', style: { 'z-index': 2 } },
    { selector: 'edge.assertion', style: { 'z-index': 5, 'target-arrow-shape': 'triangle', 'arrow-scale': 0.85 } },
    { selector: 'edge.assertion-hit', style: { width: 14, 'line-opacity': 0, 'z-index': 4, 'transition-duration': 0 } },
    { selector: 'edge.mention.hover', style: { width: 2.4, 'line-opacity': 1 } },
    { selector: 'edge.assertion.hover', style: { width: 3.4, 'line-opacity': 1 } },
    { selector: 'edge:selected', style: { width: 3, 'line-opacity': 1, 'z-index': 9 } },
    { selector: 'edge.status-verified', style: { label: '✓', 'font-size': 10, color: '#433db8', 'text-background-color': '#ffffff',
      'text-background-opacity': 1, 'text-background-padding': 1.5 } },
  ]
  for (const [type, t] of Object.entries(CONCEPT_TYPES)) {
    style.splice(2, 0, { selector: `node.concept.t-${type}`, style: { 'background-color': t.color } })
  }
  for (const [kind, k] of Object.entries(EDGE_KINDS)) {
    const selector = kind === 'mention' ? 'edge.mention' : `edge.assertion.status-${kind}`
    style.push({ selector, style: {
      'line-color': k.color, 'target-arrow-color': k.color, 'line-opacity': k.opacity, width: k.width,
      'line-style': k.line === 'solid' ? 'solid' : 'dashed', ...(DASH[k.line] ? { 'line-dash-pattern': DASH[k.line] } : {}) } })
  }
  style.push(
    { selector: '.faded', style: { opacity: 0.16, 'text-opacity': 0 } },
    { selector: '.hidden', style: { display: 'none' } },
  )
  return style
}
