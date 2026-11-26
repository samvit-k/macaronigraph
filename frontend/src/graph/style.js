// Cytoscape stylesheet; Cytoscape ignores rgba() alpha, so transparency uses *-opacity properties.
import { CONCEPT_TYPES, EDGE_KINDS, edgeColor, REEL } from './vocab.js'

const FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif'
const DASH = { dashed: [6, 4], dotted: [2, 2.5] }
const MACARONI = '#e3a33b'

// Colours that must contrast with the stage; node fills keep their hues in both themes.
export const INK = {
  light: { line: '#1c2028', label: '#1c2028', faintLabel: '#4a505c', strong: '#111318', tagBg: '#ffffff', tagLine: '#d9dbe0',
    verified: '#433db8', singleBlacken: -0.45 },
  dark: { line: '#c9ced6', label: '#dfe2e8', faintLabel: '#9aa0ac', strong: '#f2f4f7', tagBg: '#1d2027', tagLine: '#3a3f4a',
    verified: '#b5b0ff', singleBlacken: 0.35 },
}

export function buildStyle(theme = 'light') {
  const ink = INK[theme] ?? INK.light
  const style = [
    {
      selector: 'node',
      style: {
        width: 'data(size)', height: 'data(size)', shape: 'ellipse',
        'background-color': '#8a909c', 'background-opacity': 1, 'z-index': 100, 'border-width': 0.85, 'border-color': theme === 'dark' ? '#000000' : ink.line, 'border-opacity': theme === 'dark' ? 0.6 : 0.45,
        label: 'data(displayLabel)', 'font-family': FONT, 'font-size': 12.5, 'font-weight': 450, color: ink.label,
        'text-valign': 'center', 'text-halign': 'right', 'text-margin-x': 6, 'text-wrap': 'ellipsis',
        'text-max-width': 180, 'min-zoomed-font-size': 8, 'overlay-opacity': 0,
        'text-opacity': 0,
        'transition-property': 'opacity, text-opacity, border-width', 'transition-duration': 140,
      },
    },
    {
      selector: 'node.reel',
      style: { 'background-color': REEL.color, 'font-weight': 550 },
    },
    { selector: 'node.concept.single', style: { 'background-blacken': ink.singleBlacken, 'font-size': 11, color: ink.faintLabel, 'border-opacity': 0.25, 'min-zoomed-font-size': 9 } },
    { selector: 'node.concept.bridge', style: { 'font-weight': 550 } },
    { selector: 'node[?expanded]', style: { 'border-width': 1.8, 'border-style': 'double', 'border-opacity': 0.8 } },

    { selector: 'node.hover, node:selected', style: { 'min-zoomed-font-size': 0, 'text-opacity': 1, 'z-index': 120,

      'font-size': 'data(lfs)', 'text-background-padding': 'data(lpad)', 'text-margin-x': 'data(lpad)', 'font-weight': 650,
      color: ink.strong, 'text-background-color': ink.tagBg, 'text-background-opacity': 0.95,
      'text-background-shape': 'roundrectangle', 'text-border-color': ink.tagLine, 'text-border-width': 'data(lbw)', 'text-border-opacity': 1 } },
    { selector: 'node.hover', style: { 'border-width': 1.8, 'border-opacity': 1 } },
    { selector: 'node:selected', style: { 'border-width': 2.4, 'border-color': ink.strong, 'border-opacity': 1 } },
    { selector: 'node.match', style: { 'font-weight': 700 } },

    { selector: 'node.cited', style: { 'underlay-color': MACARONI, 'underlay-opacity': 0.45, 'underlay-padding': 6,
      'underlay-shape': 'ellipse', 'border-width': 2, 'border-opacity': 1 } },

    { selector: 'edge', style: { width: 1, 'curve-style': 'straight', 'line-color': ink.line, 'line-opacity': 0.25,
      'overlay-opacity': 0, 'z-index': 1, 'transition-property': 'opacity, width', 'transition-duration': 140 } },
    { selector: 'edge.mention', style: { 'z-index': 2 } },
    { selector: 'edge.assertion', style: { 'z-index': 5, 'target-arrow-shape': 'triangle', 'arrow-scale': 0.85 } },
    { selector: 'edge.assertion-hit', style: { width: 14, 'line-opacity': 0, 'z-index': 4, 'transition-duration': 0 } },
    { selector: 'edge.mention.hover', style: { width: 2.4, 'line-opacity': 1 } },
    { selector: 'edge.assertion.hover', style: { width: 3.4, 'line-opacity': 1 } },
    { selector: 'edge:selected', style: { width: 3, 'line-opacity': 1, 'z-index': 9 } },
    { selector: 'edge.status-verified', style: { label: '✓', 'font-size': 10, color: ink.verified, 'text-background-color': ink.tagBg,
      'text-background-opacity': 1, 'text-background-padding': 1.5 } },
  ]
  for (const [type, t] of Object.entries(CONCEPT_TYPES)) {
    style.splice(2, 0, { selector: `node.concept.t-${type}`, style: { 'background-color': t.color } })
  }
  for (const [kind, k] of Object.entries(EDGE_KINDS)) {
    const selector = kind === 'mention' ? 'edge.mention' : `edge.assertion.status-${kind}`
    const color = edgeColor(kind, theme)
    style.push({ selector, style: {
      'line-color': color, 'target-arrow-color': color, 'line-opacity': k.opacity, width: k.width,
      'line-style': k.line === 'solid' ? 'solid' : 'dashed', ...(DASH[k.line] ? { 'line-dash-pattern': DASH[k.line] } : {}) } })
  }
  style.push(
    { selector: '.faded', style: { opacity: 0.16, 'text-opacity': 0 } },
    { selector: '.hidden', style: { display: 'none' } },
  )
  return style
}