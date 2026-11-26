// Shared vocabulary for canvas, legend, filters and inspector (colours mirror styles/tokens.css).
// A mention means a Reel names a concept; two Reels mentioning one concept is never a relationship between them.

export const REEL = { label: 'Reel', color: '#7b76e0', help: 'A saved Instagram Reel that has been processed.' }

export const CONCEPT_TYPES = {
  concept: { label: 'Concept', color: '#3a8f86' },
  method: { label: 'Method', color: '#b7791f' },
  tool: { label: 'Tool', color: '#4f7cc4' },
  organization: { label: 'Organization', color: '#9a5a8a' },
  technology: { label: 'Technology', color: '#5f8f3e' },
  resource: { label: 'Resource', color: '#c0613e' },
  person: { label: 'Person', color: '#6b7280' },
}
export const CONCEPT_ORDER = ['concept', 'method', 'tool', 'organization', 'technology', 'resource', 'person']
export const conceptType = (t) => CONCEPT_TYPES[t] ?? { label: t ? t[0].toUpperCase() + t.slice(1) : 'Concept', color: '#8a909c' }

export const EDGE_KINDS = {
  mention: {
    label: 'Mention', line: 'dotted', color: '#5d6370', dark: '#9aa1ad', width: 1.5, opacity: 0.8,
    help: 'The Reel names this concept. A shared concept between Reels is co-mention, not a relationship.',
  },
  proposed: {
    label: 'Proposed Relationship', line: 'dashed', color: '#a8794a', dark: '#d1a06d', width: 2, opacity: 0.95,
    help: 'Extracted by the local model from one Reel; its quote was checked against the passage. Not a verified fact.',
  },
  corroborated: {
    label: 'Corroborated Relationship', line: 'solid', color: '#2f7a55', dark: '#5fbf8d', width: 2.4, opacity: 0.95,
    help: 'The same relationship was proposed independently in two or more Reels. Still not human-verified.',
  },
  verified: {
    label: 'Human-Verified Relationship', line: 'solid', color: '#5b55d6', dark: '#8f89ff', width: 3, opacity: 1,
    help: 'A person reviewed this relationship and accepted it.',
  },
}
export const EDGE_ORDER = ['mention', 'proposed', 'corroborated', 'verified']
export const edgeColor = (kind, theme = 'light') => (theme === 'dark' ? EDGE_KINDS[kind]?.dark : null) ?? EDGE_KINDS[kind]?.color
export const ASSERTION_STATUSES = ['proposed', 'corroborated', 'verified']

export const PREDICATES = {
  USES: 'uses', USED_FOR: 'used for', DEPENDS_ON: 'depends on', ALTERNATIVE_TO: 'alternative to',
  CONTRASTS_WITH: 'contrasts with', IS_A: 'is a', PART_OF: 'part of', RECOMMENDS: 'recommends',
}
export const predicateLabel = (p) => PREDICATES[p] ?? (p ? p.toLowerCase().replace(/_/g, ' ') : 'related to')

export const CONTENT_TYPES = {
  transcript: { label: 'Transcript', note: 'Speech transcribed locally; may contain recognition errors.' },
  caption: { label: 'Caption', note: 'The Reel’s caption text.' },
  ocr: { label: 'On-Screen Text', note: 'Read from video frames (OCR); may be misread.' },
  ocr_title: { label: 'On-Screen Title', note: 'Read from video frames (OCR); may be misread.' },
  ocr_screen: { label: 'On-Screen Text', note: 'Read from video frames (OCR); may be misread.' },
  ocr_fragment: { label: 'On-Screen Fragment', note: 'Read from video frames (OCR); may be misread.' },
}
export const contentType = (t) => CONTENT_TYPES[t] ?? { label: t || 'Source', note: '' }
export const isOcr = (t) => typeof t === 'string' && t.startsWith('ocr')

export function timestamp(start) {
  if (!Number.isFinite(start)) return null
  const s = Math.max(0, Math.floor(start))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}