// Real-browser checks of the workspace and welcome page against a running backend (installed Chrome, throwaway profile).
// Usage: node e2e/workspace.e2e.mjs [baseUrl=http://127.0.0.1:8765] [outDir=../reports/phase5/screens] [--offline]
import { mkdirSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { chromium } from 'playwright-core'

const BASE = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : 'http://127.0.0.1:8765'
const OUT = resolve(process.argv[3] && !process.argv[3].startsWith('--') ? process.argv[3] : '../reports/phase5/screens')
const OFFLINE = process.argv.includes('--offline')

const OFFLINE_TOPIC = process.env.E2E_TOPIC ?? 'Python'

async function pickTopics() {
  const o = await (await fetch(`${BASE}/api/graph/overview`)).json()
  const out = []
  for (const c of (o.concepts ?? []).filter((x) => x.bridge).sort((a, b) => b.n_reels - a.n_reels)) {
    const n = await (await fetch(`${BASE}/api/graph/neighborhood?kind=concept&id=${encodeURIComponent(c.key)}`)).json()
    const literal = (n.reels ?? []).some((r) => (r.evidence ?? []).some((e) => (e.surface ?? '').toLowerCase() === c.name.toLowerCase()))
    if (literal && /^[\w .+-]+$/.test(c.name)) out.push(c.name)
    if (out.length === 2) break
  }
  if (out.length < 2) throw new Error('need two literal bridge concepts in the graph')
  return out
}
const esc = (t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
mkdirSync(OUT, { recursive: true })

const results = []
const created = new Set()
let preexisting = new Set()
const check = (name, ok, detail = '') => {
  results.push({ name, ok: !!ok, detail })
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? ` — ${detail}` : ''}`)
}
const shot = async (page, name) => page.screenshot({ path: resolve(OUT, `${name}.png`) })

async function nodeCount(page) {
  return page.evaluate(() => window.__mgCy?.nodes().filter((n) => n.style('display') !== 'none').length ?? 0)
}
async function clickNode(page, id, dbl = false) {
  const p = await page.evaluate((nid) => {
    const n = window.__mgCy.getElementById(nid)
    if (!n.nonempty()) return null
    const box = window.__mgCy.container().getBoundingClientRect()
    const rp = n.renderedPosition()
    return { x: box.left + rp.x, y: box.top + rp.y }
  }, id)
  if (!p) throw new Error(`node ${id} not on canvas`)
  if (dbl) await page.mouse.dblclick(p.x, p.y)
  else await page.mouse.click(p.x, p.y)
}
async function ask(page, text) {
  const before = await page.locator('[data-testid="exchange"]').count()
  await page.getByLabel('Ask your saved Reels').fill(text)
  await page.getByLabel('Ask your saved Reels').press('Enter')
  await page.locator('[data-testid="exchange"]').nth(before).waitFor({ timeout: 240000 })
  const ex = page.locator('[data-testid="exchange"]').nth(before)
  await ex.locator('.answer-box, .status-card, .clarify-card').first().waitFor({ timeout: 240000 })
  return ex
}

const browser = await chromium.launch({ channel: 'chrome', headless: true })
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2, acceptDownloads: true })
const page = await context.newPage()
const consoleErrors = []
page.on('console', (m) => {
  if (m.type() !== 'error') return

  if (OFFLINE && /status of 503/.test(m.text())) return
  consoleErrors.push(m.text())
})
page.on('response', async (r) => {
  if (r.url().endsWith('/api/ask') && r.request().method() === 'POST') {
    try { const j = await r.json(); if (j?.conversation_id) created.add(j.conversation_id) } catch {  }
  }
})

try {
  // conversations that existed before the run are never reopened, changed or deleted
  preexisting = new Set((await (await fetch(`${BASE}/api/conversations`)).json()).map((c) => c.id))
  await page.goto(`${BASE}/workspace/?e2e`, { waitUntil: 'networkidle' })
  if (OFFLINE) {
    await page.getByRole('heading', { name: 'Knowledge graph offline' }).waitFor({ timeout: 20000 })
    check('offline: graph offline card shown', true)
    check('offline: search disabled', await page.getByLabel('Search concepts and Reels').isDisabled())
    await shot(page, 'offline-workspace')
    const ex = await ask(page, `What do my Reels say about ${OFFLINE_TOPIC}?`)
    const text = await ex.innerText()
    check('offline: chat still answers (passage-search fallback)', /Generated Answer|Exact Search|Not Enough Evidence/i.test(text),
      text.slice(0, 80).replace(/\n/g, ' '))
    check('offline: fallback note shown', /passage search only/i.test(text))
    if (await ex.locator('.cite-chip').count()) {
      await ex.locator('.cite-chip').first().click()
      check('offline: citation opens source excerpt', await page.locator('.inspector .excerpt').first().isVisible())
    }
    await shot(page, 'offline-chat')
  } else {
    await page.waitForFunction(() => (window.__mgCy?.nodes().length ?? 0) > 20, null, { timeout: 30000 })
    const n0 = await nodeCount(page)
    check('overview renders real graph', n0 > 50, `${n0} nodes`)
    check('status chip shows live status', /Reels/.test(await page.getByTestId('status-chip').innerText()))
    check('graph meta shows totals', /\d+\/\d+ concepts/i.test(await page.getByTestId('graph-meta').innerText()))
    await page.waitForTimeout(700)
    await shot(page, '01-overview')

    const [A, B] = await pickTopics()
    console.log(`topics: ${A}, ${B}`)

    await page.getByLabel('Search concepts and Reels').fill(A.toLowerCase())
    await page.getByRole('option').first().waitFor()
    await shot(page, '02-search')
    await page.getByRole('option', { name: new RegExp(`^${esc(A)}`, 'i') }).first().click()
    await page.locator('.inspector .insp-title', { hasText: A }).waitFor()
    await page.locator('.inspector .excerpt').first().waitFor({ timeout: 15000 })
    check('search → concept inspector with source excerpts', true)
    check('inspector links to the original Reel', (await page.locator('.inspector a[href*="instagram.com/reel/"]').count()) > 0)
    await page.waitForTimeout(600)
    await shot(page, '03-concept-inspector')

    await page.locator('.inspector .reel-ev .link-btn').first().click()
    await page.locator('.inspector .insp-kicker', { hasText: 'Reel' }).waitFor()
    const before = await nodeCount(page)
    await page.getByRole('button', { name: 'Show connected nodes' }).click()
    await page.waitForFunction((b) => window.__mgCy.nodes().length > b, before, { timeout: 15000 })
    const after = await nodeCount(page)
    check('expand Reel adds its concepts', after > before, `${before} → ${after} nodes`)
    await page.waitForTimeout(800)
    await shot(page, '04-reel-expanded')
    await page.getByRole('button', { name: 'Collapse' }).click()
    await page.waitForTimeout(300)
    check('collapse removes them again', (await nodeCount(page)) <= before + 1, `${await nodeCount(page)} nodes`)

    const mentionOk = await page.evaluate(() => {
      const e = window.__mgCy.edges('.mention').filter((x) => x.style('display') !== 'none')[0]
      if (!e) return false
      e.emit('tap')
      return true
    })
    if (mentionOk) {
      await page.locator('.inspector .insp-kicker', { hasText: 'Mention' }).waitFor()
      check('mention edge → evidence, labelled "not a relationship"', /not a relationship/i.test(await page.locator('.inspector').innerText()))
    } else check('mention edge → evidence', false, 'no mention edge')

    const hasAssertion = await page.evaluate(() => {
      const e = window.__mgCy.edges('.assertion-hit')[0]
      if (!e) return false
      e.emit('tap')
      return true
    })
    if (hasAssertion) {
      await page.locator('.inspector .insp-kicker', { hasText: 'Extracted relationship' }).waitFor()
      const t = await page.locator('.inspector').innerText()
      check('assertion edge → status + quote', /Proposed|Corroborated|verified/i.test(t) && /Quote from the Reel/i.test(t))
      await shot(page, '05-assertion')
    }

    await page.getByRole('button', { name: /^Filter/ }).click()
    await page.getByRole('dialog', { name: 'Graph filters' }).getByLabel('Mention').uncheck()
    const hiddenMentions = await page.evaluate(() => window.__mgCy.edges('.mention').every((e) => e.style('display') === 'none'))
    check('filter hides mention edges', hiddenMentions)
    await shot(page, '06-filters')
    await page.getByRole('button', { name: 'Reset filters' }).click()
    await page.mouse.click(10, 300)
    check('reset filters restores mentions', await page.evaluate(() => window.__mgCy.edges('.mention').some((e) => e.style('display') !== 'none')))

    await page.locator('.legend .tg').click()
    check('legend collapses to a pill', await page.locator('.legend-pill').isVisible())
    await page.locator('.legend-pill').click()

    const l = await ask(page, `Which Reels mention ${A}?`)
    check('listing answered by exact search', /Exact search · no model/i.test(await l.innerText()))
    await l.locator('.cite-chip').first().click()
    await page.locator('.inspector .insp-kicker', { hasText: 'Cited source' }).waitFor()
    check('citation → original excerpt with evidence type', await page.locator('.inspector .excerpt .ev-type').first().isVisible())
    await shot(page, '07-listing-citation')

    const g = await ask(page, `What do my Reels say about ${A}?`)
    check('generated answer labelled as model output', /Generated answer · local model/i.test(await g.innerText()))
    await g.getByRole('button', { name: 'Show in graph' }).click()
    await page.waitForTimeout(900)
    const cited = await page.evaluate(() => window.__mgCy.nodes('.cited').length)
    check('answer highlights cited Reels/concepts in the graph', cited > 0, `${cited} highlighted`)
    await shot(page, '08-answer-highlight')

    await page.getByRole('button', { name: 'New' }).click()
    await ask(page, `What do my Reels say about ${A} and ${B}?`)
    const c = await ask(page, 'Is it free?')
    const ct = await c.innerText()
    check('ambiguous follow-up gets a clarification with choices', /Which one do you mean/.test(ct) && (await c.locator('.chip-btn').count()) >= 2)
    await c.locator('.chip-btn').first().click()
    check('clarification choice fills the composer (not sent silently)', /\(about /.test(await page.getByLabel('Ask your saved Reels').inputValue()))
    await shot(page, '09-clarification')
    await page.getByLabel('Ask your saved Reels').fill('')

    await page.getByRole('button', { name: 'History' }).click()
    const rows = await page.locator('.history-row').count()
    check('conversation history lists conversations', rows >= 2, `${rows} conversations`)
    const sheetW = await page.locator('.history-sheet').evaluate((el) => el.getBoundingClientRect().width)
    const panelW = await page.locator('#assistant-panel').evaluate((el) => el.getBoundingClientRect().width)
    check('history sheet spans the chat panel', sheetW >= panelW - 2, `${Math.round(sheetW)} / ${Math.round(panelW)} px`)
    check('Clear All is offered next to the heading', await page.locator('.history-head').getByRole('button', { name: 'Clear All' }).isVisible())
    // only a conversation this run created may be deleted, never one that matches by title alone
    const ownIds = [...created].filter((c) => !preexisting.has(c))
    const delId = await page.locator('.history-item').evaluateAll((els, own) => {
      const mine = els.filter((el) => own.includes(el.dataset.id))
      return (mine.find((el) => /Which Reels mention/.test(el.textContent)) || mine[0])?.dataset.id ?? null
    }, ownIds)
    if (!delId) throw new Error('no conversation created by this run is listed; refusing to delete any other')
    const delItem = page.locator(`.history-item[data-id="${delId}"]`)
    await delItem.hover()
    await page.waitForTimeout(200)
    check('hovering a conversation reveals its delete ×', Number(await delItem.locator('.history-del').evaluate((el) => getComputedStyle(el).opacity)) > 0.9)
    await shot(page, '10-history')
    const histBefore = await page.locator('.history-item').count()
    await delItem.locator('.history-del').click()
    await page.waitForFunction((b) => document.querySelectorAll('.history-item').length === b - 1, histBefore, { timeout: 10000 })
    check('× deletes that conversation (backend updated)', true)
    const testRow = page.locator('.history-item').filter({ has: page.locator('.h-title', { hasText: `say about ${A} and` }) })
      .and(page.locator(ownIds.filter((c) => c !== delId).map((c) => `[data-id="${c}"]`).join(', ') || '[data-id="none"]'))
      .first().locator('.history-row')
    await testRow.click()
    await page.locator('[data-testid="exchange"]').first().waitFor()
    check('reopening a conversation restores its turns', (await page.locator('[data-testid="exchange"]').count()) >= 1)

    const u = await ask(page, 'Who won the 2026 Super Bowl?')
    check('insufficient evidence shown as its own state', /Not Enough Evidence/.test(await u.innerText()))

    for (const [w, h] of [[1100, 820], [800, 900]]) {
      await page.setViewportSize({ width: w, height: h })
      await page.waitForTimeout(500)
      await shot(page, `11-responsive-${w}`)
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1)
      check(`no horizontal overflow at ${w}px`, !overflow)
    }
    await page.setViewportSize({ width: 1440, height: 900 })

    await page.goto(`${BASE}/welcome`, { waitUntil: 'networkidle' })
    check('welcome headline', /A knowledge graph of your saved Reels/.test(await page.locator('h1').innerText()))
    check('welcome footer is only the copyright', (await page.locator('footer').innerText()).trim() === '© 2026 Samvit Kundapur')
    const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: 'Download Extension (.zip)' }).click()])
    check('download button delivers the extension zip', dl.suggestedFilename() === 'macaronigraph-extension.zip')
    await page.getByRole('link', { name: 'Open Workspace' }).first().click()
    await page.waitForURL(/\/workspace\//)
    check('welcome → workspace link works', true)
    await page.goto(`${BASE}/welcome`, { waitUntil: 'networkidle' })
    await shot(page, '12-welcome')
    await page.setViewportSize({ width: 800, height: 1000 })
    await shot(page, '13-welcome-800')
  }
  check('no console errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '))
} catch (e) {
  check('e2e run completed', false, e.message)
  await shot(page, 'failure').catch(() => {})
} finally {
  for (const id of [...created].filter((c) => !preexisting.has(c))) {
    try { await fetch(`${BASE}/api/conversations/${id}`, { method: 'DELETE' }) } catch {  }
  }
  await browser.close()
  writeFileSync(resolve(OUT, OFFLINE ? 'e2e-offline.json' : 'e2e.json'), JSON.stringify({ base: BASE, results, deleted_conversations: [...created] }, null, 1))
  const failed = results.filter((r) => !r.ok).length
  console.log(`\n${results.length - failed}/${results.length} passed; ${created.size} test conversations deleted`)
  process.exit(failed ? 1 : 0)
}