// Pure rendering: API response objects → DOM nodes. All text is inserted as text (never as HTML).

const el = (tag, attrs = {}, ...children) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) n.append(c instanceof Node ? c : String(c));
  return n;
};

const fmtTime = (s) => (s === null || s === undefined ? "" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`);
const kindLabel = { transcript: "spoken", caption: "caption", ocr: "on-screen", ocr_screen: "on-screen",
                    ocr_title: "on-screen title", ocr_fragment: "on-screen (uncertain)" };

export function channelBadges(channels) {
  return channels.map((ch) => {
    if (ch.type === "semantic") return el("span", { class: "badge semantic", title: "semantic similarity" }, "semantic");
    if (ch.type === "shared_concept")
      return el("span", { class: "badge comention", title: ch.note }, `shared concept: ${ch.concept_name}`);
    return el("span", { class: `badge relation ${ch.status}`, title: "model-extracted relation" },
              `${ch.subject} ${ch.predicate} ${ch.object} (${ch.status})`);
  });
}

export function sourceCard(s, idPrefix) {
  const when = s.start !== null && s.start !== undefined ? ` · ${fmtTime(s.start)}–${fmtTime(s.end)}` : "";
  return el("details", { class: "source", id: `${idPrefix}-${s.sid}` },
    el("summary", {},
      el("b", {}, s.sid), " ",
      el("a", { href: s.source_url, target: "_blank", rel: "noopener" }, `Reel ${s.reel_id}`),
      s.owner ? ` @${s.owner}` : "", ` · ${kindLabel[s.content_type] || s.content_type}${when} `,
      ...channelBadges(s.channels || [])),
    el("p", { class: "passage" }, s.text),
    el("p", { class: "meta" }, `chunk ${s.chunk_id}${s.similarity !== null && s.similarity !== undefined ? ` · similarity ${s.similarity}` : ""}`));
}

function answerBlock(r, idPrefix, onCite) {
  const a = r.answer || {};
  const st = a.status;
  if (r.status === "queued" || st === "deferred")
    return el("div", { class: "answer pending" }, el("span", { class: "spinner" }), " Answer deferred: ",
              a.reason || "the local model is busy", ". The sources below are already available.");
  if (st === "unavailable" || r.status === "retrieval_only")
    return el("div", { class: "answer muted" }, a.reason ? `No answer generated: ${a.reason}` : "Retrieval only (no answer generated).");
  if (st === "expired")
    return el("div", { class: "answer muted" }, `No answer: ${a.reason || "expired"}`);
  if (st === "error")
    return el("div", { class: "answer error" }, a.text || "The answer could not be generated.", a.error ? ` (${a.error})` : "");
  if (st === "insufficient_evidence")
    return el("div", { class: "answer insufficient" }, a.text);
  const p = el("div", { class: "answer" });
  for (const s of a.sentences || []) {
    const span = el("span", { class: `sentence ${s.kind}${s.support === "weak" ? " weak" : ""}`,
                              title: s.support === "weak" ? "low word overlap with the cited sources (diagnostic)" : null },
                    s.kind === "inference" ? el("em", {}, "Inference: ") : null, s.text, " ");
    for (const c of s.citations)
      span.append(el("a", { href: `#${idPrefix}-${c}`, class: "cite", onclick: (e) => { e.preventDefault(); onCite(`${idPrefix}-${c}`); } }, `[${c}]`));
    p.append(span, " ");
  }
  if (a.conflict) p.append(el("p", { class: "note" }, "The sources disagree; each claim is cited separately."));
  if (a.missing) p.append(el("p", { class: "note" }, a.missing));
  if ((a.removed || []).length)
    p.append(el("details", { class: "removed" }, el("summary", {}, `${a.removed.length} unsupported sentence(s) removed`),
      el("ul", {}, a.removed.map((x) => el("li", {}, `${x.text || "(empty)"} — ${x.reason}`)))));
  return p;
}

function connectionsBlock(r) {
  const c = r.connections || {};
  const box = el("div", { class: "connections" });
  if ((r.concepts || []).length)
    box.append(el("p", {}, el("b", {}, "Concepts: "),
      r.concepts.map((k) => el("span", { class: `concept${k.expanded ? "" : " dim"}`,
        title: `in ${k.df} Reel(s) · specificity ${k.specificity}${k.expanded ? "" : " · too common to expand"}` }, k.name))));
  if ((c.relations || []).length)
    box.append(el("p", {}, el("b", {}, "Graph relations (model-extracted): "),
      c.relations.map((x) => el("span", { class: `badge relation ${x.status}`, title: x.note },
        `${x.subject} ${x.predicate} ${x.object} — ${x.status}`))));
  if ((c.co_mentions || []).length)
    box.append(el("p", {}, el("b", {}, "Shared-concept connections "), el("span", { class: "muted" }, "(not stated relationships): "),
      c.co_mentions.map((x) => el("span", { class: "badge comention" }, `${x.concept}: ${x.reels.length} Reels`))));
  for (const m of c.mentions || [])
    box.append(el("p", {}, el("b", {}, `Reels mentioning ${m.concept}: `), m.reels.join(", "),
      m.complete ? "" : el("span", { class: "muted" }, ` (only ${m.in_pack.length} shown as sources)`)));
  return box;
}

export function turn(r, onCite) {
  const id = `t${r.turn_id}`;
  const fallback = (r.fallback || []).map((f) => el("p", { class: "banner" },
    `Retrieval fell back from ${f.from} to ${f.to}: ${f.reason}`));
  const rewritten = r.standalone_query && r.standalone_query !== r.question
    ? el("p", { class: "muted" }, `Interpreted as: “${r.standalone_query}”`) : null;
  return el("article", { class: "turn", id },
    el("p", { class: "question" }, r.question),
    rewritten, ...fallback,
    el("p", { class: "muted" }, `mode: ${r.mode_used}${r.mode_used !== r.mode_requested ? ` (requested ${r.mode_requested})` : ""}`),
    answerBlock(r, id, onCite),
    connectionsBlock(r),
    el("details", { class: "sources", open: (r.answer || {}).status !== "ok" },
      el("summary", {}, `${(r.sources || []).length} source passage(s)`),
      (r.sources || []).map((s) => sourceCard(s, id))));
}

export function loading(question) {
  return el("article", { class: "turn loading" }, el("p", { class: "question" }, question),
            el("div", { class: "answer pending" }, el("span", { class: "spinner" }), " Retrieving and answering locally…"));
}

export function error(question, message) {
  return el("article", { class: "turn" }, el("p", { class: "question" }, question),
            el("div", { class: "answer error" }, `Request failed: ${message}`));
}

export function conversationItem(c, active, onOpen, onDelete) {
  return el("li", { class: active ? "active" : "" },
    el("button", { type: "button", class: "open", onclick: onOpen }, c.title || "(untitled)"),
    el("button", { type: "button", class: "delete", title: "Delete conversation", onclick: onDelete }, "×"));
}
