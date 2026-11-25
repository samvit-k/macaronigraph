// Chat state and flow: conversation selection, asking, polling deferred answers. Uses api.js + render.js only.
import * as api from "./api.js";
import * as view from "./render.js";

const $ = (id) => document.getElementById(id);
const state = { conversationId: null, polling: new Map() };

function openSource(anchorId) {
  const node = document.getElementById(anchorId);
  if (!node) return;
  node.closest("details.sources")?.setAttribute("open", "");
  node.setAttribute("open", "");
  node.scrollIntoView({ behavior: "smooth", block: "center" });
  node.classList.add("flash");
  setTimeout(() => node.classList.remove("flash"), 1200);
}

function show(response, replace) {
  const node = view.turn(response, openSource);
  if (replace) replace.replaceWith(node); else $("thread").append(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  if (response.status === "queued") poll(response.request_id, node);
  return node;
}

function poll(requestId, node) {
  if (state.polling.has(requestId)) return;
  let current = node;
  const timer = setInterval(async () => {
    try {
      const r = await api.requestStatus(requestId);
      if (r.status !== "queued") {
        clearInterval(timer);
        state.polling.delete(requestId);
        current = show(r, current);
      }
    } catch (e) {
      clearInterval(timer);
      state.polling.delete(requestId);
    }
  }, 4000);
  state.polling.set(requestId, timer);
}

async function refreshConversations() {
  const list = $("conversation-list");
  list.replaceChildren();
  try {
    for (const c of await api.listConversations())
      list.append(view.conversationItem(c, c.id === state.conversationId, () => openConversation(c.id), async () => {
        await api.deleteConversation(c.id);
        if (c.id === state.conversationId) newConversation();
        refreshConversations();
      }));
  } catch { /* conversation list is optional */ }
}

async function openConversation(id) {
  state.conversationId = id;
  $("thread").replaceChildren();
  const c = await api.getConversation(id);
  for (const t of c.turns) show(t);
  refreshConversations();
}

function newConversation() {
  state.conversationId = null;
  $("thread").replaceChildren();
  refreshConversations();
}

$("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("ask-button").disabled = true;
  const placeholder = view.loading(question);
  $("thread").append(placeholder);
  try {
    const r = await api.ask({ question, conversation_id: state.conversationId, mode: $("mode").value,
                              generate: $("generate").checked, options: { co_mention: $("co-mention").checked } });
    state.conversationId = r.conversation_id;
    show(r, placeholder);
    $("question").value = "";
    refreshConversations();
  } catch (err) {
    placeholder.replaceWith(view.error(question, err.message));
  } finally {
    $("ask-button").disabled = false;
  }
});

$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("ask-form").requestSubmit(); }
});
$("new-chat").addEventListener("click", newConversation);
refreshConversations();
