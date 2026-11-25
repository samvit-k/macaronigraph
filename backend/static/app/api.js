// HTTP client for the Phase 4B API. No DOM access here (replaceable by any frontend).

async function request(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty or non-JSON body */ }
  if (!res.ok) {
    const detail = data && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return data;
}

export const ask = (payload) => request("POST", "/api/ask", payload);
export const requestStatus = (id) => request("GET", `/api/ask/${encodeURIComponent(id)}`);
export const listConversations = () => request("GET", "/api/conversations");
export const getConversation = (id) => request("GET", `/api/conversations/${encodeURIComponent(id)}`);
export const deleteConversation = (id) => request("DELETE", `/api/conversations/${encodeURIComponent(id)}`);
