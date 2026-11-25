// Confirmation for a permanent action (adapted from SanD Hacks): consequences stated exactly, runs only after an
// explicit click, errors shown in place and never retried silently.
import { useEffect, useRef, useState } from 'react'

export default function ConfirmDialog({ title, children, confirmLabel, busyLabel = 'Working…', onConfirm, onCancel }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const cancelRef = useRef(null)
  useEffect(() => { cancelRef.current?.focus() }, [])
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onCancel() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onCancel])
  const run = async () => {
    setBusy(true)
    setError(null)
    try {
      await onConfirm()
    } catch (err) {
      setError(err?.message ?? 'Nothing was changed.')
      setBusy(false)
    }
  }
  return (
    <div className="dlg-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onCancel() }}>
      <div className="dlg" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title">
        <h2 id="confirm-title">{title}</h2>
        <div className="dlg-body">{children}</div>
        {error && <div className="warn-card" role="alert">{error}</div>}
        <div className="dlg-foot">
          <button type="button" className="btn btn-sm" ref={cancelRef} onClick={onCancel} disabled={busy}>Cancel</button>
          <button type="button" className="btn btn-sm btn-danger" disabled={busy} onClick={run}>{busy ? busyLabel : confirmLabel}</button>
        </div>
      </div>
    </div>
  )
}