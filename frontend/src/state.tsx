import {createContext, useCallback, useContext, useEffect, useState, type ReactNode} from 'react'
import {api} from './services/api'
import {defaultModel, type Model, type ModelInfo, type Prediction} from './types/api'

// A saved scan = an API prediction plus a unique id and the time it was made.
type Scan = Prediction & {id: string; date: string}
// Key used to store the scan history in the browser's localStorage.
const historyKey = 'roadsense.scans.v1'
/** Read the saved history; return an empty list if it is missing or corrupted. */
function savedScans(): Scan[] {
  try {
    const result = JSON.parse(localStorage.getItem(historyKey) || '[]')
    return Array.isArray(result) ? result : []
  } catch { return [] }
}
// React Context lets every page read the same shared state without passing props down manually.
const ScanContext = createContext<ReturnType<typeof useScanState> | null>(null)

/** All shared app state and the actions that change it. */
function useScanState() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [status, setStatus] = useState<'loading'|'ready'|'offline'>('loading')
  const [active, setActive] = useState<Model | ''>('')
  const [photo, setPhoto] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [limit, setLimit] = useState(8 * 1024 * 1024)
  const [scans, setScans] = useState(savedScans)
  // Increasing `attempt` re-runs the effect below, i.e. "retry connecting to the API".
  const [attempt, setAttempt] = useState(0)
  // On start (and on retry): ask the API which models are available and the upload size limit.
  useEffect(() => {
    // AbortController cancels the request if the component unmounts before it finishes.
    const controller = new AbortController()
    Promise.all([api.models(controller.signal), api.health(controller.signal)])
      .then(([available, health]) => {
        setModels(available)
        setLimit(health.max_upload_bytes)
        // Keep the previously chosen model if still available, otherwise use the default model (ResNet50),
        // and fall back to the first available one if the default is missing.
        setActive(previous => available.find(m => m.available && m.model === previous)?.model
          || available.find(m => m.available && m.model === defaultModel)?.model
          || available.find(m => m.available)?.model || '')
        setStatus('ready')
      }).catch(() => {if (!controller.signal.aborted) setStatus('offline')})
    return () => controller.abort()
  }, [attempt])
  // Create a temporary browser URL so the chosen photo can be shown in an <img>,
  // and free it again (revokeObjectURL) when the photo changes, to avoid memory leaks.
  useEffect(() => {
    if (!photo) {setPreview(''); return}
    const url = URL.createObjectURL(photo)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [photo])
  // Save the history to the browser whenever it changes.
  useEffect(() => {
    try { localStorage.setItem(historyKey, JSON.stringify(scans)) } catch { /* Private browsing. */ }
  }, [scans])
  // Add a new scan to the front of the history and keep only the 20 most recent.
  // useCallback keeps the same function object between renders, so effects using it don't re-run.
  const record = useCallback((prediction: Prediction) => setScans(previous => [
    {...prediction, id: crypto.randomUUID?.() || String(Date.now()), date: new Date().toISOString()}, ...previous,
  ].slice(0, 20)), [])
  return {models, status, active, setActive, photo, setPhoto, preview, limit, scans, record,
    clear: () => setScans([]), retry: () => {setStatus('loading'); setAttempt(n => n + 1)}}
}
/** Wrap the app so all pages share one copy of the state. */
export function ScanProvider({children}: {children: ReactNode}) {
  return <ScanContext.Provider value={useScanState()}>{children}</ScanContext.Provider>
}
/** Hook used by pages/components to read the shared state, e.g. `const {photo} = useScan()`. */
export function useScan() {
  const context = useContext(ScanContext)
  if (!context) throw new Error('ScanProvider is required')
  return context
}
