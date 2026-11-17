import {createContext, useCallback, useContext, useEffect, useState, type ReactNode} from 'react'
import {api} from './services/api'
import type {Model, ModelInfo, Prediction} from './types/api'

type Scan = Prediction & {id: string; date: string}
const historyKey = 'roadsense.scans.v1'
function savedScans(): Scan[] {
  try {
    const result = JSON.parse(localStorage.getItem(historyKey) || '[]')
    return Array.isArray(result) ? result : []
  } catch { return [] }
}
const ScanContext = createContext<ReturnType<typeof useScanState> | null>(null)

function useScanState() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [status, setStatus] = useState<'loading'|'ready'|'offline'>('loading')
  const [active, setActive] = useState<Model | ''>('')
  const [photo, setPhoto] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [limit, setLimit] = useState(8 * 1024 * 1024)
  const [scans, setScans] = useState(savedScans)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    Promise.all([api.models(controller.signal), api.health(controller.signal)])
      .then(([available, health]) => {
        setModels(available)
        setLimit(health.max_upload_bytes)
        setActive(previous => available.find(m => m.available && m.model === previous)?.model
          || available.find(m => m.available)?.model || '')
        setStatus('ready')
      }).catch(() => {if (!controller.signal.aborted) setStatus('offline')})
    return () => controller.abort()
  }, [attempt])
  useEffect(() => {
    if (!photo) {setPreview(''); return}
    const url = URL.createObjectURL(photo)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [photo])
  useEffect(() => {
    try { localStorage.setItem(historyKey, JSON.stringify(scans)) } catch { /* Private browsing. */ }
  }, [scans])
  const record = useCallback((prediction: Prediction) => setScans(previous => [
    {...prediction, id: crypto.randomUUID?.() || String(Date.now()), date: new Date().toISOString()}, ...previous,
  ].slice(0, 20)), [])
  return {models, status, active, setActive, photo, setPhoto, preview, limit, scans, record,
    clear: () => setScans([]), retry: () => {setStatus('loading'); setAttempt(n => n + 1)}}
}
export function ScanProvider({children}: {children: ReactNode}) {
  return <ScanContext.Provider value={useScanState()}>{children}</ScanContext.Provider>
}
export function useScan() {
  const context = useContext(ScanContext)
  if (!context) throw new Error('ScanProvider is required')
  return context
}
