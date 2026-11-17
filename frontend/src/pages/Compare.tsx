import {useEffect, useRef, useState} from 'react'
import {Layers3} from 'lucide-react'
import {PhotoPicker} from '../components/PhotoPicker'
import {Empty, ErrorNotice, Loading, PageHeading, QualityTag, percent, Scores} from '../components/UI'
import {api, ApiError} from '../services/api'
import {useScan} from '../state'
import {names, type Comparison, type Model} from '../types/api'

export default function Compare() {
  const {models, photo, setActive, active} = useScan()
  const [result, setResult] = useState<Comparison>()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const invalidPhoto = error instanceof ApiError && error.status === 422
  const req = useRef<AbortController | null>(null)
  useEffect(() => {
    setResult(undefined); setError(null);setLoading(false)
    return () => req.current?.abort()
  }, [photo])
  async function compare() {
    if (!photo) return
    req.current?.abort()
    const controller = new AbortController()
    req.current = controller
    setLoading(true);setResult(undefined);setError(null)
    try {const comparison = await api.compare(photo, controller.signal);if (!controller.signal.aborted) setResult(comparison)}
    catch (e) {if (!controller.signal.aborted) setError(e as Error)}
    finally {if (!controller.signal.aborted) setLoading(false)}
  }
  const highest = result && result.predictions.length > 1 ? Math.max(...result.predictions.map(p => p.confidence)) : null
  return <>
    <PageHeading title="Compare models">Run the same photo through each available RoadSense model.</PageHeading>
    <div className="analyze-grid">
      <PhotoPicker/>
      <div className="stack">
        <button className="button primary" onClick={compare} disabled={!photo || loading || invalidPhoto || !models.some(m => m.available)}><Layers3 size={18}/> Run comparison</button>
        {loading && <Loading>Comparing the available models…</Loading>}
        {error && <ErrorNotice>{error.message}</ErrorNotice>}
        {!loading && !result && !error && <Empty icon={Layers3} title="Model comparison">{models.some(m => m.available) ? 'Add a photo to see how the available models classify it.' : 'Connect a trained model to compare classifications.'}</Empty>}
        {result && new Set(result.predictions.map(p => p.prediction)).size > 1 && <p className="notice">The models assigned different road quality classes to this image.</p>}
        {result && (Object.keys(names) as Model[]).map(model => {
          const prediction = result.predictions.find(p => p.model === model)
          const info = models.find(m => m.model === model)
          return <div className="card stack compact" key={model}>
            <div className="row spread"><h2>{names[model]}</h2>{prediction && <QualityTag quality={prediction.prediction}/>}</div>
            {prediction ? <>
              <p className="meta">{percent(prediction.confidence)} confidence · {prediction.inference_ms.toFixed(1)} ms{info?.size_mb != null ? ` · ${info.size_mb.toFixed(2)} MB` : ''}</p>
              {highest === prediction.confidence && <p className="subtle">Highest confidence in this comparison.</p>}
              {info?.use_scope === 'cambodian_capture_date_pilot' && <p className="subtle">Local pilot model</p>}
              <Scores prediction={prediction}/>
              <button className="button secondary" disabled={active===model || !info?.available} onClick={() => setActive(model)}>{active===model ? 'Active model' : 'Set as active model'}</button>
            </> : <p className="subtle">{result.errors[model] || 'No prediction returned.'}</p>}
          </div>
        })}
      </div>
    </div>
  </>
}
