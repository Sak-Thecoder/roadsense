import {useEffect, useState} from 'react'
import {Link} from 'react-router-dom'
import {ErrorNotice, Loading, Outcome, PageHeading, Scores, Empty} from '../components/UI'
import {api} from '../services/api'
import {useScan} from '../state'
import type {Prediction} from '../types/api'

export default function Result() {
  const {active, models, photo, preview, record} = useScan()
  const [result, setResult] = useState<Prediction>()
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    if (!photo || !active) return
    const controller = new AbortController()
    setResult(undefined); setError('')
    api.predict(active, photo, controller.signal).then(value => {
      if (!controller.signal.aborted) {setResult(value);record(value)}
    }).catch(e => {if (!controller.signal.aborted) setError(e.message)})
    return () => controller.abort()
  }, [active, photo, record, attempt])
  return <>
    <PageHeading title="Scan result">Review the model’s road quality classification.</PageHeading>
    {photo && active ? <div className="analyze-grid">
      <div className="photo-frame has-photo">{preview && <img src={preview} alt="Analyzed road surface"/>}</div>
      <div className="stack">
        {!result && !error && <Loading>Analyzing road surface…</Loading>}
        {error && <><ErrorNotice>{error}</ErrorNotice><button className="button secondary" onClick={() => setAttempt(n => n+1)}>Retry</button></>}
        {result && <>
          <Outcome prediction={result}/><Scores prediction={result}/>
          {models.find(model => model.model === result.model)?.use_scope === 'cambodian_capture_date_pilot'
            && <p className="notice">Pilot prediction. Accuracy on other road locations is unknown.</p>}
          <p className="subtle">{result.confidence < .6 ? 'Low confidence. Try a clearer photo or compare models. ' : ''}Confidence is a model score. Images of the same road can produce different results.</p>
          <Link className="button primary" to="/compare">Compare this photo</Link>
        </>}
        <Link className="button secondary" to="/">Back to analyze</Link>
      </div>
    </div> : <><Empty title="Choose an image and a model">Start a scan on the Analyze page.</Empty><Link className="button primary" to="/">Go to Analyze</Link></>}
  </>
}
