import {ArrowRight, ChartNoAxesColumn, Layers3, ScanLine} from 'lucide-react'
import {Link, useNavigate} from 'react-router-dom'
import {PhotoPicker} from '../components/PhotoPicker'
import {ActionLink, ErrorNotice, Loading, PageHeading, QualityTag} from '../components/UI'
import {useScan} from '../state'
import {names, type Model} from '../types/api'

/** Analyze page (home): choose a photo and a model, then go to the result page. */
export default function Home() {
  const {models, status, active, setActive, photo, scans, retry} = useScan()
  const navigate = useNavigate()
  // Whether the chosen model is the small Cambodian pilot (shown with a warning).
  const pilot = models.find(model => model.model === active)?.use_scope === "cambodian_capture_date_pilot"
  return <>
    <PageHeading title="Analyze road">Capture or upload an image to classify the visible road surface.</PageHeading>
    <div className="analyze-grid">
      <div className="stack compact"><PhotoPicker/><ActionLink to="/capture">Open capture workspace</ActionLink></div>
      <div className="stack">
        <section className="card model-panel">
          <div className="section-label"><ScanLine size={17}/> SINGLE-IMAGE SCAN</div>
          <h2>Prediction model</h2>
          <label className="field-label" htmlFor="active-model">Active model</label>
          <select id="active-model" value={active} onChange={e => setActive(e.target.value as Model)} disabled={!models.some(m => m.available)}>
            {!active && <option value="">No model available</option>}
            {models.map(model => <option key={model.model} value={model.model} disabled={!model.available}>{model.name}{model.available ? '' : ' · unavailable'}</option>)}
          </select>
          {status === 'loading' && <Loading>Checking model availability…</Loading>}
          {status === 'offline' && <><ErrorNotice>The API is unavailable.</ErrorNotice><button className="text-button" onClick={retry}>Retry connection</button></>}
          {status === 'ready' && !active && <>
            <p className="subtle">Your photo can be previewed now. Analyze becomes available when a trained model is connected.</p>
            <button className="text-button" onClick={retry}>Refresh models</button>
          </>}
          {pilot && <p className="subtle">Local pilot trained on one Cambodian capture session. Accuracy on other locations is unknown.</p>}
          <button className="button primary full" disabled={!active || !photo || status !== 'ready'} onClick={() => navigate('/result')}>Analyze photo <ArrowRight size={17}/></button>
        </section>
        <section className="card"><div className="row spread"><h2>Recent scans</h2><ActionLink to="/history">History</ActionLink></div>
          {scans.length === 0 ? <p className="subtle">Completed scans will appear here.</p> :
            <ul className="recent">{scans.slice(0,3).map(scan => <li className="row spread" key={scan.id}><span>{names[scan.model]}</span><QualityTag quality={scan.prediction}/></li>)}</ul>}
        </section>
        <div className="quick-links">
          <Link to="/compare" className="card quick"><Layers3 size={18}/>Compare models</Link>
          <Link to="/benchmark" className="card quick"><ChartNoAxesColumn size={18}/>Benchmark</Link>
        </div>
      </div>
    </div>
  </>
}
