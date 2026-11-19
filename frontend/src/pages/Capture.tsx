import {ArrowRight} from 'lucide-react'
import {Link, useNavigate} from 'react-router-dom'
import {PhotoPicker} from '../components/PhotoPicker'
import {PageHeading} from '../components/UI'
import {useScan} from '../state'
import {names} from '../types/api'

/** Larger camera/upload workspace before analysing a photo. */
export default function Capture() {
  const {active, photo, status} = useScan()
  const navigate = useNavigate()
  return <>
    <PageHeading title="Capture road photo">Frame the visible road surface and check the preview before submitting.</PageHeading>
    <div className="analyze-grid">
      <PhotoPicker/>
      <section className="card stack">
        <h2>Ready to analyze?</h2>
        <p className="subtle">{active ? `Active model · ${names[active]}` : 'No prediction model is currently available.'}</p>
        <button className="button primary" disabled={!active || !photo || status !== 'ready'} onClick={() => navigate('/result')}>
          Analyze photo <ArrowRight size={17}/>
        </button>
        <Link className="button secondary" to="/">Back to Analyze</Link>
      </section>
    </div>
  </>
}
