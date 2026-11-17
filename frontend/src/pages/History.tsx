import {Clock3} from 'lucide-react'
import {Empty, PageHeading, QualityTag, percent} from '../components/UI'
import {useScan} from '../state'
import {names} from '../types/api'

export default function History() {
  const {scans, clear} = useScan()
  return <>
    <PageHeading title="Scan history">Recent classifications saved in this browser.</PageHeading>
    <section className="card">
      {!scans.length ? <Empty icon={Clock3} title="No scans yet">Your completed scans will appear here.</Empty> :
        <><div className="row spread"><h2>Recent scans</h2><button className="text-button" onClick={clear}>Clear history</button></div>
        <ul className="recent">{scans.map(scan=><li key={scan.id} className="row spread">
          <div><strong>{names[scan.model]}</strong><p className="subtle">{new Date(scan.date).toLocaleString()} · {percent(scan.confidence)} confidence</p></div>
          <QualityTag quality={scan.prediction}/>
        </li>)}</ul></>}
    </section>
  </>
}
