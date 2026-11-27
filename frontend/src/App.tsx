import {lazy, Suspense} from 'react'
import {ChartNoAxesColumn, Clock3, Layers3, ScanLine} from 'lucide-react'
import {NavLink, Navigate, Route, Routes} from 'react-router-dom'
import Home from './pages/Home'
import Capture from './pages/Capture'
import Result from './pages/Result'
import Compare from './pages/Compare'
import {Loading} from './components/UI'
// Lazy loading: the Benchmark page (with the charting library) is downloaded only when first opened.
const Benchmark = lazy(() => import('./pages/Benchmark'))
import History from './pages/History'
import {useScan} from './state'

// Navigation menu entries: URL, text and icon.
const links = [
  {to:'/', label:'Analyze', icon:ScanLine}, {to:'/compare', label:'Compare', icon:Layers3},
  {to:'/benchmark', label:'Benchmark', icon:ChartNoAxesColumn}, {to:'/history', label:'History', icon:Clock3},
]

/** Page layout: header with API status, navigation bar, and the page that matches the URL. */
export default function App() {
  const {status, models} = useScan()
  // Number of models the API reported as ready to use.
  const count = models.filter(m => m.available).length
  return <div className="app-shell">
    <header className="site-header"><NavLink to="/" className="brand">
      <img className="brand-icon" src="/favicon.svg" alt="" width={38} height={38}/><span>RoadSense <small>AUPP · Road surface quality</small></span>
    </NavLink>
    <span className={`status ${status}`}><span/>{status === 'loading' ? 'Connecting' : status === 'offline' ? 'API offline' : `${count} model${count===1?'':'s'} available`}</span></header>
    <nav className="navigation" aria-label="Main navigation">
      {links.map(({to,label,icon:Icon}) => <NavLink to={to} end={to==='/'} key={to}><Icon size={20} aria-hidden/><span>{label}</span></NavLink>)}
    </nav>
    <main className="main"><Routes>
      <Route path="/" element={<Home/>}/><Route path="/capture" element={<Capture/>}/>
      <Route path="/result" element={<Result/>}/><Route path="/compare" element={<Compare/>}/>
      <Route path="/benchmark" element={<Suspense fallback={<Loading>Opening benchmark…</Loading>}><Benchmark/></Suspense>}/><Route path="/history" element={<History/>}/>
      <Route path="*" element={<Navigate to="/" replace/>}/>
    </Routes></main>
  </div>
}
