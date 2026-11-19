import {useEffect, useRef, useState} from 'react'
import {Camera, RotateCcw, Upload} from 'lucide-react'
import {useScan} from '../state'
import {ErrorNotice} from './UI'

/**
 * Lets the user upload a photo or take one with the device camera.
 * The chosen photo is stored in shared state, so every page sees the same photo.
 */
export function PhotoPicker() {
  const {preview, photo, setPhoto, limit} = useScan()
  const [error, setError] = useState('')
  const [camera, setCamera] = useState<'off'|'pending'|'live'>('off')
  const video = useRef<HTMLVideoElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const stream = useRef<MediaStream | null>(null)
  // Increased every time the camera is stopped. Async camera callbacks compare against it
  // and ignore their result if the user has already closed or restarted the camera.
  const generation = useRef(0)
  /** Turn off the camera (stopping the tracks switches the camera light off). */
  function stop() {
    generation.current++
    stream.current?.getTracks().forEach(track => track.stop())
    stream.current = null
  }
  // Returning `stop` from an effect with [] makes it run when the component is removed.
  useEffect(() => stop, [])
  // Once the <video> element exists, connect it to the live camera stream.
  useEffect(() => {if (camera === 'live' && video.current) video.current.srcObject = stream.current}, [camera])

  /** Check type and size in the browser first (the server checks again), then store the photo. */
  function choose(file: File) {
    setError('')
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setError('Choose a JPEG, PNG, or WebP image.'); return
    }
    if (file.size > limit) {setError(`Choose an image under ${Math.round(limit / 1024 / 1024)} MB.`);return}
    stop(); setCamera('off');setPhoto(file)
  }
  /** Ask permission and start the camera; 'environment' prefers the rear camera on phones. */
  async function openCamera() {
    setError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access requires HTTPS or localhost and a supported browser. You can upload a photo instead.')
      return
    }
    stop()
    const request = generation.current
    setCamera('pending')
    try {
      const media = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false})
      // The camera was closed while we waited for permission: release it straight away.
      if (generation.current !== request) {media.getTracks().forEach(t => t.stop());return}
      stream.current = media; setCamera('live')
    } catch {
      if (generation.current !== request) return
      setCamera('off')
      setError('The camera could not start. Check camera permissions or upload a photo instead.')
    }
  }
  /** Take a still photo: draw the current video frame on a canvas and save it as a JPEG file. */
  function capture() {
    const element = video.current
    if (!element || !element.videoWidth || !element.videoHeight) {setError('The camera is still starting. Try again in a moment.');return}
    const canvas = document.createElement('canvas')
    canvas.width = element.videoWidth
    canvas.height = element.videoHeight
    const context = canvas.getContext('2d')
    if (!context) {setError('Could not capture a frame. Upload a photo instead.');return}
    context.drawImage(element, 0, 0)
    const request = generation.current
    canvas.toBlob(blob => {
      if (request !== generation.current) return
      if (blob) choose(new File([blob], 'roadsense-capture.jpg', {type:'image/jpeg'}))
      else setError('Could not capture a frame. Upload a photo instead.')
    }, 'image/jpeg', .87)  // 87% JPEG quality
  }
  return <section className="photo-section" aria-label="Road image">
    <div className={`photo-frame ${preview || camera === 'live' ? 'has-photo' : ''}`}>
      {camera === 'live' ? <video ref={video} autoPlay playsInline muted aria-label="Live road camera"/>
        : preview ? <img src={preview} alt="Selected road surface"/>
        : <div className="photo-empty">
          <div className="road-art" aria-hidden><div className="road-bed"/><i/><i/><i/></div>
          <span className="viewfinder" aria-hidden/>
          <div className="photo-copy"><Camera size={24} aria-hidden/><strong>Add a road photo</strong>
            <span>Keep the road surface clearly visible inside the frame.</span></div>
        </div>}
      {camera === 'pending' && <span className="camera-overlay" role="status">Waiting for camera permission…</span>}
    </div>
    <div className="photo-actions">
      {camera === 'live' ? <button className="button primary" onClick={capture}><Camera size={18}/> Capture photo</button>
        : <button className="button primary" onClick={openCamera} disabled={camera === 'pending'}>{photo ? <RotateCcw size={17}/> : <Camera size={18}/>} {photo ? 'Retake photo' : 'Take photo'}</button>}
      <button className="button secondary" onClick={() => input.current?.click()}><Upload size={18}/> Upload photo</button>
      <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" hidden aria-label="Choose a road photo"
        onChange={e => {const file=e.target.files?.[0]; if(file) choose(file); e.target.value=''}}/>
    </div>
    <p className="photo-formats">JPEG, PNG or WebP · Up to {Math.round(limit / 1024 / 1024)} MB</p>
    {camera !== 'off' && <button className="text-button" onClick={() => {stop();setCamera('off')}}>Close camera</button>}
    {error && <ErrorNotice>{error}</ErrorNotice>}
  </section>
}
