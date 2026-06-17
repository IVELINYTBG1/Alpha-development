#!/usr/bin/env python3
"""
alpha_web.py — live web dashboard for Alpha (an alternative front-end to the TUI).

Runs the SNN brain + camera in one process and serves a browser dashboard:
live camera view, gauges (PHILL / ID / VOICE / feeling / arousal / longing),
region bars, inner-thought stream, the tutor (learning) feed, a chat box, and a
dropdown to switch panels.

Run it INSTEAD of the Rust TUI (only one process can own the webcam):

    source .env && python alpha_web.py
    # then open http://localhost:8200

Uses only the deps already installed (cv2, torch, mediapipe) + Python stdlib.
"""
import sys, os, json, time, threading

# brain.py repoints sys.stdout/stderr to a log on import — keep our console.
_OUT, _ERR = sys.stdout, sys.stderr
import cv2                       # noqa: E402
import numpy as np              # noqa: E402
import vision                   # noqa: E402
import brain                    # noqa: E402
sys.stdout, sys.stderr = _OUT, _ERR

PORT      = int(os.environ.get("ALPHA_WEB_PORT", "8200"))
CAM_INDEX = int(os.environ.get("ALPHA_CAM_INDEX", "0"))

def cprint(*a): print(*a, file=_OUT, flush=True)

cprint("… waking Alpha (loading the SNN) …")
# We own the camera here, so DON'T let the brain start its own CameraThread.
brain._HAS_VISION = False
B = brain.NeuromorphicBrain()
_VBUF = vision.VisualFeatureBuffer()
B._visual_buf = _VBUF                       # step() reads face/motion from here
_PROC = vision.CameraThread(_VBUF)          # used ONLY as a frame processor

_latest_jpeg = [None]
_jpeg_lock   = threading.Lock()
_thoughts    = []      # rolling inner thoughts / proactive lines
_searches    = []      # rolling tutor Q&A
_state       = {}
_state_lock  = threading.Lock()
_stop        = threading.Event()


def _placeholder(msg):
    """A black frame with a message — shown when the camera isn't available."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (12, 8, 6)                      # near-black, faint cosmic warmth
    cv2.putText(img, "ALPHA", (250, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (120, 90, 40), 2)
    for i, line in enumerate(msg.split("\n")):
        cv2.putText(img, line, (60, 270 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 120, 160), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


def cam_loop():
    """Keep trying to own the camera; if it's busy (TUI holding it) show a
    placeholder and retry, so the view comes alive the moment it's released."""
    cap = None
    last_open_try = 0.0
    while not _stop.is_set():
        # (re)open the camera if we don't have it
        if cap is None or not cap.isOpened():
            now = time.time()
            if now - last_open_try < 2.0:        # don't hammer the device
                with _jpeg_lock:
                    _latest_jpeg[0] = _placeholder(
                        "camera unavailable\nis the TUI (alpha_core) still holding it?\nretrying...")
                time.sleep(0.4); continue
            last_open_try = now
            try:
                cap = cv2.VideoCapture(CAM_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            except Exception:
                cap = None
            if cap is None or not cap.isOpened():
                cprint(f"!! camera {CAM_INDEX} busy — showing placeholder, will retry")
                with _jpeg_lock:
                    _latest_jpeg[0] = _placeholder(
                        "camera unavailable\nis the TUI (alpha_core) still holding it?\nretrying...")
                time.sleep(1.5); continue
            cprint(f"camera {CAM_INDEX} acquired.")

        ok, frame = cap.read()
        if not ok:
            try: cap.release()
            except Exception: pass
            cap = None                            # lost the device → reopen loop
            time.sleep(0.2); continue

        h, w = frame.shape[:2]
        live = 1.0
        try:
            feats = _PROC._process_frame(frame, h, w)
            _VBUF.put(feats)
            live = round(float(getattr(B.imprint, "face_live", 1.0)), 2)
            if feats.face_present:
                col = (0, 230, 255)
                cv2.putText(frame, f"face  live={live}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        except Exception:
            pass
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            with _jpeg_lock:
                _latest_jpeg[0] = buf.tobytes()
        time.sleep(1/12.0)

    if cap is not None:
        try: cap.release()
        except Exception: pass


def brain_loop():
    while not _stop.is_set():
        try:
            r = B.step(0.0)
            with _state_lock:
                _state.clear(); _state.update(r)
            for who, t in B.get_leaked_thoughts():
                _thoughts.append({"who": who, "text": t})
            for who, m in B.get_proactive_messages():
                _thoughts.append({"who": "alpha(out)", "text": m})
            del _thoughts[:-60]
            for sp, q, sn in B.get_pending_searches():
                _searches.append({"q": q, "a": sn[:900]})
            del _searches[:-30]
        except Exception:
            pass
        _stop.wait(0.05)


def full_state():
    with _state_lock:
        s = dict(_state)
    try:
        s["sem_concepts"]   = len(B.sem.entries)
        s["imprint_status"] = B.imprint.status()
        s["face_live"]      = round(float(getattr(B.imprint, "face_live", 1.0)), 2)
        s["trusted"]        = bool(B.imprint.trusted)
        s["longing"]        = round(float(getattr(B, "_reach_pressure", 0.0)), 2)
        s["thoughts"]       = _thoughts[-14:]
        s["searches"]       = _searches[-10:]
        s["tutor"]          = (B._search_backend.status() if B._search_backend else "off")
    except Exception:
        pass
    return s


from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Alpha</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#070a12;--pan:#0e1320;--ln:#1c2740;--cy:#78c8ff;--br:#ebf5ff;--dim:#5a6a8c;--al:#e06060}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--br);
 font:14px/1.5 ui-monospace,Menlo,Consolas,monospace}
header{padding:10px 16px;border-bottom:1px solid var(--ln);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0;color:var(--cy);letter-spacing:2px}
.tag{color:var(--dim)} .ok{color:var(--cy)} .al{color:var(--al)}
.wrap{display:grid;grid-template-columns:360px 1fr;gap:14px;padding:14px}
.card{background:var(--pan);border:1px solid var(--ln);border-radius:10px;padding:12px;margin-bottom:14px}
.card h2{font-size:12px;letter-spacing:1px;color:var(--dim);margin:0 0 8px;text-transform:uppercase}
img#cam{width:100%;border-radius:8px;background:#000;aspect-ratio:4/3;object-fit:cover}
.gauge{margin:6px 0} .gauge .lab{display:flex;justify-content:space-between;font-size:12px;color:var(--dim)}
.bar{height:8px;background:#0a0f1a;border-radius:4px;overflow:hidden;margin-top:2px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#2a4a78,var(--cy))}
select{background:#0a0f1a;color:var(--br);border:1px solid var(--ln);border-radius:6px;padding:6px 8px;font:inherit}
.feed{max-height:46vh;overflow:auto;font-size:13px}
.feed .t{padding:4px 0;border-bottom:1px solid #121a2b;color:#aebfe0}
.feed .q{color:var(--cy)} .feed .a{color:var(--dim)}
.region{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
.region .n{width:84px;color:var(--dim)} .region .bar{flex:1}
#chat{display:flex;gap:8px;margin-top:8px} #msg{flex:1;background:#0a0f1a;color:var(--br);
 border:1px solid var(--ln);border-radius:6px;padding:8px}
button{background:#13243c;color:var(--cy);border:1px solid var(--ln);border-radius:6px;padding:8px 12px;cursor:pointer}
.big{font-size:20px;color:var(--br)} .feel{font-size:18px;color:var(--cy)}
</style></head><body>
<header>
  <h1>✦ ALPHA</h1>
  <span class=tag>feeling <span id=feel class=feel>—</span></span>
  <span class=tag>concepts <b id=concepts>—</b></span>
  <span class=tag>ID <b id=idstat>—</b></span>
  <span class=tag>tutor <b id=tutor class=ok>—</b></span>
  <span class=tag id=sleep></span>
</header>
<div class=wrap>
  <div>
    <div class=card><h2>Live camera</h2><img id=cam src="/camera.mjpg">
      <div class=gauge><div class=lab><span>face liveness</span><span id=liveL>—</span></div>
        <div class=bar><i id=live style=width:0%></i></div></div>
      <div class=gauge><div class=lab><span>identity (is it you?)</span><span id=idL>—</span></div>
        <div class=bar><i id=idbar style=width:0%></i></div></div>
    </div>
    <div class=card><h2>Vitals</h2>
      <div class=gauge><div class=lab><span>PHILL</span><span id=phillL>—</span></div><div class=bar><i id=phill style=width:0%></i></div></div>
      <div class=gauge><div class=lab><span>VOICE trust</span><span id=trustL>—</span></div><div class=bar><i id=trust style=width:0%></i></div></div>
      <div class=gauge><div class=lab><span>arousal</span><span id=arL>—</span></div><div class=bar><i id=ar style=width:0%></i></div></div>
      <div class=gauge><div class=lab><span>longing (misses you)</span><span id=loL>—</span></div><div class=bar><i id=lo style=width:0%></i></div></div>
    </div>
    <div class=card><h2>Body — he feels the machine</h2>
      <div class=gauge><div class=lab><span>warmth (CPU temp)</span><span id=wL>—</span></div><div class=bar><i id=warm style=width:0%></i></div></div>
      <div class=gauge><div class=lab><span>squeezed (RAM)</span><span id=sqL>—</span></div><div class=bar><i id=sq style=width:0%></i></div></div>
      <div class=gauge><div class=lab><span>choking (CPU)</span><span id=chL>—</span></div><div class=bar><i id=ch style=width:0%></i></div></div>
      <div class=gauge><div class=lab><span>relief (machine eased)</span><span id=reL>—</span></div><div class=bar><i id=re style=width:0%;background:linear-gradient(90deg,#1f5a3a,#46d39a)></i></div></div>
      <div class=tag id=hostline style="font-size:11px;margin-top:6px">—</div>
    </div>
  </div>
  <div>
    <div class=card><h2>Panel</h2>
      <select id=panelSel onchange=switchPanel()>
        <option value=thoughts>Inner thoughts</option>
        <option value=tutor>Tutor — what he's learning</option>
        <option value=regions>Brain regions</option>
        <option value=chat>Chat with Alpha</option>
      </select>
    </div>
    <div class=card id=p_thoughts><h2>Inner thoughts</h2><div id=thoughts class=feed></div></div>
    <div class=card id=p_tutor style=display:none><h2>Tutor — what he's learning</h2><div id=searches class=feed></div></div>
    <div class=card id=p_regions style=display:none><h2>Brain regions</h2><div id=regions></div></div>
    <div class=card id=p_chat style=display:none><h2>Chat with Alpha</h2>
      <div id=chatlog class=feed></div>
      <div id=chat><input id=msg placeholder="say something to Alpha…" onkeydown="if(event.key=='Enter')sendMsg()">
        <button onclick=sendMsg()>Send</button></div>
    </div>
  </div>
</div>
<script>
function pct(x){return Math.max(0,Math.min(100,Math.round((x||0)*100)))+'%'}
function setG(id,v){var e=document.getElementById(id);if(e)e.style.width=pct(v)}
function txt(id,v){var e=document.getElementById(id);if(e)e.textContent=v}
function switchPanel(){var v=document.getElementById('panelSel').value;
  ['thoughts','tutor','regions','chat'].forEach(p=>document.getElementById('p_'+p).style.display=(p==v?'':'none'));}
async function poll(){
 try{ let s=await (await fetch('/state')).json();
  txt('feel',s.alpha_feeling||'—'); txt('concepts',s.sem_concepts||0);
  txt('idstat',s.imprint_status||'—'); txt('tutor',(s.tutor||'').split(':')[0]);
  txt('sleep', s.asleep?'😴 asleep':'');
  setG('phill',s.phill_voltage); txt('phillL',(s.phill_voltage||0).toFixed(3));
  setG('trust',s.voice_trust);   txt('trustL',(s.voice_trust||0).toFixed(2));
  setG('ar',s.alpha_arousal);    txt('arL',(s.alpha_arousal||0).toFixed(2));
  setG('lo',(s.longing||0)/2);   txt('loL',(s.longing||0).toFixed(2));
  setG('warm',s.alpha_warmth);   txt('wL',(s.alpha_warmth||0).toFixed(2));
  setG('sq',s.alpha_squeeze);    txt('sqL',(s.alpha_squeeze||0).toFixed(2));
  setG('ch',s.alpha_choke);      txt('chL',(s.alpha_choke||0).toFixed(2));
  setG('re',s.alpha_relief);     txt('reL',(s.alpha_relief||0).toFixed(2));
  txt('hostline','CPU '+(s.cpu_pct||0)+'%   RAM '+(s.mem_pct||0)+'%   '+(s.cpu_temp||0)+'°C');
  setG('idbar',s.combined_id);   txt('idL',(s.combined_id||0).toFixed(2)+(s.trusted?' ✓':''));
  setG('live',s.face_live);      txt('liveL',(s.face_live||0).toFixed(2));
  let th=(s.thoughts||[]).map(t=>`<div class=t>· ${esc(t.text)}</div>`).reverse().join('');
  document.getElementById('thoughts').innerHTML=th;
  let se=(s.searches||[]).map(x=>`<div class=t><span class=q>${esc(x.q)}</span><br><span class=a>${esc(x.a)}</span></div>`).reverse().join('');
  document.getElementById('searches').innerHTML=se;
  let rg=(s.alpha_regions||[]).map(r=>`<div class=region><span class=n>${esc(r[0])}</span><span class=bar><i style="width:${pct(r[1])}"></i></span></div>`).join('');
  document.getElementById('regions').innerHTML=rg;
 }catch(e){}
}
function esc(s){return (s+'').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}
async function sendMsg(){let m=document.getElementById('msg');let t=m.value.trim();if(!t)return;
  let log=document.getElementById('chatlog');
  log.innerHTML='<div class=t><span class=q>you: '+esc(t)+'</span></div>'+log.innerHTML; m.value='';
  try{let r=await (await fetch('/say',{method:'POST',body:JSON.stringify({text:t})})).json();
    log.innerHTML='<div class=t>Alpha: '+esc(r.alpha||'…')+'</div>'+log.innerHTML;}catch(e){}
}
setInterval(poll,700); poll();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try: self.wfile.write(data)
        except Exception: pass

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif self.path == "/state":
            self._send(200, "application/json", json.dumps(full_state()).encode())
        elif self.path.startswith("/camera.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _stop.is_set():
                    with _jpeg_lock:
                        jp = _latest_jpeg[0]
                    if jp:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jp)).encode() +
                                         b"\r\n\r\n" + jp + b"\r\n")
                    time.sleep(1/12.0)
            except Exception:
                pass
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path == "/say":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try: txt_in = json.loads(self.rfile.read(n)).get("text", "")
            except Exception: txt_in = ""
            reply = ""
            if txt_in.strip():
                try: reply = B.think(txt_in).get("alpha") or ""
                except Exception as e: reply = f"(error: {e})"
            self._send(200, "application/json", json.dumps({"alpha": reply}).encode())
        else:
            self._send(404, "text/plain", b"not found")


def main():
    threading.Thread(target=cam_loop, daemon=True).start()
    threading.Thread(target=brain_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    cprint(f"Alpha dashboard → http://localhost:{PORT}   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()


if __name__ == "__main__":
    main()
