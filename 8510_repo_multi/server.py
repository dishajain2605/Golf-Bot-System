#!/usr/bin/env python3
"""
Golf Ball Bot - Server
Run:  python server.py
Open: http://localhost:8000
"""

import asyncio, base64, collections, json, logging, queue
import threading, time
from pathlib import Path
from typing import Optional

import cv2, httpx, numpy as np, uvicorn, yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse

# ── Logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")

# ── Config ────────────────────────────────────────────
CFG_PATH = Path(__file__).parent / "config.yaml"
DEFAULTS = {
    "esp_ip":           "192.168.4.1",
    "model_path":       "models/best.pt",
    "confidence":       0.40,
    "inference_size":   416,
    "inference_every":  3,
    "left_zone":        0.38,
    "right_zone":       0.62,
    "close_area":       0.08,
    "arrived_hold_sec": 2.0,
    "smooth_frames":    4,
    "default_speed":    180,
    "cmd_debounce_ms":  120,
    "rotate_180":       True,
    "invert_steering":  False,
    "stream_quality":   65,
    "stream_timeout_sec": 8,
    "server_port":      8000,
}

def load_cfg():
    c = dict(DEFAULTS)
    if CFG_PATH.exists():
        try:
            c.update(yaml.safe_load(open(CFG_PATH, encoding="utf-8")) or {})
        except Exception as e:
            log.warning(f"config.yaml error: {e}")
    else:
        yaml.dump(c, open(CFG_PATH, "w", encoding="utf-8"), default_flow_style=False)
    return c

cfg = load_cfg()

# ── Find model ────────────────────────────────────────
def find_model(explicit=""):
    base = Path(__file__).parent
    candidates = []
    if explicit:
        candidates += [Path(explicit), base / explicit]
    candidates += [
        base / cfg["model_path"],
        base / "models" / "best.pt",
        base / "models" / "last.pt",
    ]
    for p in base.rglob("weights/best.pt"):
        candidates.append(p)
    for p in base.glob("*.pt"):
        candidates.append(p)
    for p in candidates:
        try:
            if p.exists() and p.stat().st_size > 1000:
                return p.resolve()
        except Exception:
            pass
    return None

# ── State ─────────────────────────────────────────────
class BotState:
    def __init__(self):
        self.esp_ip       = cfg["esp_ip"]
        self.mode         = "manual"
        self.auto_running = False
        self.conf         = float(cfg["confidence"])
        self.speed        = int(cfg["default_speed"])
        self.model        = None
        self.model_name   = ""
        self.esp_ok       = False
        self.fps          = 0.0
        self.inf_ms       = 0.0
        self.frames       = 0
        self.dets         = 0
        self.last_dir     = ""
        self.last_dir_ts  = 0.0
        self.pos_history  = collections.deque(maxlen=int(cfg["smooth_frames"]))
        self.arrived_at   = 0.0

    def status(self):
        return {
            "type":         "status",
            "esp_ip":       self.esp_ip,
            "mode":         self.mode,
            "auto_running": self.auto_running,
            "conf":         self.conf,
            "speed":        self.speed,
            "esp_ok":       self.esp_ok,
            "model_loaded": self.model is not None,
            "model_name":   self.model_name,
            "fps":          round(self.fps, 1),
            "inf_ms":       round(self.inf_ms, 1),
            "frames":       self.frames,
            "dets":         self.dets,
        }

st = BotState()

# ── Queues ────────────────────────────────────────────
_fq  = queue.Queue(maxsize=1)
_mq  = queue.Queue(maxsize=2)
_lq  = queue.Queue(maxsize=60)
_iq  = queue.Queue(maxsize=1)

# Latest detections + timestamp (module-level globals)
_latest_dets    = []
_latest_dets_ts = 0.0
_dets_lock      = threading.Lock()

def qlog(msg, level="info"):
    log.info(f"[{level}] {msg}")
    try:
        _lq.put_nowait({"ts": time.strftime("%H:%M:%S"), "msg": msg, "level": level})
    except queue.Full:
        pass

# ── Model ─────────────────────────────────────────────
def load_model(explicit=""):
    st.model = None
    st.model_name = ""
    try:
        from ultralytics import YOLO
    except ImportError:
        qlog("ultralytics not installed. Run: pip install ultralytics", "error")
        return False
    p = find_model(explicit)
    if not p:
        qlog(f"No model found — put best.pt in: {Path(__file__).parent / 'models'}", "warn")
        return False
    try:
        qlog(f"Loading {p.name} ...", "info")
        m = YOLO(str(p))
        qlog("Warming up model ...", "info")
        inf_size = int(cfg.get("inference_size", 416))
        dummy = np.zeros((inf_size, inf_size, 3), dtype=np.uint8)
        for _ in range(3):
            m.predict(dummy, conf=0.1, verbose=False)
        st.model      = m
        st.model_name = p.name
        qlog(f"Model ready: {p.name}", "ok")
        return True
    except Exception as e:
        st.model = None
        qlog(f"Model load failed: {e}", "error")
        return False

# ── Tracking ──────────────────────────────────────────
# Simple 3-zone logic. Speed is always st.speed (throttle bar).
# No variable speed on turns — throttle controls everything uniformly.
def decide(cx, cy, bw, bh):
    """Returns (direction, label)."""
    st.pos_history.append((cx, cy, bw * bh))

    # weighted average — newer frames count more
    n = len(st.pos_history)
    w = list(range(1, n + 1))
    tw = sum(w)
    avg_cx   = sum(st.pos_history[i][0] * w[i] for i in range(n)) / tw
    avg_area = sum(st.pos_history[i][2] * w[i] for i in range(n)) / tw

    close_area   = float(cfg.get("close_area",       0.08))
    arrived_hold = float(cfg.get("arrived_hold_sec", 2.0))
    left_zone    = float(cfg.get("left_zone",        0.38))
    right_zone   = float(cfg.get("right_zone",       0.62))
    invert_turns = bool(cfg.get("invert_steering", False))

    # arrived — ball very close
    if avg_area >= close_area:
        st.arrived_at = time.time()
    if time.time() - st.arrived_at < arrived_hold:
        st.pos_history.clear()
        return "stop", f"ARRIVED area={avg_area:.3f}"

    if avg_cx < left_zone:
        return ("right" if invert_turns else "left"),  f"LEFT  cx={avg_cx:.2f}"
    if avg_cx > right_zone:
        return ("left" if invert_turns else "right"),  f"RIGHT cx={avg_cx:.2f}"
    return "forward",     f"FWD   cx={avg_cx:.2f} area={avg_area:.3f}"

# ── Motor HTTP ────────────────────────────────────────
# Firmware endpoint: GET /cmd?dir=X&speed=Y
_http_client = None
_http_lock   = threading.Lock()

def get_client():
    global _http_client
    with _http_lock:
        if _http_client is None or _http_client.is_closed:
            _http_client = httpx.Client(timeout=0.4)
        return _http_client

def motor_send(direction, force=False):
    now = time.time() * 1000
    deb = float(cfg.get("cmd_debounce_ms", 120))
    if not force and direction == st.last_dir:
        return
    if not force and (now - st.last_dir_ts) < deb:
        return
    st.last_dir    = direction
    st.last_dir_ts = now
    try:
        # firmware expects ?dir=X&speed=Y
        get_client().get(
            f"http://{st.esp_ip}/cmd?dir={direction}&speed={st.speed}"
        )
    except Exception:
        pass

def motor_enqueue(d):
    while True:
        try: _mq.get_nowait()
        except queue.Empty: break
    try: _mq.put_nowait(d)
    except queue.Full: pass

# ── Draw ──────────────────────────────────────────────
G = (0, 230, 100)
Y = (0, 200, 255)
R = (60, 60, 255)

def draw_box(frame, cx, cy, bw, bh, conf, direction, arrived):
    fh, fw = frame.shape[:2]
    x1 = max(0, int((cx - bw/2) * fw))
    y1 = max(0, int((cy - bh/2) * fh))
    x2 = min(fw, int((cx + bw/2) * fw))
    y2 = min(fh, int((cy + bh/2) * fh))
    col = Y if arrived else G
    cv2.rectangle(frame, (x1,y1),(x2,y2), col, 2)
    cs = max(8, int(min(x2-x1, y2-y1) * 0.2))
    for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*cs,py),col,2)
        cv2.line(frame,(px,py),(px,py+dy*cs),col,2)
    cv2.putText(frame, f"GOLF {conf*100:.0f}%",
                (x1+4, max(y1-6,12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    bcx, bcy = int(cx*fw), int(cy*fh)
    cv2.circle(frame, (bcx,bcy), 5, col, -1)
    cv2.line(frame, (bcx,bcy),(fw//2,fh//2),(0,140,60),1,cv2.LINE_AA)
    arrows = {"forward":"^ FWD","left":"< LEFT","right":"> RIGHT","stop":"[] STOP"}
    arr = arrows.get(direction, "")
    if arr:
        cv2.putText(frame, arr, (fw-110,22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, Y if arrived else G, 2, cv2.LINE_AA)

def draw_hud(frame, fps, inf_ms, mode, running, spd, no_ball):
    fh, fw = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar,(0,0),(fw,26),(0,0,0),-1)
    cv2.addWeighted(bar,0.6,frame,0.4,0,frame)
    cv2.putText(frame, f"FPS:{fps:.0f} INF:{inf_ms:.0f}ms SPD:{spd}",
                (6,17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, G, 1)
    ms = "MANUAL" if mode=="manual" else ("AUTO/MOVE" if running else "AUTO/TRACK")
    cv2.putText(frame, ms, (fw//2-55,17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, G if running else Y, 1)
    if mode=="auto" and no_ball:
        cv2.putText(frame,"NO BALL — STOPPED",(fw//2-80,fh-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,R,1)

# ── Inference thread ──────────────────────────────────
_inf_stop   = threading.Event()
_inf_thread = None

def inference_loop():
    global _latest_dets, _latest_dets_ts
    qlog("Inference thread started", "info")
    inf_size = int(cfg.get("inference_size", 416))
    while not _inf_stop.is_set():
        try:
            frame = _iq.get(timeout=1.0)
        except queue.Empty:
            continue
        if st.model is None:
            continue
        t0 = time.perf_counter()
        dets = []
        try:
            res = st.model.predict(
                source=frame, conf=st.conf,
                imgsz=inf_size, verbose=False
            )[0]
            fh, fw = frame.shape[:2]
            for box in res.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1,y1,x2,y2 = float(xyxy[0]),float(xyxy[1]),float(xyxy[2]),float(xyxy[3])
                dets.append({
                    "cx":   ((x1+x2)/2)/fw,
                    "cy":   ((y1+y2)/2)/fh,
                    "w":    (x2-x1)/fw,
                    "h":    (y2-y1)/fh,
                    "conf": float(box.conf[0]),
                })
            st.dets += len(dets)
        except Exception as e:
            qlog(f"Inference error: {e}", "error")
        st.inf_ms = (time.perf_counter()-t0)*1000
        with _dets_lock:
            _latest_dets    = dets
            _latest_dets_ts = time.time()
    qlog("Inference thread stopped", "info")

def start_inference():
    global _inf_thread, _inf_stop
    if _inf_thread and _inf_thread.is_alive():
        _inf_stop.set(); _inf_thread.join(3)
    _inf_stop   = threading.Event()
    _inf_thread = threading.Thread(target=inference_loop, daemon=True, name="inf")
    _inf_thread.start()

# ── Capture thread ────────────────────────────────────
_cap_lock   = threading.Lock()
_cap_stop   = threading.Event()
_cap_thread = None
_last_frame_ts = 0.0

def capture_loop(stop_event):
    global _last_frame_ts, _latest_dets, _latest_dets_ts

    qlog("Capture thread started", "info")
    fps_cnt = 0; fps_ts = time.time()
    fc = 0; retry = 2.0
    inf_every  = int(cfg.get("inference_every", 3))
    enc_params = [cv2.IMWRITE_JPEG_QUALITY, int(cfg.get("stream_quality", 65))]

    import urllib.request

    while not stop_event.is_set():
        stream_url = f"http://{st.esp_ip}:81/stream"
        qlog(f"Connecting to {stream_url} ...", "info")
        st.esp_ok  = False
        response   = None
        frames_got = 0

        try:
            req = urllib.request.Request(
                stream_url,
                headers={"Connection": "keep-alive"}
            )
            response = urllib.request.urlopen(req, timeout=8)
        except Exception as e:
            qlog(f"Stream open failed: {e}", "warn")
            time.sleep(retry)
            retry = min(retry * 1.4, 10.0)
            continue

        buf = b""
        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"

        try:
            while not stop_event.is_set():
                chunk = response.read(4096)
                if not chunk:
                    break
                buf += chunk

                while True:
                    s = buf.find(SOI)
                    if s == -1:
                        buf = b""; break
                    e = buf.find(EOI, s + 2)
                    if e == -1:
                        buf = buf[s:]; break

                    jpeg = buf[s:e+2]
                    buf  = buf[e+2:]

                    if len(jpeg) < 500:
                        continue

                    arr   = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    if not st.esp_ok:
                        st.esp_ok = True
                        qlog("Stream connected", "ok")

                    _last_frame_ts = time.time()
                    st.frames += 1
                    fc        += 1
                    frames_got += 1
                    retry       = 2.0

                    # Camera orientation can be tuned from config without touching code.
                    if bool(cfg.get("rotate_180", True)):
                        frame = cv2.rotate(frame, cv2.ROTATE_180)

                    # push to inference thread every N frames
                    if fc % inf_every == 0 and st.model is not None:
                        try: _iq.get_nowait()
                        except queue.Empty: pass
                        try: _iq.put_nowait(frame.copy())
                        except queue.Full: pass

                    # get detections — expire after 0.6s to avoid stale data
                    with _dets_lock:
                        age  = time.time() - _latest_dets_ts
                        dets = list(_latest_dets) if age < 0.6 else []

                    # tracking decision
                    direction = "stop"
                    pos_label = "NO BALL"
                    best      = None
                    arrived   = False

                    if dets:
                        best = max(dets, key=lambda d: d["conf"])
                        direction, pos_label = decide(
                            best["cx"], best["cy"],
                            best["w"],  best["h"]
                        )
                        arrived = "ARRIVED" in pos_label
                    else:
                        # no ball → stop, clear smoothing history
                        st.pos_history.clear()

                    # send motor command
                    # if st.mode == "auto" and st.auto_running:
                    #     motor_send(direction)
                    # elif st.mode == "auto":
                    #     motor_send("stop", force=True)
                    
                    if st.mode == "auto" and st.auto_running:
                        motor_send(direction)
                    elif st.mode == "auto":
                        motor_send("stop")

                    # annotate
                    if best:
                        draw_box(frame, best["cx"], best["cy"],
                                 best["w"], best["h"],
                                 best["conf"], direction, arrived)
                    draw_hud(frame, st.fps, st.inf_ms,
                             st.mode, st.auto_running,
                             st.speed, best is None)

                    # FPS
                    fps_cnt += 1
                    now = time.time()
                    if now - fps_ts >= 1.0:
                        st.fps    = fps_cnt / (now - fps_ts)
                        fps_cnt   = 0
                        fps_ts    = now

                    # encode + push to browser
                    ok, buf_enc = cv2.imencode(".jpg", frame, enc_params)
                    if not ok:
                        continue

                    payload = {
                        "type":         "frame",
                        "img":          base64.b64encode(buf_enc.tobytes()).decode("ascii"),
                        "fps":          round(st.fps, 1),
                        "inf_ms":       round(st.inf_ms, 1),
                        "detections":   dets,
                        "dir":          direction,
                        "pos":          pos_label,
                        "auto_running": st.auto_running,
                        "esp_ok":       st.esp_ok,
                        "model_ok":     st.model is not None,
                        "speed":        st.speed,
                    }
                    try: _fq.get_nowait()
                    except queue.Empty: pass
                    try: _fq.put_nowait(payload)
                    except queue.Full: pass

        except Exception as ex:
            qlog(f"Stream read error: {ex}", "warn")
        finally:
            try: response.close()
            except Exception: pass

        if stop_event.is_set():
            break

        st.esp_ok = False
        if frames_got == 0:
            qlog(f"No frames received, retry {retry:.0f}s", "warn")
        else:
            qlog("Stream dropped, reconnecting ...", "warn")
            if st.mode == "auto":
                motor_send("stop", force=True)

        time.sleep(retry)
        retry = min(retry * 1.4, 10.0)

    qlog("Capture stopped", "info")

def start_capture():
    global _cap_thread, _cap_stop
    with _cap_lock:
        if _cap_thread and _cap_thread.is_alive():
            _cap_stop.set()
            _cap_thread.join(10)
            if _cap_thread.is_alive():
                qlog("Capture restart skipped: previous capture thread is still closing", "warn")
                return
        _cap_stop   = threading.Event()
        _cap_thread = threading.Thread(
            target=capture_loop,
            args=(_cap_stop,),
            daemon=True,
            name="cap",
        )
        _cap_thread.start()

# ── WebSocket manager ─────────────────────────────────
class WSM:
    def __init__(self): self._c = set()
    async def connect(self, ws):
        await ws.accept(); self._c.add(ws)
        qlog(f"Browser connected ({len(self._c)} client)", "ok")
    def disconnect(self, ws): self._c.discard(ws)
    async def broadcast(self, data):
        dead = set()
        for ws in self._c:
            try: await ws.send_json(data)
            except: dead.add(ws)
        self._c -= dead

wm = WSM()

# ── FastAPI ───────────────────────────────────────────
app = FastAPI()

@app.on_event("startup")
async def startup():
    qlog("Starting server ...", "info")
    threading.Thread(target=load_model, daemon=True).start()
    start_inference()
    start_capture()
    asyncio.create_task(_t_frames())
    asyncio.create_task(_t_motor())
    asyncio.create_task(_t_logs())
    asyncio.create_task(_t_status())
    asyncio.create_task(_t_watchdog())
    qlog(f"Dashboard → http://localhost:{cfg.get('server_port',8000)}", "ok")

@app.get("/", response_class=HTMLResponse)
async def index(): return HTMLResponse(HTML)

@app.get("/bg.png")
async def get_bg(): 
    bg_path = Path(__file__).parent / "bg.png"
    if bg_path.exists():
        return FileResponse(str(bg_path), media_type="image/png")
    from fastapi.responses import Response
    return Response(status_code=404)

@app.get("/favicon.ico")
async def get_favicon(): 
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/health")
async def health(): return st.status()

@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await wm.connect(ws)
    await ws.send_json(st.status())
    try:
        while True:
            raw = await ws.receive_text()
            try: await handle(json.loads(raw))
            except Exception: pass
    except WebSocketDisconnect:
        wm.disconnect(ws)

async def handle(msg):
    a = msg.get("action","")

    if a == "cmd":
        if st.mode == "manual":
            st.last_dir = ""
            motor_enqueue(msg.get("dir","stop"))

    elif a == "set_mode":
        st.mode         = msg.get("mode","manual")
        st.auto_running = False
        st.arrived_at   = 0.0
        st.pos_history.clear()
        motor_enqueue("stop")
        qlog(f"Mode → {st.mode.upper()}", "info")

    elif a == "auto_init":
        if st.model is None:
            qlog("Load model first", "error"); return
        st.auto_running = True
        st.arrived_at   = 0.0
        st.pos_history.clear()
        qlog("Auto INITIALIZED — motors live", "ok")

    elif a == "auto_stop":
        st.auto_running = False
        st.arrived_at   = 0.0
        motor_enqueue("stop")
        qlog("Auto STOPPED", "info")

    elif a == "set_conf":
        st.conf = float(msg.get("value", 0.40))
        qlog(f"Conf → {st.conf:.2f}", "info")

    elif a == "set_speed":
        st.speed = int(msg.get("value", 180))
        qlog(f"Speed → {st.speed} ({st.speed*100//255}%)", "info")

    elif a == "set_ip":
        ip = msg.get("ip","").strip()
        if ip:
            st.esp_ip = ip; cfg["esp_ip"] = ip
            qlog(f"ESP IP → {ip}", "info")
            start_capture()

    elif a == "load_model":
        path = msg.get("path","").strip()
        threading.Thread(target=load_model, args=(path,), daemon=True).start()

# ── Background tasks ──────────────────────────────────
async def _t_frames():
    while True:
        await asyncio.sleep(0.030)
        if not wm._c: continue
        try: await wm.broadcast(_fq.get_nowait())
        except queue.Empty: pass

async def _t_motor():
    while True:
        await asyncio.sleep(0.04)
        try:
            d = _mq.get_nowait()
            try:
                async with httpx.AsyncClient(timeout=0.4) as c:
                    await c.get(f"http://{st.esp_ip}/cmd?dir={d}&speed={st.speed}")
            except Exception: pass
        except queue.Empty: pass

async def _t_logs():
    while True:
        await asyncio.sleep(0.15)
        if not wm._c: continue
        entries = []
        while True:
            try: entries.append(_lq.get_nowait())
            except queue.Empty: break
        if entries:
            await wm.broadcast({"type":"logs","entries":entries})

async def _t_status():
    while True:
        await asyncio.sleep(1.0)
        if wm._c: await wm.broadcast(st.status())

async def _t_watchdog():
    await asyncio.sleep(12)
    while True:
        await asyncio.sleep(4)
        timeout = float(cfg.get("stream_timeout_sec", 8))
        if _last_frame_ts > 0 and time.time() - _last_frame_ts > timeout:
            qlog("Watchdog: no frame — restarting stream", "warn")
            start_capture()

# ── Dashboard HTML ────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Golf Bot</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Bebas+Neue&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f4f5f7;--bg2:#ffffff;--bg3:#e9edf0;--bdr:#d1d5db;
  --G:#059669;--Gd:rgba(5,150,105,.12);--Gg:0 0 10px rgba(5,150,105,.3);
  --R:#ef4444;--Rd:rgba(239,68,68,.12);--Rg:0 0 10px rgba(239,68,68,.3);
  --A:#d97706;--Ad:rgba(217,119,6,.12);--Ag:0 0 10px rgba(217,119,6,.3);
  --tx:#374151;--txd:#6b7280;
  --mono:'IBM Plex Mono',monospace;--hud:'Bebas Neue',monospace;
}
html,body{height:100%;background:var(--bg);color:var(--tx);font-family:var(--mono);overflow:hidden}
body::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.02) 3px,rgba(0,0,0,.02) 4px);pointer-events:none;z-index:9999}
.view{display:none;height:100%}.view.active{display:block}
.root{display:grid;grid-template-rows:48px 1fr;grid-template-columns:1fr 290px;height:100vh;gap:1px;background:var(--bdr)}
header{grid-column:1/-1;background:var(--bg2);display:flex;align-items:center;justify-content:space-between;padding:0 16px;border-bottom:1px solid var(--bdr)}
.ht{font-family:var(--hud);font-size:21px;letter-spacing:5px;color:var(--A);text-shadow:var(--Ag)}
.hs{display:flex;gap:14px;font-size:10px;align-items:center}
.dot{width:7px;height:7px;border-radius:50%;background:var(--txd);display:inline-block;margin-right:5px;transition:all .3s}
.dot.on{background:var(--G);box-shadow:var(--Gg);animation:blink 2s infinite}
.dot.warn{background:var(--A);box-shadow:var(--Ag)}
.dot.err{background:var(--R);box-shadow:var(--Rg)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.clk-txt{font-family:var(--hud);font-size:19px;letter-spacing:2px}
.cam{background:#e0e5ec;position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden}
#camImg{width:100%;height:100%;object-fit:contain}
#noSig{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px}
#noSig .ni{font-size:44px;opacity:.3}
#noSig .nt{font-family:var(--hud);font-size:13px;letter-spacing:4px;color:var(--txd)}
#noSig .ns{font-size:10px;color:var(--txd);margin-top:4px}
.cc{position:absolute;width:18px;height:18px;border-color:rgba(217,119,6,.5);border-style:solid}
.cc.tl{top:10px;left:10px;border-width:2px 0 0 2px}.cc.tr{top:10px;right:10px;border-width:2px 2px 0 0}
.cc.bl{bottom:10px;left:10px;border-width:0 0 2px 2px}.cc.br{bottom:10px;right:10px;border-width:0 2px 2px 0}
#dtag{position:absolute;top:12px;left:50%;transform:translateX(-50%);font-family:var(--hud);font-size:12px;letter-spacing:3px;padding:2px 14px;background:var(--Gd);border:1px solid var(--G);color:var(--G);border-radius:2px;opacity:0;transition:opacity .25s;box-shadow:0 2px 10px var(--Gd)}
#dtag.vis{opacity:1}
.chud{position:absolute;bottom:10px;left:12px;font-size:9px;color:var(--tx);opacity:.8;line-height:2;background:rgba(255,255,255,0.7);padding:4px 8px;border-radius:4px;}
.panel{background:var(--bg2);display:flex;flex-direction:column;overflow-y:auto;padding:10px;gap:9px}
.panel::-webkit-scrollbar{width:4px}.panel::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:2px}
.card{background:var(--bg2);border:1px solid var(--bdr);border-radius:4px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}
.ct{font-family:var(--hud);font-size:11px;letter-spacing:3px;color:var(--tx);margin-bottom:8px;display:flex;align-items:center;gap:6px}
.ct::after{content:'';flex:1;height:1px;background:var(--bdr)}
.inp{width:100%;background:var(--bg3);border:1px solid var(--bdr);color:var(--tx);font-family:var(--mono);font-size:11px;padding:6px 8px;border-radius:3px;outline:none;transition:all .2s;box-shadow:inset 0 1px 2px rgba(0,0,0,0.02)}
.inp:focus{border-color:var(--A);background:var(--bg2);box-shadow:0 0 0 2px var(--Ad)}
.row{display:flex;gap:6px}
.btn{padding:8px 12px;border:1px solid;border-radius:3px;font-family:var(--mono);font-size:10px;letter-spacing:1px;cursor:pointer;transition:all .2s;background:white;user-select:none;display:inline-flex;align-items:center;justify-content:center;gap:6px;font-weight:600;}
.btn:hover{transform:translateY(-1px);box-shadow:0 2px 5px rgba(0,0,0,0.05)}
.btn:active{transform:translateY(0);box-shadow:none}
.btn-g{border-color:var(--G);color:var(--G)}.btn-g:hover{background:var(--Gd);border-color:var(--G)}
.btn-r{border-color:var(--R);color:var(--R)}.btn-r:hover{background:var(--Rd);border-color:var(--R)}
.btn-a{border-color:var(--A);color:var(--A)}.btn-a:hover{background:var(--Ad);border-color:var(--A)}
.btn-full{width:100%;text-align:center;padding:10px}
.lit-g{background:var(--G)!important;border-color:var(--G)!important;color:white!important;box-shadow:var(--Gg)!important}
.lit-r{background:var(--R)!important;border-color:var(--R)!important;color:white!important;box-shadow:var(--Rg)!important}
.mt{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.mb{padding:9px;text-align:center;font-family:var(--hud);font-size:12px;letter-spacing:2px;cursor:pointer;border:1px solid var(--bdr);color:var(--txd);background:var(--bg3);border-radius:3px;transition:all .2s;user-select:none}
.mb.active{border-color:var(--A);color:var(--A);background:white;box-shadow:0 2px 4px var(--Ad)}
.dp{display:grid;grid-template-areas:". f ." "l s r" ". b .";grid-template-columns:repeat(3,1fr);gap:5px;width:165px;margin:0 auto}
.db{aspect-ratio:1;display:flex;align-items:center;justify-content:center;font-size:16px;cursor:pointer;border:1px solid var(--bdr);background:var(--bg2);color:var(--txd);border-radius:4px;transition:all .14s;user-select:none;-webkit-tap-highlight-color:transparent;box-shadow:0 1px 2px rgba(0,0,0,0.05)}
.db:hover{border-color:var(--A);color:var(--A);background:var(--bg3)}
.db:active,.db.pr{background:var(--A);border-color:var(--A);color:white;box-shadow:0 2px 8px var(--Ag);transform:scale(.95)}
.db.sb{font-size:9px;font-family:var(--hud)}
[data-d=forward]{grid-area:f}[data-d=backward]{grid-area:b}[data-d=left]{grid-area:l}[data-d=right]{grid-area:r}[data-d=stop]{grid-area:s}
.sr{display:flex;align-items:center;gap:8px;font-size:10px;margin-bottom:6px}
.sr input[type=range]{flex:1;accent-color:var(--A);cursor:pointer;height:6px;border-radius:3px}
.sv{width:38px;text-align:right;color:var(--A);font-family:var(--hud);font-size:15px}
.ag{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.ast{margin-top:8px;background:var(--bg3);border:1px solid var(--bdr);padding:8px 10px;font-size:11px;line-height:2;border-radius:3px;color:var(--txd)}
.ast b{color:var(--tx)}
.tg{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.ti{background:var(--bg3);border:1px solid var(--bdr);padding:8px 10px;border-radius:4px;box-shadow:inset 0 1px 0 rgba(255,255,255,0.5)}
.tl{font-size:9px;color:var(--txd);letter-spacing:1px;font-weight:600}
.tv{font-family:var(--hud);font-size:18px;color:var(--A);margin-top:2px}
.logBox{background:var(--bg3);border:1px solid var(--bdr);padding:8px;font-size:10px;line-height:1.85;height:110px;overflow-y:auto;border-radius:4px;box-shadow:inset 0 1px 3px rgba(0,0,0,0.05)}
.logBox::-webkit-scrollbar{width:4px}.logBox::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:2px}
.le{display:block}.le.ok{color:var(--G)}.le.error{color:var(--R)}.le.warn{color:var(--A)}.le.info{color:var(--tx)}
.sec{display:none}.sec.vis{display:block}

/* Distance App Specific */
.d-root{display:grid;grid-template-rows:48px 1fr;height:100vh;background:var(--bg2);gap:0}
.d-main{background:var(--bg);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px;padding:16px;height:100%}
.state-row{display:flex;align-items:center;gap:10px;font-family:var(--hud);font-size:14px;letter-spacing:4px;color:var(--txd)}
.pulse{width:10px;height:10px;border-radius:50%;background:var(--txd)}
.pulse.lit{background:var(--A);box-shadow:var(--Ag);animation:blink 1.4s infinite}
.gauge-wrap{position:relative;display:flex;align-items:center;justify-content:center;background:white;padding:30px;border-radius:50%;box-shadow:0 10px 40px rgba(0,0,0,0.05);border:1px solid var(--bdr)}
.g-svg{filter:drop-shadow(0 0 10px rgba(5,150,105,.1))}
.g-center{position:absolute;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0}
#distNum{font-family:var(--hud);font-size:90px;line-height:1;color:var(--tx);transition:color .4s,text-shadow .4s}
.g-unit{font-family:var(--hud);font-size:24px;letter-spacing:4px;color:var(--txd);margin-top:-6px}
#distConf{font-size:11px;color:var(--txd);letter-spacing:2px;margin-top:8px;font-weight:600}
#lockBadge{display:none;font-family:var(--hud);font-size:13px;letter-spacing:3px;padding:6px 20px;background:var(--G);color:white;border-radius:4px;box-shadow:0 4px 15px var(--Gd)}
@keyframes lockPop{0%,100%{transform:scale(1)}50%{transform:scale(1.05)}}
.badge-glow{animation:lockPop 2s infinite}
.tiles{display:grid;grid-template-columns:repeat(3,130px);gap:12px}
.tile{background:white;border:1px solid var(--bdr);border-radius:6px;padding:12px;box-shadow:0 2px 6px rgba(0,0,0,0.02)}
@keyframes popIn{0%{transform:scale(.3);opacity:0}65%{transform:scale(1.08)}100%{transform:scale(1);opacity:1}}
.pop{animation:popIn .55s cubic-bezier(.34,1.56,.64,1) forwards}
.nav-btn-wrap{display:flex;gap:6px;}
</style>
</head>
<body>

<!-- HOME PAGE -->
<div id="viewHome" class="view active">
  <div style="height:100vh; background: linear-gradient(rgba(244, 245, 247, 0.88), rgba(244, 245, 247, 0.88)), url('/bg.png') center/cover no-repeat; display:flex; flex-direction:column;">
    <header style="justify-content:center; background:rgba(255,255,255,0.9); backdrop-filter:blur(5px);">
      <div class="ht" style="font-size:24px;">⬡ GOLF BOT SYSTEM</div>
    </header>
    <div style="flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:20px; text-align:center;">
      <div style="font-family:var(--hud); font-size:48px; color:var(--tx); margin-bottom:16px;">WELCOME</div>
      <div style="font-family:var(--mono); color:var(--txd); font-size:14px; max-width:500px; line-height:1.6; margin-bottom:40px;">
        Select an interface below to access the hardware controls, distance measurement panel, or club recommendation system.
      </div>
      
      <div class="tiles" style="grid-template-columns:repeat(3, 220px); gap:20px; width:auto;">
        <div class="tile" style="cursor:pointer; display:flex; flex-direction:column; align-items:center; padding:30px 20px; transition:all 0.2s; border:2px solid transparent;" 
             onmouseover="this.style.borderColor='var(--A)'; this.style.transform='translateY(-5px)'" 
             onmouseout="this.style.borderColor='transparent'; this.style.transform='translateY(0)'"
             onclick="switchView('viewMain')">
          <div style="font-size:40px; margin-bottom:15px; color:var(--A)">⬡</div>
          <div style="font-family:var(--hud); font-size:24px; color:var(--tx)">DASHBOARD</div>
          <div style="font-size:11px; color:var(--txd); margin-top:8px;">Control & Auto-Track</div>
        </div>
        
        <div class="tile" style="cursor:pointer; display:flex; flex-direction:column; align-items:center; padding:30px 20px; transition:all 0.2s; border:2px solid transparent;" 
             onmouseover="this.style.borderColor='var(--G)'; this.style.transform='translateY(-5px)'" 
             onmouseout="this.style.borderColor='transparent'; this.style.transform='translateY(0)'"
             onclick="switchView('viewDist')">
          <div style="font-size:40px; margin-bottom:15px; color:var(--G)">◎</div>
          <div style="font-family:var(--hud); font-size:24px; color:var(--tx)">DISTANCE SENSOR</div>
          <div style="font-size:11px; color:var(--txd); margin-top:8px;">Measure & Validate</div>
        </div>

        <div class="tile" style="cursor:pointer; display:flex; flex-direction:column; align-items:center; padding:30px 20px; transition:all 0.2s; border:2px solid transparent;" 
             onmouseover="this.style.borderColor='var(--R)'; this.style.transform='translateY(-5px)'" 
             onmouseout="this.style.borderColor='transparent'; this.style.transform='translateY(0)'"
             onclick="switchView('viewClubs')">
          <div style="font-size:40px; margin-bottom:15px; color:var(--R)">⛳</div>
          <div style="font-family:var(--hud); font-size:24px; color:var(--tx)">CLUB ASSISTANT</div>
          <div style="font-size:11px; color:var(--txd); margin-top:8px;">Club Recommendations</div>
        </div>
      </div>
    </div>
    <div style="padding:20px; text-align:center; font-size:10px; color:var(--txd); border-top:1px solid var(--bdr);">
      GOLF BOT HARDWARE INTEGRATION SUITE
    </div>
  </div>
</div>

<!-- MAIN DASHBOARD -->
<div id="viewMain" class="view">
<div class="root">
<header>
  <div style="display:flex; align-items:center; gap:16px;">
    <button class="btn btn-a" style="padding:4px 8px; font-size:10px;" onclick="switchView('viewHome')">⌂ HOME</button>
    <div class="ht">⬡ GOLF BOT</div>
  </div>
  <div class="hs">
    <span><span class="dot" id="dSrv"></span><span id="lSrv">SERVER</span></span>
    <span><span class="dot" id="dEsp"></span><span id="lEsp">ESP-CAM</span></span>
    <span><span class="dot" id="dMdl"></span><span id="lMdl">NO MODEL</span></span>
    <span><span class="dot" id="dDet"></span><span id="lDet">IDLE</span></span>
  </div>
  <div class="clk-txt">--:--:--</div>
</header>
<div class="cam">
  <div id="noSig">
    <div class="ni">📡</div>
    <div class="nt">AWAITING FEED</div>
    <div class="ns">Connect to GolfBot WiFi · run python server.py</div>
  </div>
  <img id="camImg" style="display:none"/>
  <div class="cc tl"></div><div class="cc tr"></div>
  <div class="cc bl"></div><div class="cc br"></div>
  <div id="dtag">◉ BALL LOCKED</div>
  <div class="chud">FPS <span id="mF">--</span><br>INF <span id="mI">--</span>ms</div>
</div>
<div class="panel">
  <div class="card">
    <div class="ct">ESP-CAM IP</div>
    <div class="row">
      <input class="inp" id="ipInp" value="192.168.4.1"/>
      <button class="btn btn-a" onclick="applyIP()">SET</button>
    </div>
  </div>
  <div class="card">
    <div class="ct">Throttle</div>
    <div style="font-size:10px;color:var(--tx);margin-bottom:4px;font-weight:600;">
      Speed: <span id="spdPct">71</span>% &nbsp;(<span id="spdVal">180</span>/255)
    </div>
    <div class="sr">
      <span style="font-size:9px;color:var(--txd);font-weight:600">SLOW</span>
      <input type="range" id="spdS" min="0" max="255" value="180" oninput="setSpd(this)"/>
      <span style="font-size:9px;color:var(--txd);font-weight:600">FAST</span>
    </div>
  </div>
  <div class="card">
    <div class="ct">Mode</div>
    <div class="mt">
      <div class="mb active" id="bMan" onclick="setMode('manual')">MANUAL</div>
      <div class="mb"        id="bAut" onclick="setMode('auto')">AUTO</div>
    </div>
  </div>
  <div class="card sec vis" id="secM">
    <div class="ct">D-Pad <span style="font-size:8px;font-weight:normal;color:var(--txd)">(WASD/arrows)</span></div>
    <div class="dp">
      <div class="db" data-d="forward"  id="dbF">▲</div>
      <div class="db" data-d="left"     id="dbL">◀</div>
      <div class="db sb" data-d="stop"  id="dbS">STOP</div>
      <div class="db" data-d="right"    id="dbR">▶</div>
      <div class="db" data-d="backward" id="dbB">▼</div>
    </div>
    <div style="text-align:center;font-size:9px;color:var(--txd);margin-top:8px">Hold=move · Release=stop</div>
  </div>
  <div class="card sec" id="secA">
    <div class="ct">Auto Track</div>
    <div class="row" style="margin-bottom:10px">
      <input class="inp" id="mdlP" placeholder="auto-detect (leave blank)"/>
      <button class="btn btn-a" onclick="loadMdl()">LOAD</button>
    </div>
    <div class="sr" style="margin-bottom:12px">
      <span style="color:var(--txd);font-size:10px;font-weight:600">CONF</span>
      <input type="range" id="confS" min="10" max="90" value="40" oninput="setConf(this)"/>
      <span class="sv" id="confV">0.40</span>
    </div>
    <div class="ag">
      <button class="btn btn-g btn-full" id="btnI" onclick="autoInit()">▶ INITIALIZE</button>
      <button class="btn btn-r btn-full" id="btnSt" onclick="autoStop()">■ STOP</button>
    </div>
    <div class="ast">
      State: <b id="aState">TRACKING ONLY</b><br>
      Ball:  <b id="aPos">---</b><br>
      Cmd:   <b id="aCmd">IDLE</b>
    </div>
  </div>
  <div class="card">
    <div class="ct">Telemetry</div>
    <div class="tg">
      <div class="ti"><div class="tl">DIRECTION</div><div class="tv" id="tDir">---</div></div>
      <div class="ti"><div class="tl">DETECTIONS</div><div class="tv" id="tDet">0</div></div>
      <div class="ti"><div class="tl">CONFIDENCE</div><div class="tv" id="tConf">--</div></div>
      <div class="ti"><div class="tl">FRAMES</div><div class="tv" id="tFrm">0</div></div>
    </div>
  </div>
  <div class="card">
    <div class="ct">Server Log</div>
    <div id="logBox" class="logBox"><span class="le info">Connecting...</span></div>
  </div>
</div>
</div>
</div>

<!-- DISTANCE SENSOR -->
<div id="viewDist" class="view">
<div class="d-root">
<header>
  <div style="display:flex; align-items:center; gap:16px;">
    <button class="btn btn-g" style="padding:4px 8px; font-size:10px;" onclick="switchView('viewHome')">⌂ HOME</button>
    <div class="ht" style="color:var(--G)">⬡ DISTANCE SENSOR</div>
  </div>
  <div class="hs">
    <span><span class="dot" id="d_dSrv"></span><span id="d_lSrv">SERVER</span></span>
    <span><span class="dot" id="d_dEsp"></span><span id="d_lEsp">ESP-CAM</span></span>
    <span><span class="dot" id="d_dLk"></span><span id="d_lLk">UNLOCKED</span></span>
  </div>
  <div class="clk-txt">--:--:--</div>
</header>
<div class="d-main">
  <div class="state-row">
    <div class="pulse lit" id="pulse"></div>
    <span id="stateText">CONNECTING TO GOLF BOT SERVER...</span>
  </div>
  <div class="gauge-wrap">
    <svg class="g-svg" width="290" height="290" viewBox="0 0 290 290">
      <circle cx="145" cy="145" r="136" fill="none" stroke="#e9edf0" stroke-width="4"/>
      <g id="ticks" stroke="#9ca3af"></g>
      <path id="arcBg" fill="none" stroke="#f4f5f7" stroke-width="18" stroke-linecap="round"/>
      <path id="arcFill" fill="none" stroke="#059669" stroke-width="18" stroke-linecap="round" style="opacity:0;transition:stroke .5s"/>
      <circle cx="145" cy="145" r="76" fill="#ffffff" stroke="#e9edf0" stroke-width="2"/>
      <text x="145" y="272" fill="#6b7280" font-size="8" text-anchor="middle" font-family="IBM Plex Mono">10 cm ←——————→ 30 cm</text>
    </svg>
    <div class="g-center">
      <div id="distNum">--</div>
      <div class="g-unit">CM</div>
      <div id="distConf">AWAITING BALL</div>
    </div>
  </div>
  <div id="lockBadge">🔒 DISTANCE LOCKED</div>
  <div class="tiles">
    <div class="tile"><div class="tl">BALL AREA</div><div class="tv" id="d_tArea">---</div></div>
    <div class="tile"><div class="tl">CONFIDENCE</div><div class="tv" id="d_tConf">---</div></div>
    <div class="tile"><div class="tl">STATUS</div><div class="tv" id="d_tStat">IDLE</div></div>
  </div>
  <div class="btn-row">
    <button class="btn btn-g" style="padding:10px 20px;font-size:12px;" onclick="resetLock()">↺ RESET MEASURE</button>
  </div>
  <div class="logBox" id="d_logBox" style="width:414px"><span class="le info">Distance panel ready...</span></div>
</div>
</div>
</div>

<!-- CLUB SUGGESTION -->
<div id="viewClubs" class="view">
  <div style="height:100vh; background:var(--bg); display:flex; flex-direction:column;">
    <header style="justify-content:center; position:relative;">
      <button class="btn btn-r" style="position:absolute; left:16px; padding:4px 8px; font-size:10px;" onclick="switchView('viewHome')">⌂ HOME</button>
      <div class="ht" style="font-size:24px; color:var(--R)">⛳ CLUB ASSISTANT</div>
    </header>
    <div style="flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:20px;">
      <div class="card" style="width:450px; text-align:left; padding:30px;">
        <div class="ct" style="font-size:16px; justify-content:center; margin-bottom:24px;">CALCULATE REMAINING DISTANCE</div>
        
        <div style="margin-bottom:18px;">
          <label style="font-family:var(--hud); color:var(--tx); font-size:15px; display:block; margin-bottom:6px;">TOTAL PIN DISTANCE (X) in Yards</label>
          <input type="number" id="inpX" class="inp" style="font-size:20px; padding:12px; border-radius:4px;" placeholder="e.g. 350" oninput="calcClub()">
        </div>

        <div style="margin-bottom:24px;">
          <label style="font-family:var(--hud); color:var(--tx); font-size:15px; display:block; margin-bottom:6px;">ROVER TRAVELLED (Y) in Yards</label>
          <input type="number" id="inpY" class="inp" style="font-size:20px; padding:12px; border-radius:4px;" placeholder="e.g. 150" oninput="calcClub()">
        </div>

        <div style="background:var(--bg3); padding:20px; border-radius:6px; text-align:center; border:1px solid var(--bdr); margin-bottom:24px; display:flex; flex-direction:column; align-items:center;">
          <div style="font-size:12px; color:var(--txd); letter-spacing:2px; margin-bottom:6px; font-weight:600;">REMAINING DISTANCE (Z = X - Y)</div>
          <div style="display:flex; align-items:baseline; gap:8px;">
            <div id="outZ" style="font-family:var(--hud); font-size:52px; color:var(--tx); line-height:1;">--</div>
            <div style="font-family:var(--hud); font-size:18px; color:var(--txd);">YARDS</div>
          </div>
        </div>

        <div style="text-align:center;">
          <div style="font-size:12px; color:var(--txd); letter-spacing:2px; margin-bottom:10px; font-weight:600;">RECOMMENDED CLUBS</div>
          <div id="outClub" style="font-family:var(--hud); font-size:32px; color:var(--R); padding:15px; border:2px solid var(--R); border-radius:6px; background:var(--Rd); transition:all 0.3s;">---</div>
        </div>

      </div>
    </div>
  </div>
</div>

<script>
function switchView(id){ document.querySelectorAll('.view').forEach(e=>e.classList.remove('active')); document.getElementById(id).classList.add('active'); }
setInterval(()=>{ const t=new Date().toLocaleTimeString('en',{hour12:false}); document.querySelectorAll('.clk-txt').forEach(el=>el.textContent=t); },1000);

// --- Club Calculation Logic ---
function calcClub() {
  const vx = document.getElementById('inpX').value;
  const vy = document.getElementById('inpY').value;
  
  const x = parseFloat(vx) || 0;
  const y = parseFloat(vy) || 0;
  
  const elZ = document.getElementById('outZ');
  const elClub = document.getElementById('outClub');
  
  if (vx.trim() === '' && vy.trim() === '') {
    elZ.textContent = '--';
    elClub.textContent = '---';
    return;
  }

  const z = x - y;
  elZ.textContent = Math.round(z);
  
  let rec = '';
  if (z >= 260) rec = 'Beyond Range / Multiple Shots';
  else if (z >= 230) rec = 'Driver (1 Wood)';
  else if (z >= 210) rec = '3 Wood';
  else if (z >= 190) rec = '5 Wood';
  else if (z >= 180) rec = '3 Hybrid';
  else if (z >= 170) rec = '4 Hybrid / 4 Iron';
  else if (z >= 160) rec = '5 Iron';
  else if (z >= 150) rec = '6 Iron';
  else if (z >= 140) rec = '7 Iron';
  else if (z >= 130) rec = '8 Iron';
  else if (z >= 120) rec = '9 Iron';
  else if (z >= 110) rec = 'Pitching Wedge (PW)';
  else if (z >= 90) rec = 'Gap Wedge (GW)';
  else if (z >= 80) rec = 'Sand Wedge (SW)';
  else if (z >= 60) rec = 'Lob Wedge (LW)';
  else if (z > 0) rec = 'Putter / Short Wedge';
  else rec = 'HOLE ALREADY REACHED!';

  elClub.textContent = rec;
}

// --- Main App Logic ---
const ui={mode:'manual',ws:null};

function connectWS(){
  ui.ws=new WebSocket(`ws://${location.host}/ws`);
  ui.ws.onopen=()=>{
    dot('dSrv','on'); lbl('lSrv','ONLINE'); addLog('Server connected','ok',"logBox");
    dot('d_dSrv','on'); lbl('d_lSrv','SERVER OK'); setState('AWAITING ESP-CAM...', true); addLog('Connected to core backend','ok',"d_logBox");
  };
  ui.ws.onclose=()=>{
    dot('dSrv','err'); lbl('lSrv','OFFLINE'); addLog('Reconnecting...','warn',"logBox");
    dot('d_dSrv','err'); lbl('d_lSrv','OFFLINE'); dot('d_dEsp',''); lbl('d_lEsp','ESP-CAM');
    d_espOk=false; if(!d_locked) setState('BOT SERVER OFFLINE — RECONNECTING...', true);
    addLog('Disconnected. Retrying in 2s...','warn',"d_logBox"); setTimeout(connectWS,2000);
  };
  ui.ws.onerror=()=>{}; ui.ws.onmessage=e=>{try{handle(JSON.parse(e.data));}catch(_){}};
}
function send(o){if(ui.ws&&ui.ws.readyState===1)ui.ws.send(JSON.stringify(o));}

function handle(m){
  if(m.type==='frame'){ onFrame(m); d_onFrame(m); }
  else if(m.type==='status') onStatus(m);
  else if(m.type==='logs') m.entries.forEach(e=>{ addLog(e.msg,e.level,"logBox"); addLog(e.msg,e.level,"d_logBox"); });
}

function onFrame(m){
  const img=document.getElementById('camImg'); if(img.style.display==='none'){img.style.display='block';document.getElementById('noSig').style.display='none';}
  img.src='data:image/jpeg;base64,'+m.img;
  document.getElementById('mF').textContent=m.fps; document.getElementById('mI').textContent=m.inf_ms;
  document.getElementById('tDir').textContent=(m.dir||'---').toUpperCase();
  document.getElementById('tDet').textContent=(m.detections||[]).length;
  document.getElementById('aPos').textContent=m.pos||'---'; document.getElementById('aCmd').textContent=(m.dir||'idle').toUpperCase();
  const dets=m.detections||[];
  if(dets.length>0){
    const best=dets.reduce((a,b)=>a.conf>b.conf?a:b);
    document.getElementById('tConf').textContent=(best.conf*100).toFixed(0)+'%';
    document.getElementById('dtag').classList.add('vis'); dot('dDet','on');lbl('lDet','LOCKED');
  } else {
    document.getElementById('tConf').textContent='--'; document.getElementById('dtag').classList.remove('vis');
    dot('dDet', m.model_ok ? 'warn':''); lbl('lDet', m.model_ok ? 'SCANNING':'IDLE');
  }
  dot('dMdl',m.model_ok?'on':'warn');lbl('lMdl',m.model_ok?'MODEL OK':'NO MODEL');
  dot('dEsp',m.esp_ok?'on':'err');lbl('lEsp',m.esp_ok?'ONLINE':'OFFLINE');
}

function onStatus(m){
  document.getElementById('tFrm').textContent=m.frames||0; document.getElementById('ipInp').value=m.esp_ip||'';
  const spd=m.speed||180; document.getElementById('spdS').value=spd; document.getElementById('spdVal').textContent=spd; document.getElementById('spdPct').textContent=Math.round(spd*100/255);
  const ar=m.auto_running; document.getElementById('aState').textContent=ar?'TRACKING + MOVING':'TRACKING ONLY';
  document.getElementById('btnI').classList.toggle('lit-g',ar); document.getElementById('btnSt').classList.toggle('lit-r',!ar&&ui.mode==='auto');
}
function applyIP(){send({action:'set_ip',ip:document.getElementById('ipInp').value.trim()});}
function setSpd(el){const v=parseInt(el.value); document.getElementById('spdVal').textContent=v; document.getElementById('spdPct').textContent=Math.round(v*100/255); send({action:'set_speed',value:v});}
function setMode(m){ui.mode=m;send({action:'set_mode',mode:m}); document.getElementById('bMan').classList.toggle('active',m==='manual'); document.getElementById('bAut').classList.toggle('active',m==='auto'); document.getElementById('secM').classList.toggle('vis',m==='manual'); document.getElementById('secA').classList.toggle('vis',m==='auto');}
function autoInit(){send({action:'auto_init'});document.getElementById('btnI').classList.add('lit-g');document.getElementById('btnSt').classList.remove('lit-r');document.getElementById('aState').textContent='TRACKING + MOVING';}
function autoStop(){send({action:'auto_stop'});document.getElementById('btnSt').classList.add('lit-r');document.getElementById('btnI').classList.remove('lit-g');document.getElementById('aState').textContent='TRACKING ONLY';}
function loadMdl(){const p=document.getElementById('mdlP').value.trim();send({action:'load_model',path:p});addLog(p?`Loading: ${p}`:'Auto-searching...','info',"logBox");}
function setConf(el){const v=parseInt(el.value)/100;document.getElementById('confV').textContent=v.toFixed(2);send({action:'set_conf',value:v});}
document.querySelectorAll('.db').forEach(btn=>{
  const d=btn.dataset.d; const press=()=>{if(ui.mode!=='manual')return;btn.classList.add('pr');send({action:'cmd',dir:d});};
  const rel=()=>{btn.classList.remove('pr');if(d!=='stop'&&ui.mode==='manual')send({action:'cmd',dir:'stop'});};
  btn.addEventListener('mousedown',press); btn.addEventListener('touchstart',e=>{e.preventDefault();press();},{passive:false});
  btn.addEventListener('mouseup',rel);btn.addEventListener('mouseleave',rel);btn.addEventListener('touchend',rel);
});
const KM={ArrowUp:'forward',KeyW:'forward',ArrowDown:'backward',KeyS:'backward',ArrowLeft:'left',KeyA:'left',ArrowRight:'right',KeyD:'right',Space:'stop'}; const held=new Set();
document.addEventListener('keydown',e=>{ if(ui.mode!=='manual')return;const d=KM[e.code];if(!d||held.has(e.code))return; held.add(e.code);send({action:'cmd',dir:d}); const el={forward:'dbF',backward:'dbB',left:'dbL',right:'dbR',stop:'dbS'}[d]; if(el)document.getElementById(el).classList.add('pr'); if(e.code==='Space')e.preventDefault(); });
document.addEventListener('keyup',e=>{ if(ui.mode!=='manual')return;const d=KM[e.code];held.delete(e.code); if(d&&d!=='stop')send({action:'cmd',dir:'stop'}); const el={forward:'dbF',backward:'dbB',left:'dbL',right:'dbR',stop:'dbS'}[d]; if(el)document.getElementById(el).classList.remove('pr'); });

function addLog(msg,lv='info',boxId){
  const b=document.getElementById(boxId); if(!b) return; const e=document.createElement('span'); e.className=`le ${lv}`;
  const ts=new Date().toLocaleTimeString('en',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
  e.textContent=`[${ts}] ${msg}`; b.appendChild(e);b.appendChild(document.createElement('br')); b.scrollTop=b.scrollHeight;
  while(b.children.length>120)b.removeChild(b.firstChild);
}
function dot(id,s){const e=document.getElementById(id);if(e){e.className='dot';if(s)e.classList.add(s);}}
function lbl(id,t){const e=document.getElementById(id);if(e)e.textContent=t;}

/* --- DISTANCE LOGIC --- */
const S=225, E=495, DMIN=10, DMAX=30, CX=145, CY=145, R=106;
function arc(a1, a2){ const r=a=>( (a-90)*Math.PI/180 ); const x1=CX+R*Math.cos(r(a1)), y1=CY+R*Math.sin(r(a1)); const x2=CX+R*Math.cos(r(a2)), y2=CY+R*Math.sin(r(a2)); const lg=(a2-a1)>180?1:0; return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${R} ${R} 0 ${lg} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`; }
(()=>{ document.getElementById('arcBg').setAttribute('d', arc(S,E)); const g=document.getElementById('ticks'); for(let i=0;i<=20;i++){ const a=S+(i/20)*(E-S); const major=i%5===0; const r1=major?114:118, r2=126, rad=(a-90)*Math.PI/180; const ln=document.createElementNS('http://www.w3.org/2000/svg','line'); ln.setAttribute('x1',(CX+r1*Math.cos(rad)).toFixed(1)); ln.setAttribute('y1',(CY+r1*Math.sin(rad)).toFixed(1)); ln.setAttribute('x2',(CX+r2*Math.cos(rad)).toFixed(1)); ln.setAttribute('y2',(CY+r2*Math.sin(rad)).toFixed(1)); ln.setAttribute('stroke-width',major?2:1); ln.setAttribute('stroke', major?'#6b7280':'#9ca3af'); g.appendChild(ln); if(major&&i>0&&i<20){ const lr=102, lx=CX+lr*Math.cos(rad), ly=CY+lr*Math.sin(rad); const t=document.createElementNS('http://www.w3.org/2000/svg','text'); t.setAttribute('x',lx.toFixed(1)); t.setAttribute('y',ly.toFixed(1)); t.setAttribute('fill','#6b7280'); t.setAttribute('font-size','8'); t.setAttribute('text-anchor','middle'); t.setAttribute('dominant-baseline','middle'); t.setAttribute('font-family','IBM Plex Mono'); t.setAttribute('font-weight','600'); t.textContent=Math.round(DMIN+(i/20)*(DMAX-DMIN)); g.appendChild(t); } } })();
function setArc(dist){ const t=Math.max(0,Math.min(1,(dist-DMIN)/(DMAX-DMIN))); const fill=document.getElementById('arcFill'); if(t<=0){fill.style.opacity='0';return;} fill.style.opacity='1'; fill.setAttribute('d', arc(S, Math.min(S+t*(E-S), E-0.3))); const col = dist<=15?'#059669':dist<=22?'#d97706':'#ef4444'; fill.setAttribute('stroke', col); document.getElementById('distNum').style.color=col; document.getElementById('distNum').style.textShadow=`0 0 10px ${col}44`; }
let d_locked=false, d_espOk=false;
function calcDist(w,h){ const area=w*h; if(area<=0)return null; return Math.round(Math.max(DMIN,Math.min(DMAX, 4.0/Math.sqrt(area)))); }
function showDist(dist, conf){ const el=document.getElementById('distNum'); el.textContent=dist; el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop'); document.getElementById('distConf').textContent=`BALL CONF: ${(conf*100).toFixed(0)}%`; setArc(dist); document.getElementById('lockBadge').style.display='block'; document.getElementById('lockBadge').classList.add('badge-glow'); dot('d_dLk','on'); lbl('d_lLk','LOCKED'); document.getElementById('d_tStat').textContent='LOCKED'; setState('DISTANCE CAPTURED — LOCKED', false); addLog(`Measured: ${dist} cm (conf ${(conf*100).toFixed(0)}%)`, 'ok', "d_logBox"); }
function resetLock(){ d_locked=false; const dn=document.getElementById('distNum'); dn.textContent='--'; dn.style.color='var(--tx)'; dn.style.textShadow='none'; document.getElementById('arcFill').style.opacity='0'; document.getElementById('distConf').textContent='AWAITING BALL'; document.getElementById('lockBadge').style.display='none'; document.getElementById('lockBadge').classList.remove('badge-glow'); document.getElementById('d_tArea').textContent='---'; document.getElementById('d_tConf').textContent='---'; document.getElementById('d_tStat').textContent='SCANNING'; dot('d_dLk',''); lbl('d_lLk','UNLOCKED'); setState(d_espOk?'SCANNING FOR GOLF BALL...':'AWAITING ESP-CAM...', true); addLog('Reset — waiting for next ball detection','warn',"d_logBox"); }
function setState(txt, pulse){ document.getElementById('stateText').textContent=txt; if(pulse) document.getElementById('pulse').classList.add('lit'); else document.getElementById('pulse').classList.remove('lit'); }
function d_onFrame(m){
  if(m.esp_ok!==d_espOk){ d_espOk=m.esp_ok; dot('d_dEsp', d_espOk?'on':'err'); lbl('d_lEsp', d_espOk?'ESP ONLINE':'ESP OFFLINE'); if(!d_espOk&&!d_locked) setState('AWAITING ESP-CAM...', true); }
  if(d_locked) return;
  if(!m.esp_ok){ setState('AWAITING ESP-CAM...', true); return; }
  const dets=m.detections||[];
  if(dets.length===0){ setState('SCANNING FOR GOLF BALL...', true); document.getElementById('d_tStat').textContent='SCANNING'; return; }
  const best=dets.reduce((a,b)=>a.conf>b.conf?a:b); const dist=calcDist(best.w, best.h); if(dist===null) return;
  document.getElementById('d_tArea').textContent=(best.w*best.h).toFixed(4); document.getElementById('d_tConf').textContent=(best.conf*100).toFixed(0)+'%'; setState('LOCKING MEASUREMENT...', true);
  setTimeout(()=>{ if(d_locked) return; showDist(dist, best.conf); d_locked=true; }, 380);
}

connectWS();
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(cfg.get("server_port", 8000))
    log.info(f"Starting → http://localhost:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="warning", reload=False)
