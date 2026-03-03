"""
SALIDA AUTOMATICA v2 (CON DETENER / KEEP-ALIVE)
================================================
Flujo de barrera:
  1) Vehículo detectado en zona SALIENDO (↑) dentro del polígono → 1 pulso SUBIR
  2) Un pulso SUBIR por cada vehículo nuevo detectado
  3) Si hay vehículos aún en zona después de que la barrera subió completamente:
     → Ciclo keep-alive: DETENER (frena barrera) → espera → SUBIR (re-sube)
     → Se repite mientras haya vehículos en la zona
  4) Cuando no hay vehículos → se deja que la barrera baje sola

Mejoras:
  - Modo noche automático (CLAHE + conf baja)
  - Tracking + dirección de movimiento (solo detecta salida ↑)
  - Latencia reducida: YOLO 256px, webhook en hilo separado
  - UI compacta: fullscreen, LEDs, panel colapsable

Requisitos:
  pip install opencv-python mss requests pillow numpy
"""

import threading
import time
import queue
from datetime import datetime, timedelta
from pathlib import Path
import json
import math

import cv2
import numpy as np
import mss
import requests
import tkinter as tk

try:
    from PIL import Image, ImageTk
except Exception:
    raise SystemExit("Falta Pillow: pip install pillow")

# ── OpenCV ────────────────────────────────────────────────────────────────────
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# ── PARÁMETROS ────────────────────────────────────────────────────────────────
SCALE                   = 0.55
YOLO_INPUT_SIZE         = 256
CONF_THRES_DAY          = 0.35
CONF_THRES_NIGHT        = 0.22
NMS_THRES               = 0.40
VEHICLE_CLASSES         = {"car", "bus", "truck", "motorbike", "bicycle"}
NIGHT_BRIGHTNESS_THRESH = 60

# Tracking
MAX_TRACK_AGE           = 8
MAX_MATCH_DIST          = 80
MIN_FRAMES_CONFIRM      = 2
MIN_MOVEMENT_UP         = 8

# Temporización barrera
BARRIER_RISE_WAIT       = 5.5   # seg tras SUBIR para dar tiempo a que suba

# Keep-alive: ciclo DETENER→SUBIR mientras haya vehículos en zona
KEEPALIVE_INTERVAL      = 6.0   # seg entre ciclos keep-alive (después de subida completa)
KEEPALIVE_DET_DELAY     = 1.5   # seg de espera entre ráfaga DETENER y ráfaga SUBIR

# Ráfagas de pulsos (simula mantener presionado el botón ~5seg)
PULSES_PER_ACTION       = 6     # pulsos por acción (SUBIR o DETENER)
PULSE_DELAY             = 0.5   # seg entre pulsos consecutivos (6 × 0.5 ≈ 3-5seg total)

# Limpieza de IDs disparados (para no crecer infinito)
FIRED_TTL_SEC           = 30.0  # recordar track_id disparado por X seg (ampliado para keep-alive)

DEFAULT_MONITOR_INDEX   = 2

# ── RUTAS ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
YOLO_CFG     = str(BASE_DIR / "yolov4-tiny.cfg")
YOLO_WEIGHTS = str(BASE_DIR / "yolov4-tiny.weights")
YOLO_NAMES   = str(BASE_DIR / "coco.names")
STATE_FILE   = BASE_DIR / "webhook_state.json"
CONFIG_FILE  = BASE_DIR / "roi_poly_config.json"

# ── WEBHOOKS SUBIR ────────────────────────────────────────────────────────────
RAW_OPEN_IDS = [
    "7332b6c692284bddbc8b41154ea350ab",
    "6290377639b949f3a5ac40303c463e0d",
    "0f56cd5e899e4abcacd2d8adaf39b761",
    "272c9e2f6ca74540bba26f3e93254309",
    "838199dbb0af4de9a6221ee7eff8b970",
    "70cad7d984684251b5afb4bb86587bad",
    "99cf90d857c74b6d8746adb320936c73",
    "b64adbb325a8424e9c7fedbc11a97382",
    "bbaa9067ab064938aced336942829f84",
    "b9b2124e43a34bbd9b2f4ed4672778c5",
]

open_ids = list(dict.fromkeys(RAW_OPEN_IDS))

_open_state = {
    "last_idx":  0,
    "exhausted": {w: None for w in open_ids},
    "calls":     {w: 0    for w in open_ids},
}

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept":     "application/json,text/plain,*/*",
    "Connection": "keep-alive",
}

# ── SESIÓN HTTP (reutiliza conexiones TCP) ───────────────────────────────────
_http_session = requests.Session()
_http_session.headers.update(HTTP_HEADERS)

# ── PERSISTENCIA ROI + POLY ───────────────────────────────────────────────────
def save_roi_poly(monitor_index, roi_xywh, poly):
    CONFIG_FILE.write_text(json.dumps({
        "monitor_index": int(monitor_index),
        "roi":  [int(v) for v in roi_xywh],
        "poly": poly.astype(int).tolist(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

def load_roi_poly():
    if not CONFIG_FILE.exists():
        return None
    try:
        d    = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        mon  = int(d["monitor_index"])
        roi  = tuple(int(v) for v in d["roi"])
        poly = np.array(d["poly"], dtype=np.int32)
        if len(roi) != 4 or poly.shape[0] < 3:
            return None
        return mon, roi, poly
    except Exception:
        return None

# ── SELECCIÓN POLÍGONO ────────────────────────────────────────────────────────
class PolySelector:
    def __init__(self, title):
        self.title  = title
        self.points = []
        self.cancel = False

    def mouse_cb(self, event, x, y, flags, param):
        if self.cancel:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def select(self, image):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self.mouse_cb)
        while True:
            vis = image.copy()
            for p in self.points:
                cv2.circle(vis, p, 4, (0, 255, 255), -1)
            if len(self.points) >= 2:
                cv2.polylines(vis, [np.array(self.points, np.int32)],
                              False, (0, 255, 255), 2)
            cv2.putText(vis,
                "Izq:add  Der:undo  ENTER:ok  C:limpiar  ESC:cancelar",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50),
                2, cv2.LINE_AA)
            cv2.imshow(self.title, vis)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                self.cancel = True
                break
            elif k in (10, 13) and len(self.points) >= 3:
                break
            elif k == ord('c'):
                self.points = []
        cv2.destroyWindow(self.title)
        return None if self.cancel else np.array(self.points, np.int32)

def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

# ── TRACKER ───────────────────────────────────────────────────────────────────
class Track:
    _nid = 0

    def __init__(self, cx, cy):
        self.id        = Track._nid
        Track._nid    += 1
        self.cx        = cx
        self.cy        = cy
        self.age       = 0
        self.frames    = 1
        self.dy_acc    = 0.0
        self.confirmed = False

    def update(self, cx, cy):
        self.dy_acc   += cy - self.cy
        self.cx, self.cy = cx, cy
        self.age       = 0
        self.frames   += 1
        if self.frames >= MIN_FRAMES_CONFIRM:
            self.confirmed = True

    @property
    def moving_up(self):
        return self.dy_acc < -MIN_MOVEMENT_UP


class CentroidTracker:
    def __init__(self):
        self.tracks = []

    def update(self, dets):
        for t in self.tracks:
            t.age += 1
        unmatched = list(range(len(dets)))
        if self.tracks and dets:
            used = set()
            for di in list(unmatched):
                cx, cy  = dets[di]
                bt, bd  = None, MAX_MATCH_DIST
                for ti, t in enumerate(self.tracks):
                    if ti in used:
                        continue
                    d = math.hypot(cx - t.cx, cy - t.cy)
                    if d < bd:
                        bd = d
                        bt = ti
                if bt is not None:
                    self.tracks[bt].update(*dets[di])
                    used.add(bt)
                    unmatched.remove(di)
        for di in unmatched:
            self.tracks.append(Track(*dets[di]))
        self.tracks = [t for t in self.tracks if t.age <= MAX_TRACK_AGE]
        return self.tracks

# ── PERSISTENCIA ESTADO WEBHOOKS ──────────────────────────────────────────────
def _load_state():
    try:
        if not STATE_FILE.exists():
            return
        d    = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        now  = datetime.now()
        si   = d.get("open", {})
        _open_state["last_idx"] = int(si.get("last_idx", 0)) % max(1, len(open_ids))
        for wid in open_ids:
            v = si.get("exhausted", {}).get(wid)
            exp = datetime.fromisoformat(v) if v else None
            _open_state["exhausted"][wid] = exp if (exp and exp > now) else None
            c = si.get("calls", {}).get(wid)
            if c is not None:
                _open_state["calls"][wid] = int(c)
    except Exception:
        pass

def _save_state():
    try:
        def _ser(state):
            return {
                "last_idx":  state["last_idx"],
                "exhausted": {w: (v.isoformat() if v else None)
                              for w, v in state["exhausted"].items()},
                "calls":     state["calls"],
            }
        STATE_FILE.write_text(
            json.dumps({"open": _ser(_open_state)}, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception:
        pass

_load_state()

# ── COLA Y WORKER DE WEBHOOKS ────────────────────────────────────────────────
_wh_queue        = queue.Queue(maxsize=120)  # más grande para ráfagas
_WH_MAX_RETRIES  = 3
_WH_RETRY_DELAY  = 0.5   # seg entre reintentos
_wh_results      = {}     # track_id -> {"status": "pending"|"ok"|"fail", "retries": int}
_wh_results_lock = threading.Lock()


def _enqueue_burst(track_id, tag, count=None):
    """Encola una ráfaga de N pulsos para simular presión sostenida."""
    if count is None:
        count = PULSES_PER_ACTION
    enqueued = 0
    for i in range(count):
        try:
            _wh_queue.put_nowait((track_id, f"{tag}[{i+1}/{count}]", 0))
            enqueued += 1
        except queue.Full:
            break
    return enqueued


def _wh_worker(engine_ref):
    """Worker persistente: procesa webhooks de la cola con reintentos.
    Despacha a pulso_subir() o pulso_detener() según el tag.
    Espera PULSE_DELAY entre pulsos exitosos (simula presión sostenida)."""
    while True:
        try:
            item = _wh_queue.get(timeout=2.0)
        except queue.Empty:
            continue

        if item is None:          # Poison pill → salir
            break

        track_id, tag, retries = item

        # Despachar según tipo de acción
        if "DETENER" in tag:
            ok, reason, ms = pulso_detener()
        else:
            ok, reason, ms = pulso_subir()

        with _wh_results_lock:
            if ok:
                _wh_results[track_id] = {"status": "ok", "retries": retries}
                if engine_ref:
                    engine_ref.set_wh(f"{tag} OK ({ms}ms)")
            else:
                if retries < _WH_MAX_RETRIES:
                    _wh_results[track_id] = {"status": "pending", "retries": retries + 1}
                    time.sleep(_WH_RETRY_DELAY)
                    try:
                        _wh_queue.put_nowait((track_id, tag, retries + 1))
                    except queue.Full:
                        _wh_results[track_id] = {"status": "fail", "retries": retries + 1}
                        if engine_ref:
                            engine_ref.set_wh(f"{tag} FAIL(cola llena):{reason}")
                else:
                    _wh_results[track_id] = {"status": "fail", "retries": retries}
                    if engine_ref:
                        engine_ref.set_wh(f"{tag} FAIL:{reason} ({retries}x)")

        _wh_queue.task_done()

        # Pausa entre pulsos consecutivos (simula presión sostenida)
        if ok:
            time.sleep(PULSE_DELAY)


# ── WEBHOOK ENGINE ────────────────────────────────────────────────────────────
def _fire(state, ids_list):
    """Dispara un webhook del pool. Rota si 406 o error red. Devuelve (ok, reason, ms)."""
    now = datetime.now()

    # Limpiar agotados expirados
    for wid in ids_list:
        u = state["exhausted"].get(wid)
        if u and now >= u:
            state["exhausted"][wid] = None

    n = len(ids_list)
    if n == 0:
        return False, "no_webhooks", 0

    last_reason = "all_exhausted"
    for step in range(n):
        i   = (state["last_idx"] + step) % n
        wid = ids_list[i]

        u = state["exhausted"].get(wid)
        if u and now < u:
            last_reason = "exhausted"
            continue

        url = f"https://us-apia.coolkit.cc/v2/smartscene2/webhooks/execute?id={wid}"
        try:
            t0 = time.time()
            r  = _http_session.get(url, timeout=3, allow_redirects=True)
            ms = int((time.time() - t0) * 1000)

            if not (200 <= r.status_code < 300):
                last_reason = f"http_{r.status_code}"
                continue

            try:
                d   = r.json()
                err = d.get("error")
                if err == 0:
                    state["calls"][wid] = state["calls"].get(wid, 0) + 1
                    state["last_idx"]   = (i + 1) % n
                    _save_state()
                    return True, "ok", ms
                if err == 406:
                    tom = (now + timedelta(days=1)).replace(hour=0, minute=3, second=0, microsecond=0)
                    state["exhausted"][wid] = tom
                    _save_state()
                    last_reason = "err_406"
                    continue
                last_reason = f"err_{err}"
                continue
            except Exception:
                state["calls"][wid] = state["calls"].get(wid, 0) + 1
                state["last_idx"]   = (i + 1) % n
                _save_state()
                return True, f"ok_{r.status_code}", ms

        except Exception as e:
            last_reason = f"exc_{type(e).__name__}"
            continue

    return False, last_reason, 0

def pulso_subir():
    return _fire(_open_state, open_ids)

# ── WEBHOOKS DETENER ─────────────────────────────────────────────────────────
# IMPORTANTE: Reemplaza estos placeholders con tus 10 IDs reales de DETENER
RAW_STOP_IDS = [
    "PLACEHOLDER_DETENER_01",
    "PLACEHOLDER_DETENER_02",
    "PLACEHOLDER_DETENER_03",
    "PLACEHOLDER_DETENER_04",
    "PLACEHOLDER_DETENER_05",
    "PLACEHOLDER_DETENER_06",
    "PLACEHOLDER_DETENER_07",
    "PLACEHOLDER_DETENER_08",
    "PLACEHOLDER_DETENER_09",
    "PLACEHOLDER_DETENER_10",
]

stop_ids = list(dict.fromkeys(RAW_STOP_IDS))

_stop_state = {
    "last_idx":  0,
    "exhausted": {w: None for w in stop_ids},
    "calls":     {w: 0    for w in stop_ids},
}

def pulso_detener():
    return _fire(_stop_state, stop_ids)

# ── YOLO ──────────────────────────────────────────────────────────────────────
with open(YOLO_NAMES, encoding="utf-8") as _f:
    class_names = [l.strip() for l in _f]

net        = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
_lnames    = net.getLayerNames()
out_layers = [_lnames[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ── MOTOR DE DETECCIÓN ────────────────────────────────────────────────────────
class DetectionEngine:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread     = None
        self._lock      = threading.Lock()
        self.last_frame = None
        # estado publicado
        self._status  = "LISTO"
        self._wh_txt  = "N/A"
        self._night   = False
        self._veh_in  = 0
        self._barrier = False
        self._phase   = "LIBRE"
        self._fps     = 0.0
        self._yolo_ms = 0
        # ROI/POLY
        self.monitor_index = DEFAULT_MONITOR_INDEX
        self.roi_xywh      = None
        self.poly          = None
        self.grab_rect     = None
        self._poly_sc      = None
        self._sc_cache     = None

    # ── API thread-safe ──────────────────────────────────────────────────────
    def set_status(self, t):
        with self._lock:
            self._status = t

    def set_wh(self, t):
        with self._lock:
            self._wh_txt = t

    def get_state(self):
        with self._lock:
            return (self._status, self._wh_txt, self._night,
                    self._veh_in, self._barrier, self._phase,
                    self._fps, self._yolo_ms)

    def get_frame(self):
        with self._lock:
            return None if self.last_frame is None else self.last_frame.copy()

    def _poly_at(self, scale):
        if self._sc_cache != scale:
            self._poly_sc  = (self.poly.astype(np.float32) * scale).astype(np.int32)
            self._sc_cache = scale
        return self._poly_sc

    # ── selección ROI/POLY ───────────────────────────────────────────────────
    def _select_roi(self, sct, monitor):
        img = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
        roi = cv2.selectROI("Selecciona ROI de salida", img, False)
        cv2.destroyWindow("Selecciona ROI de salida")
        return tuple(map(int, roi))

    def _select_poly(self, sct, rect):
        img  = cv2.cvtColor(np.array(sct.grab(rect)), cv2.COLOR_BGRA2BGR)
        poly = PolySelector("Dibuja zona SALIDA (vehiculos suben ↑)").select(img)
        if poly is None:
            raise RuntimeError("Selección cancelada")
        return poly

    def ensure_roi_poly(self):
        loaded = load_roi_poly()
        sct    = mss.mss()
        if loaded is None:
            self.monitor_index = (DEFAULT_MONITOR_INDEX
                                  if DEFAULT_MONITOR_INDEX < len(sct.monitors)
                                  else 1)
            mon = sct.monitors[self.monitor_index]
            self.set_status("Selecciona ROI…")
            rx, ry, rw, rh = self._select_roi(sct, mon)
            if rw == 0 or rh == 0:
                raise RuntimeError("ROI inválido")
            rect = {"top": ry + mon["top"], "left": rx + mon["left"],
                    "width": rw, "height": rh}
            poly = self._select_poly(sct, rect)
            save_roi_poly(self.monitor_index, (rx, ry, rw, rh), poly)
            self.roi_xywh  = (rx, ry, rw, rh)
            self.poly      = poly
            self.grab_rect = rect
        else:
            self.monitor_index, roi, poly = loaded
            if self.monitor_index >= len(sct.monitors):
                self.monitor_index = 1
            mon = sct.monitors[self.monitor_index]
            rx, ry, rw, rh = roi
            self.roi_xywh  = roi
            self.poly      = poly
            self.grab_rect = {"top": ry + mon["top"], "left": rx + mon["left"],
                              "width": rw, "height": rh}
        self.set_status("Config cargada ✓")

    # ── start / stop ─────────────────────────────────────────────────────────
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.set_status("ACTIVANDO…")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        try:
            _wh_queue.put_nowait(None)   # Poison pill → detener worker
        except queue.Full:
            pass
        self.set_status("DESACTIVADO")

    # ── YOLO detect helper ────────────────────────────────────────────────────
    def _yolo_detect_centroids(self, frame, conf_thres):
        """
        Retorna lista de centroides (cx, cy) de vehículos detectados.
        frame: BGR ya escalado a tamaño de trabajo (SCALE aplicado).
        """
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(out_layers)

        boxes = []
        confs = []
        for out in outs:
            for det in out:
                scores = det[5:]
                cls_id = int(np.argmax(scores))
                conf   = float(scores[cls_id])
                if conf < conf_thres:
                    continue
                label = class_names[cls_id] if 0 <= cls_id < len(class_names) else ""
                if label not in VEHICLE_CLASSES:
                    continue

                cx = float(det[0]) * W
                cy = float(det[1]) * H
                w  = float(det[2]) * W
                h  = float(det[3]) * H
                x  = int(cx - w/2)
                y  = int(cy - h/2)
                boxes.append([x, y, int(w), int(h)])
                confs.append(conf)

        idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_thres, NMS_THRES)
        cents = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                cx = x + w//2
                cy = y + h//2
                cents.append((int(cx), int(cy)))
        return cents

    # ── loop principal ────────────────────────────────────────────────────────
    def _run(self):
        try:
            self.ensure_roi_poly()
        except Exception as e:
            self.set_status(f"Error: {e}")
            return

        sct     = mss.mss()
        tracker = CentroidTracker()
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # ── lógica SUBIR + DETENER keep-alive ─────────────────────────────────
        phase        = "LIBRE"   # UI: LIBRE / SUBIENDO / KEEPALIVE
        last_open_t  = 0.0
        fired_ids    = {}        # track_id -> timestamp último disparo
        fps_s        = 0.0
        prev_t       = time.time()

        # Keep-alive state machine: idle → wait_subir → wait_detener → ...
        ka_step      = "idle"    # "idle" | "wait_subir" | "wait_detener"
        ka_t         = 0.0       # timestamp última acción keep-alive

        # Worker persistente para webhooks (reemplaza thread-per-request)
        wh_worker_thread = threading.Thread(target=_wh_worker, args=(self,), daemon=True)
        wh_worker_thread.start()

        self.set_status("ACTIVO")

        while not self.stop_event.is_set():
            try:
                now = time.time()

                # Grab ROI
                img = np.array(sct.grab(self.grab_rect))  # BGRA
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Scale para velocidad
                if SCALE != 1.0:
                    frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE,
                                       interpolation=cv2.INTER_AREA)

                # Modo noche automático (brightness promedio)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_b = float(np.mean(gray))
                night = mean_b < NIGHT_BRIGHTNESS_THRESH

                if night:
                    # CLAHE para levantar contraste (en gris)
                    eq = clahe.apply(gray)
                    frame = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

                conf_th = CONF_THRES_NIGHT if night else CONF_THRES_DAY

                # YOLO
                t0 = time.time()
                cents = self._yolo_detect_centroids(frame, conf_th)
                inf_ms = int((time.time() - t0) * 1000)

                # Filtrar centroides dentro del polígono
                poly_sc = self._poly_at(SCALE)
                cents_in = [c for c in cents if point_in_poly(c, poly_sc)]
                tracks = tracker.update(cents_in)

                # Conteo veh en zona
                veh_in_zone = 0
                valid_exit_tracks = []
                for t in tracks:
                    if t.confirmed and point_in_poly((t.cx, t.cy), poly_sc):
                        veh_in_zone += 1
                        # SOLO SALIDA (↑)
                        if t.moving_up:
                            valid_exit_tracks.append(t)

                # Limpiar fired_ids viejos
                if fired_ids:
                    dead = [tid for tid, ts in fired_ids.items() if (now - ts) > FIRED_TTL_SEC]
                    for tid in dead:
                        fired_ids.pop(tid, None)
                        with _wh_results_lock:
                            _wh_results.pop(tid, None)

                # Disparar ráfaga SUBIR 1 vez por track id (vehículo)
                for t in valid_exit_tracks:
                    tid = t.id
                    with _wh_results_lock:
                        result = _wh_results.get(tid)

                    if tid not in fired_ids:
                        # Nunca disparado → encolar ráfaga de SUBIR
                        fired_ids[tid] = now
                        last_open_t = now
                        with _wh_results_lock:
                            _wh_results[tid] = {"status": "pending", "retries": 0}
                        n = _enqueue_burst(tid, "SUBIR")
                        if n == 0:
                            self.set_wh("SUBIR FAIL: cola llena")
                            fired_ids.pop(tid, None)
                        else:
                            self.set_wh(f"SUBIR: {n} pulsos encolados")
                    elif result and result["status"] == "fail":
                        # Falló y agotó reintentos → re-encolar ráfaga
                        fired_ids[tid] = now
                        last_open_t = now
                        with _wh_results_lock:
                            _wh_results[tid] = {"status": "pending", "retries": 0}
                        _enqueue_burst(tid, "SUBIR-RETRY")

                # ── Estado barrera + Keep-alive ────────────────────────────────
                time_since_subir = now - last_open_t
                barrier_rising   = last_open_t > 0 and time_since_subir <= BARRIER_RISE_WAIT
                barrier_up       = last_open_t > 0 and time_since_subir > BARRIER_RISE_WAIT

                if veh_in_zone > 0 and barrier_up:
                    # Barrera ya subió y hay vehículos → ciclo keep-alive con ráfagas
                    if ka_step == "idle":
                        # Iniciar ciclo: enviar ráfaga DETENER
                        n = _enqueue_burst("KA", "DETENER")
                        self.set_wh(f"KA: DETENER x{n} pulsos")
                        ka_t = now
                        ka_step = "wait_subir"

                    elif ka_step == "wait_subir":
                        # Ráfaga DETENER enviada, esperar antes de ráfaga SUBIR
                        if (now - ka_t) >= KEEPALIVE_DET_DELAY:
                            n = _enqueue_burst("KA", "SUBIR-KA")
                            self.set_wh(f"KA: SUBIR x{n} pulsos")
                            last_open_t = now   # reset timer de barrera
                            ka_t = now
                            ka_step = "wait_detener"

                    elif ka_step == "wait_detener":
                        # Ráfaga SUBIR enviada, esperar intervalo para próximo DETENER
                        if (now - ka_t) >= KEEPALIVE_INTERVAL:
                            n = _enqueue_burst("KA", "DETENER")
                            self.set_wh(f"KA: DETENER x{n} pulsos")
                            ka_t = now
                            ka_step = "wait_subir"

                    phase = "KEEPALIVE"

                elif barrier_rising:
                    phase = "SUBIENDO"
                    # No tocar keep-alive mientras sube (NUNCA enviar DETENER aquí)

                else:
                    # Sin vehículos o barrera ya bajó → reset keep-alive
                    if ka_step != "idle":
                        ka_step = "idle"
                        ka_t = 0.0
                    phase = "LIBRE"

                barrier_open = phase != "LIBRE"

                # HUD
                cv2.polylines(frame, [poly_sc], True, (0, 255, 255), 2)
                dt = now - prev_t
                prev_t = now
                fps_s = fps_s * 0.85 + (1.0 / max(1e-6, dt)) * 0.15

                mode_t = "NOCHE" if night else "DIA"
                if phase == "KEEPALIVE":
                    pc = (0, 165, 255)   # naranja: keep-alive activo
                elif phase == "SUBIENDO":
                    pc = (0, 200, 255)   # cyan: subiendo
                else:
                    pc = (80, 80, 80)    # gris: libre

                cv2.putText(frame, f"FPS:{fps_s:.1f}  YOLO:{inf_ms}ms  {mode_t}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (50, 255, 50), 1, cv2.LINE_AA)
                cv2.putText(frame, f"FASE: {phase}  Veh:{veh_in_zone}  KA:{ka_step}",
                            (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            pc, 2, cv2.LINE_AA)

                with self._lock:
                    self._night   = night
                    self._veh_in  = veh_in_zone
                    self._barrier = barrier_open
                    self._phase   = phase
                    self._fps     = fps_s
                    self._yolo_ms = inf_ms
                    self.last_frame = frame

            except Exception as e:
                # MUY IMPORTANTE: evita que el hilo muera y el programa se “desactive solo”
                self.set_wh(f"LOOP_ERR:{type(e).__name__}")
                time.sleep(0.2)

        self.set_status("DESACTIVADO")


# ── UI ────────────────────────────────────────────────────────────────────────
COLORS = {
    "bg":      "#0F172A",
    "bg2":     "#1E293B",
    "card":    "#1E293B",
    "card_br": "#334155",
    "accent":  "#3B82F6",
    "green":   "#22C55E",
    "red":     "#EF4444",
    "yellow":  "#EAB308",
    "orange":  "#F97316",
    "cyan":    "#06B6D4",
    "text":    "#F1F5F9",
    "sub":     "#94A3B8",
    "dim":     "#475569",
    "btn_stop": "#DC2626",
    "btn_off":  "#374151",
}

# Fuente base (intenta Segoe UI, luego fallback)
_FONT = "Segoe UI"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SALIDA AUTOMATICA v2")
        self.configure(bg=COLORS["bg"])
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))
        self.bind("<F11>",    lambda e: self.attributes(
            "-fullscreen", not self.attributes("-fullscreen")))

        self.engine       = DetectionEngine()
        self._tkimg       = None
        self._panel_vis   = True

        self._build_ui()
        self.after(40, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── helpers UI ────────────────────────────────────────────────────────────
    def _make_card(self, parent, width=160):
        """Crea un frame tipo tarjeta con borde redondeado simulado."""
        card = tk.Frame(parent, bg=COLORS["card"], highlightbackground=COLORS["card_br"],
                        highlightthickness=1, padx=12, pady=8)
        card.configure(width=width)
        return card

    def _make_btn(self, parent, text, bg, fg, cmd, width=16):
        """Crea un botón estilizado con hover."""
        b = tk.Button(parent, text=text, bg=bg, fg=fg,
                      relief="flat", bd=0, padx=18, pady=8,
                      font=(_FONT, 10, "bold"), width=width,
                      activebackground=bg, activeforeground=fg,
                      cursor="hand2", command=cmd)
        # Hover: aclarar color
        def _enter(e, btn=b, base=bg):
            try:
                r, g, bl = btn.winfo_rgb(base)
                lighter = f"#{min(r//256+25,255):02x}{min(g//256+25,255):02x}{min(bl//256+25,255):02x}"
                btn.configure(bg=lighter)
            except Exception:
                pass
        def _leave(e, btn=b, base=bg):
            btn.configure(bg=base)
        b.bind("<Enter>", _enter)
        b.bind("<Leave>", _leave)
        return b

    # ── construcción UI ──────────────────────────────────────────────────────
    def _build_ui(self):
        C = COLORS

        # ── HEADER ───────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=C["bg2"], height=48)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="SALIDA AUTOMATICA",
                 bg=C["bg2"], fg=C["accent"],
                 font=(_FONT, 14, "bold")).pack(side="left", padx=16, pady=10)
        tk.Label(header, text="v2",
                 bg=C["bg2"], fg=C["dim"],
                 font=(_FONT, 10)).pack(side="left", pady=10)

        self._btn_toggle = tk.Button(
            header, text="▲ Panel", bg=C["bg2"], fg=C["sub"],
            relief="flat", bd=0, font=(_FONT, 9), cursor="hand2",
            command=self._toggle_panel)
        self._btn_toggle.pack(side="right", padx=10)

        tk.Label(header, text="ESC ventana  |  F11 fullscreen",
                 bg=C["bg2"], fg=C["dim"],
                 font=(_FONT, 8)).pack(side="right", padx=10)

        # ── PANEL DE CONTROL ─────────────────────────────────────────────────
        self._ctrl = tk.Frame(self, bg=C["bg"], pady=8)
        self._ctrl.pack(fill="x")

        # ── Fila 1: Tarjetas de estado ────────────────────────────────────────
        cards_row = tk.Frame(self._ctrl, bg=C["bg"])
        cards_row.pack(fill="x", padx=16, pady=(4, 6))

        # -- Tarjeta SISTEMA --
        c_sys = self._make_card(cards_row)
        c_sys.pack(side="left", padx=(0, 8), fill="y")
        tk.Label(c_sys, text="SISTEMA", bg=C["card"], fg=C["dim"],
                 font=(_FONT, 8, "bold")).pack(anchor="w")
        sys_row = tk.Frame(c_sys, bg=C["card"])
        sys_row.pack(anchor="w", pady=(4, 0))
        self._led_sys_cv = tk.Canvas(sys_row, width=14, height=14,
                                     bg=C["card"], highlightthickness=0)
        self._led_sys_cv.pack(side="left", padx=(0, 6))
        self._led_sys_o = self._led_sys_cv.create_oval(1, 1, 13, 13, fill=C["red"])
        self._lbl_sys = tk.Label(sys_row, text="DESACTIVADO",
                                 bg=C["card"], fg=C["text"],
                                 font=(_FONT, 11, "bold"))
        self._lbl_sys.pack(side="left")

        # -- Tarjeta BARRERA --
        c_bar = self._make_card(cards_row)
        c_bar.pack(side="left", padx=(0, 8), fill="y")
        tk.Label(c_bar, text="BARRERA", bg=C["card"], fg=C["dim"],
                 font=(_FONT, 8, "bold")).pack(anchor="w")
        bar_row = tk.Frame(c_bar, bg=C["card"])
        bar_row.pack(anchor="w", pady=(4, 0))
        self._led_bar_cv = tk.Canvas(bar_row, width=14, height=14,
                                     bg=C["card"], highlightthickness=0)
        self._led_bar_cv.pack(side="left", padx=(0, 6))
        self._led_bar_o = self._led_bar_cv.create_oval(1, 1, 13, 13, fill=C["dim"])
        self._lbl_bar = tk.Label(bar_row, text="LIBRE",
                                 bg=C["card"], fg=C["sub"],
                                 font=(_FONT, 11, "bold"))
        self._lbl_bar.pack(side="left")
        self._lbl_phase = tk.Label(c_bar, text="",
                                   bg=C["card"], fg=C["dim"],
                                   font=(_FONT, 8))
        self._lbl_phase.pack(anchor="w")

        # -- Tarjeta MODO --
        c_mode = self._make_card(cards_row)
        c_mode.pack(side="left", padx=(0, 8), fill="y")
        tk.Label(c_mode, text="MODO", bg=C["card"], fg=C["dim"],
                 font=(_FONT, 8, "bold")).pack(anchor="w")
        self._lbl_mode = tk.Label(c_mode, text="DIA",
                                  bg=C["card"], fg=C["yellow"],
                                  font=(_FONT, 13, "bold"))
        self._lbl_mode.pack(anchor="w", pady=(2, 0))

        # -- Tarjeta VEHICULOS --
        c_veh = self._make_card(cards_row)
        c_veh.pack(side="left", padx=(0, 8), fill="y")
        tk.Label(c_veh, text="VEHICULOS EN ZONA", bg=C["card"], fg=C["dim"],
                 font=(_FONT, 8, "bold")).pack(anchor="w")
        self._lbl_veh = tk.Label(c_veh, text="0",
                                 bg=C["card"], fg=C["sub"],
                                 font=(_FONT, 22, "bold"))
        self._lbl_veh.pack(anchor="w")

        # -- Tarjeta WEBHOOK --
        c_wh = self._make_card(cards_row)
        c_wh.pack(side="left", fill="both", expand=True)
        tk.Label(c_wh, text="ULTIMO WEBHOOK", bg=C["card"], fg=C["dim"],
                 font=(_FONT, 8, "bold")).pack(anchor="w")
        self._lbl_wh = tk.Label(c_wh, text="N/A",
                                bg=C["card"], fg=C["sub"],
                                font=(_FONT, 10))
        self._lbl_wh.pack(anchor="w", pady=(4, 0))

        # ── Fila 2: Botones ───────────────────────────────────────────────────
        btn_row = tk.Frame(self._ctrl, bg=C["bg"])
        btn_row.pack(pady=(4, 4))

        self._btn_act = self._make_btn(btn_row, "ACTIVAR", C["accent"], "white",
                                       self._on_activate)
        self._btn_act.pack(side="left", padx=4)

        self._btn_deact = self._make_btn(btn_row, "DESACTIVAR", C["btn_off"], C["sub"],
                                         self._on_deactivate)
        self._btn_deact.pack(side="left", padx=4)

        self._btn_reconf = self._make_btn(btn_row, "RECONFIGURAR", C["bg2"], C["sub"],
                                          self._on_reconfig, width=14)
        self._btn_reconf.pack(side="left", padx=4)

        # ── AREA VIDEO ────────────────────────────────────────────────────────
        self._vid_frame = tk.Frame(self, bg="#000")
        self._vid_frame.pack(fill="both", expand=True)
        self._vid_lbl = tk.Label(self._vid_frame, bg="#000")
        self._vid_lbl.pack(fill="both", expand=True)

        # ── BARRA INFERIOR (info técnica) ─────────────────────────────────────
        self._statusbar = tk.Frame(self, bg=C["bg2"], height=28)
        self._statusbar.pack(fill="x", side="bottom")
        self._statusbar.pack_propagate(False)

        self._lbl_fps = tk.Label(self._statusbar, text="FPS: --",
                                 bg=C["bg2"], fg=C["dim"],
                                 font=(_FONT, 8))
        self._lbl_fps.pack(side="left", padx=(16, 12))

        self._lbl_yolo = tk.Label(self._statusbar, text="YOLO: --",
                                  bg=C["bg2"], fg=C["dim"],
                                  font=(_FONT, 8))
        self._lbl_yolo.pack(side="left", padx=(0, 12))

        tk.Label(self._statusbar, text="SALIDA AUTOMATICA v2",
                 bg=C["bg2"], fg=C["dim"],
                 font=(_FONT, 8)).pack(side="right", padx=16)

    # ── callbacks ────────────────────────────────────────────────────────────
    def _toggle_panel(self):
        if self._panel_vis:
            self._ctrl.pack_forget()
            self._btn_toggle.configure(text="▼ Panel")
        else:
            self._ctrl.pack(fill="x", before=self._vid_frame)
            self._btn_toggle.configure(text="▲ Panel")
        self._panel_vis = not self._panel_vis

    def _on_activate(self):
        self._lbl_sys.configure(text="ACTIVANDO...", fg=COLORS["yellow"])
        self.engine.start()

    def _on_deactivate(self):
        self.engine.stop()
        self._vid_lbl.configure(image="")
        self._tkimg = None
        self._refresh("DESACTIVADO", "LIBRE", False, False, 0, "N/A", 0.0, 0)

    def _on_reconfig(self):
        self.engine.stop()
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        self._on_activate()

    def _on_close(self):
        try:
            self.engine.stop()
        except Exception:
            pass
        self.destroy()

    # ── refresco estado ───────────────────────────────────────────────────────
    def _refresh(self, status, phase, night, barrier, veh_in, wh_txt, fps, yolo_ms):
        C = COLORS

        # -- Sistema --
        if "ACTIVO" in status:
            sc, st = C["green"], C["green"]
        elif any(x in status for x in ("ACTIVANDO", "Config", "Selecciona")):
            sc, st = C["yellow"], C["yellow"]
        else:
            sc, st = C["red"], C["text"]
        self._led_sys_cv.itemconfig(self._led_sys_o, fill=sc)
        self._lbl_sys.configure(text=status, fg=st)

        # -- Barrera --
        if phase == "KEEPALIVE":
            bc   = C["orange"]
            btxt = "ABIERTA"
            ptxt = "Keep-alive activo (DETENER/SUBIR)"
        elif phase == "SUBIENDO":
            bc   = C["cyan"]
            btxt = "SUBIENDO"
            ptxt = "Barrera subiendo..."
        else:
            bc   = C["dim"]
            btxt = "CERRADA"
            ptxt = ""
        self._led_bar_cv.itemconfig(self._led_bar_o, fill=bc)
        self._lbl_bar.configure(text=btxt, fg=bc)
        self._lbl_phase.configure(text=ptxt, fg=bc)

        # -- Modo --
        if night:
            self._lbl_mode.configure(text="NOCHE", fg=C["sub"])
        else:
            self._lbl_mode.configure(text="DIA", fg=C["yellow"])

        # -- Vehiculos --
        self._lbl_veh.configure(
            text=str(veh_in),
            fg=C["green"] if veh_in > 0 else C["sub"])

        # -- Webhook --
        if "OK" in wh_txt:
            wh_fg = C["green"]
        elif "FAIL" in wh_txt:
            wh_fg = C["red"]
        else:
            wh_fg = C["sub"]
        self._lbl_wh.configure(text=wh_txt, fg=wh_fg)

        # -- Barra inferior: FPS y YOLO --
        self._lbl_fps.configure(text=f"FPS: {fps:.1f}")
        self._lbl_yolo.configure(text=f"YOLO: {yolo_ms}ms")

    # ── tick UI ───────────────────────────────────────────────────────────────
    def _tick(self):
        status, wh_txt, night, veh_in, barrier, phase, fps, yolo_ms = self.engine.get_state()
        self._refresh(status, phase, night, barrier, veh_in, wh_txt, fps, yolo_ms)

        frame = self.engine.get_frame()
        if frame is not None and "ACTIVO" in status:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vw  = self._vid_lbl.winfo_width()
            vh  = self._vid_lbl.winfo_height()
            if vw > 50 and vh > 50:
                ih, iw = rgb.shape[:2]
                s      = min(vw / iw, vh / ih)
                rgb    = cv2.resize(rgb, (int(iw * s), int(ih * s)),
                                    interpolation=cv2.INTER_AREA)
            img = Image.fromarray(rgb)
            self._tkimg = ImageTk.PhotoImage(img)
            self._vid_lbl.configure(image=self._tkimg)
        elif "DESACTIVADO" in status:
            self._vid_lbl.configure(image="")
            self._tkimg = None

        self.after(40, self._tick)


if __name__ == "__main__":
    App().mainloop()