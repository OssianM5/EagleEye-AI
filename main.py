import cv2
import numpy as np

# --- CONFIGURARE ---
#CAM_INDEX = 1  # Schimbă în 0 dacă se deschide camera greșită
#cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
CAM_INDEX = 2
FRAME_W, FRAME_H = 640, 480
ALPHA = 0.15          # Viteza de urmărire (0.05 = foarte lent, 0.3 = rapid)
BLUR_SCALE = 0.2      # Factor de micșorare pentru blur rapid (optimizare CPU)
MODEL_PATH = "face_detection_yunet_2023mar.onnx"  # Opțional

# Inițializare cameră USB (V4L2 pe Linux/KR260)
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# Inițializare detector AI (YuNet dacă există fișierul, altfel Haar Cascade)
use_yunet = False
try:
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (FRAME_W, FRAME_H), 0.7, 0.3, 5000)
    use_yunet = True
    print("Model activ: YuNet ONNX")
except Exception:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("Model activ: Haar Cascade (fallback)")

# Dimensiuni fereastră de focus (25% din arie = 50% W x 50% H)
focus_w = FRAME_W // 2
focus_h = FRAME_H // 2

# Poziția inițială a centrului de focus (centrul imaginii)
focus_cx = float(FRAME_W // 2)
focus_cy = float(FRAME_H // 2)
target_cx, target_cy = focus_cx, focus_cy

def detect_face(frame):
    """Returnează (x, y, w, h) pentru cea mai mare față detectată sau None."""
    if use_yunet:
        _, faces = detector.detect(frame)
        if faces is not None and len(faces) > 0:
            # Alegem fața cu aria cea mai mare
            best = max(faces, key=lambda f: f[2] * f[3])
            return int(best[0]), int(best[1]), int(best[2]), int(best[3])
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces) > 0:
            best = max(faces, key=lambda f: f[2] * f[3])
            return tuple(best)
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # 1. Detecție față
    face = detect_face(frame)

    # Calculăm limitele curente ale ferestrei de focus
    fx1 = int(focus_cx - focus_w / 2)
    fy1 = int(focus_cy - focus_h / 2)
    fx2 = fx1 + focus_w
    fy2 = fy1 + focus_h

    # 2. Verificăm dacă fața a ieșit din zona de focus
    if face is not None:
        bx, by, bw, bh = face
        face_cx = bx + bw / 2.0
        face_cy = by + bh / 2.0

        # Condiție: dacă oricare margine a feței iese în zona cu ceață
        out_of_focus = (bx < fx1) or (bx + bw > fx2) or (by < fy1) or (by + bh > fy2)

        if out_of_focus:
            target_cx = face_cx
            target_cy = face_cy

    # 3. Deplasare lină (EMA) + limitare în interiorul cadrului (clamping)
    focus_cx = ALPHA * target_cx + (1.0 - ALPHA) * focus_cx
    focus_cy = ALPHA * target_cy + (1.0 - ALPHA) * focus_cy

    focus_cx = np.clip(focus_cx, focus_w / 2, w - focus_w / 2)
    focus_cy = np.clip(focus_cy, focus_h / 2, h - focus_h / 2)

    fx1 = int(focus_cx - focus_w / 2)
    fy1 = int(focus_cy - focus_h / 2)
    fx2 = fx1 + focus_w
    fy2 = fy1 + focus_h

    # 4. Creare efect de ceață (rapid: downscale -> blur -> upscale)
    small = cv2.resize(frame, (0, 0), fx=BLUR_SCALE, fy=BLUR_SCALE)
    blurred_small = cv2.GaussianBlur(small, (15, 15), 0)
    output = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 5. Suprapunere zonă clară de focus (25% arie)
    output[fy1:fy2, fx1:fx2] = frame[fy1:fy2, fx1:fx2]

    # Chenar vizual pentru zona de focus (opțional)
    cv2.rectangle(output, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)

    if face is not None:
        bx, by, bw, bh = face
        cv2.rectangle(output, (bx, by), (bx + bw, by + bh), (255, 200, 0), 1)

    cv2.imshow("EagleEye AI - KR260", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()