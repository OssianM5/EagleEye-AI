cat << 'EOF' > /home/ubuntu/EagleEye-AI/main_dpu.py
import time
import cv2
import numpy as np
from pynq_dpu import DpuOverlay

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
ALPHA = 0.15
BLUR_SCALE = 0.2
CONF_THRESH = 0.65
NMS_THRESH = 0.3
MODEL_PATH = "densebox_320_320.xmodel"

overlay = DpuOverlay("dpu.bit")
overlay.load_model(MODEL_PATH)

dpu = overlay.runner
input_tensors = dpu.get_input_tensors()
output_tensors = dpu.get_output_tensors()

in_shape = tuple(input_tensors[0].dims)
in_h, in_w = in_shape[1], in_shape[2]
in_fixpos = input_tensors[0].get_attr("fix_point")
in_scale = 2 ** in_fixpos

out_shapes = [tuple(t.dims) for t in output_tensors]
out_fixpos = [t.get_attr("fix_point") for t in output_tensors]
conf_idx = 0 if out_shapes[0][-1] == 2 else 1
bbox_idx = 1 - conf_idx

conf_scale = 2 ** (-out_fixpos[conf_idx])
bbox_scale = 2 ** (-out_fixpos[bbox_idx])

input_data = [np.empty(in_shape, dtype=np.int8, order="C")]
output_data = [
    np.empty(out_shapes[0], dtype=np.int8, order="C"),
    np.empty(out_shapes[1], dtype=np.int8, order="C"),
]

def detect_face_dpu(frame):
    orig_h, orig_w = frame.shape[:2]
    img = cv2.resize(frame, (in_w, in_h)).astype(np.float32) - 128.0
    if in_fixpos > 2:
        img = img / 128.0
    input_data[0][0] = np.clip(np.round(img * in_scale), -128, 127).astype(np.int8)

    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)

    raw_conf = output_data[conf_idx][0].astype(np.float32) * conf_scale
    raw_bbox = output_data[bbox_idx][0].astype(np.float32) * bbox_scale

    exp_s = np.exp(raw_conf - np.max(raw_conf, axis=-1, keepdims=True))
    probs = (exp_s / np.sum(exp_s, axis=-1, keepdims=True))[..., 1]

    ys, xs = np.where(probs > CONF_THRESH)
    if len(xs) == 0:
        return None

    scale_x = orig_w / float(in_w)
    scale_y = orig_h / float(in_h)
    boxes, confidences = [], []

    for gy, gx in zip(ys, xs):
        bb = raw_bbox[gy, gx]
        x1 = max(0, int((gx * 4 + bb[0]) * scale_x))
        y1 = max(0, int((gy * 4 + bb[1]) * scale_y))
        x2 = min(orig_w, int((gx * 4 + bb[2]) * scale_x))
        y2 = min(orig_h, int((gy * 4 + bb[3]) * scale_y))
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0:
            boxes.append([x1, y1, w, h])
            confidences.append(float(probs[gy, gx]))

    if not boxes:
        return None
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
    if len(indices) == 0:
        return None

    valid = [boxes[i] for i in np.array(indices).flatten()]
    return tuple(max(valid, key=lambda b: b[2] * b[3]))

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

focus_w, focus_h = FRAME_W // 2, FRAME_H // 2
focus_cx, focus_cy = float(FRAME_W // 2), float(FRAME_H // 2)
target_cx, target_cy = focus_cx, focus_cy
prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        face = detect_face_dpu(frame)

        fx1 = int(focus_cx - focus_w / 2)
        fy1 = int(focus_cy - focus_h / 2)
        fx2, fy2 = fx1 + focus_w, fy1 + focus_h

        if face is not None:
            bx, by, bw, bh = face
            if (bx < fx1) or (bx + bw > fx2) or (by < fy1) or (by + bh > fy2):
                target_cx = bx + bw / 2.0
                target_cy = by + bh / 2.0

        focus_cx = np.clip(ALPHA * target_cx + (1.0 - ALPHA) * focus_cx, focus_w / 2, w - focus_w / 2)
        focus_cy = np.clip(ALPHA * target_cy + (1.0 - ALPHA) * focus_cy, focus_h / 2, h - focus_h / 2)

        fx1 = int(focus_cx - focus_w / 2)
        fy1 = int(focus_cy - focus_h / 2)
        fx2, fy2 = fx1 + focus_w, fy1 + focus_h

        small = cv2.resize(frame, (0, 0), fx=BLUR_SCALE, fy=BLUR_SCALE)
        blurred = cv2.GaussianBlur(small, (15, 15), 0)
        output = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
        output[fy1:fy2, fx1:fx2] = frame[fy1:fy2, fx1:fx2]

        cv2.rectangle(output, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
        if face is not None:
            bx, by, bw, bh = face
            cv2.rectangle(output, (bx, by), (bx + bw, by + bh), (255, 200, 0), 1)

        now = time.time()
        fps = 1.0 / max(1e-6, now - prev_time)
        prev_time = now
        cv2.putText(output, f"KR260 DPU | {fps:.1f} FPS", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("EagleEye AI - KR260 DPU", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    del dpu
    del overlay
EOF