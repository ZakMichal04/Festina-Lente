"""Most GUI <-> szkielet.py bez modyfikacji szkieletu."""

import os
import time
import threading
from collections import deque

import cv2
import numpy as np
import torch

import szkielet

MODEL_PATH = "yolov8s-pose.pt"
CONF = 0.6
ANGLE_CONF = 0.5
IMGSZ = 640
SKIP = 2
ROI_MARGIN = 0.25
TTS_ODSTEP = 3.0


def run_pose_detection(
    source=0,
    on_plank_detected=None,
    on_plank_lost=None,
    on_session_end=None,
    on_frame=None,
    stop_event=None,
    bez_tts=False,
    bez_komend=False,
    tts_odstep=TTS_ODSTEP,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Program działa na: {device}")
    if device == "cpu":
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)
        cv2.setNumThreads(max(1, num_cores // 2))
        print(f"CPU: {num_cores} rdzeni wykryto, ustawiono wątki.")

    try:
        model = szkielet.load_model(MODEL_PATH, device, IMGSZ)
        dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
        model(dummy, imgsz=IMGSZ, verbose=False)
        print("Model działa poprawnie")
    except Exception as e:
        print(f"Błąd w wczytywaniu modelu: {e}")
        if on_session_end:
            on_session_end()
        return

    is_camera = source == 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Nie można otworzyć źródła wideo.")
        if on_session_end:
            on_session_end()
        return

    if is_camera:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        video_fps = 30.0
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_duration = 1.0 / video_fps
    embedded = on_frame is not None
    session_stop = stop_event if stop_event is not None else threading.Event()

    camera_reader = None
    if is_camera:
        camera_reader = szkielet.CameraReader(cap)
        camera_reader.start()

    mowca = None
    if not bez_tts:
        mowca = szkielet.MowcaBledow(min_odstep=tts_odstep)
        mowca.start()

    sluchacz = None
    if not bez_komend:
        sluchacz = szkielet.SluchaczKomend()
        sluchacz.stop_event = session_stop
        sluchacz.start()

    frame_count = 0
    last_annotated = None
    roi_box = None
    fps_counter = deque(maxlen=30)
    error_buffer: deque[list] = deque(maxlen=szkielet.SMOOTHING_FRAMES)
    plank_active = False

    def notify_plank_state(is_plank: bool):
        nonlocal plank_active
        if is_plank and not plank_active:
            plank_active = True
            if on_plank_detected:
                on_plank_detected()
        elif not is_plank and plank_active:
            plank_active = False
            if on_plank_lost:
                on_plank_lost()

    try:
        while True:
            if session_stop.is_set():
                break

            t0 = time.perf_counter()

            if is_camera:
                ok, frame = camera_reader.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    if not camera_reader._ok:
                        break
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            current_plank = False
            if frame_count % SKIP == 0:
                if roi_box is not None:
                    rx1, ry1, rx2, ry2 = roi_box
                    infer_img = frame[ry1:ry2, rx1:rx2]
                    off_x, off_y = rx1, ry1
                else:
                    infer_img = frame
                    off_x, off_y = 0, 0

                results = model(
                    infer_img,
                    conf=CONF,
                    imgsz=IMGSZ,
                    half=False,
                    device=device,
                    verbose=False,
                )

                if results and results[0].keypoints is not None:
                    annotated_frame = frame.copy()
                    annotated_region = results[0].plot(boxes=False)
                    annotated_frame[off_y:off_y + annotated_region.shape[0],
                                    off_x:off_x + annotated_region.shape[1]] = annotated_region

                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        bx = results[0].boxes.xyxy[0].cpu().numpy()
                        box_full = (bx[0] + off_x, bx[1] + off_y, bx[2] + off_x, bx[3] + off_y)
                        roi_box = szkielet.calculate_roi(
                            box_full, frame.shape[1], frame.shape[0], ROI_MARGIN,
                        )
                    else:
                        roi_box = None

                    kpts_xy = results[0].keypoints.xy
                    kpts_conf = results[0].keypoints.conf

                    if kpts_xy is not None and kpts_conf is not None and kpts_xy.shape[0] > 0:
                        xy = kpts_xy[0].cpu().numpy().copy()
                        xy[:, 0] += off_x
                        xy[:, 1] += off_y
                        conf = kpts_conf[0].cpu().numpy()

                        angles: dict[str, float] = {}
                        for angle_name, (iA, iB, iC) in szkielet.ANGLE_DEFS:
                            ptA, confA = szkielet.get_point(xy, conf, iA)
                            ptB, confB = szkielet.get_point(xy, conf, iB)
                            ptC, confC = szkielet.get_point(xy, conf, iC)
                            if confA >= ANGLE_CONF and confB >= ANGLE_CONF and confC >= ANGLE_CONF:
                                angles[angle_name] = szkielet.compute_angle(ptA, ptB, ptC)

                        plank = szkielet.is_plank_position(angles)
                        if plank:
                            tilt = szkielet.compute_body_tilt(xy, conf, ANGLE_CONF)
                            if not szkielet.is_body_horizontal(tilt):
                                plank = False

                        current_plank = plank
                        errors = szkielet.evaluate_plank(angles) if plank else []
                        error_buffer.append(errors)

                        smoothed_errors = []
                        if error_buffer:
                            threshold = len(error_buffer) * 0.5
                            counts: dict[str, int] = {}
                            for frame_errs in error_buffer:
                                for err in frame_errs:
                                    counts[err] = counts.get(err, 0) + 1
                            smoothed_errors = [
                                msg for msg in szkielet.ERROR_PRIORITY
                                if counts.get(msg, 0) >= threshold
                            ]

                        fps_counter.append(time.perf_counter() - t0)
                        fps = len(fps_counter) / sum(fps_counter) if fps_counter else 0

                        szkielet.draw_angles_on_skeleton(
                            annotated_frame, xy, conf, angles, ANGLE_CONF,
                        )
                        szkielet.draw_hud(annotated_frame, angles, plank, smoothed_errors, fps)

                        if mowca is not None:
                            mowca.zglos_bledy(smoothed_errors, t0)

                    notify_plank_state(current_plank)
                    last_annotated = annotated_frame
                else:
                    roi_box = None
                    notify_plank_state(False)
                    last_annotated = frame
            else:
                if last_annotated is None:
                    last_annotated = frame

            display_frame = last_annotated if last_annotated is not None else frame
            if embedded:
                on_frame(display_frame)
            else:
                cv2.imshow("Deska", display_frame)

            frame_count += 1
            elapsed = time.perf_counter() - t0

            if embedded:
                if is_camera:
                    time.sleep(0.001)
                else:
                    time.sleep(max(0.001, frame_duration - elapsed))
            else:
                wait_ms = 1 if is_camera else max(1, int((frame_duration - elapsed) * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        if camera_reader is not None:
            camera_reader.stop()
        if mowca is not None:
            mowca.stop()
        if sluchacz is not None:
            sluchacz.stop()
        cap.release()
        if not embedded:
            cv2.destroyAllWindows()
        if on_session_end:
            on_session_end()