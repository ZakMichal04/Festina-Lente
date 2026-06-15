import cv2
from ultralytics import YOLO
import argparse
import sys
import os
import math
import torch
import time
import threading
from collections import deque
import numpy as np
#OpenVINO jest ładowany z biblioteki Ultralytics

#Definicja kątów ciała
ANGLE_DEFS = [
    ("Bark L",   (7,  5,  (11, 12))),
    ("Bark P",   (8,  6,  (11, 12))),
    ("Biodro",   ((5, 6), (11, 12), 13)),
    ("Kolano L", (11, 13, 15)),
    ("Kolano P", (12, 14, 16)),
    ("Lokiec L", (5,  7,  9)),
    ("Lokiec P", (6,  8,  10)),
]

#Progi kątów dla deski (kamera z boku)
#Każdy wpis: (min, max, komunikat błędu gdy poza zakresem)
PLANK_THRESHOLDS = {
    "Biodro":   (145, 205, "Biodra za nisko lub za wysoko! Wyprostuj biodra."),
    "Kolano L": (145, 185, "Lewe kolano zgięte! Wyprostuj lewą nogę."),
    "Kolano P": (145, 185, "Prawe kolano zgięte! Wyprostuj prawą nogę."),
    "Lokiec L": (75,  105, "Lewy łokieć: ustaw pod kątem ~90°."),
    "Lokiec P": (75,  105, "Prawy łokieć: ustaw pod kątem ~90°."),
}

PLANK_THRESHOLDS_45_DEGREE = {
    "Biodro":   (165, 195, "Biodra za nisko lub za wysoko! Wyprostuj biodra."),
    "Kolano L": (155, 185, "Lewe kolano zgięte! Wyprostuj lewą nogę."),
    "Kolano P": (155, 185, "Prawe kolano zgięte! Wyprostuj prawą nogę."),
    "Lokiec L": (75,  105, "Lewy łokieć: ustaw pod kątem ~90°."),
    "Lokiec P": (75,  105, "Prawy łokieć: ustaw pod kątem ~90°."),
}


# Detekcja deski: program uznaje że użytkownik robi deskę gdy kąt bioder i kolan mieści się w zakresie
PLANK_DETECTION_RANGES = {
    "Biodro":   (140, 210),
    "Kolano L": (130, 200),
    "Kolano P": (130, 200),
}

# Liczba ostatnich klatek do uśredniania błędów 
SMOOTHING_FRAMES = 90


#Funkcje do liczenia kątów i innych wartości
def compute_angle(A, B, C):
    """Oblicza kąt ABC (w stopniach) na podstawie trzech punktów (x,y)"""
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot = BA[0] * BC[0] + BA[1] * BC[1]
    norm_BA = math.hypot(*BA)
    norm_BC = math.hypot(*BC)
    if norm_BA == 0 or norm_BC == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (norm_BA * norm_BC)))
    return math.degrees(math.acos(cos_angle))


def get_point(xy, conf, idx):
    """
    Pobiera punkt (x,y) i pewność.
    idx może być int lub tuple dwóch indeksów – wtedy uśrednia oba punkty
    """
    if isinstance(idx, tuple):
        p1, p2 = xy[idx[0]], xy[idx[1]]
        x = (float(p1[0]) + float(p2[0])) / 2.0
        y = (float(p1[1]) + float(p2[1])) / 2.0
        min_c = min(float(conf[idx[0]]), float(conf[idx[1]]))
        return (x, y), min_c
    return (float(xy[idx][0]), float(xy[idx][1])), float(conf[idx])


def is_plank_position(angles: dict) -> bool:
    """
    Sprawdza czy wykryte kąty wskazują na pozycję deski.
    Wymaga że przynajmniej dwa z trzech kluczowych kątów są w zakresie.
    Przynajmniej dwa z 3 kątów musi być 
    """
    hits = 0
    for name, (lo, hi) in PLANK_DETECTION_RANGES.items():
        angle = angles.get(name)
        if angle is not None and lo <= angle <= hi:
            hits += 1
    return hits >= 2


def evaluate_plank(angles: dict) -> list[str]:
    """
    Zwraca listę komunikatów o błędach w pozycji deski.
    Pusta lista = poprawna deska.
    """
    errors = []
    for name, (lo, hi, msg) in PLANK_THRESHOLDS.items():
        angle = angles.get(name)
        if angle is not None and not (lo <= angle <= hi):
            errors.append(msg)
    return errors


def draw_hud(frame, angles: dict, is_plank: bool, errors: list[str], fps: float):
    """Rysowanie huda z pistem testowym"""
    h, w = frame.shape[:2]

    # FPS w prawym górnym rogu
    fps_text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, fps_text, (w - tw - 8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if not is_plank:
        cv2.putText(frame, "Brak deski", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        return

    # Status główny
    if errors:
        status_text = "DESKA: Popraw pozycje!"
        status_color = (0, 100, 255)
    else:
        status_text = "DESKA: Swietnie!"
        status_color = (0, 220, 0)

    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

    # Błędy pozycji
    for i, err in enumerate(errors):
        y = 60 + i * 26
        cv2.putText(frame, f"  ! {err}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)


def draw_angles_on_skeleton(frame, xy, conf, angles: dict, angle_conf: float):
    """Rysowanie kątów na szielecie"""
    for angle_name, (iA, iB, iC) in ANGLE_DEFS:
        _, confB = get_point(xy, conf, iB)
        ptB_raw = xy[iB] if not isinstance(iB, tuple) else (
            ((xy[iB[0]][0] + xy[iB[1]][0]) / 2),
            ((xy[iB[0]][1] + xy[iB[1]][1]) / 2),
        )
        angle = angles.get(angle_name)
        if angle is not None and confB >= angle_conf:
            text_pos = (int(ptB_raw[0]) if not isinstance(iB, tuple) else int(ptB_raw[0]),
                        int(ptB_raw[1]) if not isinstance(iB, tuple) else int(ptB_raw[1]))
            text_pos = (text_pos[0], text_pos[1] - 10)
            color = (0, 255, 0)
            # Pokoloruj na czerwono jeśli kąt poza progiem deski
            if angle_name in PLANK_THRESHOLDS:
                lo, hi, _ = PLANK_THRESHOLDS[angle_name]
                if not (lo <= angle <= hi):
                    color = (0, 80, 255)
            cv2.putText(frame, f"{angle_name}: {angle:.1f}",
                        text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)



class CameraReader(threading.Thread):
    """
    Czyta klatki z kamery w osobnym wątku,dla plików wideo i program czyta synchronicznie
    żeby nie pomijać klatek i zachować oryginalną prędkość odtwarzania.
    """
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ok = True
        self.stopped = False

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self._ok = False
                break
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return self._ok, self._frame

    def stop(self):
        self.stopped = True


def get_source_from_menu():
    print("\nWybierz źródło")
    print("1. Kamera")
    print("2. Film (plik wideo)")
    while True:
        choice = input("Twój wybór (1/2): ").strip()
        if choice == '1':
            return 0
        elif choice == '2':
            path = input("Podaj ścieżkę do pliku wideo: ").strip()
            if os.path.isfile(path):
                return path
            else:
                print("Plik nie istnieje. Spróbuj ponownie.")
        else:
            print("Nieprawidłowy wybór. Wpisz 1 lub 2.")


def load_model(model_path, device, imgsz):
    """
    Wczytuje model YOLO na CPU eksportuje go jednorazowo do OpenVino
    (szybsza inferecja na procesorach Intela co pozwoli mi na wrzucić bardziej wymagający model oraz lepszą rozdzielczość działa tylko na cpu bo na gpu to tu nie ma co działać) 
    i ładuje wersję OpenVINO.
    """
    if os.path.isdir(model_path):
        return YOLO(model_path, task="pose")

    if device != 'cpu':
        model = YOLO(model_path)
        model.to(device)
        return model

    # Na CPU jeśli nie ma jeszcze wersji OpenVINO to zrobić jej export
    ov_dir = model_path.replace(".pt", "_openvino_model")
    if not os.path.isdir(ov_dir):
        YOLO(model_path).export(format="openvino", imgsz=imgsz)
    return YOLO(ov_dir, task="pose")


def calculate_roi(box, frame_w, frame_h, margin):
    """
    Wyznacza prostokąt wokół osoby (z lekkim marginesem) we współrzędnych pełnej klatki
    Zwraca (x1, y1, x2, y2) albo None gdy obszar jest zbyt mały
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    mx = bw * margin
    my = bh * margin
    nx1 = max(0, int(x1 - mx))
    ny1 = max(0, int(y1 - my))
    nx2 = min(frame_w, int(x2 + mx))
    ny2 = min(frame_h, int(y2 + my))
    if nx2 - nx1 < 20 or ny2 - ny1 < 20:
        return None
    return (nx1, ny1, nx2, ny2)


def main():
    parser = argparse.ArgumentParser(description="analiza deski")
    parser.add_argument("--model",type=str,default="yolov8n-pose.pt",
                        help="Ścieżka do modelu YOLO Pose")
    parser.add_argument("--conf",type=float,default=0.6,
                        help="Próg pewności detekcji (0-1)")
    parser.add_argument("--angle-conf",type=float,default=0.5,
                        help="Minimalna pewność punktów do obliczenia kąta")
    # Mniejszy obraz = szybsza inferecja na CPU
    parser.add_argument("--imgsz",type=int,default=640,
                        help="Rozmiar obrazu wejściowego dla YOLO")
    #analiza co 2 klatki  
    parser.add_argument("--skip",type=int,default=2,
                        help="Analiza co N-tą klatkę")
    parser.add_argument("--roi-margin",type=float,default=0.25,
                        help="Margines ROI wokół osoby (np. 0.25 to +25 procent)")
    args = parser.parse_args() #Przesłanie argumentów z góry do funkcji

    #Wykrycie czy ma pracować na karcie czy na procesorze
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(f"Program działa na: {device}")
    if device == 'cpu':
        # Ustawianie wątków dla aplikacji
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)
        cv2.setNumThreads(max(1, num_cores // 2))
        print(f"CPU: {num_cores} rdzeni wykryto, ustawiono wątki.")

    try:
        #Sprawdzanie czy wszystko działa
        model = load_model(args.model, device, args.imgsz)
        dummy = np.zeros((args.imgsz, args.imgsz, 3),dtype=np.uint8)
        model(dummy, imgsz=args.imgsz, verbose=False)
        print("Model działa poprawnie")
    except Exception as e:
        print(f"Błąd w wczytywaniu modelu w funkcji main: {e}")
        sys.exit(1)

    source = get_source_from_menu()
    is_camera = (source == 0)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Nie można otworzyć źródła wideo.")
        sys.exit(1)

    if is_camera:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        video_fps = 30.0
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"Plik wideo: {video_fps:.2f} FPS")

    #długość trwania pojedyńczej klatki
    frame_duration = 1.0 / video_fps

    print("\nRozpoczynam przetwarzanie. Naciśnij 'q' lub Esc, aby zakończyć.\n")

    camera_reader = None
    if is_camera:
        camera_reader = CameraReader(cap)
        camera_reader.start()

    frame_count = 0
    last_annotated = None
    roi_box = None  # bieżący obszar ROI
    fps_counter = deque(maxlen=30)
    error_buffer: deque[list] = deque(maxlen=SMOOTHING_FRAMES)

    while True:
        t0 = time.perf_counter()

        if is_camera:
            ok, frame = camera_reader.read()
            if not ok or frame is None:
                time.sleep(0.005)
                if not camera_reader._ok:
                    print("Koniec strumienia.")
                    break
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                print("Koniec pliku wideo.")
                break

        if frame_count % args.skip == 0:
            #jeśli mamy obszar z poprzedniej klatki z roi to analizujemy tylko jego wycinek
            if roi_box is not None:
                rx1, ry1, rx2, ry2 = roi_box
                infer_img = frame[ry1:ry2, rx1:rx2]
                off_x, off_y = rx1, ry1
            else:
                infer_img = frame
                off_x, off_y = 0, 0

            results = model(
                infer_img,
                conf=args.conf,
                imgsz=args.imgsz,
                half=False,        
                device=device,
                verbose=False,
            )

            if results and results[0].keypoints is not None:
                #Wklejanie szkieletu z wycinka ROI z powrotem w pełną klatkę
                annotated_frame = frame.copy()
                annotated_region = results[0].plot(boxes=False)
                annotated_frame[off_y:off_y + annotated_region.shape[0],
                                off_x:off_x + annotated_region.shape[1]] = annotated_region

                #Aktualizacja ROI z nowego boxa
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    bx = results[0].boxes.xyxy[0].cpu().numpy()
                    box_full = (bx[0] + off_x, bx[1] + off_y, bx[2] + off_x, bx[3] + off_y)
                    roi_box = calculate_roi(box_full, frame.shape[1], frame.shape[0], args.roi_margin)
                else:
                    roi_box = None

                kpts_xy   = results[0].keypoints.xy
                kpts_conf = results[0].keypoints.conf

                if kpts_xy is not None and kpts_conf is not None and kpts_xy.shape[0] > 0:
                    xy   = kpts_xy[0].cpu().numpy().copy()
                    xy[:, 0] += off_x
                    xy[:, 1] += off_y
                    conf = kpts_conf[0].cpu().numpy()

                    # Oblicz wszystkie kąty
                    angles: dict[str, float] = {}
                    for angle_name, (iA, iB, iC) in ANGLE_DEFS:
                        ptA, confA = get_point(xy, conf, iA)
                        ptB, confB = get_point(xy, conf, iB)
                        ptC, confC = get_point(xy, conf, iC)
                        if confA >= args.angle_conf and confB >= args.angle_conf and confC >= args.angle_conf:
                            angles[angle_name] = compute_angle(ptA, ptB, ptC)

                    plank = is_plank_position(angles)
                    errors = evaluate_plank(angles) if plank else []
                    error_buffer.append(errors)

                    #Wygładzanie błędów które wystąpiły w przynajmniej 50% klatek
                    smoothed_errors = []
                    if error_buffer:
                        all_errors = set(e for frame_errs in error_buffer for e in frame_errs)
                        threshold = len(error_buffer) * 0.5
                        smoothed_errors = [
                            e for e in all_errors
                            if sum(1 for fe in error_buffer if e in fe) >= threshold
                        ]

                    # FPS na HUD
                    fps_counter.append(time.perf_counter() - t0)
                    fps = len(fps_counter) / sum(fps_counter) if fps_counter else 0

                    # Rysuj kąty i HUD
                    draw_angles_on_skeleton(annotated_frame, xy, conf, angles, args.angle_conf)
                    draw_hud(annotated_frame, angles, plank, smoothed_errors, fps)

                last_annotated = annotated_frame
            else:
                # Jak nic nie wykryło to resetuje ROI
                roi_box = None
                last_annotated = frame
        else:
            if last_annotated is None:
                last_annotated = frame

        cv2.imshow("Deska", last_annotated)
        frame_count += 1

        #Synchronizacja czasu
        #Dla pliku wideo: czekamy tyle ile powinna trwać jedna klatka,
        #uwzględniając czas spędzony na inferecji. Dzięki temu film
        #odtwarza się z oryginalną prędkością nawet gdy CPU jest wolny.
        #Dla kamery: tylko 1ms żeby nie blokować odczytu.
        elapsed = time.perf_counter() - t0
        if is_camera:
            wait_ms = 1
        else:
            wait_ms = max(1, int((frame_duration - elapsed) * 1000))

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q') or key == 27:
            break

    if camera_reader is not None:
        camera_reader.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()