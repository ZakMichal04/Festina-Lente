import cv2
from ultralytics import YOLO
import argparse
import sys
import os
import math

#  definicje kątów dla deski
ANGLE_DEFS = [
    ("Bark L",     (7, 5, (11, 12))),
    ("Bark P",     (8, 6, (11, 12))),
    ("Biodro",     ((5, 6), (11, 12), 13)),
    ("Kolano L",   (11, 13, 15)),
    ("Kolano P",   (12, 14, 16)),
    ("Lokiec L",   (5, 7, 9)),
    ("Lokiec P",   (6, 8, 10)),
]

def compute_angle(A, B, C):
    """
    przyjmuje: 3 punkty z ANGLE_DEFS
    Oblicza kąt pomiędzy 3 sąsiednimi punktami
    zwraca: Obliczony kąt w stopniach
    """
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot = BA[0]*BC[0] + BA[1]*BC[1]
    norm_BA = math.hypot(*BA)
    norm_BC = math.hypot(*BC)
    if norm_BA == 0 or norm_BC == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (norm_BA * norm_BC)))
    return math.degrees(math.acos(cos_angle))

def get_point(xy, conf, idx):
    """
    Pobiera współrzędne (x,y) i minimalną pewność.
    xy – tablica numpy [17,2], conf – tablica [17]
    idx: int (pojedynczy punkt) lub tuple (dwa indeksy do uśrednienia)
    Zwraca: ((x,y), min_conf)
    """
    if isinstance(idx, tuple):
        p1 = xy[idx[0]]
        p2 = xy[idx[1]]
        x = (p1[0] + p2[0]) / 2.0
        y = (p1[1] + p2[1]) / 2.0
        min_c = min(conf[idx[0]], conf[idx[1]])
        return (x, y), min_c
    else:
        return (xy[idx][0], xy[idx][1]), conf[idx]

def get_source_from_menu():
    """
    Wypisuje czy obraz ma działać na podstawie Kamery czy na podstawie filmu video
    Gdy plik nie istnieje wypisuje że nie ma takiego pliku
    """
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

def main():
    parser = argparse.ArgumentParser(description="Analiza deski – kąty w czasie rzeczywistym")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt",
                        help="Ścieżka do modelu YOLO Pose")
    parser.add_argument("--conf", type=float, default=0.2,
                        help="Próg pewności detekcji (0-1)")
    parser.add_argument("--angle-conf", type=float, default=0.5,
                        help="Minimalna pewność punktów do obliczenia kąta")
    args = parser.parse_args()

    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Błąd wczytywania modelu: {e}")
        sys.exit(1)

    source = get_source_from_menu()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Nie można otworzyć źródła wideo.")
        sys.exit(1)

    print("\nRozpoczynam przetwarzanie. Naciśnij 'q' lub Esc, aby zakończyć.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Koniec strumienia.")
            break

        results = model(frame, conf=args.conf, verbose=False)

        if results and results[0].keypoints is not None:
            annotated_frame = results[0].plot(boxes=False)

            kpts_xy = results[0].keypoints.xy
            kpts_conf = results[0].keypoints.conf

            if kpts_xy is not None and kpts_conf is not None:
                num_persons = kpts_xy.shape[0]
                for person_idx in range(num_persons):
                    xy = kpts_xy[person_idx]
                    conf = kpts_conf[person_idx]

                    # oblicznie zwykłych kątów
                    for angle_name, (iA, iB, iC) in ANGLE_DEFS:
                        ptA, confA = get_point(xy, conf, iA)
                        ptB, confB = get_point(xy, conf, iB)
                        ptC, confC = get_point(xy, conf, iC)

                        if confA >= args.angle_conf and confB >= args.angle_conf and confC >= args.angle_conf:
                            angle = compute_angle(ptA, ptB, ptC)
                            text_pos = (int(ptB[0]), int(ptB[1]) - 10)
                            cv2.putText(annotated_frame, f"{angle_name}: {angle:.1f}",
                                        text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    #nachylenie tułowia w lewym górnym
                    shoulder_mid, c_shoulder = get_point(xy, conf, (5, 6))
                    hip_mid, c_hip = get_point(xy, conf, (11, 12))
                    if c_shoulder >= args.angle_conf and c_hip >= args.angle_conf:
                        dx = shoulder_mid[0] - hip_mid[0]
                        dy = shoulder_mid[1] - hip_mid[1]
                        norm = math.hypot(dx, dy)
                        if norm > 0:
                            cos_torso = dx / norm
                            cos_torso = max(-1.0, min(1.0, cos_torso))
                            torso_angle = math.degrees(math.acos(cos_torso))
                            cv2.putText(annotated_frame, f"Nachylenie tulowia: {torso_angle:.1f}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            annotated_frame = frame

        cv2.imshow("Deska - analiza katow", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()