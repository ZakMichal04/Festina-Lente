import cv2
from ultralytics import YOLO
import argparse
import sys
import os


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


def main():
    parser = argparse.ArgumentParser(description="Wyświetlanie szkieletu z menu wyboru źródła")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt",
                        help="Ścieżka do modelu YOLO Pose (domyślnie yolov8n-pose)")
    parser.add_argument("--conf", type=float, default=0.2,
                        help="Próg pewności detekcji (0-1)")
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
        else:
            annotated_frame = frame

        cv2.imshow("YOLO Pose - Szkielet człowieka", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()