import threading
import tkinter as tk

import cv2
import customtkinter as ctk
from PIL import Image

import pose_session
from mdi import MDIWorkspace

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
VIDEO_WIDTH = 768
VIDEO_HEIGHT = 528


class PlankGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plank Master Pro")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 720)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        self._timer_seconds = 0
        self._timer_job = None
        self._session_active = False
        self._timer_label = None
        self._status_label = None
        self._btn_action = None
        self._camera_label = None
        self._camera_image = None
        self._stop_event = None
        self._tts_enabled = True
        self._voice_commands_enabled = True

        self.header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 8))

        self.lbl_logo = ctk.CTkLabel(self.header_frame, text="⚡", font=("Roboto Medium", 36), text_color="#3B8ED0")
        self.lbl_logo.pack(side="left", padx=(0, 10))

        self.lbl_title = ctk.CTkLabel(self.header_frame, text="PLANK MASTER", font=("Roboto Medium", 26))
        self.lbl_title.pack(side="left")

        self.toolbar = ctk.CTkFrame(self.root, fg_color="transparent")
        self.toolbar.grid(row=1, column=0, sticky="ew", padx=24, pady=(0, 10))

        ctk.CTkButton(
            self.toolbar, text="Start", width=110, command=self.open_menu_window,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            self.toolbar, text="Sesja", width=110, command=self.open_session_window,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            self.toolbar, text="Statystyki", width=110, command=self.open_stats_window,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            self.toolbar, text="Opcje", width=110, command=self.open_options_window,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            self.toolbar, text="Wyjście", width=110, fg_color="#333", hover_color="#444",
            command=self.root.quit,
        ).pack(side="right")

        self.workspace = MDIWorkspace(self.root)
        self.workspace.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))

        self.open_menu_window()

    def _format_time(self, seconds):
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _stop_timer(self):
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None

    def _tick_timer(self):
        self._timer_seconds += 1
        if self._timer_label is not None and self._timer_label.winfo_exists():
            self._timer_label.configure(text=self._format_time(self._timer_seconds))
        self._timer_job = self.root.after(1000, self._tick_timer)

    def _resume_timer(self):
        if self._timer_job is None:
            self._tick_timer()

    def _set_status(self, text):
        if self._status_label is not None and self._status_label.winfo_exists():
            self._status_label.configure(text=text)

    def _schedule(self, callback):
        self.root.after(0, callback)

    def _frame_to_ctk_image(self, frame, max_w, max_h):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(max_w / w, max_h / h)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(rgb)
        size = (img.width, img.height)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
        return ctk_img

    def _update_camera_frame(self, frame):
        if self._camera_label is None or not self._camera_label.winfo_exists():
            return
        self._camera_image = self._frame_to_ctk_image(frame, VIDEO_WIDTH, VIDEO_HEIGHT)
        self._camera_label.configure(image=self._camera_image, text="")

    def _on_frame(self, frame):
        frame_copy = frame.copy()
        self._schedule(lambda: self._update_camera_frame(frame_copy))

    def _on_plank_detected(self):
        if not self._session_active:
            return
        self._resume_timer()
        self._set_status("Deska wykryta — liczę czas")

    def _on_plank_lost(self):
        if not self._session_active:
            return
        self._stop_timer()
        self._set_status("Utrata pozycji — wstrzymano")

    def _stop_camera_session(self):
        if self._stop_event is not None:
            self._stop_event.set()

    def _on_session_end(self):
        def finish():
            self._stop_timer()
            self._session_active = False
            self._stop_event = None
            self._camera_label = None
            self._camera_image = None
            self.workspace.set_focus_window(None)
            self.workspace.close_window("camera")
            self._set_status("Sesja zakończona")
            if self._btn_action is not None and self._btn_action.winfo_exists():
                self._btn_action.configure(state="normal")
        self._schedule(finish)

    def reset_session(self):
        self._stop_camera_session()
        self._stop_timer()
        self._timer_seconds = 0
        self._session_active = False
        self.workspace.set_focus_window(None)
        if self._timer_label is not None and self._timer_label.winfo_exists():
            self._timer_label.configure(text="00:00")
        self._set_status("Gotowy do startu")
        if self._btn_action is not None and self._btn_action.winfo_exists():
            self._btn_action.configure(state="normal")

    def open_camera_window(self):
        def build(parent):
            self._camera_label = ctk.CTkLabel(
                parent, text="Ładowanie kamery…",
                width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
                fg_color="#111", corner_radius=8,
            )
            self._camera_label.pack(expand=True)

        self.workspace.open_window(
            "camera", "Kamera",
            build,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            on_close=self._stop_camera_session,
        )
        self.workspace.set_focus_window("camera")

    def start_camera_session(self):
        if self._session_active:
            return
        self._session_active = True
        self._stop_timer()
        self._timer_seconds = 0
        if self._timer_label is not None and self._timer_label.winfo_exists():
            self._timer_label.configure(text="00:00")
        self._set_status("Czekam na wykrycie deski…")
        if self._btn_action is not None:
            self._btn_action.configure(state="disabled")

        self.open_session_window()
        self.open_camera_window()
        self.root.after(50, self.workspace.arrange_for_session)

        self._stop_event = threading.Event()
        threading.Thread(
            target=pose_session.run_pose_detection,
            kwargs={
                "source": 0,
                "on_plank_detected": lambda: self._schedule(self._on_plank_detected),
                "on_plank_lost": lambda: self._schedule(self._on_plank_lost),
                "on_session_end": lambda: self._schedule(self._on_session_end),
                "on_frame": self._on_frame,
                "stop_event": self._stop_event,
                "bez_tts": not self._tts_enabled,
                "bez_komend": not self._voice_commands_enabled,
            },
            daemon=True,
        ).start()

    def open_menu_window(self):
        def build(parent):
            menu_frame = ctk.CTkFrame(parent, fg_color="transparent")
            menu_frame.pack(fill="both", expand=True, padx=8, pady=8)

            ctk.CTkLabel(
                menu_frame, text="Witaj w Plank Master",
                font=("Roboto Medium", 20),
            ).pack(pady=(10, 24))

            ctk.CTkButton(
                menu_frame, text="▶ ROZPOCZNIJ SESJĘ",
                font=("Roboto Medium", 16), height=64, corner_radius=30,
                command=self.open_session_window,
            ).pack(pady=10, fill="x")

            secondary_frame = ctk.CTkFrame(menu_frame, fg_color="transparent")
            secondary_frame.pack(pady=10, fill="x")

            ctk.CTkButton(
                secondary_frame, text="Statystyki",
                font=("Roboto Medium", 14), fg_color="transparent",
                border_width=2, border_color="#3B8ED0", text_color="#3B8ED0", height=48,
                command=self.open_stats_window,
            ).pack(side="left", padx=(0, 10), expand=True, fill="x")

            ctk.CTkButton(
                secondary_frame, text="Opcje",
                font=("Roboto Medium", 14), fg_color="transparent",
                border_width=2, border_color="#555", text_color="#CCC", height=48,
                command=self.open_options_window,
            ).pack(side="left", padx=(10, 0), expand=True, fill="x")

        self.workspace.open_window("menu", "Start", build, width=520, height=400)

    def open_session_window(self):
        def build(parent):
            session_card = ctk.CTkFrame(parent, corner_radius=20)
            session_card.pack(fill="both", expand=True)

            ctk.CTkLabel(
                session_card, text="Trwająca sesja",
                font=("Roboto Thin", 16), text_color="#AAA",
            ).pack(pady=(20, 0))

            self._timer_label = ctk.CTkLabel(
                session_card, text=self._format_time(self._timer_seconds),
                font=("Roboto Medium", 72), text_color="#3B8ED0",
            )
            self._timer_label.pack(pady=20)

            self._status_label = ctk.CTkLabel(
                session_card, text="Gotowy do startu",
                font=("Roboto Light", 13), text_color="#888",
            )
            self._status_label.pack(pady=(0, 10))

            control_frame = ctk.CTkFrame(session_card, fg_color="transparent")
            control_frame.pack(pady=10)

            self._btn_action = ctk.CTkButton(
                control_frame, text="START", fg_color="#2CC985", hover_color="#26AF74",
                width=120, command=self.start_camera_session,
            )
            self._btn_action.pack(side="left", padx=10)
            if self._session_active:
                self._btn_action.configure(state="disabled")

            ctk.CTkButton(
                control_frame, text="Reset", fg_color="#E74C3C", hover_color="#C0392B",
                width=100, command=self.reset_session,
            ).pack(side="left", padx=10)

        self.workspace.open_window(
            "session", "Sesja", build, width=500, height=460,
            on_close=self.reset_session,
        )

    def open_stats_window(self):
        def build(parent):
            stats_card = ctk.CTkFrame(parent, corner_radius=20)
            stats_card.pack(fill="both", expand=True)

            ctk.CTkLabel(
                stats_card, text="Podsumowanie aktywności",
                font=("Roboto Medium", 18),
            ).pack(pady=(20, 16))

            stats_data = [
                ("Najdłuższa deska", "03:15 min"),
                ("Sesji w tym tygodniu", "5"),
                ("Łączny czas", "1h 25m"),
                ("Twój poziom", "Średni"),
            ]

            for icon_text, value in stats_data:
                row = ctk.CTkFrame(stats_card, fg_color="transparent")
                row.pack(fill="x", padx=24, pady=8)
                ctk.CTkLabel(row, text=icon_text, font=("Roboto Light", 14), text_color="#CCC").pack(side="left")
                ctk.CTkLabel(row, text=value, font=("Roboto Medium", 14), text_color="white").pack(side="right")

        self.workspace.open_window("stats", "Statystyki", build, width=500, height=400)

    def open_options_window(self):
        def build(parent):
            options_card = ctk.CTkFrame(parent, corner_radius=20)
            options_card.pack(fill="both", expand=True)

            ctk.CTkLabel(
                options_card, text="Ustawienia",
                font=("Roboto Medium", 18),
            ).pack(pady=(20, 16))

            self._sw_tts = ctk.CTkSwitch(
                options_card, text="Komunikaty głosowe (TTS)",
                command=self._on_tts_toggle,
            )
            self._sw_tts.pack(pady=12, anchor="w", padx=32)
            if self._tts_enabled:
                self._sw_tts.select()

            self._sw_voice = ctk.CTkSwitch(
                options_card, text="Sterowanie głosem (stop / wyjdź)",
                command=self._on_voice_toggle,
            )
            self._sw_voice.pack(pady=12, anchor="w", padx=32)
            if self._voice_commands_enabled:
                self._sw_voice.select()

            sw_dark = ctk.CTkSwitch(options_card, text="Tryb ciemny", command=self.toggle_dark_mode)
            sw_dark.pack(pady=12, anchor="w", padx=32)
            sw_dark.select()

        self.workspace.open_window("options", "Opcje", build, width=500, height=360)

    def _on_tts_toggle(self):
        if hasattr(self, "_sw_tts") and self._sw_tts.winfo_exists():
            self._tts_enabled = bool(self._sw_tts.get())

    def _on_voice_toggle(self):
        if hasattr(self, "_sw_voice") and self._sw_voice.winfo_exists():
            self._voice_commands_enabled = bool(self._sw_voice.get())

    def toggle_dark_mode(self):
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("Light")
        else:
            ctk.set_appearance_mode("Dark")
