import customtkinter as ctk
import tkinter as tk

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


class PlankGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plank Master Pro")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self.header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))

        self.lbl_logo = ctk.CTkLabel(self.header_frame, text="⚡", font=("Roboto Medium", 36), text_color="#3B8ED0")
        self.lbl_logo.pack(side="left", padx=(0, 10))

        self.lbl_title = ctk.CTkLabel(self.header_frame, text="PLANK MASTER", font=("Roboto Medium", 24))
        self.lbl_title.pack(side="left")


        self.main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)

        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        self.show_main_menu()

    def clear_container(self):
        """Usuwa stare widoki."""
        for widget in self.main_container.winfo_children():
            widget.destroy()

    def show_main_menu(self):
        self.clear_container()


        menu_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        menu_frame.grid(row=0, column=0)


        btn_start = ctk.CTkButton(menu_frame,
                                  text="▶ ROZPOCZNIJ SESJĘ",
                                  font=("Roboto Medium", 16),
                                  height=60,
                                  corner_radius=30,
                                  command=self.show_session)
        btn_start.pack(pady=20, fill="x")


        secondary_frame = ctk.CTkFrame(menu_frame, fg_color="transparent")
        secondary_frame.pack(pady=10, fill="x")

        btn_stats = ctk.CTkButton(secondary_frame,
                                  text="Statystyki",
                                  font=("Roboto Medium", 14),
                                  fg_color="transparent",
                                  border_width=2,
                                  border_color="#3B8ED0",
                                  text_color="#3B8ED0",
                                  height=45,
                                  command=self.show_stats)
        btn_stats.pack(side="left", padx=(0, 10), expand=True, fill="x")

        btn_options = ctk.CTkButton(secondary_frame,
                                    text="Opcje",
                                    font=("Roboto Medium", 14),
                                    fg_color="transparent",
                                    border_width=2,
                                    border_color="#555",
                                    text_color="#CCC",
                                    height=45,
                                    command=self.show_options)
        btn_options.pack(side="left", padx=(10, 0), expand=True, fill="x")


        btn_exit = ctk.CTkButton(menu_frame,
                                 text="Wyjście",
                                 font=("Roboto Medium", 13),
                                 fg_color="#333",
                                 hover_color="#444",
                                 height=35,
                                 command=self.root.quit)
        btn_exit.pack(pady=(100, 0))

    def show_session(self):
        self.clear_container()


        session_card = ctk.CTkFrame(self.main_container, corner_radius=20)
        session_card.pack(pady=20, padx=10, fill="both", expand=True)

        lbl_subtitle = ctk.CTkLabel(session_card, text="Trwająca sesja", font=("Roboto Thin", 16), text_color="#AAA")
        lbl_subtitle.pack(pady=(30, 0))


        self.timer_label = ctk.CTkLabel(session_card, text="00:00", font=("Roboto Medium", 72), text_color="#3B8ED0")
        self.timer_label.pack(pady=30)


        control_frame = ctk.CTkFrame(session_card, fg_color="transparent")
        control_frame.pack(pady=20)

        self.btn_action = ctk.CTkButton(control_frame, text="START", fg_color="#2CC985", hover_color="#26AF74",
                                        width=120)
        self.btn_action.pack(side="left", padx=10)

        btn_reset = ctk.CTkButton(control_frame, text="Reset", fg_color="#E74C3C", hover_color="#C0392B", width=100)
        btn_reset.pack(side="left", padx=10)


        btn_back = ctk.CTkButton(session_card, text="< Powrót", font=("Roboto Medium", 12), fg_color="transparent",
                                 text_color="#AAA", command=self.show_main_menu)
        btn_back.pack(side="bottom", pady=20)

    def show_stats(self):
        self.clear_container()


        stats_card = ctk.CTkFrame(self.main_container, corner_radius=20)
        stats_card.pack(pady=20, padx=10, fill="both", expand=True)

        lbl_subtitle = ctk.CTkLabel(stats_card, text="Podsumowanie aktywności", font=("Roboto Medium", 18))
        lbl_subtitle.pack(pady=(30, 20))


        stats_data = [
            ("Najdłuższa deska", "03:15 min"),
            ("Sesji w tym tygodniu", "5"),
            ("️Łączny czas  ", "1h 25m"),
            ("Twój poziom", "Średni")
        ]

        for icon_text, value in stats_data:
            row = ctk.CTkFrame(stats_card, fg_color="transparent")
            row.pack(fill="x", padx=30, pady=10)

            ctk.CTkLabel(row, text=icon_text, font=("Roboto Light", 14), text_color="#CCC").pack(side="left")
            ctk.CTkLabel(row, text=value, font=("Roboto Medium", 14), text_color="white").pack(side="right")

        btn_back = ctk.CTkButton(stats_card, text="< Powrót", font=("Roboto Medium", 12), fg_color="transparent",
                                 text_color="#AAA", command=self.show_main_menu)
        btn_back.pack(side="bottom", pady=20)

    def show_options(self):
        self.clear_container()

        options_card = ctk.CTkFrame(self.main_container, corner_radius=20)
        options_card.pack(pady=20, padx=10, fill="both", expand=True)

        lbl_subtitle = ctk.CTkLabel(options_card, text="Ustawienia", font=("Roboto Medium", 18))
        lbl_subtitle.pack(pady=(30, 20))


        sw_sound = ctk.CTkSwitch(options_card, text="Dźwięk start/stop")
        sw_sound.pack(pady=15, anchor="w", padx=40)
        sw_sound.select()

        sw_dark = ctk.CTkSwitch(options_card, text="Tryb ciemny", command=self.toggle_dark_mode)
        sw_dark.pack(pady=15, anchor="w", padx=40)
        sw_dark.select()

        btn_back = ctk.CTkButton(options_card, text="< Powrót", font=("Roboto Medium", 12), fg_color="transparent",
                                 text_color="#AAA", command=self.show_main_menu)
        btn_back.pack(side="bottom", pady=20)

    def toggle_dark_mode(self):
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("Light")
        else:
            ctk.set_appearance_mode("Dark")


if __name__ == "__main__":
    root = ctk.CTk()
    app = PlankGUI(root)
    root.mainloop()