import customtkinter as ctk

from gui import PlankGUI

if __name__ == "__main__":
    root = ctk.CTk()
    app = PlankGUI(root)
    root.mainloop()
