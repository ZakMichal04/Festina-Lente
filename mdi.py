import customtkinter as ctk


class MDIChildWindow(ctk.CTkFrame):
    def __init__(self, workspace, title, on_close=None, width=420, height=480):
        super().__init__(
            workspace, width=width, height=height,
            corner_radius=8, border_width=1, border_color="#333",
        )
        self.workspace = workspace
        self.title = title
        self.on_close = on_close
        self._width = width
        self._height = height
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._win_start_x = 0
        self._win_start_y = 0
        self._dimmed = False
        self._focus_key = None
        self._saved_styles = {}
        self._dim_bind_id = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.title_bar = ctk.CTkFrame(self, height=32, corner_radius=6, fg_color="#2B2B2B")
        self.title_bar.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        self.title_bar.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.title_bar, text=title, font=("Roboto Medium", 12), anchor="w",
        )
        self.title_label.grid(row=0, column=0, sticky="ew", padx=10)

        self.btn_close = ctk.CTkButton(
            self.title_bar, text="✕", width=28, height=24,
            fg_color="transparent", hover_color="#C0392B",
            command=self.close,
        )
        self.btn_close.grid(row=0, column=1, padx=(0, 4))

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        for widget in (self, self.title_bar, self.title_label):
            widget.bind("<Button-1>", self._focus)
        self.title_bar.bind("<B1-Motion>", self._drag)
        self.title_label.bind("<B1-Motion>", self._drag)

    def place_at(self, x, y):
        self.place(x=x, y=y)

    def place_centered(self):
        self.workspace.update_idletasks()
        ws_w = max(self._width, self.workspace.winfo_width())
        ws_h = max(self._height, self.workspace.winfo_height())
        x = max(0, (ws_w - self._width) // 2)
        y = max(0, (ws_h - self._height) // 2)
        self.place(x=x, y=y)

    @staticmethod
    def _mute_hex(hex_color: str) -> str:
        color = hex_color.lstrip("#")
        if len(color) != 6:
            return hex_color
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        gray = 0x77
        factor = 0.55
        r = int(r * factor + gray * (1 - factor))
        g = int(g * factor + gray * (1 - factor))
        b = int(b * factor + gray * (1 - factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _mute_color(self, color):
        if isinstance(color, (tuple, list)):
            return tuple(self._mute_color(c) for c in color)
        if isinstance(color, str) and color.startswith("#"):
            return self._mute_hex(color)
        named = {
            "white": "#BBBBBB",
            "transparent": "transparent",
        }
        return named.get(color, "#999999")

    def _save_and_set(self, widget, prop, value):
        key = str(widget)
        self._saved_styles.setdefault(key, {})
        if prop not in self._saved_styles[key]:
            self._saved_styles[key][prop] = widget.cget(prop)
        widget.configure(**{prop: value})

    def _restore_widget_styles(self, widget):
        key = str(widget)
        saved = self._saved_styles.get(key)
        if saved:
            widget.configure(**saved)

    def _restore_tree(self, widget):
        self._restore_widget_styles(widget)
        for child in widget.winfo_children():
            self._restore_tree(child)

    def _apply_dim_to_tree(self, widget):
        if isinstance(widget, ctk.CTkLabel):
            self._save_and_set(widget, "text_color", self._mute_color(widget.cget("text_color")))
        elif isinstance(widget, ctk.CTkFrame):
            fg = widget.cget("fg_color")
            if fg not in ("transparent", ["transparent", "transparent"]):
                self._save_and_set(widget, "fg_color", "#1C1C1C")
        elif isinstance(widget, ctk.CTkButton):
            if widget.cget("fg_color") == "transparent":
                self._save_and_set(widget, "text_color", self._mute_color(widget.cget("text_color")))
                self._save_and_set(widget, "border_color", self._mute_color(widget.cget("border_color")))

        for child in widget.winfo_children():
            self._apply_dim_to_tree(child)

    def _on_dimmed_click(self, _event=None):
        if self._dimmed and self._focus_key:
            self.workspace.set_focus_window(self._focus_key)

    def set_dimmed(self, dimmed: bool, focus_key: str | None = None):
        if self._dimmed == dimmed and self._focus_key == focus_key:
            return

        if self._dimmed and not dimmed:
            self.configure(border_color="#333")
            self.title_bar.configure(fg_color="#2B2B2B")
            self._restore_widget_styles(self.title_label)
            self._restore_tree(self.content)
            self._saved_styles.clear()
            if self._dim_bind_id is not None:
                self.unbind("<Button-1>", self._dim_bind_id)
                self._dim_bind_id = None

        self._dimmed = dimmed
        self._focus_key = focus_key

        if dimmed:
            self.configure(border_color="#2A2A2A")
            self.title_bar.configure(fg_color="#232323")
            self._save_and_set(self.title_label, "text_color", "#888888")
            self._apply_dim_to_tree(self.content)
            self._dim_bind_id = self.bind("<Button-1>", self._on_dimmed_click, add="+")

    def _focus(self, _event=None):
        if self._dimmed and self._focus_key:
            self.workspace.set_focus_window(self._focus_key)
            return
        self.lift()

    def _start_drag(self, event):
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root
        self._win_start_x = self.winfo_x()
        self._win_start_y = self.winfo_y()

    def _drag(self, event):
        x = self._win_start_x + (event.x_root - self._drag_start_x)
        y = self._win_start_y + (event.y_root - self._drag_start_y)
        max_x = max(0, self.workspace.winfo_width() - self._width)
        max_y = max(0, self.workspace.winfo_height() - self._height)
        self.place(x=min(max(0, x), max_x), y=min(max(0, y), max_y))

    def close(self):
        if self.on_close:
            self.on_close()
        self.destroy()


class MDIWorkspace(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="#1A1A1A", corner_radius=0, **kwargs)
        self._windows = {}
        self._cascade_step = 0
        self._focus_key = None

    def next_cascade_position(self):
        offset = self._cascade_step
        self._cascade_step = (self._cascade_step + 30) % 120
        return 24 + offset, 24 + offset

    def set_focus_window(self, focus_key: str | None):
        self._focus_key = focus_key
        for key, window in list(self._windows.items()):
            if window.winfo_exists():
                window.set_dimmed(focus_key is not None and key != focus_key, focus_key)
                if key == focus_key:
                    window.lift()

    def open_window(self, key, title, builder, width=420, height=480, on_close=None,
                    centered=False):
        existing = self._windows.get(key)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            if self._focus_key:
                self.set_focus_window(self._focus_key)
            return existing

        def handle_close():
            self._windows.pop(key, None)
            if self._focus_key == key:
                self._focus_key = None
            if on_close:
                on_close()

        window = MDIChildWindow(self, title, on_close=handle_close, width=width, height=height)
        for widget in (window.title_bar, window.title_label):
            widget.bind("<ButtonPress-1>", window._start_drag, add="+")
        builder(window.content)

        if centered:
            window.place_centered()
        else:
            # Zamiast w ciemno układać w kaskadę, spróbuj znaleźć wolne miejsce
            self.update_idletasks()
            ws_w = max(400, self.winfo_width())
            ws_h = max(300, self.winfo_height())
            placed = []

            for k, w in self._visible_windows():
                if w != window:
                    try:
                        px = int(w.place_info().get('x', w.winfo_x()))
                        py = int(w.place_info().get('y', w.winfo_y()))
                        placed.append((px, py, w._width, w._height))
                    except Exception:
                        pass

            pos = self._find_free_position(window, placed, ws_w, ws_h, pad=16, gap=14)
            if pos is not None:
                window.place_at(pos[0], pos[1])
            else:
                x, y = self.next_cascade_position()
                window.place_at(x, y)

        self._windows[key] = window
        if self._focus_key:
            self.set_focus_window(self._focus_key)
        return window

    def close_window(self, key):
        window = self._windows.get(key)
        if window is not None and window.winfo_exists():
            window.close()

    def get_window(self, key):
        window = self._windows.get(key)
        if window is not None and window.winfo_exists():
            return window
        return None

    def _visible_windows(self):
        return [
            (key, window)
            for key, window in self._windows.items()
            if window.winfo_exists()
        ]

    @staticmethod
    def _rects_overlap(x1, y1, w1, h1, x2, y2, w2, h2, gap):
        return not (
                x1 + w1 + gap <= x2
                or x2 + w2 + gap <= x1
                or y1 + h1 + gap <= y2
                or y2 + h2 + gap <= y1
        )

    def _find_free_position(self, win, placed, ws_w, ws_h, pad, gap):
        w, h = win._width, win._height
        step = 28

        max_x = max(pad, ws_w - w - pad)
        max_y = max(pad, ws_h - h - pad)

        for y in range(pad, max_y + 1, step):
            for x in range(pad, max_x + 1, step):
                collision = False

                for px, py, pw, ph in placed:
                    if self._rects_overlap(x, y, w, h, px, py, pw, ph, gap):
                        collision = True
                        break

                if not collision:
                    return x, y

        # Nie znaleziono miejsca
        return None

    def _clamp_place(self, win, x, y, ws_w, ws_h, pad):
        x = max(pad, min(int(x), max(pad, ws_w - win._width - pad)))
        y = max(pad, min(int(y), max(pad, ws_h - win._height - pad)))
        win.place_at(x, y)
        return x, y, win._width, win._height

    def _collides_any(self, x, y, w, h, placed, gap):
        for px, py, pw, ph in placed:
            if self._rects_overlap(x, y, w, h, px, py, pw, ph, gap):
                return True
        return False

    def _place_clear(self, win, x, y, placed, ws_w, ws_h, pad, gap):
        w, h = win._width, win._height

        x = max(pad, min(int(x), max(pad, ws_w - w - pad)))
        y = max(pad, min(int(y), max(pad, ws_h - h - pad)))

        if not self._collides_any(x, y, w, h, placed, gap):
            win.place_at(x, y)
            return x, y, w, h

        pos = self._find_free_position(win, placed, ws_w, ws_h, pad, gap)

        if pos is None:
            # brak miejsca - zostaw tam, gdzie chociaż mieści się w ekranie (clamp)
            return self._clamp_place(win, x, y, ws_w, ws_h, pad)

        fx, fy = pos
        return self._clamp_place(win, fx, fy, ws_w, ws_h, pad)

    def _place_menu_near_session(self, menu, session_rect, placed, ws_w, ws_h, pad, gap):
        sx, sy, sw, sh = session_rect
        mw, mh = menu._width, menu._height
        candidates = [
            (sx, sy + sh + gap),
            (sx + sw + gap, sy),
            (max(pad, sx + sw - mw), sy + sh + gap),
        ]
        for mx, my in candidates:
            if my + mh <= ws_h - pad and mx + mw <= ws_w - pad:
                if not self._collides_any(mx, my, mw, mh, placed, gap):
                    menu.place_at(mx, my)
                    return mx, my, mw, mh
        return self._place_clear(menu, sx, sy + sh + gap, placed, ws_w, ws_h, pad, gap)

    def arrange_for_session(self):
        """Rozsuwa okna MDI tak, aby kamera, sesja i pozostałe widoki się nie nakładały."""
        self.update_idletasks()
        pad = 16
        gap = 14
        ws_w = max(600, self.winfo_width())
        ws_h = max(400, self.winfo_height())
        placed: list[tuple[int, int, int, int]] = []

        camera = self.get_window("camera")
        session = self.get_window("session")
        menu = self.get_window("menu")

        if camera is not None and session is not None:
            cw, ch = camera._width, camera._height
            sw, sh = session._width, session._height
            mh = menu._height if menu is not None else 0
            right_stack_h = sh + (gap + mh if menu is not None else 0)
            side_by_side = pad + cw + gap + sw + pad <= ws_w

            if side_by_side:
                block_h = max(ch, right_stack_h)
                y0 = pad + max(0, (ws_h - 2 * pad - block_h) // 2)
                cx = pad
                cy = y0 + max(0, (block_h - ch) // 2)
                sx = pad + cw + gap
                sy = y0

                placed.append(self._clamp_place(camera, cx, cy, ws_w, ws_h, pad))
                placed.append(self._clamp_place(session, sx, sy, ws_w, ws_h, pad))
                session_rect = placed[-1]

            else:
                # Najpierw ustaw kamerę
                cx = max(pad, (ws_w - cw) // 2)
                cy = pad
                placed.append(self._clamp_place(camera, cx, cy, ws_w, ws_h, pad))

                # Znajdź pierwsze wolne miejsce dla sesji
                pos = self._find_free_position(session, placed, ws_w, ws_h, pad, gap)

                if pos is None:
                    # Zabezpieczenie na wypadek ekstremalnie małego okna
                    sx, sy = pad + 20, pad + 20
                else:
                    sx, sy = pos

                placed.append(self._clamp_place(session, sx, sy, ws_w, ws_h, pad))
                session_rect = placed[-1]

            if menu is not None:
                placed.append(
                    self._place_menu_near_session(menu, session_rect, placed, ws_w, ws_h, pad, gap)
                )

        elif camera is not None:
            placed.append(self._clamp_place(
                camera,
                (ws_w - camera._width) // 2,
                (ws_h - camera._height) // 2,
                ws_w, ws_h, pad,
            ))
            session_rect = None
            if menu is not None:
                pos = self._find_free_position(menu, placed, ws_w, ws_h, pad, gap)
                mx, my = pos if pos else (pad, pad)
                placed.append(self._clamp_place(menu, mx, my, ws_w, ws_h, pad))

        elif session is not None:
            placed.append(self._clamp_place(
                session,
                (ws_w - session._width) // 2,
                (ws_h - session._height) // 2,
                ws_w, ws_h, pad,
            ))
            session_rect = placed[-1]
            if menu is not None:
                placed.append(
                    self._place_menu_near_session(menu, session_rect, placed, ws_w, ws_h, pad, gap)
                )
        else:
            session_rect = None

        reserved = {"camera", "session", "menu"}
        for key, win in self._visible_windows():
            if key in reserved:
                continue
            pos = self._find_free_position(win, placed, ws_w, ws_h, pad, gap)
            if pos is not None:
                placed.append(self._clamp_place(win, pos[0], pos[1], ws_w, ws_h, pad))
            else:
                # W przypadku ostatecznego braku miejsca, wymuś pozycję z kaskady
                x, y = self.next_cascade_position()
                placed.append(self._clamp_place(win, x, y, ws_w, ws_h, pad))

        session_win = self.get_window("session")
        if session_win is not None:
            session_win.lift()

        if self._focus_key:
            self.set_focus_window(self._focus_key)