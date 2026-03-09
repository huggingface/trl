"""
Terminal UI Dashboard Logger — ANSI true-color TUI with iTerm2 inline image support.

No external TUI libraries (no rich/textual/curses). Only stdlib + Pillow for halfblock fallback.
"""

import base64
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime


def _visible_len(s: str) -> int:
    """Length of string after stripping ANSI escape sequences."""
    return len(re.sub(r"\033(?:\[[^m]*m|\][^\x07]*\x07)", "", s))


def _fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


RST = "\033[0m"
BOLD = "\033[1m"


class DashboardLogger:
    """Thread-safe terminal dashboard with two metric columns and a scrolling log panel.

    Usage::

        dash = DashboardLogger(logo_path="logo.png", title="My App v1.0") dash.set_header("Hostname", "node-42")
        dash.set_header("GPU", "4x A100") dash.start()

        dash.log("GPU", 87, scope="training", max_value=100, unit="%") dash.log("MEM", 21000, scope="training",
        max_value=24564, unit="MB") dash.log("steps", 1420, scope="training") dash.message("Loaded checkpoint epoch 3",
        scope="training", level="info")

        dash.wait() # blocks until Ctrl-C
    """

    # ── colour palette (terminal default background, no painted BGs) ──
    C_BORDER = (88, 86, 124)
    C_TITLE = (180, 142, 255)
    C_LABEL = (150, 145, 185)
    C_VALUE = (210, 210, 220)
    C_DIM = (60, 58, 80)
    C_HDR_KEY = (180, 142, 255)
    C_HDR_VAL = (210, 210, 220)

    C_SCOPE = {"inference": (100, 180, 255), "training": (140, 220, 140)}
    C_LEVEL = {"info": (170, 170, 180), "warn": (255, 200, 50), "error": (255, 80, 80)}

    def __init__(self, logo_path="logo.png", title="Dashboard", logo_lines=10, refresh=0.5):
        self.logo_path = logo_path
        self.title = title
        self.logo_lines = logo_lines
        self._refresh = refresh
        self._logo_col = 28  # header text starts at this column (right of logo)

        self._lock = threading.Lock()
        self._metrics: dict[str, OrderedDict] = {
            "inference": OrderedDict(),
            "training": OrderedDict(),
        }
        self._header: OrderedDict = OrderedDict()
        self._messages: list[dict] = []
        self._max_messages = 200

        self._running = False
        self._thread: threading.Thread | None = None
        self._first_frame = True
        self._has_logo = False

    # ── public API ─────────────────────────────────────────────────

    def log(self, key, value, scope="training", max_value=None, unit=""):
        """Log a metric. If *max_value* is set, renders a progress bar."""
        with self._lock:
            self._metrics[scope][key] = dict(value=value, max_value=max_value, unit=unit)

    def message(self, text, scope="training", level="info"):
        """Append a timestamped message to the scrolling log panel."""
        with self._lock:
            self._messages.append(dict(text=text, scope=scope, level=level, time=datetime.now()))
            if len(self._messages) > self._max_messages:
                self._messages = self._messages[-self._max_messages :]

    def set_header(self, key, value):
        """Set a key-value pair displayed next to the logo."""
        with self._lock:
            self._header[key] = value

    def start(self):
        """Start the background render thread."""
        self._running = True
        self._first_frame = True
        self._has_logo = bool(self.logo_path and os.path.exists(self.logo_path))
        self._use_iterm2 = self._has_logo and self._supports_iterm2()
        # Pre-render halfblock logo (cached, drawn every frame)
        self._logo_hb_lines: list[str] = []
        if self._has_logo and not self._use_iterm2:
            self._logo_hb_lines = self._render_logo_halfblock(
                self.logo_path, self._logo_col - 2, self.logo_lines
            )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop rendering and restore terminal state."""
        self._running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write("\033[?25h" + RST + "\n")
        sys.stdout.flush()

    def wait(self):
        """Block until stopped (Ctrl-C safe)."""
        try:
            while self._running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.stop()

    # ── render loop ────────────────────────────────────────────────

    def _loop(self):
        sys.stdout.write("\033[?25l")  # hide cursor
        sys.stdout.flush()
        try:
            while self._running:
                self._render()
                time.sleep(self._refresh)
        finally:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    # ── gradient helpers ───────────────────────────────────────────

    @staticmethod
    def _gradient(ratio: float) -> tuple[int, int, int]:
        """Green (0) -> Yellow (0.5) -> Red (1)."""
        ratio = max(0.0, min(1.0, ratio))
        if ratio < 0.5:
            t = ratio * 2
            return int(80 + 175 * t), int(220 - 20 * t), int(80 - 40 * t)
        t = (ratio - 0.5) * 2
        return 255, int(200 - 140 * t), int(40 + 20 * t)

    def _progress_bar(self, value, max_value, unit, width):
        ratio = value / max_value if max_value else 0
        filled = int(ratio * width)
        parts = [_fg(*self.C_DIM), "["]
        for i in range(width):
            if i < filled:
                parts.append(_fg(*self._gradient(i / max(width - 1, 1))))
                parts.append("\u2588")
            else:
                parts.append(_fg(*self.C_DIM))
                parts.append(" ")
        parts.append(_fg(*self.C_DIM))
        parts.append("] ")
        pct_col = self._gradient(ratio)
        parts.append(f"{_fg(*pct_col)}{BOLD}{ratio * 100:>3.0f}%{RST}")
        val_str = f"{value}/{max_value}"
        if unit:
            val_str += f" {unit}"
        parts.append(f"  {_fg(*self.C_VALUE)}{val_str}{RST}")
        return "".join(parts)

    # ── logo rendering ───────────────────────────────────────────

    @staticmethod
    def _iterm2_image(path: str) -> str:
        data = open(path, "rb").read()
        b64 = base64.b64encode(data).decode("ascii")
        name = base64.b64encode(os.path.basename(path).encode()).decode("ascii")
        return f"\x1b]1337;File=name={name};size={len(data)};inline=1:{b64}\x07\n"

    @staticmethod
    def _supports_iterm2() -> bool:
        tp = os.environ.get("TERM_PROGRAM", "")
        # Also check LC_TERMINAL for SSH-forwarded contexts
        lc = os.environ.get("LC_TERMINAL", "")
        return any(k in ("iTerm.app", "iTerm2", "WezTerm", "mintty") for k in (tp, lc))

    def _render_logo_halfblock(self, path: str, max_cols: int, max_rows: int) -> list[str]:
        """Render a PNG using half-block characters. Each terminal row = 2 pixel rows.

        Uses upper-half-block (U+2580): foreground = top pixel, background = bottom pixel. Returns list of ANSI-colored
        strings (one per terminal row).
        """
        try:
            from PIL import Image
        except ImportError:
            return [f"{_fg(*self.C_DIM)}[install Pillow for logo]{RST}"]

        img = Image.open(path).convert("RGBA")
        # Composite onto dark background matching terminal
        bg = Image.new("RGBA", img.size, (24, 24, 32, 255))
        bg.paste(img, mask=img)
        img = bg.convert("RGB")

        # Scale to fit within max_cols x (max_rows*2) pixels
        pix_h = max_rows * 2
        aspect = img.width / img.height
        fit_w = min(max_cols, int(pix_h * aspect))
        fit_h = min(pix_h, int(max_cols / aspect))
        if fit_h % 2:
            fit_h -= 1
        fit_w = max(1, fit_w)
        fit_h = max(2, fit_h)
        img = img.resize((fit_w, fit_h), Image.LANCZOS)

        pixels = img.load()
        lines: list[str] = []
        for y in range(0, fit_h, 2):
            row: list[str] = []
            for x in range(fit_w):
                tr, tg, tb = pixels[x, y]
                br, bg_, bb = pixels[x, y + 1] if y + 1 < fit_h else (0, 0, 0)
                row.append(f"\033[38;2;{tr};{tg};{tb}m\033[48;2;{br};{bg_};{bb}m\u2580")
            lines.append("".join(row) + RST)
        return lines

    # ── box drawing ────────────────────────────────────────────────

    def _bc(self):
        return _fg(*self.C_BORDER)

    def _box_top(self, width, label="", label_color=None):
        bc = self._bc()
        if label:
            lc = _fg(*(label_color or self.C_TITLE))
            fill = max(0, width - 5 - len(label))
            return f"{bc}\u256d─ {RST}{lc}{BOLD}{label}{RST} {bc}{'─' * fill}\u256e{RST}"
        return f"{bc}\u256d{'─' * (width - 2)}\u256e{RST}"

    def _box_bot(self, width):
        return f"{self._bc()}\u2570{'─' * (width - 2)}\u256f{RST}"

    def _box_row(self, content, width):
        bc = self._bc()
        vis = _visible_len(content)
        inner = width - 4
        pad = max(0, inner - vis)
        return f"{bc}\u2502{RST} {content}{' ' * pad} {bc}\u2502{RST}"

    # ── dual-column panel ──────────────────────────────────────────

    def _dual_top(self, lw, rw, l_label, r_label):
        bc = self._bc()
        lc = _fg(*self.C_SCOPE["inference"])
        rc = _fg(*self.C_SCOPE["training"])
        l_fill = max(0, lw - 4 - len(l_label))
        r_fill = max(0, rw - 5 - len(r_label))
        return (
            f"{bc}\u256d\u2500 {RST}{lc}{BOLD}{l_label}{RST} "
            f"{bc}{'─' * l_fill}\u252c\u2500 {RST}{rc}{BOLD}{r_label}{RST} "
            f"{bc}{'─' * r_fill}\u256e{RST}"
        )

    def _dual_bot(self, lw, rw):
        bc = self._bc()
        return f"{bc}\u2570{'─' * (lw - 1)}\u2534{'─' * (rw - 2)}\u256f{RST}"

    def _dual_row(self, left, right, lw, rw):
        bc = self._bc()
        l_pad = max(0, lw - 2 - _visible_len(left))
        r_pad = max(0, rw - 3 - _visible_len(right))
        return f"{bc}\u2502{RST} {left}{' ' * l_pad}{bc}\u2502{RST} {right}{' ' * r_pad}{bc}\u2502{RST}"

    # ── metric formatting ──────────────────────────────────────────

    def _fmt_metric(self, key, info, bar_width):
        label = f"{_fg(*self.C_LABEL)}{key}{RST}"
        if info["max_value"] is not None:
            bar = self._progress_bar(info["value"], info["max_value"], info["unit"], bar_width)
            return f"{label} {bar}"
        unit = f" {info['unit']}" if info["unit"] else ""
        return f"{label} {_fg(*self.C_DIM)}\u00b7\u00b7\u00b7{RST} {_fg(*self.C_VALUE)}{info['value']}{unit}{RST}"

    # ── full frame render ──────────────────────────────────────────

    def _render(self):
        try:
            cols, rows = os.get_terminal_size()
        except OSError:
            cols, rows = 120, 40

        with self._lock:
            metrics = {s: list(m.items()) for s, m in self._metrics.items()}
            header = list(self._header.items())
            messages = list(self._messages)

        buf: list[str] = []

        # ── header text lines (displayed to the right of logo) ────
        hdr: list[str] = []
        hdr.append(f"{BOLD}{_fg(*self.C_TITLE)}{self.title}{RST}")
        hdr.append("")  # blank line after title
        for k, v in header:
            hdr.append(f"  {_fg(*self.C_HDR_KEY)}{k:<12}{RST} {_fg(*self.C_HDR_VAL)}{v}{RST}")
        while len(hdr) < self.logo_lines:
            hdr.append("")

        # ── metric panels ─────────────────────────────────────────
        half = cols // 2
        lw, rw = half, cols - half
        bar_w = max(8, half - 35)

        panel: list[str] = []
        panel.append(self._dual_top(lw, rw, "Inference", "Training"))

        inf = metrics["inference"]
        trn = metrics["training"]
        n = max(len(inf), len(trn), 1)
        for i in range(n):
            left = self._fmt_metric(*inf[i], bar_w) if i < len(inf) else ""
            right = self._fmt_metric(*trn[i], bar_w) if i < len(trn) else ""
            panel.append(self._dual_row(left, right, lw, rw))
        panel.append(self._dual_bot(lw, rw))

        # ── log panel ─────────────────────────────────────────────
        log_avail = rows - self.logo_lines - len(panel) - 3
        log_height = max(3, log_avail)
        vis_msgs = messages[-log_height:]

        log: list[str] = []
        log.append(self._box_top(cols, "Log", self.C_TITLE))
        for msg in vis_msgs:
            ts = msg["time"].strftime("%H:%M:%S")
            sc = self.C_SCOPE.get(msg["scope"], (150, 150, 150))
            lc = self.C_LEVEL.get(msg["level"], (170, 170, 180))
            tag = msg["scope"][:3].upper()
            line = f"{_fg(*self.C_DIM)}{ts} {_fg(*sc)}[{tag}]{RST} {_fg(*lc)}{msg['text']}{RST}"
            log.append(self._box_row(line, cols))
        while len(log) < log_height + 1:
            log.append(self._box_row("", cols))
        log.append(self._box_bot(cols))

        # ── compose output buffer ─────────────────────────────────
        if self._first_frame:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            if self._use_iterm2:
                # iTerm2 inline image: emit once, never re-draw
                sys.stdout.write(self._iterm2_image(self.logo_path))
                sys.stdout.flush()
                buf.append(f"\033[{self.logo_lines}A")
                for h in hdr[: self.logo_lines]:
                    buf.append(f"\033[{self._logo_col}G{h}\033[K\n")
            elif self._logo_hb_lines:
                # Halfblock logo: render logo + header text side by side
                for i in range(self.logo_lines):
                    logo_part = self._logo_hb_lines[i] if i < len(self._logo_hb_lines) else ""
                    hdr_part = hdr[i] if i < len(hdr) else ""
                    buf.append(f"\033[2K{logo_part}\033[{self._logo_col}G{hdr_part}\033[K\n")
            else:
                for h in hdr[: self.logo_lines]:
                    buf.append(f"{h}\033[K\n")
            self._first_frame = False
        else:
            buf.append("\033[H")  # cursor home (no clear!)
            if self._use_iterm2:
                # Don't erase logo lines — only overwrite text portion
                for h in hdr[: self.logo_lines]:
                    buf.append(f"\033[{self._logo_col}G{h}\033[K\n")
            elif self._logo_hb_lines:
                # Re-draw halfblock logo + header text every frame
                for i in range(self.logo_lines):
                    logo_part = self._logo_hb_lines[i] if i < len(self._logo_hb_lines) else ""
                    hdr_part = hdr[i] if i < len(hdr) else ""
                    buf.append(f"\033[2K{logo_part}\033[{self._logo_col}G{hdr_part}\033[K\n")
            else:
                for h in hdr[: self.logo_lines]:
                    buf.append(f"\033[2K{h}\n")

        # panel + log lines are below the logo — safe to fully erase each line
        for line in panel:
            buf.append(f"\033[2K{line}\n")
        for line in log:
            buf.append(f"\033[2K{line}\n")

        buf.append("\033[J")  # erase from cursor to end of screen (cleans up stale lines)

        sys.stdout.write("".join(buf))
        sys.stdout.flush()


# ── demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math
    import random

    dash = DashboardLogger(logo_path="logo-dark.png", title="TRL Dashboard v0.1", logo_lines=10)
    dash.set_header("Hostname", os.uname().nodename)
    dash.set_header("Python", sys.version.split()[0])
    dash.set_header("PID", str(os.getpid()))
    dash.start()

    step = 0
    try:
        while True:
            step += 1
            t = step * 0.05

            # training metrics
            dash.log("GPU", min(100, int(60 + 30 * math.sin(t))), scope="training", max_value=100, unit="%")
            dash.log("MEM", min(24564, int(8000 + 6000 * abs(math.sin(t * 0.3)))), scope="training", max_value=24564, unit="MB")
            dash.log("loss", round(2.0 * math.exp(-t * 0.1) + random.gauss(0, 0.02), 4), scope="training")
            dash.log("lr", f"{5e-5 * (1 - t / 200):.2e}", scope="training")
            dash.log("step", step, scope="training")

            # inference metrics
            dash.log("throughput", round(120 + 20 * math.sin(t * 0.7), 1), scope="inference", unit="tok/s")
            dash.log("queue", max(0, int(5 * math.sin(t * 0.4))), scope="inference")
            dash.log("VRAM", min(24564, int(4000 + 2000 * abs(math.sin(t * 0.2)))), scope="inference", max_value=24564, unit="MB")

            if step % 20 == 0:
                dash.message(f"Completed step {step}", scope="training", level="info")
            if step % 50 == 0:
                dash.message(f"Checkpoint saved at step {step}", scope="training", level="warn")
            if step % 35 == 0:
                dash.message(f"Generated {random.randint(50, 200)} tokens", scope="inference", level="info")
            if step % 100 == 0:
                dash.message("GPU thermal throttle detected", scope="training", level="error")

            time.sleep(0.1)
    except KeyboardInterrupt:
        dash.stop()
