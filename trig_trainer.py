#!/usr/bin/env python3
"""
Exact Trig Trainer  -  a pygame quiz for exact trigonometric values.

    pip install pygame
    python trig_trainer.py

* Angles are every multiple of pi/6 and pi/4 from 0 up to 2*pi.
* Answer by clicking / pressing 1-6 (multiple choice) or by typing pseudo-LaTeX.
* Typed answers accept things like:
      sqrt3/2      \\frac{\\sqrt{3}}{2}      -1/2      2sqrt3/3      undefined
  (the backslash is optional, and a live preview shows how your input is read)
* At the end you get your total time (time spent answering, feedback pauses excluded).
"""

import random
import re
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import pygame

W, H = 960, 720
FPS = 60

BG = (17, 19, 30)
PANEL = (31, 35, 54)
PANEL_HOVER = (45, 51, 79)
PANEL_DARK = (24, 27, 42)
TEXT = (236, 239, 250)
DIM = (136, 144, 176)
ACCENT = (104, 146, 255)
GOOD = (84, 204, 134)
BAD = (240, 98, 112)
GOOD_BG = (26, 74, 54)
BAD_BG = (84, 34, 46)

# =============================================================================
#  Exact arithmetic:  every value is  q * sqrt(r)  with q rational, r squarefree
# =============================================================================


def simplify_sqrt(n):
    """Return (k, m) such that sqrt(n) = k * sqrt(m) with m squarefree."""
    k, m, f = 1, n, 2
    while f * f <= m:
        while m % (f * f) == 0:
            m //= f * f
            k *= f
        f += 1
    return k, m


@dataclass(frozen=True)
class Val:
    q: Fraction
    r: int = 1

    @staticmethod
    def make(q, r=1):
        q = Fraction(q)
        if q == 0:
            return Val(Fraction(0), 1)
        k, m = simplify_sqrt(r)
        return Val(q * k, m)

    def __neg__(self):
        return Val(-self.q, self.r)

    def __mul__(self, o):
        return Val.make(self.q * o.q, self.r * o.r)

    def inv(self):
        if self.q == 0:
            raise ZeroDivisionError
        return Val(1 / (self.q * self.r), self.r)

    def __truediv__(self, o):
        return self * o.inv()


def val_sqrt(v):
    if v.q == 0:
        return v
    if v.r != 1 or v.q < 0:
        raise ValueError("unsupported sqrt")
    k1, m1 = simplify_sqrt(v.q.numerator)
    k2, m2 = simplify_sqrt(v.q.denominator)
    return Val.make(k1, m1) / Val.make(k2, m2)


# sin of the reference angles 0..90 degrees
_SIN_REF = {
    0: Val.make(0),
    30: Val.make(Fraction(1, 2)),
    45: Val.make(Fraction(1, 2), 2),
    60: Val.make(Fraction(1, 2), 3),
    90: Val.make(1),
}
_ONE = Val.make(1)


def sin_exact(deg):
    deg %= 360
    sign = 1 if deg <= 180 else -1
    ref = deg if deg <= 180 else deg - 180
    if ref > 90:
        ref = 180 - ref
    v = _SIN_REF[ref]
    return v if sign == 1 else -v


def cos_exact(deg):
    return sin_exact((deg + 90) % 360)


def _div(a, b):
    return None if b.q == 0 else a / b  # None  ==  undefined


def trig(func, deg):
    s, c = sin_exact(deg), cos_exact(deg)
    return {
        "sin": s,
        "cos": c,
        "tan": _div(s, c),
        "csc": _div(_ONE, s),
        "sec": _div(_ONE, c),
        "cot": _div(c, s),
    }[func]


BASIC = ["sin", "cos", "tan"]
ALL_FUNCS = BASIC + ["csc", "sec", "cot"]
ANGLES = sorted(set(range(0, 361, 30)) | set(range(0, 361, 45)))  # degrees, 0..360


def _pool(func):
    out = []
    for d in ANGLES:
        v = trig(func, d)
        if v not in out:
            out.append(v)
    return out


POOLS = {f: _pool(f) for f in ALL_FUNCS}

# =============================================================================
#  Pseudo-LaTeX parser  ->  AST  ->  Val
#  AST nodes: ('num', str) ('text', str) ('neg', n) ('paren', n) ('sqrt', n)
#             ('frac', n, d) ('mul', [n, ...])
# =============================================================================


class ParseError(Exception):
    pass


UNDEF_WORDS = {"undefined", "undef", "und", "dne", "nan", "none"}

_TOK = re.compile(
    r"\s*(?:(?P<fn>\\?(?:frac|sqrt|cdot)|\u221a)|(?P<num>\d+)|(?P<sym>[{}()/*+\-\u2212]))"
)


def tokenize(s):
    s = s.strip()
    toks, pos = [], 0
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m:
            raise ParseError("bad character")
        if m.group("fn"):
            name = m.group("fn").lstrip("\\")
            if name == "\u221a":
                name = "sqrt"
            toks.append(("sym", "*") if name == "cdot" else ("fn", name))
        elif m.group("num"):
            if len(m.group("num")) > 4:
                raise ParseError("number too large")
            toks.append(("num", m.group("num")))
        else:
            ch = m.group("sym")
            toks.append(("sym", "-" if ch == "\u2212" else ch))
        pos = m.end()
    return toks


def _join(a, b):
    return ("mul", (a[1] if a[0] == "mul" else [a]) + [b])


class Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, ch):
        if self.eat() != ("sym", ch):
            raise ParseError("expected " + ch)

    def expr(self):
        tok = self.peek()
        if tok == ("sym", "-"):
            self.eat()
            return ("neg", self.expr())
        if tok == ("sym", "+"):
            self.eat()
            return self.expr()
        return self.term()

    def _starts_factor(self, tok):
        return tok is not None and (
            tok[0] in ("num", "fn") or tok in (("sym", "("), ("sym", "{"))
        )

    def term(self):
        left = self.factor()
        while True:
            tok = self.peek()
            if tok == ("sym", "/"):
                self.eat()
                left = ("frac", left, self.factor())
            elif tok == ("sym", "*"):
                self.eat()
                left = _join(left, self.factor())
            elif self._starts_factor(tok):
                left = _join(left, self.factor())
            else:
                return left

    def factor(self):
        tok = self.eat()
        if tok is None:
            raise ParseError("unexpected end")
        kind, val = tok
        if kind == "num":
            return ("num", val)
        if kind == "fn":
            if val == "sqrt":
                return ("sqrt", self.atom())
            n = self.atom()
            d = self.atom()
            return ("frac", n, d)
        if tok == ("sym", "("):
            e = self.expr()
            self.expect(")")
            return ("paren", e)
        if tok == ("sym", "{"):
            e = self.expr()
            self.expect("}")
            return e
        if tok == ("sym", "-"):
            return ("neg", self.factor())
        raise ParseError("unexpected token")

    def atom(self):
        tok = self.eat()
        if tok is None:
            raise ParseError("unexpected end")
        if tok[0] == "num":
            return ("num", tok[1])
        if tok in (("sym", "{"), ("sym", "(")):
            e = self.expr()
            self.expect("}" if tok[1] == "{" else ")")
            return e
        raise ParseError("expected argument")


@lru_cache(maxsize=1024)
def parse_text(text):
    toks = tokenize(text)
    if not toks:
        raise ParseError("empty")
    p = Parser(toks)
    node = p.expr()
    if p.i != len(toks):
        raise ParseError("trailing input")
    return node


def evaluate(node):
    t = node[0]
    if t == "num":
        return Val.make(int(node[1]))
    if t == "neg":
        return -evaluate(node[1])
    if t == "paren":
        return evaluate(node[1])
    if t == "sqrt":
        return val_sqrt(evaluate(node[1]))
    if t == "frac":
        d = evaluate(node[2])
        if d.q == 0:
            raise ValueError("division by zero")
        return evaluate(node[1]) / d
    if t == "mul":
        r = _ONE
        for item in node[1]:
            r = r * evaluate(item)
        return r
    raise ValueError("bad node")


def val_ast(v):
    """Canonical display AST for an exact value (None means undefined)."""
    if v is None:
        return ("text", "undefined")
    if v.q == 0:
        return ("num", "0")
    a = abs(v.q)
    n, d = a.numerator, a.denominator
    if v.r == 1:
        top = ("num", str(n))
    else:
        rad = ("sqrt", ("num", str(v.r)))
        top = rad if n == 1 else ("mul", [("num", str(n)), rad])
    node = top if d == 1 else ("frac", top, ("num", str(d)))
    return ("neg", node) if v.q < 0 else node


def angle_ast(deg):
    f = Fraction(deg, 180)
    if f == 0:
        return ("num", "0")
    n, d = f.numerator, f.denominator
    top = ("text", "\u03c0") if n == 1 else ("mul", [("num", str(n)), ("text", "\u03c0")])
    return top if d == 1 else ("frac", top, ("num", str(d)))


def question_ast(func, deg, tail=" = ?"):
    return ("mul", [("text", func), ("paren", angle_ast(deg)), ("text", tail)])


# =============================================================================
#  Tiny maths-typesetting engine (draws fractions, square roots, brackets)
# =============================================================================

_fonts = {}


def font(size):
    size = max(8, int(size))
    if size not in _fonts:
        _fonts[size] = pygame.font.Font(None, size)
    return _fonts[size]


_axis = {}


def math_axis(size):
    """y of the 'maths axis' (where fraction bars sit) inside a rendered line."""
    size = int(size)
    if size not in _axis:
        r = font(size).render("\u2212", True, (255, 255, 255)).get_bounding_rect()
        _axis[size] = r.centery if r.height else int(font(size).get_height() * 0.5)
    return _axis[size]


class Box:
    def __init__(self, surf, axis):
        self.surf = surf
        self.axis = int(round(axis))

    @property
    def w(self):
        return self.surf.get_width()

    @property
    def h(self):
        return self.surf.get_height()


def _text_box(s, size, color):
    return Box(font(size).render(s, True, color), math_axis(size))


def _hbox(boxes, gap=0):
    gap = int(gap)
    above = max(b.axis for b in boxes)
    below = max(b.h - b.axis for b in boxes)
    w = sum(b.w for b in boxes) + gap * (len(boxes) - 1)
    s = pygame.Surface((max(1, w), max(1, above + below)), pygame.SRCALPHA)
    x = 0
    for b in boxes:
        s.blit(b.surf, (x, above - b.axis))
        x += b.w + gap
    return Box(s, above)


def _frac_box(n, d, size, color):
    pad = max(4, int(size * 0.12))
    bar = max(2, int(size / 20))
    gap = max(1, int(size * 0.03))
    w = max(n.w, d.w) + 2 * pad
    y = n.h + gap
    s = pygame.Surface((w, y + bar + gap + d.h), pygame.SRCALPHA)
    s.blit(n.surf, ((w - n.w) // 2, 0))
    pygame.draw.rect(s, color, (0, y, w, bar))
    s.blit(d.surf, ((w - d.w) // 2, y + bar + gap))
    return Box(s, y + bar / 2)


def _sqrt_box(c, size, color):
    t = max(2, int(size / 15))
    hook = max(12, int(size * 0.5))
    gap = max(1, int(size / 30))
    w = hook + c.w + max(3, int(size / 14))
    h = t + gap + c.h
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.blit(c.surf, (hook, t + gap))
    yt = t / 2
    pts = [(1, h * 0.62), (hook * 0.30, h * 0.54), (hook * 0.62, h - 2), (hook, yt), (w - 1, yt)]
    for a, b in zip(pts, pts[1:]):
        pygame.draw.line(s, color, a, b, t)
    for p in pts[1:-1]:
        pygame.draw.circle(s, color, p, max(1, t // 2))
    return Box(s, t + gap + c.axis)


def _paren_glyph(ch, h, size, color):
    g = font(size).render(ch, True, color)
    g = g.subsurface(g.get_bounding_rect()).copy()
    return pygame.transform.smoothscale(g, (g.get_width(), max(8, h)))


def _paren_box(inner, size, color):
    h = inner.h + 4
    left = Box(_paren_glyph("(", h, size, color), inner.axis + 2)
    right = Box(_paren_glyph(")", h, size, color), inner.axis + 2)
    inner2 = Box(inner.surf, inner.axis)
    return _hbox([left, inner2, right], gap=max(2, size * 0.05))


def layout(node, size, color=TEXT):
    size = int(size)
    t = node[0]
    if t in ("num", "text"):
        return _text_box(node[1], size, color)
    if t == "neg":
        return _hbox([_text_box("\u2212", size, color), layout(node[1], size, color)], size * 0.05)
    if t == "paren":
        return _paren_box(layout(node[1], size, color), size, color)
    if t == "sqrt":
        return _sqrt_box(layout(node[1], size, color), size, color)
    if t == "frac":
        cs = max(20, size * 0.88)
        return _frac_box(layout(node[1], cs, color), layout(node[2], cs, color), size, color)
    if t == "mul":
        parts, items = [], node[1]
        for i, it in enumerate(items):
            if i and it[0] == "num" and items[i - 1][0] == "num":
                parts.append(_text_box("\u00b7", size, color))
            parts.append(layout(it, size, color))
        return _hbox(parts, size * 0.05)
    raise ValueError(t)


def render(node, size, color=TEXT):
    return layout(node, size, color).surf


def fit(surf, max_w, max_h):
    w, h = surf.get_size()
    k = min(1.0, max_w / w, max_h / h)
    if k < 1.0:
        return pygame.transform.smoothscale(surf, (max(1, int(w * k)), max(1, int(h * k))))
    return surf


_preview_cache = {}


def render_typed(text, size, color=TEXT):
    key = (text, size, color)
    if key not in _preview_cache:
        t = text.strip()
        out = None
        if t:
            if t.lower() in UNDEF_WORDS:
                out = render(("text", "undefined"), size, color)
            else:
                try:
                    out = render(parse_text(t), size, color)
                except ParseError:
                    out = None
        _preview_cache[key] = out
    return _preview_cache[key]


# =============================================================================
#  Game
# =============================================================================


def draw_text(surf, s, size, color, **anchor):
    img = font(size).render(s, True, color)
    r = img.get_rect(**anchor)
    surf.blit(img, r)
    return r


def fmt_time(t):
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:04.1f}"


class Button:
    def __init__(self, rect, label="", size=34):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.size = size

    def draw(self, surf, mouse, active=False, primary=False):
        hover = self.rect.collidepoint(mouse)
        if active or primary:
            col = (130, 168, 255) if hover else ACCENT
        else:
            col = PANEL_HOVER if hover else PANEL
        pygame.draw.rect(surf, col, self.rect, border_radius=14)
        fg = (12, 16, 32) if (active or primary) else TEXT
        draw_text(surf, self.label, self.size, fg, center=self.rect.center)

    def hit(self, pos):
        return self.rect.collidepoint(pos)


@dataclass
class Question:
    func: str
    deg: int
    answer: object  # Val or None (undefined)
    mode: str  # 'choice' | 'typed'
    options: list


@dataclass
class Result:
    q: Question
    ok: bool
    user_ast: tuple
    seconds: float


def make_options(func, correct):
    pool = POOLS[func]
    others = [v for v in pool if v != correct]
    random.shuffle(others)
    opts = [correct]
    if correct is not None:
        neg = -correct
        if neg != correct and neg in others:  # sign-flipped distractor is the classic mistake
            others.remove(neg)
            opts.append(neg)
    opts += others[: 6 - len(opts)]
    random.shuffle(opts)
    return opts


def build_questions(n, funcs, style):
    pairs = [(f, d) for f in funcs for d in ANGLES]
    seq = []
    while len(seq) < n:
        batch = pairs[:]
        random.shuffle(batch)
        seq += batch
    qs = []
    for f, d in seq[:n]:
        mode = style if style in ("choice", "typed") else random.choice(["choice", "typed"])
        ans = trig(f, d)
        qs.append(Question(f, d, ans, mode, make_options(f, ans) if mode == "choice" else []))
    return qs


class Game:
    STYLES = ["choice", "typed"]

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Exact Trig Trainer")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.state = "setup"

        # settings
        self.count_str = "10"
        self.style_idx = 0
        self.use_recip = False  # include csc / sec / cot

        # setup widgets
        self.b_minus = Button((400, 185, 60, 60), "\u2212", 44)
        self.b_plus = Button((612, 185, 60, 60), "+", 44)
        self.count_rect = pygame.Rect(466, 185, 140, 60)
        self.b_presets = [Button((700 + i * 58, 185, 50, 60), str(n), 28) for i, n in enumerate((5, 10, 20, 50))]
        self.b_styles = [
            Button((400 + i * 262, 275, 256, 60), lbl, 28)
            for i, lbl in enumerate(("Multiple choice", "Type the answer"))
        ]
        self.recip_rect = pygame.Rect(400, 365, 518, 60)  # whole row is clickable
        self.b_start = Button((300, 618, 360, 72), "Start quiz", 46)
        self.b_again = Button((150, 636, 320, 66), "Play again", 38)
        self.b_menu = Button((490, 636, 320, 66), "Settings", 38)

        # quiz state
        self.questions = []
        self.idx = 0
        self.q = None
        self.q_surf = None
        self.opt_surfs = []
        self.opt_rects = []
        self.typed = ""
        self.error = ""
        self.chosen = None
        self.results = []
        self.correct = 0
        self.total_time = 0.0
        self.q_start = 0.0
        self.fb_t = 0.0
        self.last_ok = False
        self.bs_t0 = None
        self.bs_last = 0.0
        self.last_preview = None
        self.final_time = 0.0

    # ------------------------------------------------------------------ flow
    def count(self):
        try:
            return max(1, min(100, int(self.count_str)))
        except ValueError:
            return 10

    def start_quiz(self):
        n = self.count()
        self.count_str = str(n)
        funcs = ALL_FUNCS if self.use_recip else BASIC
        self.questions = build_questions(n, funcs, self.STYLES[self.style_idx])
        self.idx, self.correct, self.total_time = 0, 0, 0.0
        self.results = []
        self.begin_question()

    def begin_question(self):
        self.q = self.questions[self.idx]
        self.q_surf = render(question_ast(self.q.func, self.q.deg), 78)
        self.typed, self.error, self.chosen = "", "", None
        self.last_preview = None
        self.opt_surfs = [render(val_ast(v), 50) for v in self.q.options]
        self.opt_rects = []
        for i in range(len(self.q.options)):
            col, row = i % 3, i // 3
            self.opt_rects.append(pygame.Rect(60 + col * 290, 340 + row * 130, 260, 110))
        self.state = "question"
        self.q_start = time.perf_counter()

    def finish(self, ok, user_ast):
        now = time.perf_counter()
        dt = now - self.q_start
        self.total_time += dt
        self.results.append(Result(self.q, ok, user_ast, dt))
        if ok:
            self.correct += 1
        self.last_ok = ok
        self.fb_t = now
        self.state = "feedback"

    def next_question(self):
        self.idx += 1
        if self.idx >= len(self.questions):
            self.final_time = self.total_time
            self.state = "results"
        else:
            self.begin_question()

    def answer_choice(self, i):
        if 0 <= i < len(self.q.options):
            self.chosen = i
            v = self.q.options[i]
            self.finish(v == self.q.answer, val_ast(v))

    def submit_typed(self):
        t = self.typed.strip()
        if not t:
            self.error = "Type an answer first."
            return
        if t.lower() in UNDEF_WORDS:
            val, uast = None, ("text", "undefined")
        else:
            try:
                uast = parse_text(t)
                val = evaluate(uast)
            except (ParseError, ValueError, ZeroDivisionError):
                self.error = "Couldn't read that \u2013 try something like sqrt3/2, -1/2 or undefined."
                return
        self.finish(val == self.q.answer, uast)

    # ---------------------------------------------------------------- events
    def handle(self, e):
        if self.state == "setup":
            self.handle_setup(e)
        elif self.state == "question":
            self.handle_question(e)
        elif self.state == "feedback":
            if (e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)) or (
                e.type == pygame.MOUSEBUTTONDOWN and e.button == 1
            ):
                self.next_question()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self.state = "setup"
        elif self.state == "results":
            self.handle_results(e)

    def handle_setup(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            p = e.pos
            if self.b_minus.hit(p):
                self.count_str = str(max(1, self.count() - 1))
            elif self.b_plus.hit(p):
                self.count_str = str(min(100, self.count() + 1))
            elif self.b_start.hit(p):
                self.start_quiz()
            for b, n in zip(self.b_presets, (5, 10, 20, 50)):
                if b.hit(p):
                    self.count_str = str(n)
            for i, b in enumerate(self.b_styles):
                if b.hit(p):
                    self.style_idx = i
            if self.recip_rect.collidepoint(p):
                self.use_recip = not self.use_recip
        elif e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.start_quiz()
            elif e.key == pygame.K_BACKSPACE:
                self.count_str = self.count_str[:-1]
            elif e.key == pygame.K_UP:
                self.count_str = str(min(100, self.count() + 1))
            elif e.key == pygame.K_DOWN:
                self.count_str = str(max(1, self.count() - 1))
            elif e.unicode.isdigit() and len(self.count_str) < 3:
                self.count_str = (self.count_str + e.unicode).lstrip("0") or "0"

    def handle_question(self, e):
        q = self.q
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            self.state = "setup"
            return
        if q.mode == "choice":
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for i, r in enumerate(self.opt_rects):
                    if r.collidepoint(e.pos):
                        self.answer_choice(i)
                        break
            elif e.type == pygame.KEYDOWN and pygame.K_1 <= e.key <= pygame.K_6:
                self.answer_choice(e.key - pygame.K_1)
        else:
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.submit_typed()
                elif e.key == pygame.K_BACKSPACE:
                    self.typed = self.typed[:-1]
                    self.error = ""
                    self.bs_t0 = self.bs_last = time.perf_counter()
                elif len(e.unicode) == 1 and e.unicode.isprintable() and len(self.typed) < 40:
                    self.typed += e.unicode
                    self.error = ""
            elif e.type == pygame.KEYUP and e.key == pygame.K_BACKSPACE:
                self.bs_t0 = None

    def handle_results(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.b_again.hit(e.pos):
                self.start_quiz()
            elif self.b_menu.hit(e.pos):
                self.state = "setup"
        elif e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_r):
                self.start_quiz()
            elif e.key in (pygame.K_ESCAPE, pygame.K_s):
                self.state = "setup"

    # ---------------------------------------------------------------- update
    def update(self):
        now = time.perf_counter()
        if self.state == "feedback" and self.last_ok and now - self.fb_t > 0.7:
            self.next_question()
        if (
            self.state == "question"
            and self.q.mode == "typed"
            and self.bs_t0 is not None
            and pygame.key.get_pressed()[pygame.K_BACKSPACE]
            and now - self.bs_t0 > 0.4
            and now - self.bs_last > 0.04
        ):
            self.typed = self.typed[:-1]
            self.bs_last = now

    # ------------------------------------------------------------------ draw
    def draw(self):
        self.screen.fill(BG)
        if self.state == "setup":
            self.draw_setup()
        elif self.state in ("question", "feedback"):
            self.draw_question()
        else:
            self.draw_results()
        pygame.display.flip()

    def draw_setup(self):
        s, mouse = self.screen, pygame.mouse.get_pos()
        draw_text(s, "Exact Trig Trainer", 88, TEXT, center=(W // 2, 62))
        draw_text(s, "Every multiple of \u03c0/6 and \u03c0/4, from 0 up to 2\u03c0", 30, DIM, center=(W // 2, 114))

        # --- setting rows: label + short description on the left, controls on the right
        rows = (
            (215, "Questions", "How many to answer"),
            (305, "Answer style", "How you give your answer"),
            (395, "Functions", "sin, cos, tan always included"),
        )
        for y, label, sub in rows:
            draw_text(s, label, 36, TEXT, midleft=(70, y - 12))
            draw_text(s, sub, 22, DIM, midleft=(70, y + 18))
        for y in (260, 350):
            pygame.draw.line(s, PANEL, (70, y), (W - 70, y), 2)

        # questions
        self.b_minus.draw(s, mouse)
        self.b_plus.draw(s, mouse)
        pygame.draw.rect(s, PANEL_DARK, self.count_rect, border_radius=14)
        pygame.draw.rect(s, ACCENT, self.count_rect, 2, border_radius=14)
        caret = "|" if int(time.time() * 2) % 2 == 0 else " "
        draw_text(s, self.count_str + caret, 46, TEXT, center=self.count_rect.center)
        for b, n in zip(self.b_presets, (5, 10, 20, 50)):
            b.draw(s, mouse, active=(self.count_str == str(n)))

        # answer style
        for i, b in enumerate(self.b_styles):
            b.draw(s, mouse, active=(i == self.style_idx))

        # functions: a single checkbox
        r = self.recip_rect
        pygame.draw.rect(s, PANEL_HOVER if r.collidepoint(mouse) else PANEL, r, border_radius=14)
        box = pygame.Rect(r.x + 14, r.centery - 18, 36, 36)
        if self.use_recip:
            pygame.draw.rect(s, ACCENT, box, border_radius=8)
            pygame.draw.lines(s, (12, 16, 32), False,
                              [(box.x + 8, box.centery), (box.x + 15, box.bottom - 10), (box.right - 8, box.y + 10)], 5)
        else:
            pygame.draw.rect(s, PANEL_DARK, box, border_radius=8)
            pygame.draw.rect(s, DIM, box, 2, border_radius=8)
        draw_text(s, "Include sec, csc and cot", 30, TEXT, midleft=(box.right + 18, r.centery))

        # --- context-sensitive tip for the chosen answer style
        panel = pygame.Rect(70, 448, 820, 120)
        pygame.draw.rect(s, PANEL_DARK, panel, border_radius=16)
        if self.style_idx == 0:
            draw_text(s, "Multiple choice", 30, ACCENT, midtop=(W // 2, panel.y + 22))
            draw_text(s, "Pick the right value from six options \u2013 click it, or press 1\u20136.", 28, TEXT,
                      midtop=(W // 2, panel.y + 66))
        else:
            draw_text(s, "Type your answer in pseudo-LaTeX", 30, ACCENT, midtop=(W // 2, panel.y + 14))
            draw_text(s, "sqrt3/2     \\frac{\\sqrt{3}}{2}     -1/2     2sqrt3/3     undefined", 30, TEXT,
                      midtop=(W // 2, panel.y + 52))
            draw_text(s, "The backslash is optional, and a live preview shows how your input is read.", 24, DIM,
                      midtop=(W // 2, panel.y + 92))

        draw_text(s, "Type a number to set the question count (1\u2013100)  \u00b7  Enter to start", 22, DIM,
                  center=(W // 2, 590))
        self.b_start.draw(s, mouse, primary=True)

    def draw_question(self):
        s, mouse, q = self.screen, pygame.mouse.get_pos(), self.q
        now = time.perf_counter()
        n = len(self.questions)
        elapsed = self.total_time + ((now - self.q_start) if self.state == "question" else 0.0)

        draw_text(s, f"Question {self.idx + 1} / {n}", 36, TEXT, topleft=(40, 24))
        draw_text(s, f"{self.correct} correct", 30, GOOD, midtop=(W // 2, 28))
        draw_text(s, fmt_time(elapsed), 48, TEXT, topright=(W - 40, 18))
        bar = pygame.Rect(40, 76, W - 80, 8)
        pygame.draw.rect(s, PANEL, bar, border_radius=4)
        done = self.idx + (1 if self.state == "feedback" else 0)
        pygame.draw.rect(s, ACCENT, (bar.x, bar.y, int(bar.w * done / n), bar.h), border_radius=4)

        s.blit(self.q_surf, self.q_surf.get_rect(center=(W // 2, 205)))
        fb = self.state == "feedback"

        if q.mode == "choice":
            for i, (r, surf) in enumerate(zip(self.opt_rects, self.opt_surfs)):
                col, border = PANEL, None
                if fb:
                    if q.options[i] == q.answer:
                        col, border = GOOD_BG, GOOD
                    elif i == self.chosen:
                        col, border = BAD_BG, BAD
                elif r.collidepoint(mouse):
                    col = PANEL_HOVER
                pygame.draw.rect(s, col, r, border_radius=16)
                if border:
                    pygame.draw.rect(s, border, r, 3, border_radius=16)
                s.blit(fit(surf, r.w - 40, r.h - 16), fit(surf, r.w - 40, r.h - 16).get_rect(center=r.center))
                draw_text(s, str(i + 1), 24, DIM, topleft=(r.x + 12, r.y + 8))
            if not fb:
                draw_text(s, "Click an answer or press 1\u20136", 24, DIM, center=(W // 2, 625))
        else:
            box = pygame.Rect(130, 290, 700, 64)
            pygame.draw.rect(s, PANEL, box, border_radius=14)
            pygame.draw.rect(s, ACCENT if not fb else PANEL_HOVER, box, 2, border_radius=14)
            img = font(48).render(self.typed, True, TEXT if not fb else DIM)
            s.blit(img, (box.x + 18, box.centery - img.get_height() // 2))
            if not fb and int(time.time() * 2) % 2 == 0:
                cx = box.x + 20 + img.get_width()
                pygame.draw.rect(s, ACCENT, (cx, box.y + 12, 3, box.h - 24))
            if not self.typed and not fb:
                draw_text(s, "e.g.  sqrt3/2   \\frac{1}{2}   -1   undefined", 30, PANEL_HOVER,
                          midleft=(box.x + 18, box.centery))
            draw_text(s, "Enter to submit", 22, DIM, midtop=(W // 2, box.bottom + 8))

            draw_text(s, "preview", 22, DIM, midtop=(W // 2, 398))
            prev = render_typed(self.typed, 58, TEXT)
            if prev is not None:
                self.last_preview = prev
                dim = False
            else:
                prev = self.last_preview if self.typed.strip() else None
                dim = True
            if prev is not None:
                prev = fit(prev, 760, 120)
                if dim:
                    prev = prev.copy()
                    prev.set_alpha(90)
                s.blit(prev, prev.get_rect(center=(W // 2, 480)))
            if self.error:
                draw_text(s, self.error, 26, BAD, center=(W // 2, 560))

        if fb:
            panel = pygame.Rect(40, 598, 880, 104)
            ok = self.last_ok
            pygame.draw.rect(s, GOOD_BG if ok else BAD_BG, panel, border_radius=18)
            pygame.draw.rect(s, GOOD if ok else BAD, panel, 3, border_radius=18)
            if ok:
                draw_text(s, "Correct!", 54, GOOD, midleft=(panel.x + 34, panel.centery))
            else:
                draw_text(s, "Not quite", 46, BAD, midleft=(panel.x + 30, panel.centery))
                draw_text(s, "answer:", 30, DIM, midleft=(panel.x + 250, panel.centery))
                ans = fit(render(val_ast(q.answer), 50), 320, 90)
                s.blit(ans, ans.get_rect(midleft=(panel.x + 350, panel.centery)))
                draw_text(s, "Enter to continue", 24, DIM, midright=(panel.right - 28, panel.centery))

    def draw_results(self):
        s, mouse = self.screen, pygame.mouse.get_pos()
        n = len(self.questions)
        draw_text(s, "Quiz complete!", 64, ACCENT, center=(W // 2, 52))
        draw_text(s, "TOTAL TIME", 26, DIM, center=(W // 2, 112))
        draw_text(s, fmt_time(self.final_time), 132, TEXT, center=(W // 2, 180))
        pct = round(100 * self.correct / n)
        avg = self.final_time / n
        draw_text(
            s,
            f"{self.correct} / {n} correct  \u00b7  {pct}%  \u00b7  {avg:.1f}s per question",
            34,
            TEXT,
            center=(W // 2, 262),
        )

        misses = [r for r in self.results if not r.ok]
        if not misses:
            draw_text(s, "Flawless \u2013 every answer correct.", 36, GOOD, center=(W // 2, 400))
        else:
            draw_text(s, "Review your misses", 30, DIM, midtop=(W // 2, 296))
            for i, r in enumerate(misses[:4]):
                row = pygame.Rect(60, 334 + i * 70, 840, 64)
                pygame.draw.rect(s, PANEL, row, border_radius=12)
                line = fit(render(question_ast(r.q.func, r.q.deg, " = "), 36), 300, 56)
                s.blit(line, line.get_rect(midleft=(row.x + 20, row.centery)))
                ans = fit(render(val_ast(r.q.answer), 36, GOOD), 200, 56)
                s.blit(ans, ans.get_rect(midleft=(row.x + 340, row.centery)))
                draw_text(s, "you:", 26, DIM, midleft=(row.x + 560, row.centery))
                you = fit(render(r.user_ast, 36, BAD), 190, 56)
                s.blit(you, you.get_rect(midleft=(row.x + 620, row.centery)))
            if len(misses) > 4:
                draw_text(s, f"+ {len(misses) - 4} more", 26, DIM, midtop=(W // 2, 334 + 4 * 70))

        self.b_again.draw(s, mouse, primary=True)
        self.b_menu.draw(s, mouse)

    # ------------------------------------------------------------------ loop
    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                self.handle(e)
            self.update()
            self.draw()
            self.clock.tick(FPS)


if __name__ == "__main__":
    Game().run()
