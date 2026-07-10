"""Agent harness helpers for the local DeepSeek OpenAI server.

Pure Python (no MLX) so unit tests can run without loading weights.

Goals:
  - Steer greenfield turns toward complete single-file code
  - Correct genre labels (never poison from harness suffix text)
  - Drop thrash assistant history so mid-thread retries get a clean prefill
  - Stop hard thrash; allow brief mid-file comments
"""
from __future__ import annotations

import re
from typing import Any, Optional

# ── stop strings (decoded) ───────────────────────────────────────────────────

STOP_STRINGS = (
    "<｜User｜>",
    "<｜end▁of▁sentence｜>",
    "<|User|>",
    "\nUser:",
    "\nUSER:",
    "\nHuman:",
    "\n\nUser:",
    "\nUser：",
)

# ── system / user steering text ──────────────────────────────────────────────

CODING_SYSTEM = (
    "You are a coding agent running locally. CRITICAL RULES:\n"
    "1) Emit real code immediately (fenced file or real host tool calls).\n"
    "2) Web demos: ONE self-contained HTML file, ALL CSS/JS inline. "
    "No ./components imports, no multi-file ES modules.\n"
    "3) Write executable JS (functions, event listeners, game loop). "
    "Do NOT fill the file with design essays, 'So plan:', or 'The user wants…' loops.\n"
    "4) Never invent tool results / User: lines / fake tool markup.\n"
    "5) Finish the file, close the fence, stop."
)

HARNESS_SYSTEM_TAIL = (
    "[local harness — highest priority]\n"
    "- Implement immediately with real code, not plans.\n"
    "- Single HTML file; inline CSS/JS only.\n"
    "- No 'The user wants…' / 'So plan:' / 'Alright let's assemble' restarts.\n"
    "- Finish one complete file and stop."
)

# No genre words here (driving/racing/adventure) — they poison labels.
IMPLEMENT_USER_SUFFIX = (
    "\n\n[Required by local harness] Reply format:\n"
    "1) Optional one-sentence intent.\n"
    "2) One markdown fence with a complete self-contained file "
    "(web: single HTML, inline CSS/JS).\n"
    "FORBIDDEN: ./ imports, multi-file modules, placeholder prose in CSS, "
    "planning essays, 'So plan' / 'The user wants' loops.\n"
    "Match the user's requested genre/title. Write real JS. Close the fence."
)

IMPLEMENT_TEMP_CAP = 0.85

_IMPLEMENT_RE = re.compile(
    r"\b(?:create|build|implement|write|make|scaffold|generate|code|"
    r"fix|add|refactor|port|convert)\b",
    re.I,
)
_ARTIFACT_RE = re.compile(
    r"\b(?:game|app|page|site|html|css|js|javascript|python|script|file|"
    r"component|server|api|cli|tool|module|class|function|test|"
    r"adventure|website|frontend|backend|ui|racing|racer|canvas|driving)\b",
    re.I,
)
_WEB_CREATE_RE = re.compile(
    r"(?:"
    r"create\s+(?:a\s+)?(?:web|html|browser)|"
    r"web[- ]based|"
    r"web\s+\w*\s*game|"
    r"web\s+game|"
    r"text\s+adventure|"
    r"adventure\s+game|"
    r"racing\s+game|"
    r"driving\s+game|"
    r"single[- ]file\s+html|"
    r"html\s*/\s*css\s*/\s*js|"
    r"index\.html"
    r")",
    re.I,
)

# Hard plan loops (need real repetition)
_PLAN_LOOP_PHRASES = (
    ("generating single", 2),
    ("let me craft", 3),
    ("the user wants", 2),
    ("so plan:", 2),
    ("alright let's", 2),
    ("alright let us", 2),
    ("alright assemble", 2),
    ("let's assemble", 2),
    ("let's implement the game", 2),
    ("step1", 3),
    ("step 1", 3),
    ("step2", 3),
    ("drift mode toggle that changes", 2),
    ("normal steering (left right", 2),
    ("car sprite drawn", 2),
    ("we'll produce one complete", 2),
    ("finalized_game_code", 2),
)


def _content_to_str(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content or "")


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _content_to_str(m.get("content", ""))
    return ""


def original_user_text(messages: list[dict[str, Any]]) -> str:
    """User text without harness suffix (must not poison genre labels)."""
    t = last_user_text(messages)
    if "[Required by local harness]" in t:
        t = t.split("[Required by local harness]", 1)[0]
    return t.strip()


def is_implement_request(messages: list[dict[str, Any]]) -> bool:
    text = original_user_text(messages) or last_user_text(messages)
    if not text.strip():
        return False
    if _WEB_CREATE_RE.search(text):
        return True
    if _IMPLEMENT_RE.search(text) and _ARTIFACT_RE.search(text):
        return True
    if re.search(
        r"^(?:please\s+)?(?:create|build|write|implement)\b", text.strip(), re.I
    ):
        return True
    return False


def is_web_create_request(messages: list[dict[str, Any]]) -> bool:
    t = original_user_text(messages) or last_user_text(messages)
    return bool(_WEB_CREATE_RE.search(t))


def _artifact_label(user_text: str) -> str:
    low = user_text.lower()
    if "[required by local harness]" in low:
        low = low.split("[required by local harness]", 1)[0]
    # racing before driving (order matters)
    if re.search(r"\bracing\b|\bracer\b|\brace\s*car\b", low):
        return "HTML racing game"
    if re.search(r"\bdriv(?:e|ing)\b", low):
        return "HTML driving game"
    if re.search(r"\badventure\b", low):
        return "HTML adventure game"
    if re.search(r"\bplatform(?:er)?\b", low):
        return "HTML platformer game"
    if re.search(r"\bshoot(?:er|ing)\b|\bspace\s*invad", low):
        return "HTML shooter game"
    if re.search(r"\bpong\b|\bbreakout\b|\bsnake\b|\btetris\b", low):
        m = re.search(r"\b(pong|breakout|snake|tetris)\b", low)
        return f"HTML {m.group(1)} game" if m else "HTML game"
    if re.search(r"\bgame\b", low):
        return "HTML game"
    if re.search(r"\b(app|page|site|dashboard|ui)\b", low):
        return "HTML page"
    return "HTML file"


def _has_real_js_game_signal(text: str) -> bool:
    """True if text has game logic beyond the harness prefill shell.

    Prefill already includes getContext + addEventListener — those alone
    must NOT count, or plan/essay thrash detectors never fire.
    """
    return bool(
        re.search(
            r"requestAnimationFrame\s*\(|"
            r"fillRect\s*\(|fillStyle|strokeStyle|beginPath\s*\(|"
            r"function\s+(?:draw|update|loop|tick|render|frame)\b|"
            r"(?:player|car|road|speed)\s*[=.:]|"
            r"\.translate\s*\(|\.rotate\s*\(",
            text,
            re.I,
        )
    )


# Philosophy / design-essay collapse (seen after partial JS on 2-bit)
_ESSAY_MARKERS = (
    "foundational principles",
    "technical architecture",
    "novel approach",
    "philosophical",
    "seamless integration",
    "leveraging established",
    "implementation plan",
    "implementation specifics",
    "key features:",
    "we'll craft",
    "we shall craft",
    "then we shall",
    "core proposition",
    "dimensional reference",
    "immersive interactive",
    "perceptual cues",
    "target demographic",
    "iterative refinement",
    "sustainable equilibrium",
    "vision statement",
    "operating environment restrictions",
    "macroeconomic",
    "civilization progress",
    "unleashing",
    "transcending traditional",
    "symbiotic relationships",
)


def prior_assistant_is_thrash(messages: list[dict[str, Any]]) -> bool:
    """Detect incomplete/plan-spam assistant turns so we can drop them and re-prefill."""
    last = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last = _content_to_str(m.get("content", ""))
            break
    if last is None:
        return False
    if not last.strip():
        return True
    low = last.lower()
    # Plan thrash without real game loop
    plan_hits = sum(
        low.count(p)
        for p in (
            "the user wants",
            "so plan",
            "alright let's",
            "let me think",
            "drift mode toggle",
            "let's assemble",
            "let's implement",
            "step1",
            "step 1:",
            "wait... the user",
        )
    )
    if plan_hits >= 2 and not _has_real_js_game_signal(last):
        return True
    if plan_hits >= 1 and len(last) > 600 and not _has_real_js_game_signal(last):
        return True
    # Open fence + long HTML shell with no JS game logic
    if last.count("```") % 2 == 1 or "<!doctype" in low or "<html" in low:
        if len(last) > 500 and not _has_real_js_game_signal(last):
            return True
    # Relative import collapse
    if re.search(r"""from\s+['"]\./|import\s+\w+\s+from\s+['"]\./""", last):
        return True
    return False


def drop_thrash_assistants(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove thrash assistant turns so implement retries get a clean prefill."""
    if not prior_assistant_is_thrash(messages):
        return list(messages)
    return [m for m in messages if m.get("role") != "assistant"]


def assistant_prefill(messages: list[dict[str, Any]]) -> str:
    """Force start of a single-file HTML+JS shell (first assistant turn only)."""
    if not is_implement_request(messages):
        return ""
    if any(m.get("role") == "assistant" for m in messages):
        return ""
    user = original_user_text(messages)
    if not (
        is_web_create_request(messages)
        or re.search(r"\b(?:html|web|game|page|canvas)\b", user, re.I)
    ):
        return "Implementing now — complete file contents:\n\n```\n"

    label = _artifact_label(user)
    title = label.replace("HTML ", "")
    # Skeleton ends inside a real <script> so the model continues JS, not CSS comments
    return (
        f"Writing a complete single-file {label} now "
        f"(all CSS/JS inline, no separate modules).\n\n"
        "```html\n"
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f"<title>{title}</title>\n"
        "<style>\n"
        "html,body{margin:0;height:100%;background:#111;overflow:hidden;"
        "font-family:system-ui,sans-serif;color:#eee}\n"
        "canvas{display:block;margin:0 auto;background:#1a1a1a;"
        "image-rendering:pixelated}\n"
        "#hud{position:fixed;top:10px;left:12px;font:14px/1.3 monospace;"
        "color:#0f0;text-shadow:0 0 4px #000;pointer-events:none}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        '<div id="hud"></div>\n'
        '<canvas id="c" width="800" height="600"></canvas>\n'
        "<script>\n"
        '"use strict";\n'
        "const canvas = document.getElementById('c');\n"
        "const ctx = canvas.getContext('2d');\n"
        "const hud = document.getElementById('hud');\n"
        "const keys = Object.create(null);\n"
        "addEventListener('keydown', e => { keys[e.code] = true; "
        "if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space'].includes(e.code)) "
        "e.preventDefault(); });\n"
        "addEventListener('keyup', e => { keys[e.code] = false; });\n"
    )


def steer_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    norm: list[dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user"))
        norm.append({"role": role, "content": _content_to_str(m.get("content", ""))})

    has_system = any(m["role"] == "system" for m in norm)
    if not has_system:
        norm.insert(0, {"role": "system", "content": CODING_SYSTEM})
    else:
        for i in range(len(norm) - 1, -1, -1):
            if norm[i]["role"] == "system":
                base = norm[i]["content"].rstrip()
                if "[local harness" not in base.lower():
                    norm[i]["content"] = base + "\n\n" + HARNESS_SYSTEM_TAIL
                break

    if is_implement_request(norm):
        for i in range(len(norm) - 1, -1, -1):
            if norm[i]["role"] == "user":
                if "[Required by local harness]" not in norm[i]["content"]:
                    norm[i]["content"] = (
                        norm[i]["content"].rstrip() + IMPLEMENT_USER_SUFFIX
                    )
                break

    return norm


def has_code_signal(text: str) -> bool:
    return bool(
        re.search(
            r"```|<!doctype|<html\b|function\s*\(|const\s+\w+\s*=|"
            r"document\.|addEventListener\s*\(|"
            r"def\s+\w+\s*\(|class\s+\w+|import\s+\w+|from\s+\w+\s+import",
            text,
            re.I,
        )
    )


def _normalize_for_count(s: str) -> str:
    s = s.lower()
    s = s.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("‑", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _in_open_code_fence(text: str) -> bool:
    return text.count("```") % 2 == 1


def _recent_plan_density(text: str, window: int = 900) -> tuple[int, int]:
    """Return (plan_marker_count, code_token_count) over the last window."""
    w = text[-window:]
    low = w.lower()
    plan = 0
    for p in (
        "the user wants",
        "so plan",
        "alright let's",
        "alright let",
        "let me think",
        "let's assemble",
        "let's implement",
        "drift mode toggle",
        "step1",
        "step 1",
        "step2",
        "step 2",
        "wait...",
        "maybe later",
        "but let me",
        "simplify:",
        "then we shall",
        "we'll craft",
        "key features",
        "novel approach",
        "implementation plan",
        "foundational",
        "technical architecture",
    ):
        plan += low.count(p)
    # Count only "real" code tokens — not prefill boilerplate alone
    code = len(
        re.findall(
            r"requestAnimationFrame|fillRect|fillStyle|beginPath|"
            r"function\s+(?:draw|update|loop|tick|render)|"
            r"for\s*\(|while\s*\(|if\s*\([^)]{0,40}\)\s*\{|"
            r"Math\.(?:sin|cos|atan|random|floor|min|max)",
            w,
            re.I,
        )
    )
    return plan, code


class LoopDetector:
    """Stop hard thrash; allow brief mid-file comments."""

    def __init__(self, *, expect_code: bool = False) -> None:
        self.text = ""
        self._line_counts: dict[str, int] = {}
        self.expect_code = expect_code

    def feed(self, chunk: str) -> Optional[str]:
        if not chunk:
            return None
        self.text += chunk
        low = _normalize_for_count(self.text)
        n = len(self.text)
        code = has_code_signal(self.text)
        in_fence = _in_open_code_fence(self.text)

        tail = self.text[-120:]
        if re.search(r"(?m)^(User|Human|USER|HUMAN)\s*:\s*$", self.text):
            return "fake_user_turn"
        if re.search(r"(?:^|\n)(User|Human)\s*:\s*$", tail):
            return "fake_user_turn_tail"

        if self.text.count("...") + self.text.count("…") >= 6:
            return "ellipsis_spam"
        if re.search(r"(?:\.\s*){16,}", self.text) or re.search(r"\.{16,}", self.text):
            return "dot_spam"
        # mid-word ellipsis thrash (… at end of long philosophy dump)
        if self.text.count("…") >= 2 and len(self.text) > 800:
            return "ellipsis_spam"

        if re.search(r"\b([a-zA-Z]{3,})\b(?:\s+\1){6,}", self.text, re.I):
            return "word_spam"

        for needle in (
            "model sharing",
            "agent sharing",
            "sharing mode",
            "you now have access to model sharing",
        ):
            if needle in low:
                return f"hallucination:{needle}"

        if re.search(
            r"<re_call_\w+|</?re_call|\[hole\s*\d+\s*:|call_edit\s*\(|tool_call\s*>",
            self.text,
            re.I,
        ):
            return "fake_tool_markup"
        if (
            re.search(
                r"<\s*(?:function|tool|invoke|re_call)[^>]{0,80}>",
                self.text,
                re.I,
            )
            and "```" not in self.text[-200:]
        ):
            return "fake_tool_tag"
        if re.search(
            r"\b[pP]l?[0-9a-f]{6,}[-0-9a-f]*\s+Tool result\s*:",
            self.text,
            re.I,
        ):
            return "fake_tool_result_id"
        if re.search(r"(?m)^Tool result\s*:\s*", self.text):
            return "fake_tool_result_line"
        if low.count("tool result:") >= 2:
            return "fake_tool_result_spam"

        # Hard collapse
        if code and re.search(
            r"""(?:import|from)\s+[^;{'"]{0,40}['"]\./[^'"]+['"]"""
            r"""|from\s+['"]\./"""
            r"""|import\s+\w+\s+from\s+['"]\./""",
            self.text,
        ):
            return "relative_import_single_file"

        if code and re.search(
            r"set using\s+js|right:\s*\(\s*calculator\s*\)|"
            r"cursorpointer|endstil|endtrig",
            low,
        ):
            return "degenerate_placeholder"

        if code:
            curly_attrs = len(
                re.findall(
                    r"""(?:\bid|\bclass|\bsrc|\bhref)\s*=\s*[“”‘’]""",
                    self.text,
                    re.I,
                )
            )
            curly_n = len(re.findall(r"[“”‘’]", self.text))
            if curly_attrs >= 2 or curly_n >= 12:
                return "curly_quote_degeneration"

        # ── post-code essay / markdown collapse (OutRun bomb mode) ───────────
        # Model leaves broken JS and writes ## philosophy sections for pages.
        if code:
            # Any markdown H2/H3 after we started the HTML file = left code mode
            if re.search(r"(?m)^#{1,3}\s+\S", self.text):
                return "markdown_section_after_code"

            essay_hits = sum(low.count(m) for m in _ESSAY_MARKERS)
            if essay_hits >= 1 and n > 350:
                return "essay_thrash"
            if essay_hits >= 2:
                return "essay_thrash"

            # Absurd identifier / camelCase thrash
            if re.search(r"\b[A-Za-z][A-Za-z0-9_]{55,}\b", self.text):
                return "identifier_spam"

            # Run-on prose: many long words, almost no real game code in tail
            tail = self.text[-1100:]
            if len(tail) > 650:
                long_words = len(re.findall(r"\b\w{11,}\b", tail))
                code_tok = len(
                    re.findall(
                        r"requestAnimationFrame|fillRect|fillStyle|beginPath|"
                        r"function\s+(?:draw|update|loop|tick)|"
                        r"for\s*\(|Math\.(?:sin|cos|random)",
                        tail,
                        re.I,
                    )
                )
                if long_words >= 18 and code_tok < 3:
                    return "prose_collapse"
                # very few sentence breaks + huge tail = one philosophy paragraph
                breaks = tail.count(".") + tail.count("!") + tail.count("?")
                if breaks <= 3 and len(tail) > 900 and code_tok < 4:
                    return "runon_prose"

        # Plan thrash inside an open fence with little real game code
        if code and n > 400:
            plan, code_tok = _recent_plan_density(self.text)
            if plan >= 2 and code_tok < 3:
                return "plan_thrash_low_code"
            if plan >= 3 and code_tok < 8:
                return "plan_thrash_low_code"
            if low.count("the user wants") >= 2:
                return "user_wants_loop"
            if low.count("drift mode toggle that changes control") >= 2:
                return "drift_essay_loop"
            if low.count("then we shall") >= 2:
                return "shall_essay_loop"

        for phrase, limit in _PLAN_LOOP_PHRASES:
            if low.count(phrase) >= limit:
                return f"plan_phrase_loop:{phrase}"

        if len(re.findall(r"(?m)^#{1,3}\s*generating\b", self.text, re.I)) >= 2:
            return "heading_generate_loop"

        for line in chunk.splitlines():
            s = line.strip()
            if len(s) < 40:
                continue
            key = _normalize_for_count(s)
            key_stripped = re.sub(r"^#+\s*", "", key)
            for k in {key, key_stripped}:
                if len(k) < 40:
                    continue
                if re.fullmatch(
                    r"(document\.)?getelementbyid\([^)]*\)[;.]?",
                    k,
                ):
                    continue
                self._line_counts[k] = self._line_counts.get(k, 0) + 1
                if self._line_counts[k] >= 3:
                    return f"repeat_line:{s[:60]}"

        window = self.text[-1000:]
        if len(window) >= 160:
            for phrase_n in (50, 70, 90, 120):
                if len(window) < phrase_n * 3:
                    continue
                phrase = window[-phrase_n:]
                if phrase.isspace() or re.fullmatch(r"[\.\s…-]+", phrase):
                    continue
                if window.count(phrase) >= 3:
                    return f"repeat_phrase:{phrase[:40]!r}"

        paras = re.split(r"\n\s*\n", self.text)
        if len(paras) >= 4:
            last = _normalize_for_count(paras[-1] if paras[-1].strip() else paras[-2])
            if len(last) >= 100:
                same = sum(
                    1
                    for p in paras
                    if len(p.strip()) >= 100 and _normalize_for_count(p) == last
                )
                if same >= 3:
                    return f"repeat_paragraph:{last[:50]!r}"

        # Pre-code monologue only
        if not code:
            if len(re.findall(r"i(?:'m| am) ready to .{8,80}", low)) >= 3:
                return "repeat_ready_phrase"
            if low.count("let me start") >= 3:
                return "repeat_let_me_start"
            rephrase = len(
                re.findall(
                    r"you want (?:me to|a )|you(?:'re| are) asking me to|"
                    r"i(?:'ll| will) (?:now )?(?:start|begin|create|set up)|"
                    r"let me (?:begin|start|set up)|let's set up",
                    low,
                )
            )
            if rephrase >= 3 and n > 280:
                return "rephrase_restart_loop"
            monologue = len(
                re.findall(
                    r"(?:let(?:'s| us)?|i(?:'ll| will)|okay|alright)"
                    r"[\s,]+(?:check|see|start|set\s*up|create|write|build|"
                    r"implement|begin|outline)",
                    low,
                )
            )
            if monologue >= 4 and n > 400:
                return "agent_monologue_no_code"
            ceiling = 900 if self.expect_code else 1400
            if n > ceiling:
                return f"no_code_after_{ceiling}_chars"

        if code and not in_fence:
            if low.count("let me craft") >= 2 or low.count("the user wants") >= 2:
                if n > 400:
                    return "replan_after_closed_fence"

        return None


def strip_stop_suffix(text: str) -> str:
    for s in STOP_STRINGS:
        if s in text:
            text = text.split(s, 1)[0]
    text = re.split(r"(?m)^(User|Human|USER|HUMAN)\s*:\s*$", text, maxsplit=1)[0]
    text = re.split(r"(?m)\n(User|Human)\s*:\s*", text, maxsplit=1)[0]
    text = re.split(
        r"(?i)you now have access to model sharing|use agent sharing mode",
        text,
        maxsplit=1,
    )[0]
    text = re.split(
        r"(?im)<re_call_\w+|\[hole\s*\d+\s*:|"
        r"\bpl?[0-9a-f]{6,}[-0-9a-f]*\s+Tool result\s*:|"
        r"^Tool result\s*:",
        text,
        maxsplit=1,
    )[0]
    text = re.split(
        r"(?im)"
        r"^#{1,3}\s+\S|"  # any markdown heading after code
        r"\nimport\s+\w+\s+from\s+['\"]\./|"
        r"\nfrom\s+['\"]\./|"
        r"\nthe user wants\b|"
        r"\nso plan\s*:|"
        r"\nthen we shall\b|"
        r"\nfoundational principles\b|"
        r"\ntechnical architecture\b|"
        r"\nnovel approach\b|"
        r"\nalright let's (?:assemble|implement)\b",
        text,
        maxsplit=1,
    )[0]
    text = re.split(
        r"(?i)set using\s+js|right:\s*\(\s*calculator\s*\)",
        text,
        maxsplit=1,
    )[0]
    text = re.sub(r"(?:\s*\.{3,}\s*){3,}$", "\n", text)
    text = re.sub(r"</?think>\s*$", "", text)
    text = re.sub(r"(\b\w{3,}\b)(?:\s+\1){5,}\s*$", r"\1", text, flags=re.I)
    return text.rstrip()
