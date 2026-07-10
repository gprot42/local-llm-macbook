#!/usr/bin/env python3
"""Unit tests for harness.py only (no MLX, no game generation)."""
from __future__ import annotations

import unittest

from harness import (
    IMPLEMENT_USER_SUFFIX,
    LoopDetector,
    _artifact_label,
    assistant_prefill,
    drop_thrash_assistants,
    original_user_text,
    prior_assistant_is_thrash,
    steer_messages,
)


class TestLabels(unittest.TestCase):
    def test_racing_not_driving_after_steer(self):
        msgs = [{"role": "user", "content": "create a web based racing game"}]
        steered = steer_messages(msgs)
        self.assertEqual(original_user_text(steered), "create a web based racing game")
        pf = assistant_prefill(msgs)  # raw client messages
        self.assertIn("racing", pf.lower())
        self.assertNotIn("driving", pf.lower())
        # also after steer if we only use original_user_text
        pf2 = assistant_prefill(steered)
        self.assertIn("racing", pf2.lower())
        self.assertNotIn("driving", pf2.lower())

    def test_suffix_has_no_genre_words(self):
        low = IMPLEMENT_USER_SUFFIX.lower()
        for w in ("driving", "racing", "adventure"):
            self.assertNotIn(w, low)


class TestThrashDrop(unittest.TestCase):
    def test_drops_plan_essay_assistant(self):
        thrash = (
            "Writing a complete single-file HTML driving game now.\n\n"
            "```html\n<!DOCTYPE html>\n<style>body{}</style>\n"
            "The user wants a web-based racing game\n"
            "So plan:\n- Car sprite\n- Drift mode toggle that changes control scheme\n"
            "Alright let's assemble the game\n"
            "The user wants drift mode toggle that changes control scheme\n"
        )
        msgs = [
            {"role": "user", "content": "create a web based racing game"},
            {"role": "assistant", "content": thrash},
            {"role": "user", "content": "create a web racing game"},
        ]
        self.assertTrue(prior_assistant_is_thrash(msgs))
        cleaned = drop_thrash_assistants(msgs)
        self.assertFalse(any(m["role"] == "assistant" for m in cleaned))
        # After drop, prefill applies again with correct genre
        pf = assistant_prefill(cleaned)
        self.assertIn("racing", pf.lower())
        self.assertIn("const canvas", pf)

    def test_keeps_good_assistant(self):
        good = (
            "```html\n<script>\n"
            "const canvas=document.getElementById('c');\n"
            "const ctx=canvas.getContext('2d');\n"
            "function loop(){requestAnimationFrame(loop);ctx.fillRect(0,0,10,10);}\n"
            "addEventListener('keydown', e => {});\n"
            "loop();\n"
            "</script>\n```\n"
        )
        msgs = [
            {"role": "user", "content": "create a web racing game"},
            {"role": "assistant", "content": good},
        ]
        self.assertFalse(prior_assistant_is_thrash(msgs))


class TestDetector(unittest.TestCase):
    def test_stops_user_wants_plan_loop(self):
        text = (
            "```html\n<style></style>\n<body>\n"
            "The user wants a web-based racing game with bonuses.\n"
            "So plan: car sprite and drift mode.\n"
            "Alright let's assemble code step1 setup canvas.\n"
            "The user wants drift mode toggle that changes control scheme:\n"
            "Normal steering left right. Alright let's implement the game:\n"
            "So plan again carefully...\n"
        )
        det = LoopDetector(expect_code=True)
        reason = None
        for i in range(0, len(text), 40):
            reason = det.feed(text[i : i + 40])
            if reason:
                break
        self.assertIsNotNone(reason)

    def test_stops_outrun_philosophy_essay(self):
        """Real failure: partial JS then ## Foundational Principles word-salad."""
        shell = (
            "Writing a complete single-file HTML driving game now.\n\n"
            "```html\n<!DOCTYPE html>\n<script>\n"
            "const canvas = document.getElementById('c');\n"
            "const ctx = canvas.getContext('2d');\n"
            "addEventListener('keydown', e => {});\n"
            "function drawRoad(now) { const vpX = 380;\n"
            "  // broken\n}\n"
            "Then we shall do it.\n\n"
            "---\n\n"
            "## The requested out-run driving experience\n\n"
            "We'll craft a self-contained HTML file.\n\n"
            "**Key features:**\n\n"
            "- Road extends from horizon\n\n"
            "## Novel approach\n\n"
            "Instead of conventional rendering pipeline with seamless integration "
            "of perceptual cues facilitated by coherent narrative environments "
            "where users actively participate within continuously unfolding "
            "scenarios shaped dynamically through responsive feedback systems "
            "reacting real-time inputs generated via direct manipulation "
            "leveraging established web standards enabling robust scalable "
            "solutions flexible enough adapting evolving requirements.\n\n"
            "## Foundational Principles\n\n"
            "We believe that immersive interactive experiences arise from "
            "seamless integration of perceptual cues facilitated by coherent "
            "narrative environments transcending traditional boundaries "
            "imposed by geographical distances socioeconomic barriers.\n"
        )
        det = LoopDetector(expect_code=True)
        reason = None
        for i in range(0, len(shell), 50):
            reason = det.feed(shell[i : i + 50])
            if reason:
                break
        self.assertIsNotNone(reason, "must stop philosophy essay early")
        self.assertTrue(
            any(
                x in reason
                for x in (
                    "markdown_section",
                    "essay_thrash",
                    "prose_collapse",
                    "plan_thrash",
                    "shall_essay",
                )
            ),
            reason,
        )
        # Must stop well before full essay finishes
        self.assertLess(len(det.text), len(shell))

    def test_allows_real_js_continuation(self):
        text = (
            "```html\n<script>\n"
            "const canvas = document.getElementById('c');\n"
            "const ctx = canvas.getContext('2d');\n"
            "const keys = {};\n"
            "addEventListener('keydown', e => { keys[e.code] = true; });\n"
            "let x = 100, speed = 2;\n"
            "function update(dt){\n"
            "  if (keys['ArrowLeft']) x -= speed * dt;\n"
            "  if (keys['ArrowRight']) x += speed * dt;\n"
            "}\n"
            "function loop(t){\n"
            "  requestAnimationFrame(loop);\n"
            "  ctx.clearRect(0,0,canvas.width,canvas.height);\n"
            "  ctx.fillStyle = '#0f0';\n"
            "  ctx.fillRect(x,200,20,30);\n"
            "}\n"
            "loop();\n"
        )
        det = LoopDetector(expect_code=True)
        reason = None
        for i in range(0, len(text), 30):
            reason = det.feed(text[i : i + 30])
            if reason:
                break
        self.assertIsNone(reason, reason)

    def test_prefill_has_js_hooks(self):
        pf = assistant_prefill(
            [{"role": "user", "content": "create a web based racing game"}]
        )
        self.assertIn("racing", pf.lower())
        self.assertIn("getContext", pf)
        self.assertIn("keydown", pf)


if __name__ == "__main__":
    unittest.main()
