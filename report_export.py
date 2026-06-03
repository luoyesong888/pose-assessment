from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _read_image_base64(path: str) -> str:
    if not path:
        return ""
    image_path = Path(path)
    if not image_path.exists():
        return ""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _load_image(path: str, fallback_path: str | None = None) -> Image.Image | None:
    for candidate in (path, fallback_path):
        if not candidate:
            continue
        image_path = Path(candidate)
        if image_path.exists():
            try:
                return Image.open(image_path).convert("RGB")
            except Exception:
                continue
    return None


def _font_candidates() -> List[str]:
    return [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = _font_candidates()
    if bold:
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ] + candidates
    for path in candidates:
        p = Path(path)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    lines: List[str] = []
    for paragraph in (text or "").split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for ch in paragraph:
            test = current + ch
            if draw.textlength(test, font=font) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines or [""]


def build_report_html(report: Dict[str, Any]) -> str:
    sections = report.get("report_sections", {})
    summary = report.get("summary", {})
    muscle_map = report.get("muscle_map", {})
    front_result = report.get("front_result", {})
    side_result = report.get("side_result", {})

    def img_block(path: str, fallback_path: str, label: str) -> str:
        b64 = _read_image_base64(path) or _read_image_base64(fallback_path)
        if not b64:
            return f"<div class='img-missing'>{html.escape(label)}</div>"
        return f"""
        <figure class="img-card">
            <img src="data:image/jpeg;base64,{b64}" alt="{html.escape(label)}">
            <figcaption>{html.escape(label)}</figcaption>
        </figure>
        """

    def bullets(items: List[str]) -> str:
        if not items:
            return "<li>无</li>"
        return "".join(f"<li>{html.escape(item)}</li>" for item in items)

    primary = muscle_map.get("primary_muscles", [])
    secondary = muscle_map.get("secondary_muscles", [])

    return f"""
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(sections.get("title", "Report"))}</title>
<style>
body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    background: #f5efe6;
    color: #1c1510;
}}
.page {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px 48px;
}}
.hero {{
    background: #fdfaf6;
    border: 1px solid rgba(28,21,16,0.09);
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 16px 40px rgba(28,21,16,0.06);
    margin-bottom: 20px;
}}
.title {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 34px;
    margin: 0 0 8px;
}}
.sub {{
    color: #7a6e65;
    line-height: 1.7;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 18px 0 22px;
}}
.card {{
    background: #fff;
    border: 1px solid rgba(28,21,16,0.09);
    border-radius: 18px;
    padding: 16px;
}}
.label {{
    font-size: 12px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #7a6e65;
    margin-bottom: 8px;
}}
.value {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 22px;
}}
.section {{
    background: #fdfaf6;
    border: 1px solid rgba(28,21,16,0.09);
    border-radius: 22px;
    padding: 22px;
    margin: 18px 0;
}}
.section h2 {{
    margin: 0 0 14px;
    font-family: Georgia, "Times New Roman", serif;
}}
.section ul {{
    margin: 0;
    padding-left: 20px;
    line-height: 1.8;
}}
.images {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}}
.img-card {{
    margin: 0;
    background: #fff;
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(28,21,16,0.09);
}}
.img-card img {{
    width: 100%;
    display: block;
}}
.img-card figcaption {{
    padding: 10px 14px;
    color: #7a6e65;
    font-size: 12px;
}}
.img-missing {{
    min-height: 180px;
    display: grid;
    place-items: center;
    background: #fff;
    color: #7a6e65;
    border: 1px dashed rgba(28,21,16,0.2);
    border-radius: 18px;
}}
@media print {{
    body {{
        background: white;
    }}
    .page {{
        max-width: none;
        padding: 0;
    }}
    .hero, .section {{
        box-shadow: none;
        break-inside: avoid;
    }}
}}
</style>
</head>
<body>
<div class="page">
    <div class="hero">
        <div class="title">{html.escape(sections.get("title", "Therapist Report"))}</div>
        <div class="sub">{html.escape(report.get("patient_name") or report.get("patient_code") or "")} · {html.escape(report.get("created_at", ""))}</div>
        <div class="grid">
            <div class="card"><div class="label">ACL Risk</div><div class="value">{html.escape(str(summary.get("acl_risk", {}).get("label_zh", "-")))} · {summary.get("acl_risk", {}).get("score", 0):.2f}</div></div>
            <div class="card"><div class="label">Muscle Targets</div><div class="value">{html.escape((muscle_map.get("dominant_targets") or ['-'])[0])}</div></div>
            <div class="card"><div class="label">Chain Notes</div><div class="value">{len(summary.get("kinetic_chain", []))} items</div></div>
        </div>
        <div class="images">
            {img_block(front_result.get("annotated_path", ""), report.get("front_path", ""), "Front annotated view")}
            {img_block(side_result.get("annotated_path", ""), report.get("side_path", ""), "Side annotated view")}
        </div>
    </div>
    <div class="section">
        <h2>{html.escape(sections.get("overview_title", "Assessment Summary"))}</h2>
        <ul>{bullets(sections.get("overview_lines", []))}</ul>
    </div>
    <div class="section">
        <h2>{html.escape(sections.get("risk_title", "ACL Risk Assessment"))}</h2>
        <ul>{bullets(sections.get("risk_lines", []))}</ul>
    </div>
    <div class="section">
        <h2>{html.escape(sections.get("muscle_title", "Muscle Function Hypothesis"))}</h2>
        <ul>{bullets(sections.get("muscle_lines", []))}</ul>
    </div>
    <div class="section">
        <h2>{html.escape(sections.get("plan_title", "Personalized Rehab Plan"))}</h2>
        <ul>{bullets(sections.get("plan_lines", []))}</ul>
    </div>
    <div class="section">
        <h2>{html.escape(sections.get("notes_title", "Therapist Notes"))}</h2>
        <ul>{bullets(sections.get("notes_lines", []))}</ul>
    </div>
    <div class="section">
        <h2>{html.escape(sections.get("followup_title", "Follow-up Schedule"))}</h2>
        <ul>{bullets(sections.get("followup_lines", []))}</ul>
    </div>
</div>
</body>
</html>
"""


def _make_page(width: int = 1240, height: int = 1754, background=(245, 239, 230)) -> Image.Image:
    return Image.new("RGB", (width, height), background)


def _draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy, font, fill, max_width, line_gap=8):
    x, y = xy
    for line in _wrap_text(text, font, max_width, draw):
        if line:
            draw.text((x, y), line, font=font, fill=fill)
            bbox = draw.textbbox((x, y), line, font=font)
            y = bbox[3] + line_gap
        else:
            y += font.size + line_gap
    return y


def build_report_pdf_bytes(report: Dict[str, Any]) -> bytes:
    sections = report.get("report_sections", {})
    summary = report.get("summary", {})
    muscle_map = report.get("muscle_map", {})
    front_result = report.get("front_result", {})
    side_result = report.get("side_result", {})
    title_font = _load_font(36, bold=True)
    heading_font = _load_font(26, bold=True)
    body_font = _load_font(20)
    small_font = _load_font(16)

    pages: List[Image.Image] = []
    page = _make_page()
    draw = ImageDraw.Draw(page)
    margin = 72
    y = margin

    y = _draw_wrapped(draw, sections.get("title", "Therapist Report"), (margin, y), title_font, (28, 21, 16), 1100)
    y += 10
    y = _draw_wrapped(
        draw,
        f"{report.get('patient_name') or report.get('patient_code') or ''}   {report.get('created_at', '')}",
        (margin, y),
        small_font,
        (122, 110, 101),
        1100,
    )

    stats = [
        f"ACL Risk: {summary.get('acl_risk', {}).get('label_zh', '-')}",
        f"Muscle Targets: {(muscle_map.get('dominant_targets') or ['-'])[0]}",
        f"Chain Items: {len(summary.get('kinetic_chain', []))}",
    ]
    sy = y + 18
    for idx, stat in enumerate(stats):
        box_x = margin + idx * 360
        draw.rounded_rectangle((box_x, sy, box_x + 330, sy + 84), radius=18, fill="white", outline=(226, 216, 206), width=1)
        draw.text((box_x + 18, sy + 14), stat.split(":")[0], font=small_font, fill=(122, 110, 101))
        draw.text((box_x + 18, sy + 40), stat.split(": ", 1)[1], font=body_font, fill=(28, 21, 16))

    y = sy + 110
    max_w = page.width - 2 * margin

    image_y = y
    image_h = 280
    image_w = (page.width - 2 * margin - 18) // 2
    front_img = _load_image(front_result.get("annotated_path", ""), report.get("front_path", ""))
    side_img = _load_image(side_result.get("annotated_path", ""), report.get("side_path", ""))
    if front_img or side_img:
        needed_space = image_h + 26
        if image_y + needed_space > page.height - margin:
            pages.append(page)
            page = _make_page()
            draw = ImageDraw.Draw(page)
            image_y = margin
        draw.text((margin, image_y - 24), "Annotated Views", font=heading_font, fill=(28, 21, 16))
        for idx, (img, label) in enumerate(
            [
                (front_img, "Front annotated view"),
                (side_img, "Side annotated view"),
            ]
        ):
            box_x = margin + idx * (image_w + 18)
            draw.rounded_rectangle(
                (box_x, image_y, box_x + image_w, image_y + image_h),
                radius=18,
                fill="white",
                outline=(226, 216, 206),
                width=1,
            )
            if img is not None:
                fitted = ImageOps.contain(img, (image_w - 14, image_h - 38))
                px = box_x + (image_w - fitted.width) // 2
                py = image_y + 8 + ((image_h - 38) - fitted.height) // 2
                page.paste(fitted, (px, py))
            else:
                draw.text((box_x + 18, image_y + 24), label, font=body_font, fill=(122, 110, 101))
            draw.text((box_x + 14, image_y + image_h - 26), label, font=small_font, fill=(122, 110, 101))
        y = image_y + image_h + 24

    def add_section(title: str, items: List[str]):
        nonlocal page, draw, y, pages
        needed_space = 120 + max(1, len(items)) * 34
        if y + needed_space > page.height - margin:
            pages.append(page)
            page = _make_page()
            draw = ImageDraw.Draw(page)
            y = margin
        draw.rounded_rectangle((margin, y, page.width - margin, y + needed_space), radius=22, fill=(253, 250, 246), outline=(226, 216, 206), width=1)
        inner_x = margin + 24
        inner_y = y + 20
        inner_y = _draw_wrapped(draw, title, (inner_x, inner_y), heading_font, (28, 21, 16), max_w - 48)
        inner_y += 8
        if not items:
            items = ["无"]
        for item in items:
            inner_y = _draw_wrapped(draw, f"• {item}", (inner_x, inner_y), body_font, (28, 21, 16), max_w - 48)
            inner_y += 4
        y += needed_space + 18

    add_section(sections.get("overview_title", "Assessment Summary"), sections.get("overview_lines", []))
    add_section(sections.get("risk_title", "ACL Risk Assessment"), sections.get("risk_lines", []))
    add_section(sections.get("muscle_title", "Muscle Function Hypothesis"), sections.get("muscle_lines", []))
    add_section(sections.get("chain_title", "Kinetic Chain Analysis"), sections.get("chain_lines", []))
    add_section(sections.get("plan_title", "Personalized Rehab Plan"), sections.get("plan_lines", []))
    add_section(sections.get("notes_title", "Therapist Notes"), sections.get("notes_lines", []))
    add_section(sections.get("followup_title", "Follow-up Schedule"), sections.get("followup_lines", []))

    pages.append(page)
    buffer = io.BytesIO()
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:])
    return buffer.getvalue()
