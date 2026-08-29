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

    image_results = report.get("image_results", [])
    if image_results:
        def result_img_block(result, label):
            b64 = _read_image_base64(result.get("annotated_path", "")) or _read_image_base64(result.get("source_path", ""))
            if not b64:
                return f"<div class='img-missing'>{html.escape(label)}</div>"
            return f"<figure class=\"img-card\"><img src=\"data:image/jpeg;base64,{b64}\" alt=\"{html.escape(label)}\"><figcaption>{html.escape(label)}</figcaption></figure>"

        images_html = "".join(
            result_img_block(result, f"Image {index + 1} · {result.get('detected_view', 'auto')} view")
            for index, result in enumerate(image_results)
        )
    else:
        images_html = (
            img_block(front_result.get("annotated_path", ""), report.get("front_path", ""), "Front annotated view")
            + img_block(side_result.get("annotated_path", ""), report.get("side_path", ""), "Side annotated view")
        )

    def bullets(items: List[str]) -> str:
        if not items:
            return "<li>无</li>"
        return "".join(f"<li>{html.escape(item)}</li>" for item in items)

    primary = muscle_map.get("primary_muscles", [])
    secondary = muscle_map.get("secondary_muscles", [])
    acl_display = summary.get("acl_risk", {}).get("label_zh", "未评估（需动态测试）")
    coverage_display = summary.get("view_coverage", {}).get("label_zh", "未知")
    movement_display = summary.get("movement_screening", {}).get("label_zh", "待评估")
    confirmed_block = ""
    if sections.get("confirmed_plan_lines"):
        confirmed_block = f"""
    <div class="section">
        <h2>{html.escape(sections.get("confirmed_plan_title", "Confirmed Improvement Plan"))}</h2>
        <ul>{bullets(sections.get("confirmed_plan_lines", []))}</ul>
    </div>
"""
    regional_block = "".join(
        f"""
    <div class="section">
        <h2>{html.escape(item.get('title', ''))}</h2>
        <ul>{bullets(item.get('lines', []))}</ul>
    </div>
"""
        for item in sections.get("regional_sections", [])
    )

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
    background: #f7f4fa;
    color: #1a1a2e;
}}
.page {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px 48px;
}}
.hero {{
    background: #ffffff;
    border: 1px solid #e7e8ef;
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 16px 40px rgba(16,24,40,0.07);
    margin-bottom: 20px;
}}
.title {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 34px;
    margin: 0 0 8px;
    color: #792f9b;
}}
.sub {{
    color: #6b7280;
    line-height: 1.7;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 18px 0 22px;
}}
.card {{
    background: #fafafc;
    border: 1px solid #e7e8ef;
    border-radius: 18px;
    padding: 16px;
}}
.label {{
    font-size: 12px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 8px;
}}
.value {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 22px;
}}
.section {{
    background: #ffffff;
    border: 1px solid #e7e8ef;
    border-radius: 22px;
    padding: 22px;
    margin: 18px 0;
}}
.section h2 {{
    margin: 0 0 14px;
    font-family: Georgia, "Times New Roman", serif;
    color: #792f9b;
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
    background: #fafafc;
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid #e7e8ef;
}}
.img-card img {{
    width: 100%;
    display: block;
}}
.img-card figcaption {{
    padding: 10px 14px;
    color: #6b7280;
    font-size: 12px;
}}
.img-missing {{
    min-height: 180px;
    display: grid;
    place-items: center;
    background: #fafafc;
    color: #6b7280;
    border: 1px dashed #d0d2dd;
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
            <div class="card"><div class="label">Capture Coverage</div><div class="value">{html.escape(str(coverage_display))}</div></div>
            <div class="card"><div class="label">Key Observation</div><div class="value">{html.escape(str(movement_display))}</div></div>
            <div class="card"><div class="label">Priorities</div><div class="value">{len(report.get("recommendation_options", []))} items</div></div>
        </div>
        <div class="images">
            {images_html}
        </div>
    </div>
    <div class="section">
        <h2>{html.escape(sections.get("overview_title", "Assessment Summary"))}</h2>
        <ul>{bullets(sections.get("overview_lines", []))}</ul>
    </div>
    {regional_block}
    <div class="section">
        <h2>{html.escape(sections.get("risk_title", "ACL Screening Limits"))}</h2>
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
        <h2>{html.escape(sections.get("evidence_title", "RAG Evidence & Limits"))}</h2>
        <ul>{bullets(sections.get("evidence_lines", []))}</ul>
    </div>
    {confirmed_block}
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


def _make_page(width: int = 1240, height: int = 1754, background=(244, 245, 250)) -> Image.Image:
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

    y = _draw_wrapped(draw, sections.get("title", "Therapist Report"), (margin, y), title_font, (26, 26, 46), 1100)
    y += 10
    y = _draw_wrapped(
        draw,
        f"{report.get('patient_name') or report.get('patient_code') or ''}   {report.get('created_at', '')}",
        (margin, y),
        small_font,
        (107, 114, 128),
        1100,
    )

    stats = [
        f"Capture Coverage: {summary.get('view_coverage', {}).get('label_zh', '未知')}",
        f"Key Observation: {summary.get('movement_screening', {}).get('label_zh', '待评估')}",
        f"Priorities: {len(report.get('recommendation_options', []))}",
    ]
    sy = y + 18
    for idx, stat in enumerate(stats):
        box_x = margin + idx * 360
        draw.rounded_rectangle((box_x, sy, box_x + 330, sy + 84), radius=18, fill="white", outline=(231, 232, 239), width=1)
        draw.text((box_x + 18, sy + 14), stat.split(":")[0], font=small_font, fill=(107, 114, 128))
        draw.text((box_x + 18, sy + 40), stat.split(": ", 1)[1], font=body_font, fill=(26, 26, 46))

    y = sy + 110
    max_w = page.width - 2 * margin

    image_y = y
    image_h = 280
    image_w = (page.width - 2 * margin - 18) // 2
    image_results = report.get("image_results", [])
    if image_results:
        image_items = [
            (_load_image(r.get("annotated_path", ""), r.get("source_path", "")), f"Image {index + 1} · {r.get('detected_view', 'auto')} view")
            for index, r in enumerate(image_results[:4])
        ]
    else:
        image_items = [
            (_load_image(front_result.get("annotated_path", ""), report.get("front_path", "")), "Front annotated view"),
            (_load_image(side_result.get("annotated_path", ""), report.get("side_path", "")), "Side annotated view"),
        ]
    image_items = [(img, label) for img, label in image_items if img is not None]
    if image_items:
        needed_space = image_h + 26
        if image_y + needed_space > page.height - margin:
            pages.append(page)
            page = _make_page()
            draw = ImageDraw.Draw(page)
            image_y = margin
        draw.text((margin, image_y - 24), "Annotated Views", font=heading_font, fill=(26, 26, 46))
        for idx, (img, label) in enumerate(image_items):
            col, row = idx % 2, idx // 2
            box_x = margin + col * (image_w + 18)
            box_y = image_y + row * (image_h + 24)
            if box_y + image_h > page.height - margin:
                break
            draw.rounded_rectangle(
                (box_x, box_y, box_x + image_w, box_y + image_h),
                radius=18,
                fill="white",
                outline=(231, 232, 239),
                width=1,
            )
            fitted = ImageOps.contain(img, (image_w - 14, image_h - 38))
            px = box_x + (image_w - fitted.width) // 2
            py = box_y + 8 + ((image_h - 38) - fitted.height) // 2
            page.paste(fitted, (px, py))
            draw.text((box_x + 14, box_y + image_h - 26), label, font=small_font, fill=(107, 114, 128))
        rows = (len(image_items) + 1) // 2
        y = image_y + rows * (image_h + 24)

    def add_section(title: str, items: List[str]):
        nonlocal page, draw, y, pages
        needed_space = 120 + max(1, len(items)) * 34
        if y + needed_space > page.height - margin:
            pages.append(page)
            page = _make_page()
            draw = ImageDraw.Draw(page)
            y = margin
        draw.rounded_rectangle((margin, y, page.width - margin, y + needed_space), radius=22, fill=(255, 255, 255), outline=(231, 232, 239), width=1)
        inner_x = margin + 24
        inner_y = y + 20
        inner_y = _draw_wrapped(draw, title, (inner_x, inner_y), heading_font, (26, 26, 46), max_w - 48)
        inner_y += 8
        if not items:
            items = ["无"]
        for item in items:
            inner_y = _draw_wrapped(draw, f"• {item}", (inner_x, inner_y), body_font, (26, 26, 46), max_w - 48)
            inner_y += 4
        y += needed_space + 18

    add_section(sections.get("overview_title", "Assessment Summary"), sections.get("overview_lines", []))
    for regional in sections.get("regional_sections", []):
        add_section(regional.get("title", "Regional Assessment"), regional.get("lines", []))
    add_section(sections.get("risk_title", "ACL Screening Limits"), sections.get("risk_lines", []))
    add_section(sections.get("muscle_title", "Muscle Function Hypothesis"), sections.get("muscle_lines", []))
    add_section(sections.get("chain_title", "Kinetic Chain Analysis"), sections.get("chain_lines", []))
    add_section(sections.get("plan_title", "Personalized Rehab Plan"), sections.get("plan_lines", []))
    add_section(sections.get("evidence_title", "RAG Evidence & Limits"), sections.get("evidence_lines", []))
    if sections.get("confirmed_plan_lines"):
        add_section(sections.get("confirmed_plan_title", "Confirmed Improvement Plan"), sections.get("confirmed_plan_lines", []))
    add_section(sections.get("notes_title", "Therapist Notes"), sections.get("notes_lines", []))
    add_section(sections.get("followup_title", "Follow-up Schedule"), sections.get("followup_lines", []))

    pages.append(page)
    buffer = io.BytesIO()
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:])
    return buffer.getvalue()
