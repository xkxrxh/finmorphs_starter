import base64
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

from io import BytesIO
import html
import matplotlib.pyplot as plt
import av
import numpy as np
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch


if "finmorphs_evidence" not in st.session_state:
    st.session_state["finmorphs_evidence"] = {
        "photos": [],
        "videos": [],
        "voice_notes": [],
    }
if "finmorphs_session_items" not in st.session_state:
    st.session_state["finmorphs_session_items"] = []
if "finmorphs_tour" not in st.session_state:
    st.session_state["finmorphs_tour"] = {
        "active": False,
        "started_at": None,
        "notes": [],
        "observations": [],
        "assistant_response": "",
    }
if "finmorphs_gemini_files" not in st.session_state:
    st.session_state["finmorphs_gemini_files"] = {}
if "finmorphs_observation_review" not in st.session_state:
    st.session_state["finmorphs_observation_review"] = {
        "raw": "",
        "key_observations": [],
        "assumptions_to_review": [],
        "unanswered_questions": [],
    }

evidence = st.session_state["finmorphs_evidence"]
for evidence_type in ("photos", "videos"):
    evidence[evidence_type] = [
        item if isinstance(item, dict) else {
            "name": item.name,
            "data": item.getvalue(),
            "type": item.type,
        }
        for item in evidence[evidence_type]
    ]
evidence["voice_notes"] = [
    note if isinstance(note, dict) else {
        "id": str(time.time_ns()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audio": note.getvalue(),
        "type": note.type,
        "transcript": "",
        "status": "saved",
        "audio_hash": hashlib.sha256(note.getvalue()).hexdigest(),
    }
    for note in evidence["voice_notes"]
]


def register_session_item(kind, name, data, mime_type, source, transcript=""):
    """Register one reviewable item in the unified inspection session."""
    item_id = hashlib.sha256(
        f"{kind}:{name}:{hashlib.sha256(data).hexdigest()}".encode("utf-8")
    ).hexdigest()[:20]
    items = st.session_state["finmorphs_session_items"]
    if any(item["id"] == item_id for item in items):
        return item_id
    items.append({
        "id": item_id,
        "kind": kind,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": name,
        "status": "saved",
        "source": source,
        "mime_type": mime_type or "application/octet-stream",
        "data": data,
        "transcript": transcript,
    })
    return item_id


for photo in evidence["photos"]:
    register_session_item(
        "photo", photo["name"], photo["data"], photo.get("type"), "photo uploader"
    )
for video in evidence["videos"]:
    register_session_item(
        "video", video["name"], video["data"], video.get("type"), "video uploader"
    )
for note in evidence["voice_notes"]:
    register_session_item(
        "voice", note["id"], note["audio"], note.get("type"),
        "voice recorder", note.get("transcript", "")
    )


class GlassesVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.max_frames = 600

    def recv(self, frame):
        if len(self.frames) < self.max_frames:
            self.frames.append(frame.to_ndarray(format="bgr24"))
        return frame


def encode_captured_video(frames, fps=20):
    """Encode captured camera frames as an MP4 byte stream."""
    output = BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for frame_data in frames:
            frame = av.VideoFrame.from_ndarray(
                np.asarray(frame_data),
                format="bgr24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    output.seek(0)
    return output.getvalue()


def parse_json_response(text):
    """Extract a JSON object from a model response, including fenced JSON."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def saved_evidence_summary(evidence):
    """Return metadata only, keeping prompts small and excluding media bytes."""
    return {
        "photos": [item["name"] for item in evidence["photos"]],
        "videos": [item["name"] for item in evidence["videos"]],
        "voice_questions": [
            {
                "id": note["id"],
                "timestamp": note["timestamp"],
                "transcript": note.get("transcript", ""),
            }
            for note in evidence["voice_notes"]
        ],
        "session_registry": [
            {
                "id": item["id"],
                "kind": item["kind"],
                "filename": item["filename"],
                "timestamp": item["timestamp"],
                "source": item["source"],
                "status": item["status"],
            }
            for item in st.session_state.get("finmorphs_session_items", [])
        ],
    }


def append_saved_media_to_parts(client, parts, evidence):
    """Attach all persisted session media to a Gemini request."""
    for photo in evidence["photos"]:
        parts.append({
            "type": "image",
            "data": base64.b64encode(photo["data"]).decode("utf-8"),
            "mime_type": photo.get("type") or "image/jpeg",
        })
    for note in evidence["voice_notes"]:
        parts.append({
            "type": "audio",
            "data": base64.b64encode(note["audio"]).decode("utf-8"),
            "mime_type": note.get("type") or "audio/wav",
        })
    for video in evidence["videos"]:
        video_data = video["data"]
        video_key = hashlib.sha256(video_data).hexdigest()
        cached_video = st.session_state["finmorphs_gemini_files"].get(video_key)
        if cached_video:
            parts.append({
                "type": "video",
                "uri": cached_video["uri"],
                "mime_type": cached_video["mime_type"],
            })
            continue

        temp_path = None
        suffix = os.path.splitext(video["name"])[1] or ".mp4"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(video_data)
                temp_path = temp_file.name
            with st.spinner(f"Preparing saved video {video['name']}…"):
                uploaded_video = client.files.upload(file=temp_path)
                deadline = time.time() + 180
                while (
                    uploaded_video.state
                    and uploaded_video.state.name == "PROCESSING"
                    and time.time() < deadline
                ):
                    time.sleep(1)
                    uploaded_video = client.files.get(name=uploaded_video.name)
                if (
                    not uploaded_video.state
                    or uploaded_video.state.name != "ACTIVE"
                ):
                    raise RuntimeError(
                        f"Gemini did not finish processing {video['name']} within 3 minutes."
                    )
            mime_type = uploaded_video.mime_type or video.get("type") or "video/mp4"
            st.session_state["finmorphs_gemini_files"][video_key] = {
                "uri": uploaded_video.uri,
                "mime_type": mime_type,
            }
            parts.append({
                "type": "video",
                "uri": uploaded_video.uri,
                "mime_type": mime_type,
            })
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def create_pdf_chart(title, labels, values, ylabel, chart_type="bar"):
    """Create a chart in memory for embedding in the PDF."""
    fig, ax = plt.subplots(figsize=(7, 3.6))

    if chart_type == "pie":
        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
    elif chart_type == "line":
        ax.plot(labels, values, marker="o", linewidth=2.5, color="#1d4ed8")
        ax.fill_between(range(len(values)), values, alpha=0.12, color="#1d4ed8")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    elif chart_type == "horizontal":
        ax.barh(labels, values, color="#2563eb")
        ax.set_xlabel(ylabel)
        ax.grid(axis="x", alpha=0.25)
    else:
        ax.bar(labels, values, color="#2563eb")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)

    ax.set_title(title)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def create_pdf_report(report_text, scenarios, finance):
    """Build a downloadable PDF with financial charts and report text."""
    pdf_buffer = BytesIO()
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="FinMorphsTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#1d4ed8"),
        alignment=TA_CENTER,
        spaceAfter=18,
    ))

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="FinMorphs Financial Intelligence Report",
    )

    elements = [
        Paragraph("FinMorphs", styles["FinMorphsTitle"]),
        Paragraph("Financial Intelligence Report", styles["Heading1"]),
        Spacer(1, 10),
    ]

    # Convert report lines into safe PDF paragraphs.
    for line in report_text.splitlines():
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 5))
            continue

        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
            elements.append(Paragraph(html.escape(heading), styles["Heading2"]))
        elif line.startswith("|"):
            # Keep Markdown table rows readable in the PDF.
            elements.append(Paragraph(
                html.escape(line.replace("|", "   •   ")),
                styles["BodyText"]
            ))
        elif line.startswith("- "):
            elements.append(Paragraph(
                "• " + html.escape(line[2:]),
                styles["BodyText"]
            ))
        else:
            elements.append(Paragraph(
                html.escape(line),
                styles["BodyText"]
            ))

        elements.append(Spacer(1, 5))

    elements.append(PageBreak())
    elements.append(Paragraph("Financial Visualizations", styles["Heading1"]))
    elements.append(Spacer(1, 10))

    scenario_labels = [row["Scenario"] for row in scenarios]
    scenario_values = [
        row["Contribution before fixed costs (₹)"]
        for row in scenarios
    ]

    chart1 = create_pdf_chart(
        "Scenario Contribution",
        scenario_labels,
        scenario_values,
        "Contribution (₹)"
    )
    elements.append(Paragraph("1. Scenario Comparison", styles["Heading2"]))
    elements.append(Image(chart1, width=6.7 * inch, height=3.4 * inch))
    elements.append(Spacer(1, 16))

    cost_labels = ["Purchase cost", "Other variable cost", "Selling price"]
    cost_values = [
        finance["unit_cost"],
        finance["other_cost"],
        finance["selling_price"],
    ]

    chart2 = create_pdf_chart(
        "Cost and Selling Price per Unit",
        cost_labels,
        cost_values,
        "Rupees per unit"
    )
    elements.append(Paragraph("2. Unit Economics", styles["Heading2"]))
    elements.append(Image(chart2, width=6.7 * inch, height=3.4 * inch))
    elements.append(Spacer(1, 16))

    # Only show a pie chart when all components are valid and non-negative.
    pie_values = [
        finance["unit_cost"],
        finance["other_cost"],
        finance["contribution"],
    ]

    if finance["selling_price"] > 0 and all(v >= 0 for v in pie_values):
        if sum(pie_values) > 0:
            chart3 = create_pdf_chart(
                "Selling Price Breakdown",
                ["Purchase cost", "Other variable cost", "Contribution"],
                pie_values,
                "Amount (₹)",
                chart_type="pie",
            )
            elements.append(
                Paragraph("3. Selling Price Breakdown", styles["Heading2"])
            )
            elements.append(
                Image(chart3, width=6.3 * inch, height=3.4 * inch)
            )

    projection_labels = [f"Month {month}" for month in range(1, 7)]
    projection_values = [
        finance["monthly_contribution"] * month
        for month in range(1, 7)
    ]
    chart4 = create_pdf_chart(
        "Cumulative Contribution Projection",
        projection_labels,
        projection_values,
        "Cumulative contribution (₹)",
        chart_type="line",
    )
    elements.append(Paragraph("4. Six-Month Contribution Projection", styles["Heading2"]))
    elements.append(Image(chart4, width=6.7 * inch, height=3.4 * inch))
    elements.append(Spacer(1, 16))

    summary_labels = [
        "Monthly contribution",
        "Fixed costs",
        "After fixed costs",
    ]
    summary_values = [
        finance["monthly_contribution"],
        finance["fixed_costs"],
        finance["estimated_after_fixed"],
    ]
    chart5 = create_pdf_chart(
        "Monthly Financial Position",
        summary_labels,
        summary_values,
        "Amount (₹)",
        chart_type="horizontal",
    )
    elements.append(Paragraph("5. Monthly Financial Position", styles["Heading2"]))
    elements.append(Image(chart5, width=6.7 * inch, height=3.4 * inch))

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(
            A4[0] / 2,
            0.35 * inch,
            f"FinMorphs • Page {document.page}"
        )
        canvas.restoreState()

    doc.build(
        elements,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = "gemini-3.8-flash"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5").strip()

NEWS_FEEDS = {
    "Market news": "https://news.google.com/rss/search?q=stock+market+business&hl=en-IN&gl=IN&ceid=IN:en",
    "Economic news": "https://news.google.com/rss/search?q=economy+inflation+interest+rates&hl=en-IN&gl=IN&ceid=IN:en",
    "Financial news": "https://news.google.com/rss/search?q=finance+markets+companies&hl=en-IN&gl=IN&ceid=IN:en",
}


def fetch_live_news(max_items=5):
    """Fetch recent headlines from public RSS feeds without sending user data."""
    stories = []
    for category, feed_url in NEWS_FEEDS.items():
        try:
            request = urllib.request.Request(
                feed_url,
                headers={"User-Agent": "FinMorphs/1.0"},
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                root = ET.fromstring(response.read())
            for item in root.findall("./channel/item")[:max_items]:
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                published = item.findtext("pubDate", "").strip()
                if title and link:
                    stories.append({
                        "category": category,
                        "title": title,
                        "link": link,
                        "published": published,
                    })
        except (urllib.error.URLError, ET.ParseError, TimeoutError):
            continue
    return stories


def call_openai(prompt):
    """Request an optional independent analysis from OpenAI."""
    payload = json.dumps({
        "model": OPENAI_MODEL,
        "input": prompt,
        "max_output_tokens": 900,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        data = json.loads(response.read())
    return data.get("output_text", "").strip()


def call_anthropic(prompt):
    """Request an optional independent analysis from Anthropic."""
    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 900,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        data = json.loads(response.read())
    return "\n".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ).strip()

st.set_page_config(page_title="FinMorphs | Financial Intelligence", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1320px; padding-top: 1.25rem; padding-bottom: 3rem;}
.hero {padding: 1.55rem 1.7rem; border-radius: 22px; background: radial-gradient(circle at 85% 10%,#3b82f6 0,#1e3a8a 28%,#0f172a 75%); color: white; margin-bottom: .8rem; box-shadow: 0 12px 30px rgba(15,23,42,.16);}
.hero h1 {margin:0; font-size:2.35rem; letter-spacing:-.04em;} .hero p {margin:.45rem 0 0;color:#dbeafe;font-size:1.02rem;}
.hero-meta {display:flex;gap:.55rem;flex-wrap:wrap;margin-top:1.05rem;}
.hero-pill {border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.1);border-radius:999px;padding:.3rem .65rem;font-size:.78rem;color:#e0f2fe;}
.section-title {display:flex;align-items:center;gap:.65rem;margin-top:1.35rem;margin-bottom:.15rem;}
.section-number {display:inline-flex;align-items:center;justify-content:center;width:2rem;height:2rem;border-radius:10px;background:#dbeafe;color:#1d4ed8;font-weight:800;}
body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {color:#172033;}
.section-title h3 {margin:0;font-size:1.25rem;color:#172033 !important;}
.section-subtitle {color:#475569 !important;margin:.15rem 0 .75rem;font-size:.92rem;}
.form-card {border:1px solid #cbd5e1;border-radius:16px;padding:1rem 1.05rem .55rem;background:#ffffff;box-shadow:0 5px 18px rgba(30,64,175,.08);min-height:246px;color:#172033;}
.form-card h4 {margin:0 0 .15rem;color:#1e3a8a !important;font-size:1rem;}
.form-card p {margin:0 0 .7rem;color:#475569 !important;font-size:.82rem;}
[data-testid="stMetric"] {border:1px solid #cbd5e1;padding:.9rem;border-radius:14px;background:#ffffff;box-shadow:0 5px 15px rgba(15,23,42,.08);}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color:#172033 !important;}
.section-card {border:1px solid #bfdbfe;border-radius:14px;padding:1rem 1.1rem;margin:.5rem 0 1rem;background:#eff6ff;color:#172033;}
.hint {color:#475569 !important;font-size:.9rem;}
label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {color:#334155 !important;}
input, textarea, [data-baseweb="select"] > div {color:#172033 !important;}
input::placeholder, textarea::placeholder {color:#64748b !important;opacity:1;}
div.stButton > button, div.stDownloadButton > button {border-radius:10px;font-weight:700;min-height:2.65rem;}
div.stButton > button[kind="primary"] {background:linear-gradient(90deg,#2563eb,#1d4ed8);}
[data-testid="stExpander"] {border:1px solid #cbd5e1;border-radius:12px;}
@media (prefers-color-scheme: dark) {
  body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {color:#f8fafc;}
  .section-title h3 {color:#f8fafc !important;}
  .section-subtitle, .hint {color:#cbd5e1 !important;}
  .form-card {border-color:#475569;background:#1e293b;color:#f8fafc;}
  .form-card h4 {color:#bfdbfe !important;}
  .form-card p {color:#cbd5e1 !important;}
  [data-testid="stMetric"] {border-color:#475569;background:#1e293b;}
  [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color:#f8fafc !important;}
  .section-card {border-color:#3b82f6;background:#172554;color:#f8fafc;}
  label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {color:#e2e8f0 !important;}
  input, textarea, [data-baseweb="select"] > div {color:#f8fafc !important;}
  input::placeholder, textarea::placeholder {color:#94a3b8 !important;}
  [data-testid="stExpander"] {border-color:#475569;}
}
</style>
""", unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>📊 FinMorphs</h1>'
    '<p>Visual financial intelligence for real-world product inspections</p>'
    '<div class="hero-meta"><span class="hero-pill">▣ Unit economics</span>'
    '<span class="hero-pill">◉ Evidence review</span>'
    '<span class="hero-pill">↗ Scenario planning</span>'
    '<span class="hero-pill">✦ AI-assisted insights</span></div></div>',
    unsafe_allow_html=True,
)
st.caption("A decision workspace for turning product evidence and operating assumptions into a clearer financial picture.")

with st.sidebar:
    st.header("Settings")
    demo_mode = st.toggle("Demo mode (no AI request)", value=not bool(API_KEY), help="Turn this off to use Gemini. A valid GEMINI_API_KEY must be set in your local .env file.")
    multi_model_mode = st.toggle(
        "Multi-model review",
        value=False,
        disabled=not (OPENAI_API_KEY and ANTHROPIC_API_KEY),
        help="Run optional OpenAI and Anthropic reviews alongside Gemini.",
    )
    live_news_mode = st.toggle(
        "Include live news",
        value=False,
        help="Fetch current market, economic, and financial headlines from public RSS feeds.",
    )
    st.caption(f"Model: {MODEL_NAME}")
    st.caption("Uploaded media is sent to Gemini only when you generate a report with Demo mode off.")
    if not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
        st.caption("Add OPENAI_API_KEY and ANTHROPIC_API_KEY to enable multi-model review.")

st.markdown(
    '<div class="section-title"><span class="section-number">1</span>'
    '<h3>Inspection details</h3></div>'
    '<div class="section-subtitle">Set the context before reviewing the numbers. Clear inputs produce more useful financial insights.</div>',
    unsafe_allow_html=True,
)
a, b = st.columns(2)
with a:
    with st.container(border=True):
        st.markdown("#### Asset profile")
        st.caption("Identify the product, source, and place being reviewed.")
        product_name = st.text_input("Product or item name", "Packaged organic snack")
        supplier_name = st.text_input("Supplier / company (optional)", "Sample supplier")
        location = st.text_input("Inspection location (optional)", "Sample warehouse")
        source_note = st.text_input("Source of figures (optional)", "Illustrative inputs; not independently verified")
with b:
    with st.container(border=True):
        st.markdown("#### Analyst brief")
        st.caption("Describe what you want the evidence and financial model to answer.")
        observation = st.text_area("Your question / observation", "What evidence would verify sales volume, and what could limit scalability?", height=130)
        st.caption("Tip: ask Gemini to distinguish what the media shows from claims that still need independent verification.")

st.markdown(
    '<div class="section-title"><span class="section-number">2</span>'
    '<h3>Evidence vault</h3></div>'
    '<div class="section-subtitle">Attach photos, videos, or a voice note. Preview everything before adding it to the session.</div>',
    unsafe_allow_html=True,
)
image_col, video_col, audio_col = st.columns(3)
with image_col:
    uploaded_photos = st.file_uploader(
        "Product pictures",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="product_image",
    )
    for uploaded_photo in uploaded_photos:
        st.image(uploaded_photo, caption=uploaded_photo.name, use_container_width=True)
with video_col:
    uploaded_videos = st.file_uploader(
        "Short inspection videos (optional)",
        type=["mp4", "mov", "mpeg", "mpg", "avi", "webm"],
        accept_multiple_files=True,
        key="product_video",
        help="For a smoother first test, use short clips under 20 MB.",
    )
    for uploaded_video in uploaded_videos:
        st.video(uploaded_video.getvalue())
        st.caption(f"Selected: {uploaded_video.name} · {uploaded_video.size / (1024 * 1024):.1f} MB")
with audio_col:
    voice_note = st.audio_input("Record a voice note (optional)")
    if voice_note:
        st.audio(voice_note)
        st.caption("The recording will be included in Gemini analysis when AI mode is enabled.")

st.markdown(
    '<div class="section-title"><span class="section-number">3</span>'
    '<h3>Financial inputs</h3></div>'
    '<div class="section-subtitle">Build the unit economics model with transparent, editable assumptions.</div>',
    unsafe_allow_html=True,
)
st.caption("All figures recalculate automatically. Estimates remain user-provided assumptions and are not independently verified.")
c1, c2, c3 = st.columns(3)
with c1:
    unit_cost = st.number_input("Purchase cost per unit (₹)", min_value=0.0, value=40.0, step=1.0, key="unit_cost_input")
    selling_price = st.number_input("Selling price per unit (₹)", min_value=0.0, value=75.0, step=1.0, key="selling_price_input")
with c2:
    other_cost = st.number_input("Other variable cost per unit (₹)", min_value=0.0, value=8.0, step=1.0, key="other_cost_input")
    monthly_units = st.number_input("Estimated units sold per month", min_value=0, value=1200, step=100, key="monthly_units_input")
with c3:
    budget = st.number_input("Inventory budget (₹)", min_value=0.0, value=100000.0, step=5000.0, key="budget_input")
    fixed_costs = st.number_input("Monthly fixed costs (₹, optional estimate)", min_value=0.0, value=10000.0, step=1000.0, key="fixed_costs_input")

contribution = selling_price - unit_cost - other_cost
margin_pct = contribution / selling_price * 100 if selling_price > 0 else None
monthly_contribution = contribution * monthly_units
estimated_after_fixed = monthly_contribution - fixed_costs
units_budget = int(budget // unit_cost) if unit_cost > 0 else 0

st.markdown(
    '<div class="section-title"><span class="section-number">4</span>'
    '<h3>Live financial snapshot</h3></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="section-card"><strong>Decision dashboard</strong><div class="hint">These indicators update as you edit the financial inputs.</div></div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Contribution / unit", f"₹{contribution:,.2f}")
k2.metric("Contribution margin", f"{margin_pct:.1f}%" if margin_pct is not None else "N/A")
k3.metric("Monthly contribution", f"₹{monthly_contribution:,.0f}")
k4.metric("After estimated fixed costs", f"₹{estimated_after_fixed:,.0f}")
k5.metric("Units budget can buy", f"{units_budget:,}")
if contribution <= 0:
    st.warning("Contribution per unit is zero or negative with these inputs. Check the cost and selling-price values.")
st.caption("Contribution = selling price − purchase cost − other variable cost. The after-fixed-cost figure still excludes tax, interest, unexpected expenses, returns and other possible costs; it is not verified net profit.")

scenarios = []
for label, multiplier in [("Conservative", 0.5), ("Base", 1.0), ("Upside", 1.25)]:
    units = round(monthly_units * multiplier)
    scenarios.append({"Scenario": label, "Units / month": units, "Inventory cost (₹)": round(units * unit_cost, 2), "Contribution before fixed costs (₹)": round(units * contribution, 2)})
scenario_df = pd.DataFrame(scenarios)
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("#### Scenario contribution")
    st.bar_chart(scenario_df, x="Scenario", y="Contribution before fixed costs (₹)")
with chart_col2:
    st.markdown("#### Cost and selling price per unit")
    cost_df = pd.DataFrame({"Item": ["Purchase cost", "Other variable cost", "Selling price"], "₹ per unit": [unit_cost, other_cost, selling_price]})
    st.bar_chart(cost_df, x="Item", y="₹ per unit")
st.dataframe(scenario_df, use_container_width=True, hide_index=True)

st.markdown("#### Six-month contribution trend")
projection_df = pd.DataFrame({
    "Month": [f"Month {month}" for month in range(1, 7)],
    "Cumulative contribution (₹)": [
        monthly_contribution * month for month in range(1, 7)
    ],
})
st.line_chart(
    projection_df,
    x="Month",
    y="Cumulative contribution (₹)",
    use_container_width=True,
)

if estimated_after_fixed >= 0:
    st.success("The entered base case covers the estimated fixed costs before other unmodeled expenses.")
else:
    st.error("The entered base case does not cover the estimated fixed costs. Review price, costs, or volume.")

# ============================================================
# SMART GLASSES MODE — first-person camera prototype
# ============================================================

st.divider()
st.header("🥽 Smart Glasses Mode")
st.write(
    "Simulate an AI-glasses experience by capturing a photo or video, "
    "or uploading media for financial intelligence analysis."
)

st.info(
    "Prototype mode: this uses your computer's camera or uploaded media. "
    "It does not yet connect to physical AI glasses."
)

st.caption(
    "Only capture or upload images you have permission to use. "
    "Avoid capturing private conversations or sensitive information."
)

tour = st.session_state["finmorphs_tour"]
tour_control_col, tour_status_col = st.columns([1, 2])
with tour_control_col:
    if not tour["active"]:
        if st.button("▶ Start facility tour", use_container_width=True):
            tour["active"] = True
            tour["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()
    else:
        if st.button("■ End facility tour", use_container_width=True):
            tour["active"] = False
            st.rerun()
with tour_status_col:
    if tour["active"]:
        st.success(f"Tour recording active · started {tour['started_at']}")
    else:
        st.info("Tour not active. Start a tour to capture structured facility observations.")

st.markdown("#### Facility context")
context_col, observation_col = st.columns(2)
with context_col:
    tour_zone = st.text_input(
        "Current zone or station",
        placeholder="e.g. Receiving dock, packing line, cold storage",
        key="tour_zone",
    )
    movement_pattern = st.selectbox(
        "Observed workflow pattern",
        [
            "Select an observation",
            "Smooth flow",
            "Queue or waiting",
            "Backtracking or excess movement",
            "Idle equipment or staff",
            "Blocked or unsafe path",
            "Unknown",
        ],
        key="tour_movement_pattern",
    )
with observation_col:
    tour_note = st.text_area(
        "Contextual tour note",
        placeholder="What did you notice, hear, or want to verify in this area?",
        height=105,
        key="tour_note",
    )
    if st.button("＋ Save contextual note", use_container_width=True):
        if tour_note.strip():
            tour["notes"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "zone": tour_zone.strip() or "Unspecified zone",
                "note": tour_note.strip(),
            })
            register_session_item(
                "interview_note",
                f"{tour_zone.strip() or 'unspecified-zone'}-note",
                tour_note.strip().encode("utf-8"),
                "text/plain",
                f"tour note · {tour_zone.strip() or 'Unspecified zone'}",
                tour_note.strip(),
            )
            st.success("Contextual note saved.")
        else:
            st.warning("Enter a note before saving it.")

with st.expander("Record operational efficiency for this zone", expanded=False):
    efficiency_col1, efficiency_col2, efficiency_col3 = st.columns(3)
    with efficiency_col1:
        cycle_time = st.number_input("Observed cycle time (minutes)", min_value=0.0, step=0.5, key="tour_cycle_time")
    with efficiency_col2:
        walking_distance = st.number_input("Approx. walking distance (metres)", min_value=0.0, step=1.0, key="tour_walking_distance")
    with efficiency_col3:
        queue_length = st.number_input("Visible queue / waiting items", min_value=0, step=1, key="tour_queue_length")
    if st.button("＋ Save zone observation", use_container_width=True):
        if not tour_zone.strip():
            st.warning("Enter a current zone or station first.")
        else:
            tour["observations"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "zone": tour_zone.strip(),
                "movement_pattern": movement_pattern,
                "cycle_time_minutes": cycle_time,
                "walking_distance_metres": walking_distance,
                "queue_length": queue_length,
            })
            register_session_item(
                "operational_observation",
                f"{tour_zone.strip()}-observation",
                json.dumps(tour["observations"][-1]).encode("utf-8"),
                "application/json",
                f"zone observation · {tour_zone.strip()}",
            )
            st.success("Zone observation saved.")

if tour["notes"] or tour["observations"]:
    st.caption(
        f"Tour log: {len(tour['notes'])} contextual notes · "
        f"{len(tour['observations'])} zone observations"
    )

glasses_frame = st.camera_input(
    "Capture what the glasses would see",
    key="glasses_camera_frame"
)

glasses_uploads = st.file_uploader(
    "Or upload sample camera frames",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key="glasses_image_upload"
)

glasses_video_uploads = st.file_uploader(
    "Upload sample videos",
    type=["mp4", "mov", "avi", "webm"],
    accept_multiple_files=True,
    key="glasses_video_upload",
)
for uploaded_glasses_video in glasses_video_uploads:
    st.video(uploaded_glasses_video.getvalue())
    st.caption(
        f"Selected: {uploaded_glasses_video.name} · "
        f"{uploaded_glasses_video.size / (1024 * 1024):.1f} MB"
    )

st.markdown("#### Capture a video from your camera")
st.caption("Press Start, record the scene, then press Stop before saving it.")
webrtc_context = webrtc_streamer(
    key="glasses_video_capture",
    video_processor_factory=GlassesVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

captured_video = None
if (
    not webrtc_context.state.playing
    and webrtc_context.video_processor
    and webrtc_context.video_processor.frames
):
    captured_video = encode_captured_video(
        webrtc_context.video_processor.frames
    )
    st.video(captured_video)

selected_frame = glasses_frame if glasses_frame is not None else (
    glasses_uploads[0] if glasses_uploads else None
)

if st.button("Add selected evidence to session"):
    evidence = st.session_state["finmorphs_evidence"]

    existing_photos = {
        item["name"] for item in evidence["photos"]
    }
    existing_videos = {
        item["name"] for item in evidence["videos"]
    }

    for photo in uploaded_photos or []:
        if photo.name not in existing_photos:
            evidence["photos"].append({
                "name": photo.name,
                "data": photo.getvalue(),
                "type": photo.type,
            })
            register_session_item("photo", photo.name, photo.getvalue(), photo.type, "product photo uploader")
            existing_photos.add(photo.name)

    for video in uploaded_videos or []:
        if video.name not in existing_videos:
            evidence["videos"].append({
                "name": video.name,
                "data": video.getvalue(),
                "type": video.type,
            })
            register_session_item("video", video.name, video.getvalue(), video.type, "inspection video uploader")
            existing_videos.add(video.name)

    for photo in glasses_uploads or []:
        if photo.name not in existing_photos:
            evidence["photos"].append({
                "name": photo.name,
                "data": photo.getvalue(),
                "type": photo.type,
            })
            register_session_item("photo", photo.name, photo.getvalue(), photo.type, "Smart Glasses photo upload")
            existing_photos.add(photo.name)

    for video in glasses_video_uploads or []:
        if video.name not in existing_videos:
            evidence["videos"].append({
                "name": video.name,
                "data": video.getvalue(),
                "type": video.type,
            })
            register_session_item("video", video.name, video.getvalue(), video.type, "Smart Glasses video upload")
            existing_videos.add(video.name)

    if captured_video:
        captured_video_name = "smart-glasses-capture.mp4"
        if captured_video_name not in existing_videos:
            evidence["videos"].append({
                "name": captured_video_name,
                "data": captured_video,
                "type": "video/mp4",
            })
            register_session_item(
                "video", captured_video_name, captured_video, "video/mp4",
                "Smart Glasses camera capture"
            )

    st.success("Selected evidence added to this session.")

if voice_note:
    audio_bytes = voice_note.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    saved_audio_hashes = {
        note.get("audio_hash") for note in st.session_state["finmorphs_evidence"]["voice_notes"]
    }
    if audio_hash not in saved_audio_hashes:
        st.session_state["finmorphs_evidence"]["voice_notes"].append({
            "id": str(time.time_ns()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "audio": audio_bytes,
            "type": voice_note.type,
            "transcript": "",
            "status": "saved",
            "audio_hash": audio_hash,
        })
        register_session_item(
            "voice", audio_hash, audio_bytes, voice_note.type,
            "microphone recording"
        )

evidence = st.session_state["finmorphs_evidence"]
st.write(f"Photos saved: {len(evidence['photos'])}")
st.write(f"Videos saved: {len(evidence['videos'])}")
st.write(f"Voice notes saved: {len(evidence['voice_notes'])}")

with st.expander("Review or remove individual session items", expanded=False):
    if not st.session_state["finmorphs_session_items"]:
        st.caption("No saved session items yet.")
    for item in list(st.session_state["finmorphs_session_items"]):
        item_col, action_col = st.columns([5, 1])
        with item_col:
            st.markdown(
                f"**{item['kind'].replace('_', ' ').title()}** · "
                f"`{item['filename']}`"
            )
            st.caption(
                f"{item['timestamp']} · source: {item['source']} · "
                f"status: {item['status']}"
            )
            if item["kind"] in {"interview_note", "voice"} and item.get("transcript"):
                st.write(item["transcript"])
        with action_col:
            if st.button("Remove", key=f"remove_session_item_{item['id']}"):
                st.session_state["finmorphs_session_items"] = [
                    saved for saved in st.session_state["finmorphs_session_items"]
                    if saved["id"] != item["id"]
                ]
                if item["kind"] == "photo":
                    evidence["photos"] = [
                        saved for saved in evidence["photos"]
                        if hashlib.sha256(saved["data"]).hexdigest()
                        != hashlib.sha256(item["data"]).hexdigest()
                    ]
                elif item["kind"] == "video":
                    evidence["videos"] = [
                        saved for saved in evidence["videos"]
                        if hashlib.sha256(saved["data"]).hexdigest()
                        != hashlib.sha256(item["data"]).hexdigest()
                    ]
                elif item["kind"] == "voice":
                    evidence["voice_notes"] = [
                        saved for saved in evidence["voice_notes"]
                        if hashlib.sha256(saved["audio"]).hexdigest()
                        != hashlib.sha256(item["data"]).hexdigest()
                    ]
                st.rerun()

for index, note in enumerate(evidence["voice_notes"], start=1):
    st.markdown(f"**Question {index} — {note['timestamp']}**")
    st.write(note["transcript"] or "Transcript not available for this recording.")
    if note.get("assistant_response"):
        st.info(f"Glasses assistant: {note['assistant_response']}")

st.markdown("#### Evidence-to-finance review")
st.caption(
    "Ask Gemini to identify visible evidence, separate assumptions from observations, "
    "and flag financial inputs for your approval. No values change automatically."
)
if st.button(
    "🧠 Analyze saved session and suggest assumption reviews",
    use_container_width=True,
    disabled=not any(evidence.values()),
):
    review_prompt = f"""You are FinMorphs' evidence-to-finance reviewer.
Review the saved inspection session. Identify only what the media and notes
support, distinguish observations from assumptions, and suggest which current
financial inputs deserve review. Never invent a replacement value from media
unless it is explicitly visible and measurable. Return JSON only:
{{
      "key_observations": [
        {{"finding": "...", "source": "filename or note identifier", "uncertainty": "low|medium|high", "why_uncertain": "..."}}
      ],
      "assumptions_to_review": [
        {{"field": "selling_price", "current_value": 75, "suggested_value": null, "reason": "...", "source": "...", "uncertainty": "low|medium|high"}}
      ],
      "unanswered_questions": ["..."]
    }}
Allowed fields: unit_cost, selling_price, other_cost, monthly_units, budget, fixed_costs.
Current financial inputs:
unit_cost={unit_cost}, selling_price={selling_price}, other_cost={other_cost},
monthly_units={monthly_units}, budget={budget}, fixed_costs={fixed_costs}
Tour context:
{json.dumps({"notes": tour["notes"], "observations": tour["observations"]})}
Saved evidence metadata:
{json.dumps(saved_evidence_summary(evidence))}
"""
    if demo_mode:
        review = {
            "key_observations": [
                {
                    "finding": "The saved session contains media and operational notes that require evidence review.",
                    "source": "session registry",
                    "uncertainty": "high",
                    "why_uncertain": "No live model was called in Demo mode.",
                }
            ],
            "assumptions_to_review": [
                {
                    "field": "monthly_units",
                    "current_value": monthly_units,
                    "suggested_value": None,
                    "reason": "Validate volume against inventory movement or point-of-sale records.",
                },
                {
                    "field": "selling_price",
                    "current_value": selling_price,
                    "suggested_value": None,
                    "reason": "Confirm the observed or stated price with dated transaction evidence.",
                },
            ],
            "unanswered_questions": [
                "Which visible operational signals are repeatable rather than one-time conditions?"
            ],
        }
    else:
        try:
            review_client = genai.Client(api_key=API_KEY)
            review_parts = [{"type": "text", "text": review_prompt}]
            append_saved_media_to_parts(review_client, review_parts, evidence)
            with st.spinner("Reviewing saved evidence against financial assumptions…"):
                review_response = review_client.interactions.create(
                    model=MODEL_NAME,
                    input=review_parts,
                )
            review = parse_json_response(review_response.output_text or "")
            if not review:
                st.error("Gemini returned an unreadable assumption review. No financial values were changed.")
                review = {}
        except Exception as exc:
            st.error(f"Evidence review failed: {exc}")
            review = {}
    if review:
        st.session_state["finmorphs_observation_review"] = review

review = st.session_state["finmorphs_observation_review"]
for review_key in ("key_observations", "assumptions_to_review", "unanswered_questions"):
    if not isinstance(review.get(review_key), list):
        review[review_key] = []
if review["key_observations"] or review["assumptions_to_review"] or review["unanswered_questions"]:
    with st.expander("Review findings and approve changes", expanded=True):
        if review["key_observations"]:
            st.markdown("**Evidence observations**")
            for item in review["key_observations"]:
                if isinstance(item, dict):
                    st.write(
                        f"• {item.get('finding', '')} "
                        f"(source: {item.get('source', 'unknown')}; "
                        f"uncertainty: {item.get('uncertainty', 'unknown')})"
                    )
                    if item.get("why_uncertain"):
                        st.caption(item["why_uncertain"])
                else:
                    st.write(f"• {item}")
        if review["unanswered_questions"]:
            st.markdown("**Unanswered questions**")
            for item in review["unanswered_questions"]:
                st.write(f"• {item}")
        approved_suggestions = []
        for index, suggestion in enumerate(review["assumptions_to_review"]):
            field = suggestion.get("field", "")
            current_value = suggestion.get("current_value")
            suggested_value = suggestion.get("suggested_value")
            st.markdown(
                f"**{field}** · current: `{current_value}` · "
                f"suggested: `{suggested_value if suggested_value is not None else 'manual review only'}`"
            )
            st.caption(
                f"{suggestion.get('reason', 'Review this assumption against evidence.')} "
                f"Source: {suggestion.get('source', 'not specified')}. "
                f"Uncertainty: {suggestion.get('uncertainty', 'unknown')}."
            )
            if suggested_value is not None:
                if st.checkbox("Approve this numeric change", key=f"approve_review_{index}"):
                    approved_suggestions.append((field, suggested_value))
        if approved_suggestions and st.button("Apply approved financial changes", type="primary"):
            input_keys = {
                "unit_cost": "unit_cost_input",
                "selling_price": "selling_price_input",
                "other_cost": "other_cost_input",
                "monthly_units": "monthly_units_input",
                "budget": "budget_input",
                "fixed_costs": "fixed_costs_input",
            }
            for field, value in approved_suggestions:
                if field in input_keys and isinstance(value, (int, float)) and value >= 0:
                    st.session_state[input_keys[field]] = value
            st.session_state["finmorphs_observation_review"]["assumptions_to_review"] = []
            st.success("Approved values applied. Scenarios will recalculate on the next run.")
            st.rerun()

if selected_frame is not None:
    st.image(
        selected_frame,
        caption="Selected observation frame",
        use_container_width=True
    )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        analyze_observation = st.button(
            "🔍 Analyze current view",
            key="analyze_glasses_frame",
            use_container_width=True,
        )
    with action_col2:
        ask_glasses_assistant = st.button(
            "🎙️ Ask glasses assistant",
            key="ask_glasses_assistant",
            use_container_width=True,
        )

    if analyze_observation or ask_glasses_assistant:
        try:
            import base64

            api_key = os.getenv("GEMINI_API_KEY", "").strip()

            if ask_glasses_assistant and demo_mode:
                assistant_response = (
                    "Demo response: this context may indicate an operational "
                    "constraint or opportunity. Verify it with cycle times, "
                    "queue counts, inventory records, and interviews before "
                    "drawing a financial conclusion."
                )
                tour["assistant_response"] = assistant_response
                if evidence["voice_notes"]:
                    evidence["voice_notes"][-1]["assistant_response"] = assistant_response
                st.info(assistant_response)
            elif not api_key:
                st.error(
                    "Gemini API key was not found. Check your existing .env setup."
                )
            else:
                glasses_client = genai.Client(api_key=api_key)

                prompt = """
You are the hands-free AI assistant for FinMorphs, a prototype inspired
by camera-equipped smart glasses used during warehouse and factory tours.
Use only the supplied media and tour context. Separate direct evidence,
reasonable hypotheses, and unknowns. You may describe visible body language
or apparent nonverbal cues, but never identify a person's emotions,
intentions, health, honesty, or personality as fact. Do not identify people.

If this is an assistant question, answer it directly first. Then return:
1. OBSERVED EVIDENCE
2. OPERATIONAL RELEVANCE
3. POSSIBLE FINANCIAL IMPACT
4. WHAT TO VERIFY NEXT

Do not invent prices, quantities, identities, supplier claims, market
movements, or financial facts. Do not give direct buy/sell instructions.
"""

                parts = [{"type": "text", "text": prompt}]
                parts.append({
                    "type": "image",
                    "data": base64.b64encode(selected_frame.getvalue()).decode("utf-8"),
                    "mime_type": selected_frame.type or "image/jpeg",
                })
                for saved_note in evidence["voice_notes"]:
                    parts.append({
                        "type": "audio",
                        "data": base64.b64encode(saved_note["audio"]).decode("utf-8"),
                        "mime_type": saved_note.get("type") or "audio/wav",
                    })
                parts.append({
                    "type": "text",
                    "text": (
                        f"Current zone: {tour_zone or 'Not provided'}\n"
                        f"Workflow pattern: {movement_pattern}\n"
                        f"Tour note: {tour_note or 'Not provided'}\n"
                        f"Recent zone observations: {json.dumps(tour['observations'][-5:])}\n"
                        f"Mode: {'answer the spoken question' if ask_glasses_assistant else 'analyze the current view'}"
                    ),
                })

                with st.spinner("Analyzing the observation with Gemini..."):
                    response = glasses_client.interactions.create(
                        model=MODEL_NAME,
                        input=parts,
                    )

                if ask_glasses_assistant:
                    tour["assistant_response"] = response.output_text or "Gemini returned no response."
                    if evidence["voice_notes"]:
                        evidence["voice_notes"][-1]["assistant_response"] = tour["assistant_response"]
                    st.subheader("Glasses Assistant Response")
                else:
                    st.subheader("Observation Intelligence")
                st.markdown(
                    response.output_text
                    or "Gemini returned no text. Try another image."
                )

                st.caption(
                    "This is an AI-generated interpretation of one image, "
                    "not independent verification of financial or market facts."
                )

        except Exception as exc:
            st.error(f"Observation analysis failed: {exc}")

st.subheader("5 · Generate Gemini analysis and downloadable report")
st.caption("This step sends the entered question, financial values, and any selected image/audio/video to Gemini if Demo mode is off.")
if st.button("Generate Financial Intelligence Report", type="primary", use_container_width=True):
    if not product_name.strip():
        st.error("Please enter a product name first.")
        st.stop()
    if not demo_mode and not API_KEY:
        st.error("GEMINI_API_KEY is blank. Add your key to the existing .env file, save it, and restart Streamlit—or enable Demo mode.")
        st.stop()

    now = datetime.now().strftime("%d %b %Y, %I:%M %p")
    live_stories = fetch_live_news() if live_news_mode else []
    news_digest = "\n".join(
        f"- [{story['category']}] {story['title']} ({story['published']}) "
        f"{story['link']}"
        for story in live_stories
    ) or "No live news was requested or available."
    financial_summary = f"""
PRODUCT: {product_name}
SUPPLIER: {supplier_name or 'Not provided'}
LOCATION: {location or 'Not provided'}
UNIT PURCHASE COST INR: {unit_cost:.2f}
SELLING PRICE INR: {selling_price:.2f}
OTHER VARIABLE COST PER UNIT INR: {other_cost:.2f}
MONTHLY UNIT ESTIMATE: {monthly_units}
INVENTORY BUDGET INR: {budget:.2f}
MONTHLY FIXED COST ESTIMATE INR: {fixed_costs:.2f}
CONTRIBUTION PER UNIT INR: {contribution:.2f}
CONTRIBUTION MARGIN PERCENT: {margin_pct if margin_pct is not None else 'N/A'}
MONTHLY CONTRIBUTION BEFORE FIXED COSTS INR: {monthly_contribution:.2f}
ESTIMATED AFTER FIXED COSTS INR (not net profit): {estimated_after_fixed:.2f}
UNITS FUNDED BY INVENTORY BUDGET: {units_budget}
SOURCE NOTE: {source_note or 'Not provided'}
ANALYST QUESTION: {observation or 'No question provided'}
TOUR STATUS: {'Active' if tour['active'] else 'Ended or not started'}
TOUR STARTED: {tour['started_at'] or 'Not recorded'}
CONTEXTUAL TOUR NOTES: {json.dumps(tour['notes'])}
ZONE EFFICIENCY OBSERVATIONS: {json.dumps(tour['observations'])}
SAVED EVIDENCE COUNTS: {len(evidence['photos'])} photos, {len(evidence['videos'])} videos, {len(evidence['voice_notes'])} voice notes
SAVED VOICE QUESTIONS:
{json.dumps([note.get('transcript', '') or 'Audio question recorded; transcript unavailable.' for note in evidence['voice_notes']])}
EVIDENCE REVIEW:
{json.dumps(st.session_state['finmorphs_observation_review'])}
LIVE NEWS SOURCES:
{news_digest}
""".strip()
    findings = ""
    if demo_mode:
        findings = f"""### Illustrative demo response — no live AI model was called

**Product:** {product_name}

**What to verify:** Ask for dated invoices, point-of-sale records, repeat orders, inventory movement, return/expiry rates, and supplier capacity. Confirm whether sales refer to shipments to retailers or actual end-customer purchases.

**Scalability checks:** Test whether supply can increase without disproportionate cost, whether repeat demand is evidenced, and whether working capital covers stock and payment delays.

**Tour intelligence:** Review the saved facility notes and zone observations for queues, excess movement, idle capacity, or blocked flow. Any apparent nonverbal cues require human confirmation and should not be treated as fact.

**Evidence limitations:** No independent supplier or market database was queried. Media and financial inputs have not been independently verified. This response is a demonstration, not a forecast or recommendation."""
    else:
        try:
            client = genai.Client(api_key=API_KEY)
            prompt = f"""You are FinMorphs, an evidence-conscious product and business due-diligence assistant.
Analyze only the supplied evidence and facts. Do not claim access to live market databases, do not invent sales statistics or citations, and distinguish visible evidence from user claims and unknowns. Do not give buy/sell instructions or guarantee returns. State that financial estimates depend on user-supplied values. Use these headings: Field Answer; What the media supports; Facility tour signals; What remains unverified; Scalability checks; Risks and missing evidence; Recommended follow-up questions. Treat apparent sentiment or nonverbal cues as uncertain observations requiring human confirmation, never as facts about a person's emotions or intent.

FINANCIAL INPUTS AND QUESTION:\n{financial_summary}"""
            parts = [{"type": "text", "text": prompt}]
            append_saved_media_to_parts(client, parts, evidence)
            with st.spinner("Gemini is analyzing your evidence…"):
                response = client.interactions.create(model=MODEL_NAME, input=parts)
            findings = response.output_text or "Gemini returned no text. Try a shorter voice note or a clearer image/video."

            if multi_model_mode:
                consensus_prompt = f"""You are an independent second-opinion financial analyst.
Review the following user-entered financial assumptions, facility-tour observations,
and clearly labeled live headlines. Do not treat headlines as proof of product
performance. Identify points of agreement, uncertainty, and the three most useful
follow-up checks. Do not give buy/sell instructions.

{financial_summary}"""
                model_reviews = []
                with st.spinner("Getting independent second opinions from OpenAI and Anthropic…"):
                    try:
                        openai_review = call_openai(consensus_prompt)
                        if openai_review:
                            model_reviews.append(("OpenAI review", openai_review))
                    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                        st.warning(f"OpenAI review unavailable: {exc}")
                    try:
                        anthropic_review = call_anthropic(consensus_prompt)
                        if anthropic_review:
                            model_reviews.append(("Anthropic review", anthropic_review))
                    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                        st.warning(f"Anthropic review unavailable: {exc}")
                for review_name, review_text in model_reviews:
                    findings += f"\n\n### {review_name}\n{review_text}"
        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")
            st.info("Check that your API key is valid, internet is available, the model is enabled for your key, and the media file is supported. Your local financial calculations and charts still work.")
            st.stop()
            
    report = f"""# FinMorphs — Financial Intelligence Report

- Generated: {now}
- Mode: {'DEMO SIMULATION — no AI model called' if demo_mode else 'Gemini API analysis'}
- Product: {product_name}
- Supplier: {supplier_name or 'Not provided'}
- Location: {location or 'Not provided'}
- Images analyzed from saved session: {len(evidence['photos'])}
- Videos analyzed from saved session: {len(evidence['videos'])}
- Voice notes saved: {len(evidence['voice_notes'])}
- Assistant response available: {'Yes' if tour['assistant_response'] else 'No'}
- Live headlines included: {len(live_stories)}
- Tour status: {'Active' if tour['active'] else 'Ended or not started'}
- Saved tour notes: {len(tour['notes'])}
- Zone observations: {len(tour['observations'])}

## Analyst observation
{observation or 'Not provided'}

## Saved voice questions
""" + "\n".join(
        f"- Question {index}: {note.get('transcript', '') or 'Audio recorded; transcript unavailable.'}"
        for index, note in enumerate(evidence["voice_notes"], start=1)
    ) + """

## Facility tour log
""" + "\n".join(
        f"- {item['timestamp']} · {item['zone']}: {item['note']}"
        for item in tour["notes"]
    ) + """

## Zone efficiency observations
""" + "\n".join(
        f"- {item['timestamp']} · {item['zone']} · pattern: {item['movement_pattern']} · "
        f"cycle time: {item['cycle_time_minutes']} min · walking: "
        f"{item['walking_distance_metres']} m · queue: {item['queue_length']}"
        for item in tour["observations"]
    ) + f"""

## Latest glasses assistant response
{tour['assistant_response'] or 'No assistant response recorded.'}

## Live market, economic, and financial headlines
{news_digest}

## Evidence observations and assumption review
### Key observations
""" + "\n".join(
        f"- {item}" for item in st.session_state["finmorphs_observation_review"].get("key_observations", [])
    ) + """

### Assumptions to review
""" + "\n".join(
        f"- {item.get('field')}: {item.get('reason')}"
        for item in st.session_state["finmorphs_observation_review"].get("assumptions_to_review", [])
    ) + """

### Unanswered questions
""" + "\n".join(
        f"- {item}" for item in st.session_state["finmorphs_observation_review"].get("unanswered_questions", [])
    ) + f"""

## Supplied financial inputs
- Purchase cost/unit: ₹{unit_cost:,.2f}
- Selling price/unit: ₹{selling_price:,.2f}
- Other variable cost/unit: ₹{other_cost:,.2f}
- Estimated units/month: {monthly_units:,.0f}
- Inventory budget: ₹{budget:,.2f}
- Estimated fixed costs/month: ₹{fixed_costs:,.2f}
- Source note: {source_note or 'Not provided'}

## Calculated snapshot
- Contribution/unit: ₹{contribution:,.2f}
- Contribution margin: {f'{margin_pct:.2f}%' if margin_pct is not None else 'N/A'}
- Monthly contribution before fixed costs: ₹{monthly_contribution:,.2f}
- After estimated fixed costs (not net profit): ₹{estimated_after_fixed:,.2f}
- Units fundable by inventory budget at entered purchase cost: {units_budget:,}

## Scenario analysis
| Scenario | Units/month | Inventory cost (₹) | Contribution before fixed costs (₹) |
|---|---:|---:|---:|
""" + "\n".join(f"| {r['Scenario']} | {r['Units / month']:,} | {r['Inventory cost (₹)']:,.2f} | {r['Contribution before fixed costs (₹)']:,.2f} |" for r in scenarios) + f"""

## AI findings
{findings}

## Limitations
These figures are calculated from user-entered values and are not independently verified. Contribution is not net profit. Include taxes, financing, logistics, returns, spoilage, and other relevant costs before making decisions. This prototype does not automatically verify suppliers or market data.
"""
    st.session_state["finmorphs_report"] = report
    st.session_state["finmorphs_findings"] = findings
    st.session_state["finmorphs_demo"] = demo_mode
    st.session_state["finmorphs_news"] = live_stories

    st.session_state["finmorphs_scenarios"] = scenarios

    st.session_state["finmorphs_finance"] = {
    "unit_cost": unit_cost,
    "other_cost": other_cost,
    "selling_price": selling_price,
    "contribution": contribution,
    "monthly_units": monthly_units,
    "monthly_contribution": monthly_contribution,
    "fixed_costs": fixed_costs,
    "estimated_after_fixed": estimated_after_fixed,
    "budget": budget,
    "units_budget": units_budget,
    "margin_pct": margin_pct,
}

if "finmorphs_report" in st.session_state:
    st.divider()
    st.subheader("6 · Report results")
    if st.session_state.get("finmorphs_demo"):
        st.warning("Demo output is illustrative and was not generated by Gemini.")
    else:
        st.success("Gemini analysis completed. Review assumptions and evidence before relying on it.")
    st.markdown(st.session_state["finmorphs_findings"])
    if st.session_state.get("finmorphs_news"):
        with st.expander("Live news used in this report", expanded=False):
            st.caption("Headlines are contextual signals, not verified evidence of this product's performance.")
            for story in st.session_state["finmorphs_news"]:
                st.markdown(
                    f"- **{story['category']}** · [{story['title']}]({story['link']})"
                )
    with st.expander("Preview full downloadable report"):
        st.markdown(st.session_state["finmorphs_report"])
    st.download_button("Download report (.md)", data=st.session_state["finmorphs_report"], file_name="finmorphs_report.md", mime="text/markdown", use_container_width=True)
    
    # Download the report as a PDF with financial charts
    try:
        pdf_bytes = create_pdf_report(
            st.session_state["finmorphs_report"],
            st.session_state["finmorphs_scenarios"],
            st.session_state["finmorphs_finance"],
        )

        st.download_button(
            label="📄 Download Financial Intelligence Report (PDF)",
            data=pdf_bytes,
            file_name="FinMorphs_Financial_Intelligence_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_finmorphs_pdf",
        )

    except Exception as exc:
        st.error(f"Could not generate the PDF: {exc}")
    # Browser's built-in speech synthesis reads the report findings aloud; no extra API required.
    speech_json = json.dumps(st.session_state["finmorphs_findings"]).replace("<", "\\u003c")
    import streamlit.components.v1 as components
    components.html(f"""<div style="font-family:sans-serif"><button id="speak" style="background:#1d4ed8;color:white;border:0;border-radius:8px;padding:9px 12px;cursor:pointer">🔊 Read findings aloud</button> <button id="stop" style="border:0;border-radius:8px;padding:9px 12px;cursor:pointer">Stop</button></div><script>const textToRead={speech_json};document.getElementById('speak').onclick=()=>{{if(!('speechSynthesis' in window)){{alert('Speech is not supported in this browser.');return;}}window.speechSynthesis.cancel();window.speechSynthesis.speak(new SpeechSynthesisUtterance(textToRead));}};document.getElementById('stop').onclick=()=>window.speechSynthesis.cancel();</script>""", height=48)

st.divider()
st.caption("FinMorphs is a local hackathon prototype. Ask permission before recording or uploading people, private premises, or confidential supplier/customer data. Never share your API key.")
