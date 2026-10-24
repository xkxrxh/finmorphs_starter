import base64
import json
import os
import tempfile
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = "gemini-3.8-flash"

st.set_page_config(page_title="FinMorphs | Financial Intelligence", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.5rem; border-radius: 18px; background: linear-gradient(115deg,#111827,#1e40af); color: white; margin-bottom: 1rem;}
.hero h1 {margin:0; font-size:2.15rem;} .hero p {margin:.45rem 0 0;color:#dbeafe;}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25);padding:.8rem;border-radius:12px;background:rgba(128,128,128,.035);}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📊 FinMorphs</h1><p>Visual financial intelligence for real-world product inspections</p></div>', unsafe_allow_html=True)
st.write("Inspect a product, calculate unit economics, attach image/video/voice evidence, and generate a Gemini-powered due-diligence report.")
st.info("This is a prototype. Calculations depend on the values you enter; AI output can be wrong and is not financial advice.")

with st.sidebar:
    st.header("Settings")
    demo_mode = st.toggle("Demo mode (no AI request)", value=not bool(API_KEY), help="Turn this off to use Gemini. A valid GEMINI_API_KEY must be set in your local .env file.")
    st.caption(f"Model: {MODEL_NAME}")
    st.caption("Uploaded media is sent to Gemini only when you generate a report with Demo mode off.")

st.subheader("1 · Inspection details")
a, b = st.columns(2)
with a:
    product_name = st.text_input("Product or item name", "Packaged organic snack")
    supplier_name = st.text_input("Supplier / company (optional)", "Sample supplier")
    location = st.text_input("Inspection location (optional)", "Sample warehouse")
    source_note = st.text_input("Source of figures (optional)", "Illustrative inputs; not independently verified")
with b:
    observation = st.text_area("Your question / observation", "What evidence would verify sales volume, and what could limit scalability?", height=130)
    st.caption("Tip: ask Gemini to distinguish what the media shows from claims that still need independent verification.")

st.subheader("2 · Upload evidence")
image_col, video_col, audio_col = st.columns(3)
with image_col:
    product_image = st.file_uploader("Product picture", type=["jpg", "jpeg", "png", "webp"], key="product_image")
    if product_image:
        st.image(product_image, caption=product_image.name, use_container_width=True)
with video_col:
    product_video = st.file_uploader("Short inspection video (optional)", type=["mp4", "mov", "mpeg", "mpg", "avi", "webm"], key="product_video", help="For a smoother first test, use a short clip under 20 MB.")
    if product_video:
        st.video(product_video.getvalue())
        st.caption(f"Selected: {product_video.name} · {product_video.size / (1024 * 1024):.1f} MB")
with audio_col:
    voice_note = st.audio_input("Record a voice note (optional)")
    if voice_note:
        st.audio(voice_note)
        st.caption("The recording will be included in Gemini analysis when AI mode is enabled.")

st.subheader("3 · Financial inputs")
st.caption("These fields recalculate automatically whenever you change a value.")
c1, c2, c3 = st.columns(3)
with c1:
    unit_cost = st.number_input("Purchase cost per unit (₹)", min_value=0.0, value=40.0, step=1.0)
    selling_price = st.number_input("Selling price per unit (₹)", min_value=0.0, value=75.0, step=1.0)
with c2:
    other_cost = st.number_input("Other variable cost per unit (₹)", min_value=0.0, value=8.0, step=1.0)
    monthly_units = st.number_input("Estimated units sold per month", min_value=0, value=1200, step=100)
with c3:
    budget = st.number_input("Inventory budget (₹)", min_value=0.0, value=100000.0, step=5000.0)
    fixed_costs = st.number_input("Monthly fixed costs (₹, optional estimate)", min_value=0.0, value=10000.0, step=1000.0)

contribution = selling_price - unit_cost - other_cost
margin_pct = contribution / selling_price * 100 if selling_price > 0 else None
monthly_contribution = contribution * monthly_units
estimated_after_fixed = monthly_contribution - fixed_costs
units_budget = int(budget // unit_cost) if unit_cost > 0 else 0

st.subheader("4 · Live financial snapshot")
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
""".strip()
    findings = ""
    if demo_mode:
        findings = f"""### Illustrative demo response — no live AI model was called

**Product:** {product_name}

**What to verify:** Ask for dated invoices, point-of-sale records, repeat orders, inventory movement, return/expiry rates, and supplier capacity. Confirm whether sales refer to shipments to retailers or actual end-customer purchases.

**Scalability checks:** Test whether supply can increase without disproportionate cost, whether repeat demand is evidenced, and whether working capital covers stock and payment delays.

**Evidence limitations:** No independent supplier or market database was queried. Media and financial inputs have not been independently verified. This response is a demonstration, not a forecast or recommendation."""
    else:
        try:
            client = genai.Client(api_key=API_KEY)
            prompt = f"""You are FinMorphs, an evidence-conscious product and business due-diligence assistant.
Analyze only the supplied evidence and facts. Do not claim access to live market databases, do not invent sales statistics or citations, and distinguish visible evidence from user claims and unknowns. Do not give buy/sell instructions or guarantee returns. State that financial estimates depend on user-supplied values. Use these headings: Field Answer; What the media supports; What remains unverified; Scalability checks; Risks and missing evidence; Recommended follow-up questions.

FINANCIAL INPUTS AND QUESTION:\n{financial_summary}"""
            parts = [{"type": "text", "text": prompt}]
            if product_image:
                parts.append({"type": "image", "data": base64.b64encode(product_image.getvalue()).decode("utf-8"), "mime_type": product_image.type or "image/jpeg"})
            if voice_note:
                parts.append({"type": "audio", "data": base64.b64encode(voice_note.getvalue()).decode("utf-8"), "mime_type": voice_note.type or "audio/wav"})
            if product_video:
                suffix = os.path.splitext(product_video.name)[1] or ".mp4"
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(product_video.getvalue())
                        temp_path = temp_file.name
                    with st.spinner("Uploading video to Gemini and waiting for processing…"):
                        uploaded_video = client.files.upload(file=temp_path)
                        deadline = time.time() + 180
                        while uploaded_video.state and uploaded_video.state.name == "PROCESSING" and time.time() < deadline:
                            time.sleep(2)
                            uploaded_video = client.files.get(name=uploaded_video.name)
                        if not uploaded_video.state or uploaded_video.state.name != "ACTIVE":
                            raise RuntimeError("Gemini did not finish processing the video within 3 minutes. Try a shorter MP4 clip.")
                    parts.append({"type": "video", "uri": uploaded_video.uri, "mime_type": uploaded_video.mime_type or product_video.type or "video/mp4"})
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
            with st.spinner("Gemini is analyzing your evidence…"):
                response = client.interactions.create(model=MODEL_NAME, input=parts)
            findings = response.output_text or "Gemini returned no text. Try a shorter voice note or a clearer image/video."
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
- Image attached: {'Yes' if product_image else 'No'}
- Video attached: {'Yes' if product_video else 'No'}
- Voice note attached: {'Yes' if voice_note else 'No'}

## Analyst observation
{observation or 'Not provided'}

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

if "finmorphs_report" in st.session_state:
    st.divider()
    st.subheader("6 · Report results")
    if st.session_state.get("finmorphs_demo"):
        st.warning("Demo output is illustrative and was not generated by Gemini.")
    else:
        st.success("Gemini analysis completed. Review assumptions and evidence before relying on it.")
    st.markdown(st.session_state["finmorphs_findings"])
    with st.expander("Preview full downloadable report"):
        st.markdown(st.session_state["finmorphs_report"])
    st.download_button("Download report (.md)", data=st.session_state["finmorphs_report"], file_name="finmorphs_report.md", mime="text/markdown", use_container_width=True)
    # Browser's built-in speech synthesis reads the report findings aloud; no extra API required.
    speech_json = json.dumps(st.session_state["finmorphs_findings"]).replace("<", "\\u003c")
    import streamlit.components.v1 as components
    components.html(f"""<div style="font-family:sans-serif"><button id="speak" style="background:#1d4ed8;color:white;border:0;border-radius:8px;padding:9px 12px;cursor:pointer">🔊 Read findings aloud</button> <button id="stop" style="border:0;border-radius:8px;padding:9px 12px;cursor:pointer">Stop</button></div><script>const textToRead={speech_json};document.getElementById('speak').onclick=()=>{{if(!('speechSynthesis' in window)){{alert('Speech is not supported in this browser.');return;}}window.speechSynthesis.cancel();window.speechSynthesis.speak(new SpeechSynthesisUtterance(textToRead));}};document.getElementById('stop').onclick=()=>window.speechSynthesis.cancel();</script>""", height=48)

st.divider()
st.caption("FinMorphs is a local hackathon prototype. Ask permission before recording or uploading people, private premises, or confidential supplier/customer data. Never share your API key.")
