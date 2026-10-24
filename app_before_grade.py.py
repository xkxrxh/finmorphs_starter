import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

st.set_page_config(page_title="FinMorphs | Field Intelligence", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --fm-ink:#eaf1ff; --fm-muted:#9aacc9; --fm-panel:#101b2d; --fm-line:#253651; --fm-cyan:#62e6d4; }
.stApp { background: radial-gradient(ellipse at 12% 0%, #1b3453 0%, #0a1220 43%, #080e18 100%); color:var(--fm-ink); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.block-container { max-width: 1400px; padding-top:1.5rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
.hero { border:1px solid #304664; background:linear-gradient(115deg,rgba(25,46,75,.97),rgba(12,24,40,.97)); border-radius:24px; padding:1.5rem 1.7rem; margin-bottom:1rem; box-shadow:0 18px 60px #0003; }
.hero-kicker { color:var(--fm-cyan); text-transform:uppercase; font-size:.74rem; letter-spacing:.18em; font-weight:700; }
.hero-title { font-size:2.5rem; font-weight:700; line-height:1.12; margin:.4rem 0; color:#f4f8ff; }
.hero-copy { color:#b6c7df; max-width:760px; margin:0; }
.pill { display:inline-block; border:1px solid #35506c; color:#c9f7ef; border-radius:999px; padding:.35rem .7rem; font-size:.78rem; margin:.7rem .35rem 0 0; background:#122b3b; }
.panel { border:1px solid var(--fm-line); border-radius:18px; background:rgba(16,27,45,.78); padding:1rem 1.1rem; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#14233a,#0f1b2d); border:1px solid #2b405e; padding:1rem; border-radius:16px; }
[data-testid="stMetricLabel"] { color:#a9bbd4; }
[data-testid="stMetricValue"] { color:#f2f7ff; }
.stButton>button, .stDownloadButton>button { border-radius:12px; min-height:2.8rem; font-weight:700; border:1px solid #42617e; }
.stButton>button[kind="primary"] { background:linear-gradient(90deg,#2dd4bf,#66e6d5); color:#062128; border:0; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea { background:#0c1727; }
[data-testid="stFileUploader"] { border:1px dashed #45617f; border-radius:14px; padding:.4rem; }
hr { border-color:#253651; }
section[data-testid="stSidebar"] { background:#0a1423; border-right:1px solid #20334c; }
.small-muted { color:#9aacc9; font-size:.88rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <div class="hero-kicker">Field intelligence · Prototype</div>
  <div class="hero-title">FinMorphs <span style="color:#62e6d4">◈</span></div>
  <p class="hero-copy">A smart-glasses-inspired financial inspection workspace. Turn product observations and supplied numbers into a transparent, evidence-aware report.</p>
  <span class="pill">◉ Financial calculations</span><span class="pill">✦ Gemini analysis</span><span class="pill">⌁ Voice note</span><span class="pill">↗ Exportable report</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ◈ FinMorphs")
    st.caption("WORKSPACE SETTINGS")
    demo_mode = st.toggle("Demo mode (no AI request)", value=not bool(API_KEY), help="Demo mode uses clearly labelled sample findings. Turn off to call Gemini with your API key.")
    st.caption(f"Mode: {'DEMO SIMULATION' if demo_mode else 'GEMINI API'}")
    st.caption(f"Model: {MODEL}")
    st.divider()
    st.markdown("**Privacy & evidence**")
    st.caption("Inputs are not independently verified. No live supplier or sales database is connected by default. Do not upload confidential information without permission.")
    if not API_KEY and not demo_mode:
        st.error("GEMINI_API_KEY is missing. Add it to .env, then restart the app.")

st.markdown("### 01 / Inspection context")
col_a, col_b = st.columns([1.05, .95], gap="large")
with col_a:
    product_name = st.text_input("Product / item", value="Packaged organic snack", key="product_name")
    supplier_name = st.text_input("Supplier / company", value="Sample supplier", key="supplier_name")
    location = st.text_input("Inspection location", value="Sample warehouse", key="location")
    product_image = st.file_uploader("Product photo (camera simulation)", type=["jpg", "jpeg", "png", "webp"], key="product_image", help="A photo can help Gemini describe visible packaging. It cannot verify sales or supplier claims.")
    if product_image:
        st.image(product_image, caption="Image supplied for inspection", use_container_width=True)
with col_b:
    typed_observation = st.text_area("Observation / question", value="The supplier says demand is growing. What evidence would verify sales volume, and what could limit scalability?", height=145, key="typed_observation")
    voice_note = st.audio_input("Record a voice note (optional)", key="voice_note")
    if voice_note:
        st.audio(voice_note)
        st.caption("When you generate findings in Gemini mode, the recording is sent for analysis. It is not continuous, real-time conversation.")

st.markdown("### 02 / Commercial inputs")
st.caption("Enter observed values or estimates. The report will refresh its financial sections whenever these values change.")
a,b,c,d,e = st.columns(5, gap="medium")
with a: unit_cost = st.number_input("Purchase cost / unit (₹)", min_value=0.0, value=40.0, step=1.0, key="unit_cost")
with b: selling_price = st.number_input("Selling price / unit (₹)", min_value=0.0, value=75.0, step=1.0, key="selling_price")
with c: other_unit_cost = st.number_input("Other variable cost (₹)", min_value=0.0, value=8.0, step=1.0, key="other_unit_cost")
with d: monthly_units = st.number_input("Estimated units / month", min_value=0, value=1200, step=100, key="monthly_units")
with e: inventory_budget = st.number_input("Inventory budget (₹)", min_value=0.0, value=100000.0, step=5000.0, key="inventory_budget")
source_note = st.text_input("Source / confidence note", value="Illustrative hackathon inputs; not independently verified", key="source_note")

contribution_per_unit = selling_price - unit_cost - other_unit_cost
margin_pct = contribution_per_unit / selling_price * 100 if selling_price > 0 else None
monthly_contribution = contribution_per_unit * monthly_units
max_units_with_budget = int(inventory_budget // unit_cost) if unit_cost > 0 else 0

st.markdown("### 03 / Financial snapshot")
m1,m2,m3,m4 = st.columns(4, gap="medium")
m1.metric("Contribution / unit", f"₹{contribution_per_unit:,.2f}")
m2.metric("Contribution margin", f"{margin_pct:.1f}%" if margin_pct is not None else "N/A")
m3.metric("Monthly contribution*", f"₹{monthly_contribution:,.0f}")
m4.metric("Units within budget", f"{max_units_with_budget:,}")
st.caption("*Before fixed overheads and taxes. Contribution is not net profit. Results are mechanical calculations from your inputs, not verified financial advice.")
if contribution_per_unit <= 0:
    st.warning("Contribution per unit is zero or negative. Check the entered price and costs.")

finance_summary = f"""PRODUCT: {product_name}
SUPPLIER: {supplier_name or 'Not provided'}
LOCATION: {location or 'Not provided'}
UNIT PURCHASE COST INR: {unit_cost:.2f}
SELLING PRICE INR: {selling_price:.2f}
OTHER VARIABLE COST PER UNIT INR: {other_unit_cost:.2f}
ESTIMATED MONTHLY UNITS: {monthly_units}
INVENTORY BUDGET INR: {inventory_budget:.2f}
CONTRIBUTION PER UNIT INR: {contribution_per_unit:.2f}
CONTRIBUTION MARGIN: {f'{margin_pct:.2f}%' if margin_pct is not None else 'N/A'}
MONTHLY CONTRIBUTION BEFORE FIXED OVERHEADS/TAX INR: {monthly_contribution:.2f}
UNITS FUNDED BY BUDGET: {max_units_with_budget}
SOURCE NOTE: {source_note or 'Not provided'}
ANALYST QUESTION: {typed_observation or 'No question provided'}"""

st.markdown("### 04 / AI field analysis")
btn_col, hint_col = st.columns([.55, 1.45], vertical_alignment="center")
with btn_col:
    generate = st.button("✦ Generate / refresh AI findings", type="primary", use_container_width=True)
with hint_col:
    st.markdown('<div class="small-muted">Use this when you want Gemini to re-read the current product context, image/voice note and financial inputs. Financial figures below refresh automatically even before this button is pressed.</div>', unsafe_allow_html=True)

if "ai_findings" not in st.session_state:
    st.session_state.ai_findings = "No AI findings yet. Review your inputs, then select **Generate / refresh AI findings**."
    st.session_state.ai_mode = "Not generated"

if generate:
    if not product_name.strip():
        st.error("Enter a product name first.")
    elif demo_mode:
        st.session_state.ai_findings = f"""### Illustrative demo findings — not generated by Gemini

**Field answer**  
For **{product_name}**, request dated invoices, point-of-sale records, repeat-order data and inventory reconciliation before accepting a sales-growth claim.

**What remains unverified**  
Supplier statements, sales volumes, customer demand, product quality, market prices and scalability have not been independently verified. The image and voice note do not prove commercial performance.

**Scalability checks**  
Check production capacity, input availability, delivery lead times, customer concentration, working capital, returns/expiry and whether growth reflects end-customer sell-through rather than shipments to retailers.

**Evidence to collect next**  
1. Dated sales and returns records. 2. Recent invoices matched to payments. 3. Stock movement records. 4. Customer reorder evidence. 5. Capacity and lead-time commitments.

**Mode**  
Demo simulation only. No live model or supplier database was queried. Financial estimates depend on the accuracy of the supplied values."""
        st.session_state.ai_mode = "Demo simulation"
    elif not API_KEY:
        st.error("Gemini API key missing. Add GEMINI_API_KEY to your .env file and restart Streamlit, or switch Demo mode on.")
    else:
        with st.spinner("Gemini is reviewing the current inspection..."):
            try:
                client = genai.Client(api_key=API_KEY)
                prompt = f"""You are FinMorphs, an evidence-conscious field research assistant. Analyze only the provided photo/audio and analyst-supplied facts. Never claim to have checked live databases or news. Never invent supplier facts, market statistics, citations or prices. Clearly label claims as unverified. Explain that image evidence cannot prove sales, demand or supplier reliability. Do not give buy/sell instructions or guaranteed returns. Keep the answer useful and concise, with these headings: Field answer; What the image/audio supports; What remains unverified; Scalability checks; Risks and missing evidence; Recommended follow-up questions. State that financial estimates depend on accuracy of supplied inputs.\n\nCURRENT INPUTS:\n{finance_summary}"""
                parts = [types.Part.from_text(text=prompt)]
                if product_image:
                    parts.append(types.Part.from_bytes(data=product_image.getvalue(), mime_type=product_image.type or "image/jpeg"))
                if voice_note:
                    parts.append(types.Part.from_bytes(data=voice_note.getvalue(), mime_type=voice_note.type or "audio/wav"))
                response = client.models.generate_content(model=MODEL, contents=parts)
                answer = (response.text or "Gemini returned no text. Try again with a shorter voice note.").strip()
                st.session_state.ai_findings = answer
                st.session_state.ai_mode = f"Gemini API · {MODEL}"
            except Exception as exc:
                st.error(f"Gemini request failed: {exc}")
                st.info("Check your API key, model access and internet connection. The financial report still updates from your inputs.")

# Always build the report from CURRENT widgets, rather than keeping a stale report from the last button click.
now = datetime.now().strftime("%d %b %Y, %I:%M %p")
scenarios = []
for label, multiplier in [("Conservative", 0.5), ("Base", 1.0), ("Upside", 1.25)]:
    units = round(monthly_units * multiplier)
    scenarios.append({"Scenario": label, "Units / month": units, "Inventory cost (₹)": round(units * unit_cost, 2), "Contribution before fixed costs/tax (₹)": round(units * contribution_per_unit, 2)})

report = f"""# FinMorphs — Inspection Report

- **Last refreshed:** {now}
- **Product:** {product_name}
- **Supplier:** {supplier_name or 'Not provided'}
- **Location:** {location or 'Not provided'}
- **Analysis mode:** {st.session_state.ai_mode}

## Analyst observation
{typed_observation or 'No written observation provided.'}

## Supplied commercial inputs
- Unit purchase cost: ₹{unit_cost:,.2f}
- Selling price: ₹{selling_price:,.2f}
- Other variable cost per unit: ₹{other_unit_cost:,.2f}
- Estimated monthly units: {monthly_units:,}
- Inventory budget: ₹{inventory_budget:,.2f}
- Source / confidence note: {source_note or 'Not provided'}

## Calculated financial snapshot
- Contribution per unit: ₹{contribution_per_unit:,.2f}
- Contribution margin: {f'{margin_pct:.2f}%' if margin_pct is not None else 'N/A'}
- Estimated monthly contribution before fixed overheads and taxes: ₹{monthly_contribution:,.2f}
- Units fundable by budget at entered purchase cost: {max_units_with_budget:,}

These are mechanical calculations from supplied inputs, not independently verified. Contribution is not net profit.

## Scenario analysis
| Scenario | Units/month | Inventory cost (₹) | Contribution before fixed costs/tax (₹) |
|---|---:|---:|---:|
""" + "\n".join(f"| {r['Scenario']} | {r['Units / month']:,} | {r['Inventory cost (₹)']:,.2f} | {r['Contribution before fixed costs/tax (₹)']:,.2f} |" for r in scenarios) + f"""

Scenario multipliers are illustrative sensitivity tests, not forecasts.

## AI findings
{st.session_state.ai_findings}

## Limitations and evidence checklist
- No live supplier, point-of-sale, inventory or market database is connected by default.
- Verify sales using dated records; distinguish sell-in from end-customer sell-through.
- Include overheads, taxes, financing, logistics, spoilage/returns and payment delays before estimating net profit.
- Treat AI output as decision support for human review, not a guaranteed outcome or personalized investment recommendation.
"""

st.divider()
st.markdown("### 05 / Live report & export")
st.caption("This report is rebuilt from the current form values on every Streamlit rerun. If you edit a number, the figures and downloadable report change immediately; press Generate / refresh to update Gemini findings too.")
with st.expander("Preview current full report", expanded=True):
    st.markdown(report)
left_export, right_export = st.columns([.55, 1.45], vertical_alignment="center")
with left_export:
    st.download_button("↓ Download current report (.md)", data=report, file_name="finmorphs_inspection_report.md", mime="text/markdown", use_container_width=True)
with right_export:
    st.markdown(f'<div class="small-muted">Current status: <b>{st.session_state.ai_mode}</b> · Report refresh: {now}</div>', unsafe_allow_html=True)

with st.expander("🔊 Read findings aloud"):
    st.caption("Uses your browser's built-in speech synthesis; it does not require another API.")
    import streamlit.components.v1 as components
    import json
    safe_text = json.dumps(st.session_state.ai_findings).replace("<", "\\u003c")
    components.html(f"""<div style="font-family:Arial,sans-serif;background:#101b2d;padding:12px;border-radius:12px"><button id="speak" style="padding:10px 14px;border-radius:8px;border:0;background:#62e6d4;color:#071b21;font-weight:bold">▶ Read findings</button> <button id="stop" style="padding:10px 14px;border-radius:8px;border:0">■ Stop</button><script>const txt={safe_text};document.getElementById('speak').onclick=()=>{{if(!('speechSynthesis'in window)){{alert('Speech output is not supported in this browser.');return}}speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance(txt))}};document.getElementById('stop').onclick=()=>speechSynthesis.cancel();</script></div>""", height=64)

st.caption("FinMorphs is a hackathon prototype. Use recordings only with appropriate consent. Never commit or publish your API key. Demo findings are simulations; Gemini findings are model-generated and may be wrong.")
