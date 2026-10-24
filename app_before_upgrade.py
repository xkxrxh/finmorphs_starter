import base64
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

st.set_page_config(
    page_title="FinMorphs | Financial Intelligence",
    page_icon="👓",
    layout="wide",
)
st.title("FinMorphs — Debug Check")
st.success("The app has started rendering successfully.")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = "gemini-3.8-flash"

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1180px;}
      .hero {padding: 1.25rem 1.4rem; border-radius: 18px; background: linear-gradient(115deg, #111827, #1e3a5f); color: white; margin-bottom: 1rem;}
      .hero h1 {margin: 0; font-size: 2.1rem;}
      .hero p {margin: .45rem 0 0 0; color: #dbeafe;}
      .small-note {font-size: .88rem; color: #64748b;}
      [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: .85rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>👓 FinMorphs</h1><p>Context-aware financial intelligence for real-world inspections</p></div>',
    unsafe_allow_html=True,
)
st.write("**Hackathon prototype:** simulate a smart-glasses inspection, ask a product question, analyze the supplied numbers, and generate a due-diligence report.")

with st.sidebar:
    st.header("Prototype settings")
    demo_mode = st.toggle(
        "Demo mode — no AI API call",
        value=not bool(GEMINI_API_KEY),
        help="Uses clearly labelled illustrative AI output. Turn this off to use your Gemini API key.",
    )
    st.caption("Model: Gemini 3.8 Flash" if not demo_mode else "Demo simulation is active")
    st.divider()
    st.markdown("**Data honesty**")
    st.caption("Calculations use the values entered below. This prototype does not automatically access live sales, supplier, or market databases.")

st.subheader("1 · Start an inspection")
left, right = st.columns([1.05, 1])
with left:
    product_name = st.text_input("Product or item name", value="Packaged organic snack")
    supplier_name = st.text_input("Supplier / company (optional)", value="Sample supplier")
    location = st.text_input("Inspection location (optional)", value="Sample warehouse")
    product_image = st.file_uploader(
        "Upload a product photo (simulates the glasses camera)",
        type=["jpg", "jpeg", "png", "webp"],
        help="For the first demo, use a clear product label or product photo.",
    )
    if product_image:
        st.image(product_image, caption="Inspection image", use_container_width=True)
with right:
    typed_observation = st.text_area(
        "Analyst's observation / question",
        value="The supplier says demand is growing. What evidence would I need to verify sales volume, and what could limit scalability?",
        height=140,
    )
    voice_note = st.audio_input("Or record a short voice note (optional)")
    if voice_note:
        st.audio(voice_note)
        st.caption("The recorded audio can be sent to Gemini when Demo mode is off.")

st.subheader("2 · Enter available commercial data")
st.caption("Use supplier records or clearly marked estimates. Do not treat an unverified claim as a confirmed fact.")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    unit_cost = st.number_input("Unit purchase cost (₹)", min_value=0.0, value=40.0, step=1.0)
with c2:
    selling_price = st.number_input("Selling price / unit (₹)", min_value=0.0, value=75.0, step=1.0)
with c3:
    other_unit_cost = st.number_input("Other variable cost / unit (₹)", min_value=0.0, value=8.0, step=1.0)
with c4:
    monthly_units = st.number_input("Estimated units sold / month", min_value=0, value=1200, step=100)
with c5:
    inventory_budget = st.number_input("Inventory budget (₹)", min_value=0.0, value=100000.0, step=5000.0)

source_note = st.text_input(
    "Where did these figures come from? (optional)",
    value="Illustrative hackathon inputs; not independently verified",
)

# Deterministic financial calculations. These are not model-generated.
contribution_per_unit = selling_price - unit_cost - other_unit_cost
margin_pct = (contribution_per_unit / selling_price * 100) if selling_price > 0 else None
monthly_contribution = contribution_per_unit * monthly_units
capital_for_month = unit_cost * monthly_units
max_units_with_budget = int(inventory_budget // unit_cost) if unit_cost > 0 else 0


def build_scenarios(units, purchase_cost, contribution):
    """Build all scenario rows from the current financial inputs."""
    rows = []
    for label, multiplier in (
        ("Conservative", 0.5),
        ("Base", 1.0),
        ("Upside", 1.25),
    ):
        scenario_units = round(units * multiplier)
        rows.append({
            "Scenario": label,
            "Units / month": scenario_units,
            "Inventory cost (₹)": round(scenario_units * purchase_cost, 2),
            "Contribution before fixed costs/tax (₹)": round(
                scenario_units * contribution, 2
            ),
        })
    return rows


# Keep the comparison table and both charts driven by the live widget values.
# Streamlit reruns this block whenever an input changes.
scenarios = build_scenarios(
    monthly_units,
    unit_cost,
    contribution_per_unit,
)

st.subheader("3 · Financial snapshot")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Contribution / unit", f"₹{contribution_per_unit:,.2f}")
m2.metric("Estimated contribution margin", f"{margin_pct:.1f}%" if margin_pct is not None else "N/A")
m3.metric("Monthly contribution estimate", f"₹{monthly_contribution:,.0f}")
m4.metric("Units fundable by budget", f"{max_units_with_budget:,}")
st.caption("Contribution is selling price minus entered unit purchase and other variable costs. It is not net profit: fixed overheads, tax, financing, returns, spoilage, and other costs may still apply.")

if contribution_per_unit <= 0:
    st.warning("With the entered values, contribution per unit is zero or negative. Verify costs and pricing before considering further analysis.")

st.markdown("#### Scenario comparison: contribution and monthly sales volume")
scenario_chart_data = pd.DataFrame(scenarios).set_index("Scenario")
chart_left, chart_right = st.columns(2, gap="large")
with chart_left:
    st.markdown("**Estimated monthly contribution before fixed costs and tax**")
    st.bar_chart(
        scenario_chart_data[["Contribution before fixed costs/tax (₹)"]],
        x_label="Scenario",
        y_label="Contribution (₹)",
        use_container_width=True,
    )
    st.caption(
        "What this shows: the estimated money left after purchase cost and "
        "other variable cost for each scenario. It is not net profit."
    )
    st.markdown(
        "**Required for this graph:** monthly units estimate, selling price "
        "per unit, unit purchase cost, and other variable cost per unit. "
        "The scenarios use 50% (Conservative), 100% (Base), and 125% "
        "(Upside) of the monthly units estimate."
    )
with chart_right:
    st.markdown("**Projected monthly sales volume by scenario**")
    st.line_chart(
        scenario_chart_data[["Units / month"]],
        x_label="Scenario",
        y_label="Units",
        use_container_width=True,
    )
    st.caption(
        "What this shows: the assumed number of units sold per month under "
        "each scenario. This is a scenario comparison, not a historical "
        "time-series trend."
    )
    st.markdown(
        "**Required for this graph:** an estimated monthly unit volume. "
        "The displayed values are calculated as 50% (Conservative), 100% "
        "(Base), and 125% (Upside) of that estimate. Add dated sales history "
        "if you want a genuine month-by-month trend line."
    )

st.dataframe(
    scenario_chart_data.reset_index(),
    use_container_width=True,
    hide_index=True,
)

st.subheader("4 · Generate the inspection answer and report")
if st.button("Generate Financial Intelligence Report", type="primary", use_container_width=True):
    if not product_name.strip():
        st.error("Enter a product name first.")
        st.stop()

    image_bytes = product_image.getvalue() if product_image else None
    audio_bytes = voice_note.getvalue() if voice_note else None
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")

    finance_summary = f"""
PRODUCT: {product_name}
SUPPLIER: {supplier_name or 'Not provided'}
LOCATION: {location or 'Not provided'}
UNIT PURCHASE COST (INR): {unit_cost:.2f}
SELLING PRICE (INR): {selling_price:.2f}
OTHER VARIABLE COST / UNIT (INR): {other_unit_cost:.2f}
ESTIMATED MONTHLY SALES UNITS: {monthly_units}
INVENTORY BUDGET (INR): {inventory_budget:.2f}
CALCULATED CONTRIBUTION / UNIT (INR): {contribution_per_unit:.2f}
CALCULATED CONTRIBUTION MARGIN: {margin_pct:.2f}%
ESTIMATED MONTHLY CONTRIBUTION BEFORE FIXED OVERHEADS/TAX (INR): {monthly_contribution:.2f}
INVENTORY UNITS FUNDED BY BUDGET AT PURCHASE COST: {max_units_with_budget}
DATA SOURCE NOTE: {source_note or 'Not provided'}
ANALYST OBSERVATION / QUESTION: {typed_observation or 'No written question provided'}
""".strip()

    with st.spinner("Preparing the inspection findings..."):
        if demo_mode:
            ai_findings = f"""### Illustrative demo response — not generated by a live AI model

**Product context:** {product_name}. The image and question are shown in this demo, but this response is a simulation because Demo mode is active.

**What to verify:** Request dated invoices, point-of-sale records, repeat-order information, current inventory records, supplier capacity, customer concentration, returns/expiry data, and comparable selling prices. Match the evidence period and unit definitions before accepting a monthly sales claim.

**Scalability checks:** Confirm that supply can increase without disproportionate cost, that demand is supported by actual sell-through rather than only shipments to retailers, and that working capital covers additional stock and payment delays.

**Evidence status:** The financial figures below are supplied demo inputs, not independently verified market data. No live market or supplier database was queried. Do not interpret this simulation as an investment recommendation."""
        else:
            if not GEMINI_API_KEY:
                st.error("No GEMINI_API_KEY found. Add it to your .env file, or turn on Demo mode.")
                st.stop()
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                prompt = f"""
You are the evidence-conscious research assistant in FinMorphs, a prototype for analysts inspecting physical products and businesses.
Analyze only the image/audio and the analyst-provided facts below. Do not claim you queried live databases or current news. Do not invent sales statistics, citations, prices, market sizes, or factual claims.
Clearly label user-entered values as supplied inputs and any unverified supplier statements as claims. If the image does not make the product identifiable, say so. Provide a concise response the analyst could hear in the field, followed by structured due-diligence findings.
Use these headings: Field Answer; What the image/audio supports; What remains unverified; Scalability checks; Risks and missing evidence; Recommended follow-up questions.
Do not give a guaranteed return or a buy/sell instruction. State that financial estimates depend on the accuracy of the supplied values.

INSPECTION AND FINANCIAL INPUTS:
{finance_summary}
""".strip()
                parts = [{"type": "text", "text": prompt}]
                if image_bytes:
                    parts.append({
                        "type": "image",
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                        "mime_type": product_image.type or "image/jpeg",
                    })
                if audio_bytes:
                    parts.append({
                        "type": "audio",
                        "data": base64.b64encode(audio_bytes).decode("utf-8"),
                        "mime_type": voice_note.type or "audio/wav",
                    })
                response = client.interactions.create(model=MODEL_NAME, input=parts)
                ai_findings = response.output_text or "The model returned no text. Try a shorter audio note or a clearer image."
            except Exception as exc:
                st.error(f"The AI request failed: {exc}")
                st.info("You can still turn on Demo mode to rehearse the complete presentation flow.")
                st.stop()

    report = f"""# FinMorphs — Inspection Report

- **Generated:** {now}
- **Product:** {product_name}
- **Supplier:** {supplier_name or 'Not provided'}
- **Location:** {location or 'Not provided'}
- **Mode:** {'DEMO SIMULATION — no model call' if demo_mode else 'Gemini API analysis'}

## Analyst observation
{typed_observation or 'No written observation provided.'}

## Supplied commercial inputs
- Unit purchase cost: ₹{unit_cost:,.2f}
- Selling price: ₹{selling_price:,.2f}
- Other variable cost per unit: ₹{other_unit_cost:,.2f}
- Estimated monthly units: {monthly_units:,}
- Inventory budget: ₹{inventory_budget:,.2f}
- Data source note: {source_note or 'Not provided'}

## Calculated financial snapshot
- Contribution per unit: ₹{contribution_per_unit:,.2f}
- Contribution margin: {f'{margin_pct:.2f}%' if margin_pct is not None else 'N/A'}
- Estimated monthly contribution before fixed overheads and taxes: ₹{monthly_contribution:,.2f}
- Units fundable by the budget at the entered purchase cost: {max_units_with_budget:,}

These are mechanical calculations from supplied inputs. They are not independently verified, and contribution is not net profit.

## Scenario analysis
| Scenario | Units/month | Inventory cost (₹) | Contribution before fixed costs/tax (₹) |
|---|---:|---:|---:|
""" + "\n".join(
        f"| {row['Scenario']} | {row['Units / month']:,} | {row['Inventory cost (₹)']:,.2f} | {row['Contribution before fixed costs/tax (₹)']:,.2f} |"
        for row in scenarios
    ) + f"""

The conservative/base/upside quantities are illustrative multipliers of the supplied monthly estimate, not a forecast. Validate demand and available cash before interpreting them.

## AI findings
{ai_findings}

## Limitations and next evidence to collect
- No automatic access to live supplier, POS, inventory, or market databases is included in this prototype.
- Verify sales figures using dated sales records and distinguish sell-in from end-customer sell-through.
- Include fixed overheads, taxes, financing, logistics, spoilage/returns, and payment delays before calculating net returns.
- Treat all generated analysis as decision support for human review, not a guaranteed outcome or personalized investment recommendation.
"""

    st.session_state["last_report"] = report
    st.session_state["last_findings"] = ai_findings
    st.session_state["last_demo_mode"] = demo_mode
    st.session_state["last_scenarios"] = scenarios

if "last_report" in st.session_state:
    st.divider()
    st.subheader("5 · Inspection result")
    if st.session_state.get("last_demo_mode"):
        st.warning("Demo simulation is active. The output is illustrative and was not produced by a live AI model.")
    else:
        st.success("Gemini API analysis completed. Verify evidence and assumptions before relying on the findings.")

    with st.expander("View AI field answer and findings", expanded=True):
        st.markdown(st.session_state["last_findings"])

    st.download_button(
        "Download full report (.md)",
        data=st.session_state["last_report"],
        file_name="finmorphs_inspection_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # Browser-native speech for a convincing demo; no separate TTS API required.
    speak_text = json.dumps(st.session_state["last_findings"]).replace("<", "\\u003c")
    import streamlit.components.v1 as components
    components.html(
        f"""
        <div style="font-family: sans-serif; padding: 4px 0;">
          <button id="speakBtn" style="background:#1d4ed8;color:white;border:0;border-radius:9px;padding:10px 14px;cursor:pointer;font-size:14px;">
            🔊 Read AI findings aloud
          </button>
          <button id="stopBtn" style="background:#e5e7eb;color:#111827;border:0;border-radius:9px;padding:10px 14px;cursor:pointer;font-size:14px;margin-left:6px;">
            Stop
          </button>
        </div>
        <script>
          const spokenText = {speak_text};
          document.getElementById('speakBtn').addEventListener('click', () => {{
            if (!('speechSynthesis' in window)) {{ alert('Speech output is not supported in this browser.'); return; }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(spokenText);
            utterance.rate = 1.02;
            window.speechSynthesis.speak(utterance);
          }});
          document.getElementById('stopBtn').addEventListener('click', () => window.speechSynthesis.cancel());
        </script>
        """,
        height=58,
    )

st.divider()
st.caption("FinMorphs is a hackathon prototype. Use recording only with appropriate consent. Do not upload confidential supplier/customer information without permission. Never publish your API key.")
