# FinMorphs — Hackathon starter

A Streamlit prototype that simulates the software workflow behind financial-intelligence smart glasses. It accepts site-visit details, photos, videos, voice notes, and financial inputs; runs an evidence-linked Gemini pipeline when configured; lets an analyst review findings and approve assumption changes; and exports a professional Markdown/PDF due-diligence report.

## What's included

- **Demo mode**: runs without an API key and clearly labels the AI findings as a simulation.
- **Gemini mode**: passes the saved photos, videos, voice notes, inspection details, and financial inputs to `gemini-3.8-flash`, with source markers for each media item.
- **Evidence-linked findings**: separates visible facts, hypotheses, and unknowns, then records the source, financial relevance, and verification action for each finding.
- **Analyst review**: accepts, rejects, or flags findings for verification and allows a human-approved financial assumption change.
- **Transparent scenarios**: recalculates conservative, base, and upside scenarios after an approved assumption change.
- **Professional report**: includes a visit summary, evidence register, findings, implications, scenario comparison, analyst review, verification checklist, limitations, and a human-review reminder.
- Deterministic calculations for contribution per unit, contribution margin, monthly contribution, inventory budget, and simple scenario comparisons.
- Downloadable Markdown report and browser-native read-aloud button.

## Run on Windows

1. Install Python 3.11 or newer from https://www.python.org/downloads/ and Visual Studio Code from https://code.visualstudio.com/.
2. Extract this ZIP into a folder, for example `C:\\Users\\YourName\\Desktop\\finmorphs_starter`.
3. In VS Code, open that folder (**File → Open Folder**).
4. Open **Terminal → New Terminal** and run:

   ```powershell
   py -m venv .venv
   .venv\\Scripts\\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, use Command Prompt terminal and run `.venv\\Scripts\\activate.bat`, then the install commands.

5. Run the app in demo mode first:

   ```powershell
   streamlit run app.py
   ```

6. Your browser should open the local site. If it does not, open the `Local URL` shown in the terminal, usually `http://localhost:8501`.
7. Enter a product name and sample financial data, start a site visit if useful, upload photos/videos or record a voice note, and click **Generate Financial Intelligence Report**. Download both the Markdown and PDF reports.

## Enable the live multimodal AI

1. Open https://aistudio.google.com/apikey and create a Gemini API key.
2. In the project folder, copy `.env.example` to a file named `.env`.
3. Open `.env` and put your key after `GEMINI_API_KEY=`. Do not share screenshots of this file or publish it to GitHub.
4. Restart the app if it was running. Turn **Demo mode (no AI request)** off in the sidebar and generate a report.
5. API/model access, quotas, and possible billing depend on your Google AI Studio account and current terms. Check usage before repeated demo requests.

## Recommended Gemini workflow

1. Add the site-visit context and the analyst's question.
2. Upload the evidence and confirm that the saved evidence counts are correct.
3. Click **Analyze saved session and suggest assumption reviews**.
4. Review each finding's source, evidence type, uncertainty, financial area, and verification record.
5. Approve a numeric assumption only when the supporting evidence and follow-up checks justify it.
6. Compare the before/after scenarios.
7. Generate the final report and keep the human-review reminder with the deliverable.

Gemini interprets evidence and proposes investigation points; it does not invent
financial impacts from an image. The app performs the scenario arithmetic only
after an analyst-approved assumption change. Keep `.env` local and never commit
API keys or confidential evidence.

## Optional multi-model and live-news review

Copy the optional keys from `.env.example` into `.env` to enable independent
OpenAI and Anthropic second opinions in the sidebar. Set `OPENAI_MODEL` and
`ANTHROPIC_MODEL` if your accounts use different model IDs. The report keeps
Gemini as the primary multimodal analysis and appends the other model reviews
as separate, clearly labeled opinions.

The **Include live news** toggle fetches current market, economic, and financial
headlines from public RSS feeds and includes the headline titles and source
links in the report. Headlines are contextual signals, not proof of product
performance or investment advice. Network availability, feed changes, provider
terms, and API quotas can affect results.

## What this prototype does NOT do

- It does not connect to real smart glasses or continuously stream live video.
- It does not query live supplier, point-of-sale, inventory, or market databases.
- Financial figures come from the inputs you enter; only arithmetic is performed automatically.
- Demo mode does not make a live AI call and does not actually analyze the image/audio.

For a hackathon presentation, describe this as a **browser-based simulation of the companion app**. The next integrations would be real evidence retrieval, session storage, and a supported wearable SDK or live audio/video API.
