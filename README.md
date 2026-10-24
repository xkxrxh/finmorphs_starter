# FinMorphs — Hackathon starter

A Streamlit prototype that simulates the software workflow behind financial-intelligence smart glasses. It accepts a product photo, a typed observation and an optional microphone recording; calculates transparent unit economics; runs multimodal analysis through the Gemini API when configured; and exports a Markdown inspection report.

## What's included

- **Demo mode**: runs without an API key and clearly labels the AI findings as a simulation.
- **Gemini mode**: passes the uploaded image, optional recorded audio and inspection details to `gemini-3.8-flash`.
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
7. Enter a product name and sample financial data, optionally upload a picture/record audio, and click **Generate Financial Intelligence Report**. Download the report and try the read-aloud button.

## Enable the live multimodal AI

1. Open https://aistudio.google.com/apikey and create a Gemini API key.
2. In the project folder, copy `.env.example` to a file named `.env`.
3. Open `.env` and put your key after `GEMINI_API_KEY=`. Do not share screenshots of this file or publish it to GitHub.
4. Restart the app if it was running. Turn **Demo mode — no AI API call** off in the sidebar and generate a report.
5. API/model access, quotas, and possible billing depend on your Google AI Studio account and current terms. Check usage before repeated demo requests.

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
