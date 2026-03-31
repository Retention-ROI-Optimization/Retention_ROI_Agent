# Retention ROI Dashboard + LLM Analyst

## Install

```bash
pip install -r requirements.txt
```

## Run dashboard

```bash
streamlit run dashboard/app.py
```

## Enable LLM summary / Q&A

Recommended: set your OpenAI API key as an environment variable.

```bash
export OPENAI_API_KEY="your-api-key"
streamlit run dashboard/app.py
```

You can also paste the API key in the dashboard sidebar at runtime.

## Added LLM features

- Per-view AI summary below the charts/tables
- Per-view question input for metric-specific answers
- LangChain + ChatOpenAI based integration
- Graceful fallback when API key or packages are missing
