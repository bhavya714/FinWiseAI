# app.py — Streamlit-ready Fraud Prevention App

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="TradeShield — SEBI Fraud Prevention MVP")

# ---------- Helper / Mock Data ----------
MOCK_ADVISOR_DB = pd.DataFrame([
    {"advisor_id": "AD-1001", "name": "Alpha Invest LLP", "registered": True},
    {"advisor_id": "AD-1002", "name": "GreenEdge Advisors", "registered": True},
    {"advisor_id": "AD-9001", "name": "QuickRich Promos", "registered": False},
    {"advisor_id": "AD-9002", "name": "InstantAllot Pvt Ltd", "registered": False},
])

HISTORICAL_ANNOUNCEMENTS = [
    "Company reports quarterly revenue growth of 5% driven by core operations and steady margins.",
    "Board approves dividend of INR 2 per share. No material changes to management or business model.",
    "Company announces strategic partnership with logistics provider to expand distribution channels.",
    "Company issues profit warning due to one-time exceptional items affecting FY results.",
    "Company files share buyback of up to 2% to improve capital efficiency."
]

RED_FLAG_KEYWORDS = [
    "guaranteed", "100% return", "insider", "firm allotment", "get rich", "quick gains",
    "pump", "moon", "payout", "no risk", "private offering", "unlisted", "whatsapp", "telegram"
]

# ---------- App UI ----------
st.title("📡 TradeShield — SEBI Fraud Prevention Prototype")
st.markdown("""
**Modules:** Advisor Verifier · Social Tip Analyzer · Announcement Verifier  
This is an MVP to demonstrate AI + rule-based checks to flag likely fraudulent activity (mock data).
""")

col1, col2, col3 = st.columns(3)

# ---------------- Advisor Verifier ----------------
with col1:
    st.header("🔎 Advisor Verifier")
    st.write("Check advisor name or ID against a mock regulatory database.")
    q = st.text_input("Enter Advisor Name or ID (e.g., AD-1001 or QuickRich Promos)")
    if st.button("Verify Advisor"):
        if not q:
            st.info("Enter an advisor name or ID to check.")
        else:
            found = MOCK_ADVISOR_DB[
                MOCK_ADVISOR_DB.apply(lambda r: q.lower() in str(r["advisor_id"]).lower() or q.lower() in r["name"].lower(), axis=1)
            ]
            if found.empty:
                st.error("❌ Advisor NOT found in registry. **Flag: Unregistered / suspicious**")
                st.write("Recommended action: Verify documents, ask for registration number, check SEBI website.")
            else:
                r = found.iloc[0]
                if r["registered"]:
                    st.success(f"✅ Registered: {r['name']} ({r['advisor_id']})")
                else:
                    st.warning(f"⚠️ Found but NOT registered: {r['name']} ({r['advisor_id']}). Investigate further.")

    st.markdown("---")
    st.write("**Registry (mock sample):**")
    st.table(MOCK_ADVISOR_DB)

# ---------------- Social Tip Analyzer ----------------
with col2:
    st.header("💬 Social Tip Analyzer")
    st.write("Paste social posts (one per line) or upload a CSV with a `text` column. The app flags posts with pump/scam indicators.")
    uploaded = st.file_uploader("Upload CSV of social posts (optional)", type=["csv"], key="posts")
    raw_text = st.text_area("Or paste social posts (one per line):", height=140)

    posts_df = None
    if uploaded is not None:
        try:
            posts_df = pd.read_csv(uploaded)
            if "text" not in posts_df.columns:
                st.error("CSV must have a `text` column. You can also paste posts manually.")
                posts_df = None
        except Exception as e:
            st.error("Couldn't read CSV: " + str(e))
            posts_df = None

    if posts_df is None and raw_text.strip():
        posts = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
        posts_df = pd.DataFrame({"text": posts})

    if posts_df is not None:
        def score_post(text):
            txt = str(text).lower()
            kw_count = sum(1 for k in RED_FLAG_KEYWORDS if k in txt)
            hype = 1 if ("guarantee" in txt or "guaranteed" in txt or "firm allotment" in txt) else 0
            return kw_count + hype

        posts_df["risk_score"] = posts_df["text"].apply(score_post)
        posts_df["flag"] = posts_df["risk_score"].apply(lambda s: "High" if s>=2 else ("Medium" if s==1 else "Low"))
        st.write("### Analyzed posts")
        st.dataframe(posts_df)
        st.write("### High / Medium risk posts")
        st.dataframe(posts_df[posts_df["flag"]!="Low"].reset_index(drop=True))

        if st.checkbox("Show simulated market activity linked to posts"):
            t = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="T")
            vol = np.random.normal(1000, 100, size=len(t))
            if (posts_df["flag"]=="High").any():
                idx = len(t)//2
                vol[idx:idx+3] += 4000
            fig, ax = plt.subplots()
            ax.plot(t, vol)
            ax.set_title("Simulated Market Volume (spike may indicate pump activity)")
            ax.set_ylabel("Volume")
            st.pyplot(fig)

# ---------------- Announcement Verifier ----------------
with col3:
    st.header("📢 Announcement Verifier")
    st.write("Paste a company's announcement. The tool compares it to historical filings (mock) and applies rule-based checks.")
    ann = st.text_area("Paste company announcement here:", height=180)
    if st.button("Verify Announcement"):
        if not ann.strip():
            st.info("Paste an announcement to verify.")
        else:
            docs = HISTORICAL_ANNOUNCEMENTS + [ann]
            vec = TfidfVectorizer(stop_words='english').fit_transform(docs)
            sim = cosine_similarity(vec[-1], vec[:-1]).flatten()
            max_sim = sim.max()
            avg_sim = sim.mean()
            text_low = ann.lower()
            rf_flags = [k for k in RED_FLAG_KEYWORDS if k in text_low]
            credibility = 1.0
            if max_sim < 0.15:
                credibility -= 0.5
            if len(rf_flags) > 0:
                credibility -= 0.3
            credibility = max(0.0, credibility)

            st.write(f"**Credibility score (0-1)**: {credibility:.2f}")
            st.write(f"Max similarity to past filings: {max_sim:.2f} | Average similarity: {avg_sim:.2f}")
            if rf_flags:
                st.warning("Detected suspicious keywords: " + ", ".join(rf_flags))
            if credibility < 0.6:
                st.error("❌ Low credibility — treat as suspicious and verify with official exchange filings / company contacts.")
            else:
                st.success("✅ Announcement is reasonably consistent with historical filings (still verify with official sources).")

    st.markdown("---")
    st.write("**Example historical filings (mock)**")
    for i, h in enumerate(HISTORICAL_ANNOUNCEMENTS, 1):
        st.write(f"{i}. {h[:140]}...")

st.markdown("---")
st.write("**How to demo:** show Advisor Verifier (try unregistered name), paste social posts with pump language, paste a suspicious announcement with 'firm allotment' or 'guaranteed returns'.")
