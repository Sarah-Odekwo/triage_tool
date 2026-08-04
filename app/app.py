import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()  # for local development

# Try Streamlit secrets first (for deployment), fall back to .env (for local)
api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
client = Groq(api_key=api_key)

st.set_page_config(page_title="FirstGlance — AI Ticket Triage", layout="wide")

CLASSIFICATION_PROMPT = """You are a support ticket triage assistant for Monzo, a UK digital bank.

Classify the following support ticket. Respond with ONLY a valid JSON object, no other text, no markdown formatting, in this exact structure:

{{
  "category": "Billing" | "Technical" | "Account" | "Other",
  "urgency": "Low" | "Medium" | "High",
  "rationale": "one short sentence explaining your classification"
}}

Category definitions:
- Billing: money movement, charges, fees, refunds, subscriptions, currency conversion
- Technical: app bugs, crashes, card not working due to system issues, login/payment failures
- Account: identity verification, account access/locking, personal detail changes, closing accounts. If a card is blocked and the cause is unclear, default to Account.
- Other: general feedback, feature requests, vague complaints that don't fit above

Urgency definitions:
- High: customer has no access to funds, suspects fraud, or describes a time-critical failure
- Medium: a real problem, not blocking access to money right now
- Low: general questions, feedback, no time pressure

Ticket: "{ticket_text}"
"""

URGENCY_COLOURS = {"High": "#FF3B30", "Medium": "#FF9500", "Low": "#34C759"}
CATEGORY_COLOURS = {"Billing": "#5856D6", "Technical": "#007AFF", "Account": "#FF5A5F", "Other": "#8E8E93"}


def classify_ticket(ticket_text):
    prompt = CLASSIFICATION_PROMPT.format(ticket_text=ticket_text)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    raw_output = response.choices[0].message.content.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"category": "PARSE_ERROR", "urgency": "PARSE_ERROR", "rationale": raw_output}


def urgency_badge(urgency):
    colour = URGENCY_COLOURS.get(urgency, "#999999")
    return (f'<span style="background-color:{colour}; color:white; padding:4px 12px; '
            f'border-radius:12px; font-size:0.8em; font-weight:600;">{urgency}</span>')


def category_badge(category):
    colour = CATEGORY_COLOURS.get(category, "#999999")
    return (f'<span style="background-color:{colour}; color:white; padding:4px 12px; '
            f'border-radius:12px; font-size:0.8em; font-weight:600;">{category}</span>')


# --- Header banner ---
st.markdown("""
<div style="background-color:#FF5A5F; padding:24px 28px; border-radius:10px; margin-bottom:24px;">
<h1 style="color:white; margin:0; font-size:1.8em;">FirstGlance</h1>
<p style="color:white; margin:6px 0 0 0; opacity:0.9; font-size:0.95em;">
Support queues move fast. FirstGlance gives a team lead a faster read on what needs attention first, with the reasoning shown, not hidden.<br>
A product portfolio project by Sarah Odekwo, modelled on the kind of support challenges a company like Monzo might face.
</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Try It Live", "Queue View"])

# --- Tab 1: Live classification ---
with tab1:
    st.subheader("See how it triages a ticket")
    st.caption("Paste a real-sounding support message below. FirstGlance will sort it by category and "
               "urgency, and tell you why, not just what.")

    ticket_input = st.text_area(
        "Ticket text",
        height=120,
        placeholder="e.g. My card was declined twice today and I don't know why.",
        label_visibility="collapsed"
    )

    if st.button("Classify Ticket", type="primary"):
        if ticket_input.strip():
            with st.spinner("Reading the ticket..."):
                result = classify_ticket(ticket_input)

            if result["category"] == "PARSE_ERROR":
                st.error("Could not parse a clean classification. Raw model output shown below.")
                st.code(result["rationale"])
            else:
                with st.container(border=True):
                    badge_col, _ = st.columns([2, 3])
                    with badge_col:
                        st.markdown(
                            f"{urgency_badge(result['urgency'])} &nbsp; {category_badge(result['category'])}",
                            unsafe_allow_html=True
                        )
                    st.write("")
                    st.markdown(f"**Rationale:** {result['rationale']}")
        else:
            st.warning("Please paste a ticket first.")

# --- Tab 2: Full queue demo ---
with tab2:
    st.subheader("A queue, sorted for you")
    st.caption("150 sample tickets, triaged automatically. This is what a lead would see first thing "
               "in the morning.")

    try:
        df = pd.read_csv("data/full_comparison.csv")

        urgency_order = {"High": 0, "Medium": 1, "Low": 2}
        df["_sort"] = df["predicted_urgency"].map(urgency_order)
        df = df.sort_values("_sort")

        # --- Summary metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tickets", len(df))
        col2.metric("High Priority", len(df[df["predicted_urgency"] == "High"]))
        col3.metric("Medium Priority", len(df[df["predicted_urgency"] == "Medium"]))
        col4.metric("Low Priority", len(df[df["predicted_urgency"] == "Low"]))

        st.write("")

        # --- Chart ---
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Tickets by category**")
            st.bar_chart(df["predicted_category"].value_counts())
        with chart_col2:
            st.markdown("**Tickets by urgency**")
            st.bar_chart(df["predicted_urgency"].value_counts())

        st.write("")

        # --- Filters ---
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            category_filter = st.multiselect(
                "Filter by category",
                options=sorted(df["predicted_category"].unique()),
                default=list(df["predicted_category"].unique())
            )
        with filter_col2:
            urgency_filter = st.multiselect(
                "Filter by urgency",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"]
            )

        df_filtered = df[
            df["predicted_category"].isin(category_filter) &
            df["predicted_urgency"].isin(urgency_filter)
        ]

        st.write("")
        st.markdown(f"**Showing {len(df_filtered)} tickets**")

        # --- Ticket cards ---
        for _, row in df_filtered.iterrows():
            with st.container(border=True):
                header_col, badge_col = st.columns([3, 2])
                with header_col:
                    st.markdown(f"**{row['ticket_id']}**")
                with badge_col:
                    st.markdown(
                        f"{urgency_badge(row['predicted_urgency'])} &nbsp; "
                        f"{category_badge(row['predicted_category'])}",
                        unsafe_allow_html=True
                    )
                st.write(row["ticket_text"])
                st.caption(f"Rationale: {row['rationale']}")

    except FileNotFoundError:
        st.warning("No classified dataset found. Run the classification pipeline first.")