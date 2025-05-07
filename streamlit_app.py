import streamlit as st
import pandas as pd
import altair as alt
import os
import re
import io

# ========== Page Setup ==========
st.set_page_config(page_title="Topic and Sentiment Explorer", layout="wide")

# ========== Authentication ==========
def show_login():
    st.title("🔐 Login Required")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if username in st.secrets["users"] and st.secrets["users"][username] == password:
                st.session_state.authenticated = True
            else:
                st.error("Invalid username or password")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    show_login()
    st.stop()

# ========== Load Data ==========
@st.cache_data
def load_data():
    return pd.read_pickle("youtube_topics_with_sentiment.pkl")

df = load_data()

# ========== Sidebar Filters ==========
st.sidebar.title("Filters")

# Channel filter
if "Channel" in df.columns:
    unique_channels = sorted(df["Channel"].dropna().unique())
    selected_channels = st.sidebar.multiselect("Filter by Channel", unique_channels, default=unique_channels)
else:
    selected_channels = []

sentiment_filter = st.sidebar.selectbox("Filter by Sentiment Label", ["All"] + sorted(df["Sentiment_Label"].unique()))

keyword_query = st.sidebar.text_input("Search Text by Keyword")
with st.sidebar.expander("⚙️ Advanced Search: Text"):
    col1, col2 = st.columns(2)
    with col1:
        fuzzy_match = st.checkbox("Fuzzy Match (Text)", value=True, key="fuzzy_match_key_text")
    with col2:
        regex_enabled = st.checkbox("Use Regex (Text)", value=False, key="regex_enabled_key_text")
keyword_topic = st.sidebar.text_input("Search Topic Keywords")
with st.sidebar.expander("⚙️ Advanced Search: Topic"):
    col1, col2 = st.columns(2)
    with col1:
        fuzzy_topic = st.checkbox("Fuzzy Match (Topic)", value=True, key="fuzzy_match_key_topic")
    with col2:
        regex_topic = st.checkbox("Use Regex (Topic)", value=False, key="regex_enabled_key_topic")
case_sensitive = st.sidebar.checkbox("Case Sensitive", value=False)

# Enforce mutual exclusivity
if regex_enabled and st.session_state.get("fuzzy_match_key_text"):
    st.session_state["fuzzy_match_key_text"] = False
if fuzzy_match and st.session_state.get("regex_enabled_key_text"):
    st.session_state["regex_enabled_key_text"] = False
if regex_topic and st.session_state.get("fuzzy_match_key_topic"):
    st.session_state["fuzzy_match_key_topic"] = False
if fuzzy_topic and st.session_state.get("regex_enabled_key_topic"):
    st.session_state["regex_enabled_key_topic"] = False

unique_topics = df["Topic"].unique()
selected_topics = st.sidebar.multiselect("Select Topic(s)", sorted(unique_topics), default=sorted(unique_topics))

# ========== Apply Filters ==========
filtered_df = df.copy()
if selected_channels:
    filtered_df = filtered_df[filtered_df["Channel"].isin(selected_channels)]
filtered_df["Published Date"] = pd.to_datetime(filtered_df["Published Date"], errors="coerce")
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df["Sentiment_Label"] == sentiment_filter]

# Apply keyword filters
if keyword_query:
    try:
        pattern = keyword_query if regex_enabled else re.escape(keyword_query)
        filtered_df = filtered_df[filtered_df["Text"].str.contains(pattern, case=case_sensitive, regex=True, na=False)]
    except re.error:
        st.warning("⚠️ Invalid regular expression for text search.")

if keyword_topic:
    try:
        pattern_topic = keyword_topic if regex_topic else re.escape(keyword_topic)
        filtered_df = filtered_df[
            filtered_df["Topic_Keywords"].str.contains(pattern_topic, case=case_sensitive, regex=True, na=False)]
    except re.error:
        st.warning("⚠️ Invalid regular expression for topic keyword search.")

filtered_df = filtered_df[filtered_df["Topic"].isin(selected_topics)]

# ========== Tab Layout ==========
tabs = st.tabs(["📄 Overview", "📊 Sentiment Analysis", "🧠 Topic Modeling", "🔀 Topic & Sentiment"])

with tabs[0]:
    st.subheader("📄 Overview")
    df_display = filtered_df.copy()
    if "Column" in df_display.columns:
        df_display = df_display.drop(columns=["Column"])
    cols = ["Topic", "Sentiment_Label"] + [col for col in df_display.columns if col not in ["Topic", "Sentiment_Label"]]
    df_display = df_display[cols]
    st.dataframe(df_display.head(10), use_container_width=True)

    excel_buffer = io.BytesIO()
    filtered_df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)

    st.download_button(
        label="📥 Download Filtered Data as Excel",
        data=excel_buffer,
        file_name="filtered_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with tabs[1]:
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = filtered_df["Sentiment_Label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    sent_chart = alt.Chart(sentiment_counts).mark_bar().encode(
        x=alt.X("Sentiment", title="Sentiment"),
        y=alt.Y("Count", title="Document Count"),
        color=alt.Color("Sentiment",
                        scale=alt.Scale(domain=["positive", "neutral", "negative"], range=["green", "blue", "red"]))
    ).properties(height=300)
    st.altair_chart(sent_chart, use_container_width=True)

    st.subheader("📈 Sentiment Over Time")
    time_data_sent = (
        filtered_df.dropna(subset=["Published Date"])
        .groupby([pd.Grouper(key="Published Date", freq="M"), "Sentiment_Label"])
        .size()
        .reset_index(name="Count")
    )
    normalize_sent = st.checkbox("Normalize Sentiment Over Time", value=False)
    if normalize_sent:
        totals_sent = time_data_sent.groupby("Published Date")["Count"].sum().reset_index(name="Total")
        time_data_sent = time_data_sent.merge(totals_sent, on="Published Date")
        time_data_sent["Proportion"] = time_data_sent["Count"] / time_data_sent["Total"]
        y_sent = alt.Y("Proportion:Q", title="Proportion")
    else:
        y_sent = alt.Y("Count:Q", title="Mentions")

    time_chart_sent = alt.Chart(time_data_sent).mark_line(point=True).encode(
        x=alt.X("Published Date:T", title="Date"),
        y=y_sent,
        color=alt.Color("Sentiment_Label:N",
                        scale=alt.Scale(domain=["positive", "neutral", "negative"], range=["green", "blue", "red"])),
        tooltip=["Published Date:T", "Sentiment_Label", "Count"]
    ).properties(height=400)
    st.altair_chart(time_chart_sent, use_container_width=True)

with tabs[2]:
    st.subheader("📈 Topic Distribution (Pie Chart)")
    topic_counts = filtered_df["Topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    pie_chart = alt.Chart(topic_counts).mark_arc().encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Topic", type="nominal"),
        tooltip=["Topic", alt.Tooltip("Topic_Keywords:N", title="Keywords"), "Count"]
    ).properties(height=400)
    st.altair_chart(pie_chart, use_container_width=True)

    st.subheader("📈 Topic Frequency Over Time")
    time_data_topic = (
        filtered_df.dropna(subset=["Published Date"])
        .groupby([pd.Grouper(key="Published Date", freq="M"), "Topic"])
        .size()
        .reset_index(name="Count")
    )
    normalize_topic = st.checkbox("Normalize Topic Over Time", value=False)
    if normalize_topic:
        totals_topic = time_data_topic.groupby("Published Date")["Count"].sum().reset_index(name="Total")
        time_data_topic = time_data_topic.merge(totals_topic, on="Published Date")
        time_data_topic["Proportion"] = time_data_topic["Count"] / time_data_topic["Total"]
        y_topic = alt.Y("Proportion:Q", title="Proportion")
    else:
        y_topic = alt.Y("Count:Q", title="Mentions")

    time_chart_topic = alt.Chart(time_data_topic).mark_line(point=True).encode(
        x=alt.X("Published Date:T", title="Date"),
        y=y_topic,
        color=alt.Color("Topic:N"),
        tooltip=["Published Date:T", "Topic", "Count"]
    ).properties(height=400)
    st.altair_chart(time_chart_topic, use_container_width=True)

with tabs[3]:
    st.subheader("🔀 Sentiment per Topic")
    intersection_data = (
        filtered_df.groupby(["Topic", "Sentiment_Label"])
        .size()
        .reset_index(name="Count")
    )

    normalize_intersection = st.checkbox("Normalize Charts by Topic", value=True)

    if normalize_intersection:
        topic_totals = intersection_data.groupby("Topic")["Count"].sum().reset_index(name="Total")
        intersection_data = intersection_data.merge(topic_totals, on="Topic")
        intersection_data["Proportion"] = intersection_data["Count"] / intersection_data["Total"]
        y_val = alt.Y("Proportion:Q", title="Proportion")
        fill_val = alt.Color("Proportion:Q", title="Proportion", scale=alt.Scale(scheme="blues"))
    else:
        y_val = alt.Y("Count:Q", title="Total Mentions")
        fill_val = alt.Color("Sentiment_Label:N", title="Sentiment",
                             scale=alt.Scale(domain=["positive", "neutral", "negative"],
                                             range=["green", "blue", "red"]))

    st.markdown("**Stacked Bar Chart**")
    stacked_chart = alt.Chart(intersection_data).mark_bar().encode(
        x=alt.X("Topic:N", title="Topic"),
        y=y_val,
        color=alt.Color("Sentiment_Label:N", title="Sentiment",
                        scale=alt.Scale(domain=["positive", "neutral", "negative"], range=["green", "blue", "red"])),
        tooltip=["Topic", "Sentiment_Label", "Count"]
    ).properties(height=400)
    st.altair_chart(stacked_chart, use_container_width=True)

    st.markdown("**Heatmap (Proportional by Topic)**")
    heatmap = alt.Chart(intersection_data).mark_rect().encode(
        x=alt.X("Sentiment_Label:N", title="Sentiment"),
        y=alt.Y("Topic:N", title="Topic"),
        color=fill_val,
        tooltip=["Topic", "Sentiment_Label", "Proportion"]
    )
    st.altair_chart(heatmap, use_container_width=True)
