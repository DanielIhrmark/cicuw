import streamlit as st
import pandas as pd
import altair as alt
import os
import re
import io
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components

from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from datetime import datetime             
from sklearn.feature_extraction.text import TfidfVectorizer


# ========== Page Setup ==========
st.set_page_config(page_title="CICuW Datahhub", layout="wide")

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
tabs = st.tabs(["📄 Overview", "📊 Sentiment Analysis", "🧠 Topic Modeling", "📂Static Topic Visualizations", "🔀 Topic & Sentiment", "🫧 Bubble Graph: Frequency vs. Channel Overlap", "🧠 Channel-Specific Word Signatures (TF-IDF)"])

with tabs[0]:
    st.subheader("📄 Overview")

    # Copy and clean display data
    df_display = filtered_df.copy()
    if "Column" in df_display.columns:
        df_display = df_display.drop(columns=["Column"])

    # Define and validate preferred columns
    preferred_columns = ["Published Date", "Channel", "URL", "Topic", "Sentiment_Label", "Text", "Sentiment_Negative", "Sentiment_Neutral", "Sentiment_Positive"]
    existing_columns = [col for col in preferred_columns if col in df_display.columns]
    if not existing_columns:
        existing_columns = df_display.columns.tolist()

    # === Row count summary ===
    st.markdown(f"**🔢 Filtered documents: {len(df_display)}**")

    # === Sunburst chart: Channel (inner) > Sentiment (outer) ===
    if all(col in df_display.columns for col in ["Channel", "Sentiment_Label"]):
        channel_sentiment_counts = (
            df_display.groupby(["Channel", "Sentiment_Label"])
            .size()
            .reset_index(name="Count")
        )

        fig = px.sunburst(
            channel_sentiment_counts,
            path=["Channel", "Sentiment_Label"],
            values="Count",
            color="Sentiment_Label",
            color_discrete_map={
                "positive": "green",
                "neutral": "blue",
                "negative": "red"
            },
            title="Channel and Sentiment Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)


    # === Max rows slider (just above table) ===
    max_rows = st.slider("Max rows to display", 10, 500, 100, step=10)

    # === Data preview table ===
    st.dataframe(df_display[existing_columns].head(max_rows), use_container_width=True)

    # === Download button ===
    excel_buffer = io.BytesIO()
    df_display[existing_columns].head(max_rows).to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)

    st.download_button(
        label="📥 Download Displayed Data as Excel",
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
    
    st.subheader("🗝️ Topic Keywords Reference")
    # Create and display a simple topic-keywords reference table
    topic_keywords_table = (
        filtered_df[["Topic", "Topic_Keywords"]]
        .drop_duplicates()
        .sort_values("Topic")
        .reset_index(drop=True)
    )

    st.dataframe(topic_keywords_table, use_container_width=True)

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
    st.subheader("📂 Topic Visualizations (HTML)")

    def display_html_file(file_path, height=1000):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=height, scrolling=True)
        else:
            st.warning(f"⚠️ File not found: {file_path}")

    display_html_file("hierarchy.html", height=600)
    display_html_file("intertopic_map.html", height=600)
    display_html_file("heatmap.html", height=600)


with tabs[4]:
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

# Ensure stopwords are available
nltk.download('stopwords')
stop_words = set(stopwords.words('english') + stopwords.words('swedish'))

with tabs[5]:
    st.subheader("🫧 Bubble Graph: Frequency vs. Channel Overlap")

    available_channels = sorted(filtered_df["Channel"].dropna().unique())
    selected = st.multiselect("Select channels to compare", available_channels, default=available_channels[:3])

    if len(selected) < 2:
        st.info("Please select at least 2 channels to generate the bubble graph.")
    else:
        min_freq = st.slider("Minimum Word Frequency", 1, 20, 3)
        max_words = st.slider("Maximum Number of Words Displayed", 50, 500, 100, step=50)
        
        # === Filter Toggles ===
        show_shared_only = st.checkbox("Show only shared words (2+ channels)", value=False)
        show_unique_only = st.checkbox("Show only unique words (1 channel only)", value=False)

        word_freq_total = Counter()
        word_channel_map = defaultdict(set)

        def get_word_freq(df, channel):
            texts = df[df["Channel"] == channel]["Text"].dropna().str.cat(sep=" ")
            words = re.findall(r"\b\w+\b", texts.lower())
            words = [w for w in words if w not in stop_words and len(w) > 1]
            return Counter(words)

        for channel in selected:
            freq = get_word_freq(filtered_df, channel)
            filtered_freq = {word: count for word, count in freq.items() if count >= min_freq}
            for word, count in filtered_freq.items():
                word_freq_total[word] += count
                word_channel_map[word].add(channel)

        rows = []
        for word, chans in word_channel_map.items():
            count = word_freq_total[word]
            chan_count = len(chans)
            if count >= min_freq:
                if show_shared_only and chan_count < 2:
                    continue
                if show_unique_only and chan_count > 1:
                    continue
                rows.append({
                    "word": word,
                    "frequency": count,
                    "channels": ", ".join(sorted(chans)),
                    "channel_count": chan_count
                })

        df_bubble = pd.DataFrame(rows)

        if df_bubble.empty:
            st.warning("No words meet the frequency threshold and selected filters.")
            st.stop()

        df_bubble = df_bubble.sort_values("frequency", ascending=False).head(max_words)

        fig = px.scatter(
            df_bubble,
            x="frequency",
            y="channel_count",
            size="frequency",
            color="channel_count",
            hover_name="word",
            hover_data=["frequency", "channels"],
            text="word"
        )

        fig.update_traces(
            textposition="top center",
            marker=dict(line=dict(width=0.5, color="black")),
            textfont=dict(size=14)
        )

        fig.update_layout(
            height=700,
            title="Bubble Graph: Word Frequency vs. Channel Overlap<br><sub>X = Frequency, Y = Number of Channels</sub>",
            xaxis_title="Word Frequency (Total)",
            yaxis_title="Number of Channels Word Appears In",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

with tabs[6]:
        st.subheader("🧠 Channel-Specific Word Signatures (TF-IDF)")

    # === TF-IDF Parameters ===
    tfidf_top_n = st.slider("Top N words per channel (TF-IDF)", 5, 30, 10)
    tfidf_min_freq = st.slider("Minimum word frequency for TF-IDF", 1, 10, 3)

    # === Prepare channel texts ===
    channel_docs = []
    valid_channels = []
    for channel in selected:
        texts = filtered_df[filtered_df["Channel"] == channel]["Text"].dropna().tolist()
        full_text = " ".join(texts).lower()
        words = re.findall(r"\b\w+\b", full_text)
        words = [w for w in words if w not in stop_words and len(w) > 1]
        freq = Counter(words)
        if sum(freq.values()) >= tfidf_min_freq:
            valid_channels.append(channel)
            channel_docs.append(" ".join([w for w in words if freq[w] >= tfidf_min_freq]))

    if len(channel_docs) < 2:
        st.warning("Not enough valid channels with sufficient text to compute TF-IDF.")
    else:
        # === Run TF-IDF ===
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(channel_docs)
        terms = vectorizer.get_feature_names_out()

        for i, channel in enumerate(valid_channels):
            scores = tfidf_matrix[i].toarray().flatten()
            top_indices = scores.argsort()[::-1][:tfidf_top_n]
            top_terms = [terms[j] for j in top_indices]
            top_scores = [scores[j] for j in top_indices]

            df_tfidf = pd.DataFrame({
                "word": top_terms,
                "score": top_scores
            })

            st.markdown(f"**Top {tfidf_top_n} TF-IDF words for channel: _{channel}_**")
            tfidf_fig = px.bar(df_tfidf, x="score", y="word", orientation="h",
                               labels={"score": "TF-IDF Score", "word": "Word"},
                               height=300)
            tfidf_fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(tfidf_fig, use_container_width=True)
