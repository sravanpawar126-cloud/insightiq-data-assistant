import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import json
from google import genai

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="InsightIQ | Data Assistant",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 1.5rem;
}
.hero {
    padding: 24px;
    border-radius: 16px;
    background: linear-gradient(120deg, #172554, #2563eb);
    color: white;
    margin-bottom: 20px;
}
.hero h1 {
    color: white;
    margin-bottom: 5px;
}
.hero p {
    color: #e0e7ff;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e2e8f0;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>📊 InsightIQ</h1>
    <p>GenAI-Based Data Analytics and Data Engineering Assistant</p>
    <p>Explore data, identify quality issues, create visualizations,
    and ask AI questions about your dataset.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Project Settings")
st.sidebar.write("Upload a CSV dataset to begin.")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

st.sidebar.divider()
st.sidebar.subheader("🤖 Gemini AI")
api_key = st.sidebar.text_input(
    "Gemini API key",
    type="password",
    help="Enter your Gemini API key to enable AI questions."
)
model_name = st.sidebar.selectbox(
    "AI model",
    ["gemini-3.8-flash", "gemini-3.8-flash"]
)

# ---------------- HELPER FUNCTIONS ----------------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

def dataset_fingerprint(df):
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()

def get_quality_report(df):
    report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing Values": [int(df[col].isna().sum()) for col in df.columns],
        "Missing (%)": [
            round(float(df[col].isna().mean() * 100), 2)
            for col in df.columns
        ],
        "Unique Values": [int(df[col].nunique(dropna=True)) for col in df.columns]
    })
    return report

def dataset_summary(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(map(str, df.columns)),
        "data_types": {str(c): str(df[c].dtype) for c in df.columns},
        "missing_values": {
            str(c): int(df[c].isna().sum()) for c in df.columns
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "sample_rows": df.head(5).fillna("").astype(str).to_dict(orient="records")
    }

# ---------------- WELCOME SCREEN ----------------
if uploaded_file is None:
    st.info("👈 Upload a CSV file using the sidebar to start analysing your data.")

    st.subheader("What can InsightIQ do?")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 📈 Analytics")
        st.write("Explore dataset statistics and discover patterns.")

    with c2:
        st.markdown("### 🧹 Data Quality")
        st.write("Find missing values, duplicates, and data type issues.")

    with c3:
        st.markdown("### 🤖 Generative AI")
        st.write("Ask questions about your dataset using Gemini.")

    st.stop()

# ---------------- LOAD DATA ----------------
try:
    file_bytes = uploaded_file.getvalue()
    df_original = load_csv(file_bytes)
    df_original.columns = [str(c) for c in df_original.columns]
except Exception as e:
    st.error(f"Unable to read this CSV file: {e}")
    st.stop()

if df_original.empty:
    st.warning("The uploaded CSV has no data rows.")
    st.stop()

# Keep a separate working copy for cleaning
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = df_original.copy()

file_hash = hashlib.md5(file_bytes).hexdigest()
if st.session_state.get("current_file_hash") != file_hash:
    st.session_state.current_file_hash = file_hash
    st.session_state.cleaned_df = df_original.copy()

df = st.session_state.cleaned_df

st.caption(
    f"File: **{uploaded_file.name}** · "
    f"Original dataset: {df_original.shape[0]:,} rows × "
    f"{df_original.shape[1]:,} columns"
)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Overview",
    "🔎 Data Quality",
    "📊 Visualizations",
    "🤖 Ask AI",
    "🧹 Data Cleaning"
])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.header("Dataset Overview")
    st.write("A quick summary of your uploaded dataset.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", f"{df.shape[1]:,}")
    m3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    m4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Column Data Types")
        dtype_df = df.dtypes.astype(str).value_counts().reset_index()
        dtype_df.columns = ["Data Type", "Count"]
        fig = px.pie(
            dtype_df,
            names="Data Type",
            values="Count",
            hole=0.45,
            title="Distribution of Data Types"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Numerical Summary")
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().T, use_container_width=True)
        else:
            st.info("No numerical columns are available.")

    with st.expander("View all column names"):
        st.write(list(df.columns))

# ==================== TAB 2: DATA QUALITY ====================
with tab2:
    st.header("Data Quality Analysis")
    st.write("Check missing values, duplicates, and column characteristics.")

    quality = get_quality_report(df)
    st.subheader("Column Quality Report")
    st.dataframe(quality, use_container_width=True, hide_index=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Missing Values by Column")
        missing_df = quality[quality["Missing Values"] > 0]
        if not missing_df.empty:
            fig = px.bar(
                missing_df,
                x="Column",
                y="Missing Values",
                color="Missing Values",
                title="Missing Values"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found.")

    with right:
        st.subheader("Duplicate Row Profile")
        duplicate_count = int(df.duplicated().sum())
        unique_count = int(len(df) - duplicate_count)

        duplicate_df = pd.DataFrame({
            "Row Type": ["Unique rows", "Duplicate rows"],
            "Count": [unique_count, duplicate_count]
        })
        fig = px.pie(
            duplicate_df,
            names="Row Type",
            values="Count",
            hole=0.45,
            title="Unique vs Duplicate Rows"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Health")
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())

    if total_cells:
        completeness = 100 * (total_cells - missing_cells) / total_cells
    else:
        completeness = 0

    st.metric("Cell Completeness", f"{completeness:.2f}%")
    st.progress(min(max(completeness / 100, 0.0), 1.0))

# ==================== TAB 3: VISUALIZATIONS ====================
with tab3:
    st.header("Interactive Data Visualizations")
    st.write("Choose columns to explore patterns in your dataset.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()
    categorical_cols = [
        c for c in all_cols
        if c not in numeric_cols
    ]

    chart_type = st.selectbox(
        "Choose visualization",
        [
            "Numerical Distribution",
            "Category Comparison",
            "Scatter Plot",
            "Correlation Heatmap"
        ]
    )

    if chart_type == "Numerical Distribution":
        if numeric_cols:
            col = st.selectbox("Select numerical column", numeric_cols)
            fig = px.histogram(
                df,
                x=col,
                nbins=30,
                marginal="box",
                title=f"Distribution of {col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numerical columns found.")

    elif chart_type == "Category Comparison":
        possible_cols = categorical_cols if categorical_cols else all_cols
        col = st.selectbox("Select category column", possible_cols)
        top_n = st.slider("Number of categories to display", 5, 30, 10)

        counts = df[col].fillna("Missing").astype(str).value_counts().head(top_n)
        chart_df = counts.rename_axis(col).reset_index(name="Count")

        fig = px.bar(
            chart_df,
            x=col,
            y="Count",
            title=f"Top {top_n} values in {col}"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis column", numeric_cols, index=0)
            y_col = st.selectbox(
                "Y-axis column",
                numeric_cols,
                index=1 if len(numeric_cols) > 1 else 0
            )
            color_options = ["None"] + categorical_cols
            color_col = st.selectbox("Color by (optional)", color_options)

            if color_col == "None":
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            else:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{y_col} vs {x_col}"
                )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("A scatter plot requires at least two numerical columns.")

    elif chart_type == "Correlation Heatmap":
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Numerical Feature Correlation"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("A correlation heatmap requires at least two numerical columns.")

# ==================== TAB 4: ASK AI ====================
with tab4:
    st.header("Ask AI About Your Dataset")
    st.write(
        "Ask questions about patterns, columns, data quality, or possible "
        "analytical approaches. Your API key is used for this request."
    )

    question = st.text_area(
        "Enter your question",
        placeholder="Example: Which columns have the most missing values?",
        height=100
    )

    if st.button("✨ Ask Gemini", type="primary"):
        if not api_key:
            st.warning("Enter your Gemini API key in the sidebar first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                client = genai.Client(api_key=api_key)
                summary = dataset_summary(df)

                prompt = f"""
You are InsightIQ, a helpful data analytics assistant.
Answer the user's question using the dataset summary below.
Be clear and beginner-friendly. Do not claim to have analysed rows
or calculated results that are not included in the summary.
If the summary is insufficient, say what extra analysis is needed.

DATASET SUMMARY:
{json.dumps(summary, ensure_ascii=False, default=str)}

USER QUESTION:
{question}
"""
                with st.spinner("Gemini is analysing the dataset summary..."):
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )

                st.subheader("AI Response")
                st.markdown(response.text or "The AI returned an empty response.")

            except Exception as e:
                st.error(f"Gemini request failed: {e}")
                st.info(
                    "Check your internet connection, API key, model availability, "
                    "and whether your key has access to the selected model."
                )

    st.caption(
        "Privacy note: this feature sends a dataset summary and sample rows "
        "to Gemini when you ask a question. Avoid uploading confidential data."
    )

# ==================== TAB 5: DATA CLEANING ====================
with tab5:
    st.header("Data Cleaning")
    st.write(
        "Choose cleaning operations. These changes affect the working copy "
        "in this app, not the original CSV file on your computer."
    )

    st.subheader("Missing Values")
    missing_strategy = st.selectbox(
        "How should missing values be handled?",
        [
            "Do not change missing values",
            "Remove rows with missing values",
            "Fill numerical columns with mean",
            "Fill numerical columns with median",
            "Fill missing values with 0 / 'Missing'"
        ]
    )

    st.subheader("Duplicate Rows")
    remove_duplicates = st.checkbox("Remove duplicate rows")

    st.subheader("Outlier Detection")
    outlier_col = st.selectbox(
        "Numerical column for IQR outlier detection",
        ["None"] + numeric_cols
    )
    remove_outliers = st.checkbox(
        "Remove outlier rows from the selected column",
        disabled=(outlier_col == "None")
    )

    if st.button("🧹 Apply Cleaning", type="primary"):
        cleaned = df_original.copy()
        messages = []

        if missing_strategy == "Remove rows with missing values":
            before = len(cleaned)
            cleaned = cleaned.dropna()
            messages.append(f"Removed {before - len(cleaned)} rows with missing values.")

        elif missing_strategy == "Fill numerical columns with mean":
            for col in cleaned.select_dtypes(include="number").columns:
                if cleaned[col].isna().any():
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
            messages.append("Filled missing numerical values with column means.")

        elif missing_strategy == "Fill numerical columns with median":
            for col in cleaned.select_dtypes(include="number").columns:
                if cleaned[col].isna().any():
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            messages.append("Filled missing numerical values with column medians.")

        elif missing_strategy == "Fill missing values with 0 / 'Missing'":
            for col in cleaned.columns:
                if cleaned[col].isna().any():
                    if pd.api.types.is_numeric_dtype(cleaned[col]):
                        cleaned[col] = cleaned[col].fillna(0)
                    else:
                        cleaned[col] = cleaned[col].fillna("Missing")
            messages.append("Filled missing values with 0 or 'Missing'.")

        if remove_duplicates:
            before = len(cleaned)
            cleaned = cleaned.drop_duplicates()
            messages.append(f"Removed {before - len(cleaned)} duplicate rows.")

        if remove_outliers and outlier_col != "None":
            series = cleaned[outlier_col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            before = len(cleaned)
            cleaned = cleaned[
                series.isna() | series.between(lower, upper)
            ]
            messages.append(
                f"Removed {before - len(cleaned)} outlier rows from {outlier_col}."
            )

        st.session_state.cleaned_df = cleaned
        st.success("Cleaning completed.")
        for message in messages:
            st.write("• " + message)

        st.write(
            f"Rows: {len(df_original):,} → {len(cleaned):,} | "
            f"Columns: {df_original.shape[1]:,} → {cleaned.shape[1]:,}"
        )
        st.dataframe(cleaned.head(10), use_container_width=True)

    current_cleaned = st.session_state.cleaned_df
    csv_data = current_cleaned.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Current Cleaned CSV",
        data=csv_data,
        file_name="insightiq_cleaned_data.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption("InsightIQ | GenAI-Based Data Analytics and Data Engineering Assistant")
