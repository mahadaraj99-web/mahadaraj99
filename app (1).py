import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    if os.path.exists("data/clean.csv"):
        return pd.read_csv("data/clean.csv")
    if os.path.exists("data/raw.csv"):
        return pd.read_csv("data/raw.csv")
    np.random.seed(42)
    return pd.DataFrame({
        "category": np.random.choice(["A", "B", "C"], 100),
        "sales": np.random.uniform(100, 1000, 100).round(2),
        "orders": np.random.randint(1, 20, 100),
        "profit": np.random.uniform(10, 300, 100).round(2)
    })


df = load_data()
target_col = "profit" if "profit" in df.columns else df.columns[-1]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns if c != target_col]

st.sidebar.header("Filters")
filtered_df = df.copy()

for col in cat_cols[:2]:
    options = ["All"] + sorted(filtered_df[col].dropna().astype(str).unique().tolist())
    selected = st.sidebar.selectbox(col, options)
    if selected != "All":
        filtered_df = filtered_df[filtered_df[col].astype(str) == selected]

for col in num_cols[:2]:
    min_val = float(filtered_df[col].min())
    max_val = float(filtered_df[col].max())
    if min_val < max_val:
        val_range = st.sidebar.slider(col, min_val, max_val, (min_val, max_val))
        filtered_df = filtered_df[
            (filtered_df[col] >= val_range[0]) & (filtered_df[col] <= val_range[1])
        ]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows", len(filtered_df))
c2.metric(
    f"Mean {target_col}",
    f"{filtered_df[target_col].mean():.2f}" if len(filtered_df) > 0 else "0.00",
)
c3.metric(
    f"Sum {target_col}",
    f"{filtered_df[target_col].sum():.2f}" if len(filtered_df) > 0 else "0.00",
)
main_num = num_cols[0] if num_cols else target_col
c4.metric(
    f"Mean {main_num}",
    f"{filtered_df[main_num].mean():.2f}" if len(filtered_df) > 0 else "0.00",
)

if len(filtered_df) > 0 and num_cols:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    med_val = filtered_df[main_num].median()
    high_val = filtered_df[filtered_df[main_num] >= med_val][target_col].mean()
    low_val = filtered_df[filtered_df[main_num] < med_val][target_col].mean()
    ax1.bar(["Above Median", "Below Median"], [high_val, low_val])
    ax1.set_title(f"Average {target_col} by {main_num}")
    ax1.set_xlabel(main_num)
    ax1.set_ylabel(f"Mean {target_col}")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(filtered_df[main_num], filtered_df[target_col], alpha=0.6)
    ax2.set_title(f"{main_num} vs {target_col}")
    ax2.set_xlabel(main_num)
    ax2.set_ylabel(target_col)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.hist(filtered_df[target_col].dropna(), bins=15)
    ax3.set_title(f"Distribution of {target_col}")
    ax3.set_xlabel(target_col)
    ax3.set_ylabel("Count")
    st.pyplot(fig3)
