"""
Usage Statistics Dashboard for the SilentCodingLegend AI agent.
This page displays usage statistics and analytics for OpenRouter API calls.
"""

import streamlit as st
from src.utils import apply_dark_theme
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import numpy as np

from src.usage_tracker import UsageTracker

# Page configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="Usage Statistics",
    page_icon="📊",
    layout="wide"
)

# Apply dark theme using utility function
apply_dark_theme()

# Function to format large numbers with commas
def format_number(num):
    return f"{num:,}"

# Function to format cost
def format_cost(cost):
    return f"${cost:.4f}"

def main():
    st.title("📊 Usage Statistics Dashboard")
    
    # Get usage statistics
    tracker = UsageTracker()
    current_usage = tracker.get_monthly_summary()
    
    # Load usage logs for historical data
    usage_logs_dir = Path("/home/silentlegendkali/scl-openrouter/usage_logs")
    usage_logs_dir.mkdir(exist_ok=True)
    
    usage_files = list(usage_logs_dir.glob("usage_*.json"))
    
    # Monthly selector
    available_months = []
    for file in usage_files:
        month_str = file.stem.replace("usage_", "")
        try:
            # Parse month to ensure it's valid
            year, month = month_str.split("-")
            month_name = f"{calendar.month_name[int(month)]} {year}"
            available_months.append((month_str, month_name))
        except:
            continue
    
    # Sort months chronologically
    available_months.sort(key=lambda x: x[0])
    
    # Add "Current Month" option
    current_month_str = datetime.now().strftime("%Y-%m")
    if not any(m[0] == current_month_str for m in available_months):
        current_month_name = datetime.now().strftime("%B %Y")
        available_months.append((current_month_str, f"{current_month_name} (Current)"))
    
    # Default to the most recent month
    default_month_idx = len(available_months) - 1 if available_months else 0
    
    # Month selection
    if available_months:
        month_options = [m[1] for m in available_months]
        month_keys = [m[0] for m in available_months]
        selected_month_name = st.selectbox("Select Month:", month_options, index=default_month_idx)
        selected_month = month_keys[month_options.index(selected_month_name)]
        
        # Load the selected month's data
        usage_file = usage_logs_dir / f"usage_{selected_month}.json"
        
        if usage_file.exists():
            with open(usage_file, 'r') as f:
                monthly_data = json.load(f)
        else:
            # If file doesn't exist but it's the current month, use the current usage
            if selected_month == current_month_str:
                monthly_data = current_usage
            else:
                st.warning(f"No usage data found for {selected_month_name}")
                return
    else:
        st.warning("No usage data found")
        return
    
    # Display summary statistics
    st.header("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Tokens", 
            value=format_number(monthly_data.get("total_tokens", 0))
        )
    
    with col2:
        st.metric(
            label="Total Requests", 
            value=format_number(monthly_data.get("requests", 0))
        )
    
    with col3:
        st.metric(
            label="Total Conversations", 
            value=format_number(monthly_data.get("conversations", 0))
        )
    
    with col4:
        # Calculate estimated cost based on different model rates
        # Using an average rate for estimation
        est_cost = tracker.get_estimated_cost(0.002, 0.002)
        st.metric(
            label="Estimated Cost", 
            value=format_cost(est_cost)
        )
    
    # Display token usage breakdown
    st.subheader("Token Usage Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Token usage pie chart
        fig, ax = plt.subplots()
        labels = ['Prompt Tokens', 'Completion Tokens']
        sizes = [
            monthly_data.get("prompt_tokens", 0),
            monthly_data.get("completion_tokens", 0)
        ]
        
        # Only create the chart if we have tokens
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)
        else:
            st.info("No token usage data available for this period")
    
    with col2:
        # Create a dataframe for the token usage
        token_data = {
            'Type': ['Prompt Tokens', 'Completion Tokens', 'Total Tokens'],
            'Count': [
                monthly_data.get("prompt_tokens", 0),
                monthly_data.get("completion_tokens", 0),
                monthly_data.get("total_tokens", 0)
            ]
        }
        token_df = pd.DataFrame(token_data)
        st.table(token_df)
    
    # Model usage section
    st.header("Model Usage Statistics")
    
    # Extract model data
    models_data = monthly_data.get("models", {})
    
    if not models_data:
        st.info("No model usage data available for this period")
    else:
        # Prepare data for visualization
        model_names = []
        model_tokens = []
        model_requests = []
        
        for model_id, model_stats in models_data.items():
            model_names.append(model_id)
            model_tokens.append(model_stats.get("total_tokens", 0))
            model_requests.append(model_stats.get("requests", 0))
        
        # Create a dataframe for the model usage
        model_df = pd.DataFrame({
            'Model': model_names,
            'Total Tokens': model_tokens,
            'Requests': model_requests
        })
        
        # Sort by total tokens (highest first)
        model_df = model_df.sort_values('Total Tokens', ascending=False).reset_index(drop=True)
        
        # Display model usage table
        st.subheader("Model Usage Table")
        st.dataframe(model_df)
        
        # Model usage bar charts
        st.subheader("Model Usage Comparison")
        
        # Only display charts if we have data
        if len(model_names) > 0:
            tabs = st.tabs(["Tokens", "Requests"])
            
            with tabs[0]:  # Tokens tab
                fig, ax = plt.subplots(figsize=(10, 6))
                bar_plot = sns.barplot(x='Total Tokens', y='Model', data=model_df, ax=ax)
                
                # Add value labels to the bars
                for i, v in enumerate(model_df['Total Tokens']):
                    ax.text(v + 0.1, i, format_number(v), va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tabs[1]:  # Requests tab
                fig, ax = plt.subplots(figsize=(10, 6))
                bar_plot = sns.barplot(x='Requests', y='Model', data=model_df, ax=ax)
                
                # Add value labels to the bars
                for i, v in enumerate(model_df['Requests']):
                    ax.text(v + 0.1, i, format_number(v), va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Task type usage
    st.header("Task Type Analysis")
    
    # Extract task types from all models
    task_types = {}
    
    for model_id, model_stats in models_data.items():
        model_task_types = model_stats.get("task_types", {})
        
        for task_type, task_stats in model_task_types.items():
            if task_type not in task_types:
                task_types[task_type] = {
                    "total_tokens": 0,
                    "requests": 0
                }
            
            task_types[task_type]["total_tokens"] += task_stats.get("total_tokens", 0)
            task_types[task_type]["requests"] += task_stats.get("requests", 0)
    
    if not task_types:
        st.info("No task type data available for this period")
    else:
        # Create dataframe for task types
        task_df = pd.DataFrame({
            'Task Type': list(task_types.keys()),
            'Total Tokens': [task_stats["total_tokens"] for task_stats in task_types.values()],
            'Requests': [task_stats["requests"] for task_stats in task_types.values()]
        })
        
        # Sort by total tokens (highest first)
        task_df = task_df.sort_values('Total Tokens', ascending=False).reset_index(drop=True)
        
        # Display task type table
        st.subheader("Task Type Usage Table")
        st.dataframe(task_df)
        
        # Task type pie charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Tokens by task type
            st.subheader("Tokens by Task Type")
            fig, ax = plt.subplots()
            ax.pie(task_df['Total Tokens'], labels=task_df['Task Type'], autopct='%1.1f%%')
            ax.axis('equal')
            st.pyplot(fig)
        
        with col2:
            # Requests by task type
            st.subheader("Requests by Task Type")
            fig, ax = plt.subplots()
            ax.pie(task_df['Requests'], labels=task_df['Task Type'], autopct='%1.1f%%')
            ax.axis('equal')
            st.pyplot(fig)
    
    # Usage trends over time (future enhancement)
    st.header("Advanced Analytics")
    st.info("More detailed analytics features coming soon!")
    
    # API cost calculator
    st.header("API Cost Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Calculate Estimated Cost")
        prompt_price = st.number_input("Price per 1K Prompt Tokens ($)", value=0.002, step=0.0001, format="%.4f")
        completion_price = st.number_input("Price per 1K Completion Tokens ($)", value=0.002, step=0.0001, format="%.4f")
        
        prompt_tokens = monthly_data.get("prompt_tokens", 0)
        completion_tokens = monthly_data.get("completion_tokens", 0)
        
        calc_prompt_cost = (prompt_tokens / 1000) * prompt_price
        calc_completion_cost = (completion_tokens / 1000) * completion_price
        total_calc_cost = calc_prompt_cost + calc_completion_cost
        
        st.write(f"Prompt Tokens: {format_number(prompt_tokens)} = {format_cost(calc_prompt_cost)}")
        st.write(f"Completion Tokens: {format_number(completion_tokens)} = {format_cost(calc_completion_cost)}")
        st.write(f"**Total Estimated Cost: {format_cost(total_calc_cost)}**")
    
    with col2:
        st.subheader("Cost Breakdown by Model")
        
        # Create a dataframe with model costs
        model_costs = []
        for model_id, model_stats in models_data.items():
            model_prompt_tokens = model_stats.get("prompt_tokens", 0)
            model_completion_tokens = model_stats.get("completion_tokens", 0)
            
            model_prompt_cost = (model_prompt_tokens / 1000) * prompt_price
            model_completion_cost = (model_completion_tokens / 1000) * completion_price
            model_total_cost = model_prompt_cost + model_completion_cost
            
            model_costs.append({
                "Model": model_id,
                "Estimated Cost": model_total_cost
            })
        
        if model_costs:
            model_costs_df = pd.DataFrame(model_costs)
            model_costs_df = model_costs_df.sort_values("Estimated Cost", ascending=False)
            model_costs_df["Estimated Cost"] = model_costs_df["Estimated Cost"].apply(format_cost)
            st.table(model_costs_df)
        else:
            st.info("No model data available for cost calculation")
    
    # Export options
    st.header("Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export as CSV"):
            # Create a DataFrame from the monthly data
            export_df = pd.DataFrame({
                "Metric": ["Total Tokens", "Prompt Tokens", "Completion Tokens", "Requests", "Conversations", "Estimated Cost"],
                "Value": [
                    monthly_data.get("total_tokens", 0),
                    monthly_data.get("prompt_tokens", 0),
                    monthly_data.get("completion_tokens", 0),
                    monthly_data.get("requests", 0),
                    monthly_data.get("conversations", 0),
                    est_cost
                ]
            })
            
            # Convert to CSV
            csv = export_df.to_csv(index=False)
            
            # Create a download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"usage_stats_{selected_month}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export as JSON"):
            # Convert to JSON
            export_json = json.dumps(monthly_data, indent=2)
            
            # Create a download button
            st.download_button(
                label="Download JSON",
                data=export_json,
                file_name=f"usage_stats_{selected_month}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()