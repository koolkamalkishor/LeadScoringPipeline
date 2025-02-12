import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import sys
from sqlalchemy import create_engine
import requests
import logging

# logging.basicConfig(level=logging.INFO)
# Configure logging to output to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lead_scoring.log"),
        logging.StreamHandler()
    ]
)
# Page config
st.set_page_config(page_title="Lead Scoring Dashboard", layout="wide")

# Database connection
@st.cache_resource
def get_database_connection():
    DATABASE_URL = "postgresql://doadmin:AVNS_HnP29MJc4B156XYIhO8@db-postgresql-blr1-67276-do-user-18857318-0.l.db.ondigitalocean.com:25060/agentkamal?sslmode=require"
    return create_engine(DATABASE_URL)

# HuggingFace setup
@st.cache_resource
def setup_huggingface():
    return st.secrets["HUGGINGFACE_API_KEY"]

def query_llm(prompt, max_length=200):
    """Query HuggingFace's inference API"""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {setup_huggingface()}"}
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "return_full_text": False
                }
            }
        )
        return response.json()[0]['generated_text']
    except Exception as e:
        return f"LLM analysis temporarily unavailable: {str(e)}"

def ensure_timezone_aware(dt):
    """Ensure a datetime is timezone aware, converting to UTC if it isn't"""
    if dt is pd.NaT:
        return dt
    if dt.tzinfo is None:
        return pytz.UTC.localize(dt)
    return dt.astimezone(pytz.UTC)

@st.cache_data
def load_data():
    """Load and prepare data from database"""
    try:
        engine = get_database_connection()
        
        with engine.connect() as conn:
            # Query leads
            leads_query = '''
                SELECT 
                    "Id", "FirstName", "LastName", "Email", "Phone", "Company",
                    "Industry", "LeadSource", "Status", "IsConverted",
                    "ConvertedContactId", "ConvertedAccountId", 
                    "CreatedDate", "LastModifiedDate", "IsDeleted"
                FROM "Lead" 
                WHERE "IsDeleted" = FALSE
            '''
            leads = pd.read_sql(leads_query, conn)
            
            # Query contacts
            contacts_query = '''
                SELECT 
                    "Id", "FirstName", "LastName", "Email", "Phone",
                    "AccountId", "CreatedDate", "LastModifiedDate", "IsDeleted"
                FROM "Contact" 
                WHERE "IsDeleted" = FALSE
            '''
            contacts = pd.read_sql(contacts_query, conn)
            
            # Query accounts
            accounts_query = '''
                SELECT 
                    "Id", "Name", "Industry", "Type", "Phone",
                    "CreatedDate", "LastModifiedDate", "IsDeleted"
                FROM "Account" 
                WHERE "IsDeleted" = FALSE
            '''
            accounts = pd.read_sql(accounts_query, conn)

        # Handle DateTime columns
        datetime_columns = ['CreatedDate', 'LastModifiedDate']
        for col in datetime_columns:
            if col in leads.columns:
                leads[col] = pd.to_datetime(leads[col]).apply(ensure_timezone_aware)
            if col in contacts.columns:
                contacts[col] = pd.to_datetime(contacts[col]).apply(ensure_timezone_aware)
            if col in accounts.columns:
                accounts[col] = pd.to_datetime(accounts[col]).apply(ensure_timezone_aware)

        # Convert ID columns to string
        id_columns = ['ConvertedContactId', 'ConvertedAccountId', 'Id', 'AccountId']
        for df in [leads, contacts, accounts]:
            for col in id_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)

        # Merge datasets
        data = pd.merge(
            leads, 
            contacts, 
            left_on="ConvertedContactId", 
            right_on="Id", 
            how="left",
            suffixes=('', '_contact')
        )
        
        data = pd.merge(
            data, 
            accounts, 
            left_on="ConvertedAccountId", 
            right_on="Id", 
            how="left",
            suffixes=('', '_account')
        )

        # Process data
        data['LastActivityDate'] = pd.to_datetime(data['LastModifiedDate']).apply(ensure_timezone_aware)
        data['Industry'] = data['Industry'].fillna('Unknown')
        data['LeadSource'] = data['LeadSource'].fillna('Other')
        data['Status'] = data['Status'].fillna('New')
        data['IsConverted'] = data['IsConverted'].fillna(False)

        # Calculate scores
        data['LeadScore'] = calculate_enhanced_lead_score(data)
        data = calculate_conversion_probability(data)
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.write("Debug info:")
        st.write("Python version:", sys.version)
        st.write("Pandas version:", pd.__version__)
        return pd.DataFrame()

def calculate_enhanced_lead_score(df):
    """Calculate lead score using multiple factors"""
    score = pd.Series(0, index=df.index)
    
    # Firmographic factors (30%)
    industry_weights = {
        'Technology': 30,
        'Healthcare': 25,
        'Finance': 28,
        'Manufacturing': 20,
        'Retail': 15
    }
    score += df['Industry'].map(industry_weights).fillna(10)
    
    # Contact information completeness (20%)
    info_score = 0
    for field in ['Email', 'Phone', 'Company']:
        info_score += df[field].notna().astype(int) * 6.67
    score += info_score
    
    # Engagement factors (30%)
    current_time = ensure_timezone_aware(pd.Timestamp.now())
    days_since_activity = (current_time - df['LastActivityDate']).dt.total_seconds() / (24 * 3600)
    activity_score = 30 * np.exp(-days_since_activity / 30)
    score += activity_score
    
    # Lead source quality (20%)
    source_weights = {
        'Web': 20,
        'Phone Inquiry': 18,
        'Partner Referral': 15,
        'Purchased List': 5
    }
    score += df['LeadSource'].map(source_weights).fillna(5)
    
    # Normalize to 0-100
    if score.max() == score.min():
        return pd.Series(50, index=df.index)
    score = (score - score.min()) / (score.max() - score.min()) * 100
    return score.round(1)

def calculate_conversion_probability(df):
    """Calculate conversion probability"""
    df['ConversionProbability'] = df['LeadScore'] / 100
    return df

def generate_lead_insights(lead_data):
    """Generate AI insights for a specific lead"""
    try:
        # Ensure all required keys are present
        required_keys = ['Industry', 'LeadScore', 'LeadSource', 'Status', 'ConversionProbability']
        for key in required_keys:
            if key not in lead_data:
                raise KeyError(f"Missing key: {key}")
            
                    # Ensure ConversionProbability is a float
        if not isinstance(lead_data['ConversionProbability'], float):
            lead_data['ConversionProbability'] = float(lead_data['ConversionProbability'].replace('%', '')) / 100

        lead_score = float(lead_data['LeadScore'].replace('/100', '').strip())

        
        if lead_score >= 95:
            follow_up = "Immediate Sales Follow-Up: Flag for immediate outreach by the sales team."
            call_scheduling = "Call Scheduling: Generate reminders for sales reps to make follow-up calls."
            lead_nurturing = "Lead Nurturing: Recommend additional resources such as case studies or demos."
        elif lead_score >= 50:
            follow_up = "Nurture Campaigns: Send to a nurturing campaign with targeted emails or webinars."
            call_scheduling = "Personalized Outreach: Suggest follow-up actions like sending educational content."
            lead_nurturing = "Monitor Engagement: Continuously monitor for changes in behavior and re-prioritize."
        else:
            follow_up = "Lead Segmentation: Add to a 'low priority' list for future analysis."
            call_scheduling = "Automated Content Delivery: Place on a content drip campaign with general content."
            lead_nurturing = "Re-assessment: Reassess if there is an increase in activity and move to higher priority."

        prompt = f"""<s>[INST] As a sales intelligence assistant, analyze this lead and provide insights:

Lead Details:
- Industry: {lead_data['Industry']}
- Lead Score: {lead_data['LeadScore']}
- Lead Source: {lead_data['LeadSource']}
- Status: {lead_data['Status']}
- Conversion Probability: {lead_data['ConversionProbability']*100:.1f}%

Provide three short points:
1. Key insight about this lead
2. Specific next best action
3. Main risk factor

Keep each point under 50 words. [/INST]</s>"""
        
        logging.info(f"Prompt: {prompt}")
        
        response = query_llm(prompt, max_length=150)
        logging.info(f"Response: {response}")
        
        insights = response.split('\n')
        return {
            'insights': insights[1] if len(insights) > 1 else "",
            'next_action': insights[2] if len(insights) > 2 else "",
            'risks': insights[3] if len(insights) > 3 else "",
            'follow_up': follow_up,
            'call_scheduling': call_scheduling,
            'lead_nurturing': lead_nurturing
            
        }
    except KeyError as e:
        logging.error(f"Error generating insights: {str(e)}")
        return {
            'insights': f"Missing data: {str(e)}",
            'next_action': "Please check lead details manually",
            'risks': "Unable to assess risks at this time",
            'follow_up': "Unable to assess follow-up actions",  
            'call_scheduling': "Unable to assess call scheduling",
            'lead_nurturing': "Unable to assess lead nurturing"
            
        }
    except Exception as e:
        logging.error(f"Error generating insights: {str(e)}")
        return {
            'insights': "AI insights temporarily unavailable",
            'next_action': "Please check lead details manually",
            'risks': "Unable to assess risks at this time",
            'follow_up': "Unable to assess follow-up actions",
            'call_scheduling': "Unable to assess call scheduling",  
            'lead_nurturing': "Unable to assess lead nurturing"
        }
    

def get_bulk_insights(filtered_data):
    """Generate overall insights for the filtered dataset"""
    try:
        stats = {
            'total_leads': len(filtered_data),
            'hot_leads': len(filtered_data[filtered_data['LeadScore'] >= 80]),
            'avg_score': filtered_data['LeadScore'].mean(),
            'top_industry': filtered_data['Industry'].mode().iloc[0],
            'top_source': filtered_data['LeadSource'].mode().iloc[0]
        }
        
        prompt = f"""<s>[INST] As a sales strategy advisor, analyze these pipeline metrics:

Pipeline Stats:
- Total Leads: {stats['total_leads']}
- Hot Leads: {stats['hot_leads']}
- Average Score: {stats['avg_score']:.1f}
- Top Industry: {stats['top_industry']}
- Top Lead Source: {stats['top_source']}

Provide 3 specific, actionable recommendations to improve pipeline performance.
Keep total response under 100 words. [/INST]</s>"""
        
        return query_llm(prompt, max_length=200)
    except Exception as e:
        return "AI insights temporarily unavailable. Please analyze the dashboard metrics manually."

def calculate_next_best_action(lead_data):
    """Calculate next best action based on lead characteristics"""
    lead_score = float(lead_data['LeadScore'].replace('/100', '').strip())
    status = lead_data['Status']
    industry = lead_data['Industry']
    
    try:
        prompt = f"""<s>[INST] As a sales assistant, suggest the single most important next action for this lead:

Lead Details:
- Score: {lead_score}/100
- Status: {status}
- Industry: {industry}

Provide one specific, actionable next step in under 15 words. 
Return only plain text with no formatting, HTML, or Markdown. [/INST]</s>"""
        
        llm_response = query_llm(prompt, max_length=50)

        # Remove unwanted CSS, HTML, or Markdown artifacts
        clean_response = re.sub(r"<.*?>|ol\s*\{.*?\}", "", llm_response).strip()

        return clean_response
    
    except Exception as e:
        return "Next action calculation unavailable"

def main():
    st.title("🎯 Lead Scoring Dashboard")
    
    # Load data
    data = load_data()
    
    if data.empty:
        st.error("No data available. Please check database connection.")
        return

    # Sidebar filters
    with st.sidebar:
        st.title("📊 Filters")
        
        default_end = ensure_timezone_aware(pd.Timestamp.now())
        default_start = default_end - pd.Timedelta(days=30)
        date_range = st.date_input(
            "Date Range",
            value=(default_start.date(), default_end.date())
        )
        
        score_range = st.slider("Lead Score Range", 0, 100, (0, 100))
        
        industries = ["All"] + sorted(data["Industry"].dropna().unique().tolist())
        selected_industry = st.selectbox("Industry", industries)
        
        sources = ["All"] + sorted(data["LeadSource"].dropna().unique().tolist())
        selected_source = st.selectbox("Lead Source", sources)
    
    # Apply filters
    filtered_data = data.copy()
    filtered_data = filtered_data[
        (filtered_data['LeadScore'] >= score_range[0]) &
        (filtered_data['LeadScore'] <= score_range[1])
    ]
    if selected_industry != "All":
        filtered_data = filtered_data[filtered_data["Industry"] == selected_industry]
    if selected_source != "All":
        filtered_data = filtered_data[filtered_data["LeadSource"] == selected_source]
    
    # Overview metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Leads", len(filtered_data))
    with col2:
        hot_leads = len(filtered_data[filtered_data['LeadScore'] >= 80])
        st.metric("Hot Leads", hot_leads)
    with col3:
        warm_leads = len(filtered_data[(filtered_data['LeadScore'] >= 50) & (filtered_data['LeadScore'] < 80)])
        st.metric("Warm Leads", warm_leads)
    with col4:
        cold_leads = len(filtered_data[filtered_data['LeadScore'] < 50])
        st.metric("Cold Leads", cold_leads)
    with col5:
        avg_score = filtered_data["LeadScore"].mean()
        st.metric("Average Score", f"{avg_score:.1f}")
    with col6:
        avg_prob = filtered_data["ConversionProbability"].mean() * 100
        st.metric("Avg. Conv. Probability", f"{avg_prob:.1f}%")
    
    # AI Insights
    st.subheader("🤖 AI-Powered Pipeline Insights")
    with st.expander("View Strategic Recommendations", expanded=True):
        strategic_insights = get_bulk_insights(filtered_data)
        st.write(strategic_insights)
    
    # Lead Score Distribution
    st.subheader("📈 Lead Score Distribution")
    fig_dist = px.histogram(
        filtered_data,
        x="LeadScore",
        nbins=20,
        color_discrete_sequence=['#00A36C'],
        title="Lead Score Distribution"
    )
    fig_dist.update_layout(bargap=0.1)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Main content
    
    st.subheader("🔥 High Priority Leads")
        
    priority_leads = filtered_data.nlargest(10, "LeadScore")[
            ["FirstName", "LastName", "Company", "LeadScore", 
             "ConversionProbability", "Industry", "Status", "LeadSource"]
        ].copy()
        
        # Format data
    priority_leads["ConversionProbability"] = (priority_leads["ConversionProbability"] * 100).round(1).astype(str) + '%'
    priority_leads["LeadScore"] = priority_leads["LeadScore"].round(1).astype(str) + '/100'
        
        # Generate insights
    priority_leads['AI_Insights'] = priority_leads.apply(generate_lead_insights, axis=1)
    priority_leads['NextBestAction'] = priority_leads.apply(calculate_next_best_action, axis=1)
        
        # Display table
    display_cols = ["FirstName", "LastName", "Company", "LeadScore", 
                       "ConversionProbability", "Industry", "Status", "NextBestAction"]
    
    # Reset index and add a sequential numbering column
    priority_leads = priority_leads.reset_index(drop=True)
    priority_leads.index += 1  # Start numbering from 1
    priority_leads.index.name = "No."

    st.dataframe(priority_leads[display_cols], height=400,use_container_width=True)
        
        # Detailed analysis
    st.subheader("📊 Detailed Lead Analysis")
    for idx, lead in priority_leads.iterrows():
            with st.expander(f"📋 {lead['FirstName']} {lead['LastName']} - {lead['Company']}"):
                insights = lead['AI_Insights']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("🔍 **Key Insights**")
                    st.write(insights['insights'])
                with col2:
                    st.write("➡️ **Next Best Action**")
                    st.write(insights['next_action'])
                with col3:
                    st.write("⚠️ **Risk Factors**")
                    st.write(insights['risks'])
               
            
    
    st.subheader("📊 Lead Source Performance")
    source_analysis = filtered_data.groupby("LeadSource").agg({
            "Id": "count",
            "ConversionProbability": "mean",
            "LeadScore": "mean"
        }).round(2)
    source_analysis.columns = ["Count", "Avg. Conv. Probability", "Avg. Score"]
    source_analysis["Avg. Conv. Probability"] = (source_analysis["Avg. Conv. Probability"] * 100).round(1).astype(str) + '%'
    st.dataframe(source_analysis)
    
    # Industry Analysis
    st.subheader("🏢 Industry Analysis")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        industry_scores = filtered_data.groupby("Industry")["LeadScore"].mean().round(1)
        fig_industry = px.bar(
            industry_scores,
            title="Average Lead Score by Industry",
            color_discrete_sequence=['#00A36C']
        )
        fig_industry.update_layout(
            xaxis_title="Industry",
            yaxis_title="Average Lead Score",
            showlegend=False
        )
        st.plotly_chart(fig_industry, use_container_width=True)
    
    with col2:
        industry_conversion = filtered_data.groupby("Industry")["ConversionProbability"].mean() * 100
        fig_conversion = px.bar(
            industry_conversion,
            title="Conversion Probability by Industry",
            color_discrete_sequence=['#4B0082']
        )
        fig_conversion.update_layout(
            xaxis_title="Industry",
            yaxis_title="Conversion Probability (%)",
            showlegend=False
        )
        st.plotly_chart(fig_conversion, use_container_width=True)
    
    # Trend Analysis
    st.subheader("📈 Trend Analysis")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Lead Score Trends
        filtered_data['CreatedDate'] = pd.to_datetime(filtered_data['CreatedDate']).dt.date
        daily_scores = filtered_data.groupby('CreatedDate')['LeadScore'].mean().reset_index()
        fig_trend = px.line(
            daily_scores,
            x='CreatedDate',
            y='LeadScore',
            title="Average Lead Score Trend",
            color_discrete_sequence=['#00A36C']
        )
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Lead Score"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Conversion Probability Trends
        daily_conv = filtered_data.groupby('CreatedDate')['ConversionProbability'].mean().reset_index()
        daily_conv['ConversionProbability'] = daily_conv['ConversionProbability'] * 100
        fig_conv_trend = px.line(
            daily_conv,
            x='CreatedDate',
            y='ConversionProbability',
            title="Average Conversion Probability Trend",
            color_discrete_sequence=['#4B0082']
        )
        fig_conv_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Conversion Probability (%)"
        )
        st.plotly_chart(fig_conv_trend, use_container_width=True)
    
    # Download section
    st.subheader("📥 Export Data")
    with st.expander("Download Options"):
        col1, col2 = st.columns(2)
        with col1:
            # Download filtered data
            csv = filtered_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered Data",
                data=csv,
                file_name="lead_scoring_data.csv",
                mime="text/csv"
            )
        with col2:
            # Download high priority leads
            priority_csv = priority_leads.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Priority Leads",
                data=priority_csv,
                file_name="priority_leads.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()