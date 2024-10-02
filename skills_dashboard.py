
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    # Modify this path to your local CSV file path
    return pd.read_csv('skills_by_role.csv')

df = load_data()

# Streamlit App Layout
st.title("Skills vs Job Role Dashboard")

# Display raw data
if st.checkbox('Show raw data'):
    st.write(df)

# Heatmap: Skill Importance Across Job Roles
st.subheader('Heatmap: Skill Importance by Job Role')
if st.checkbox('Show heatmap'):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.set_index('General Skill'), cmap='RdYlGn', annot=True, ax=ax)
    st.pyplot(fig)

# Top Skills by Job Role
st.subheader('Top Skills by Job Role')
selected_role = st.selectbox("Select a Job Role", df.columns[1:])
if st.checkbox('Show top skills for the selected role'):
    role_data = df[['General Skill', selected_role]].sort_values(by=selected_role, ascending=False).head(5)
    st.bar_chart(role_data.set_index('General Skill'))

# Skill Comparison Across Job Roles
st.subheader('Skill Comparison Across Job Roles')
selected_skill = st.selectbox("Select a Skill to Compare Across Roles", df['General Skill'].unique())
if st.checkbox('Show skill comparison'):
    skill_data = df[df['General Skill'] == selected_skill].T.iloc[1:]
    skill_data.columns = ['Score']
    st.line_chart(skill_data)

# Radar Chart for Job Role Profile
st.subheader('Radar Chart: Job Role Skill Profile')
import plotly.express as px
if st.checkbox('Show radar chart'):
    role = st.selectbox("Select a Job Role for Radar Chart", df.columns[1:])
    role_skills = df[['General Skill', role]]
    
    fig = px.line_polar(role_skills, r=role, theta='General Skill', line_close=True)
    st.plotly_chart(fig)

# Job Role Clustering Based on Skills
st.subheader('Job Role Clustering')
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

if st.checkbox('Show clustering'):
    # Preprocess data
    scaler = StandardScaler()
    job_roles = df.columns[1:]
    df_scaled = pd.DataFrame(scaler.fit_transform(df[job_roles]), columns=job_roles)

    # Clustering with KMeans
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(df_scaled.T)
    
    # Show Clustering Result
    st.write("Clustering results:")
    df_clusters = pd.DataFrame({'Job Role': job_roles, 'Cluster': clusters})
    st.write(df_clusters)

    # Cluster Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=df_scaled.iloc[0], y=df_scaled.iloc[1], hue=clusters, palette='viridis', s=100)
    st.pyplot(fig)

st.sidebar.subheader("About")
st.sidebar.info("This dashboard analyzes the skill requirements for different job roles based on data provided.")
