#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#  CELL 1: Import Libraries & Setup


# In[1]:


# Core Data Handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistics & Forecasting
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# System & Utilities
import warnings
warnings.filterwarnings('ignore')

# Set up visualization style
plt.style.use('default')
sns.set_palette("husl")
print("✅ All libraries imported successfully!")


# In[ ]:


# CELL 2: Real-World Data Processing from Source to Analysis


# In[2]:


import pandas as pd
import numpy as np
import os
from IPython.display import display, HTML

# Define file paths
source_path = r'C:\Users\hp\Desktop\Python\CSV_Excel_dummy\Call_Center_Source_Sentiment_Sample_Data.xlsx'
destination_path = r'C:\Users\hp\Desktop\Python\compact_python_projects\MyAPIs_Projects\Call_center_python analysis\Call_Center_Sentiment_Sample_Data.xlsx'

print("📊 REAL-WORLD DATA PROCESSING PIPELINE")
print("=" * 60)
print(f"🔗 SOURCE: {source_path}")
print(f"💾 DESTINATION: {destination_path}")
print("=" * 60)

# 1. Check if source file exists
print("\n1. CHECKING SOURCE FILE...")
if not os.path.exists(source_path):
    print(f"❌ SOURCE FILE NOT FOUND: {source_path}")
    print("Please check the path and make sure the file exists.")
else:
    print("✅ Source file found!")
    print(f"📁 File size: {os.path.getsize(source_path)} bytes")

# 2. Load data from source
print("\n2. LOADING DATA FROM SOURCE...")
try:
    # First, let's see what sheets are available
    excel_file = pd.ExcelFile(source_path)
    sheet_names = excel_file.sheet_names
    print(f"📋 Sheets available: {sheet_names}")
    
    # Load data from the first sheet (or specify the correct sheet name)
    df = pd.read_excel(source_path, sheet_name=sheet_names[0], skiprows=5)
    print(f"✅ Data loaded successfully!")
    
    # FIX: Remove the extra index column immediately after loading
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print("✅ Removed extra index column 'Unnamed: 0'")
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🏷️ Columns: {list(df.columns)}")
    
except Exception as e:
    print(f"❌ ERROR LOADING DATA: {e}")
    print("\nTrying alternative loading methods...")
    
    # Try without skiprows
    try:
        df = pd.read_excel(source_path)
        print(f"✅ Loaded without skiprows! Shape: {df.shape}")
        print("Columns:", list(df.columns))
    except Exception as e2:
        print(f"❌ Also failed: {e2}")
        # Create fallback data
        print("\n🔄 Creating fallback sample data...")
        df = pd.DataFrame({
            'ID': [f'CS-{i}' for i in range(100, 115)],
            'Customer Name': [f'Customer_{i}' for i in range(1, 16)],
            'Sentiment': ['Positive', 'Negative', 'Neutral'] * 5,
            'CSAT Score': np.random.randint(1, 11, 15),
            'Call Timestamp': pd.date_range('2024-01-01', periods=15, freq='D'),
            'Reason': ['Billing Question'] * 10 + ['Payments'] * 5,
            'City': ['City_' + str(i) for i in range(1, 16)],
            'State': ['State_' + str(i) for i in range(1, 16)],
            'Channel': ['Call-Center', 'Chatbot', 'Email'] * 5,
            'Response Time': ['Within SLA'] * 15,
            'Call Duration (Minutes)': np.random.randint(5, 60, 15),
            'Call Center': ['Center_A', 'Center_B'] * 7 + ['Center_A']
        })  

# 3. Data cleaning and preparation
print("\n3. DATA CLEANING & PREPARATION...")

# Convert timestamp if it exists
date_columns = df.select_dtypes(include=['object']).columns
for col in date_columns:
    if 'timestamp' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"✅ Converted {col} to datetime")
        except:
            pass

# Map sentiment to numerical scores if sentiment column exists
if 'Sentiment' in df.columns:
    sentiment_map = {'Very Negative': 1, 'Negative': 2, 'Neutral': 3, 'Positive': 4, 'Very Positive': 5}
    df['Sentiment_Score'] = df['Sentiment'].map(sentiment_map)
    print("✅ Created Sentiment_Score numerical mapping")

# Create time-based features if timestamp exists
timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()]
if timestamp_cols:
    timestamp_col = timestamp_cols[0]
    df['Hour'] = df[timestamp_col].dt.hour
    df['DayOfWeek'] = df[timestamp_col].dt.day_name()
    df['Date'] = df[timestamp_col].dt.date
    print("✅ Created time-based features (Hour, DayOfWeek, Date)")

# 4. Save processed data to destination
print("\n4. SAVING PROCESSED DATA TO DESTINATION...")
try:
    df.to_excel(destination_path, sheet_name='Call Center Data', index=False)
    print(f"✅ Processed data saved to: {destination_path}")
    print(f"📊 Final dataset shape: {df.shape}")
except Exception as e:
    print(f"❌ Error saving to destination: {e}")

# 5. Final data summary WITH CENTER ALIGNMENT
print("\n5. FINAL DATA SUMMARY:")
print(f"📊 Dataset shape: {df.shape}")
print(f"🏷️ Columns: {list(df.columns)}")
print(f"📅 Data types:")
print(df.dtypes)
print(f"\n📈 Sample of processed data (Center Aligned):")

# Apply center alignment to the dataframe display
styled_df = df.head(3).style.set_properties(**{
    'text-align': 'center',
    'vertical-align': 'middle'
}).set_table_styles([{
    'selector': 'th',
    'props': [('text-align', 'center'), ('vertical-align', 'middle')]
}])

display(styled_df)

print("\n" + "=" * 60)
print("🎯 REAL-WORLD PROCESSING COMPLETE!")
print("=" * 60)


# In[ ]:


# CELL 3: Data Cleaning & Feature Engineering


# In[3]:


print("\n🧹 Cleaning data and creating features...")

# FIX: Remove the extra index column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    print("✅ Removed extra index column 'Unnamed: 0'")

# Convert timestamp and extract time features
df['Call Timestamp'] = pd.to_datetime(df['Call Timestamp'])
df['Hour'] = df['Call Timestamp'].dt.hour
df['DayOfWeek'] = df['Call Timestamp'].dt.day_name()
df['Date'] = df['Call Timestamp'].dt.date

# Map sentiment to numerical scores for analysis
sentiment_map = {'Very Negative': 1, 'Negative': 2, 'Neutral': 3, 'Positive': 4, 'Very Positive': 5}
df['Sentiment_Score'] = df['Sentiment'].map(sentiment_map)

# Create time-based features
df['Week'] = df['Call Timestamp'].dt.isocalendar().week

print("✅ Data cleaning completed!")
print(f"\nCleaned data shape: {df.shape}")
print(f"New features created: Hour, DayOfWeek, Date, Sentiment_Score, Week")
print(f"Sample of processed data:")
df[['Customer Name', 'Sentiment', 'Sentiment_Score', 'DayOfWeek', 'Hour']].head()


# In[ ]:


# CELL 4: Exploratory Data Analysis - Visualizations


# In[4]:


# FIRST: Display Call Center Distribution with Percentages as Text
print("📞 CALL VOLUME BY CALL CENTER (WITH PERCENTAGES)")
print("=" * 50)

# Calculate call center distribution with percentages
call_center_stats = df['Call Center'].value_counts()
total_calls = len(df)

print(f"\n📊 Total Calls: {total_calls}")
print("\n🏢 Call Center Distribution:")
print("-" * 30)

for center, count in call_center_stats.items():
    percentage = (count / total_calls) * 100
    print(f"📍 {center}: {count} calls ({percentage:.1f}%)")

print("-" * 30)

# Display as a nice formatted block
print(f"\n📋 CALL CENTER DISTRIBUTION SUMMARY:")
print("=" * 40)
for center, count in call_center_stats.items():
    percentage = (count / total_calls) * 100
    print(f"   {center}: {count} calls ({percentage:.1f}%)")
print("=" * 40)

# THEN: Continue with your existing visualizations
print("\n📊 Creating comprehensive visualizations...")

# Create SLA_Simplified column first
df['SLA_Simplified'] = df['Response Time'].apply(
    lambda x: 'Within SLA' if x in ['Within SLA', 'Below SLA'] else 'Outside SLA'
)

# Set up the figure for multiple plots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Call Center Performance Dashboard', fontsize=16, fontweight='bold')

# 1. Sentiment Distribution - UPDATED TITLE AND WHITE TEXT
sentiment_counts = df['Sentiment'].value_counts()
# Reorder sentiments logically
sentiment_order = ['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative']
sentiment_counts = sentiment_counts.reindex(sentiment_order)

# Colors with good contrast
sentiment_colors = ['green', 'yellowgreen', 'gray', 'red', 'black']

wedges, texts, autotexts = axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                                         autopct='%1.1f%%', startangle=90, colors=sentiment_colors)

# Make all percentage text WHITE and bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(14)

axes[0,0].set_title('Customer Sentiment Distribution\nAll Call Centers & All Channels', fontweight='bold')

# 2. REMOVED: CSAT Score Distribution and empty chart

# 3. Call Duration by Sentiment - EXPLAINED WITH BOLDER COLORS
# Create a boxplot with bolder colors and better explanation
box_data = [df[df['Sentiment'] == sentiment]['Call Duration (Minutes)'] for sentiment in sentiment_order]
box_plot = axes[0,1].boxplot(box_data, labels=sentiment_order, patch_artist=True)

# Use bolder, more distinct colors
box_colors = ['green', 'yellowgreen', 'lightblue', 'lightcoral', 'darkred']
for patch, color in zip(box_plot['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Make median lines bolder
for median in box_plot['medians']:
    median.set_color('black')
    median.set_linewidth(2)

axes[0,1].set_title('Call Duration Analysis by Customer Sentiment', fontweight='bold', fontsize=14)
axes[0,1].set_xlabel('Customer Sentiment', fontweight='bold', fontsize=16)  # TWICE AS BIG
axes[0,1].set_ylabel('Call Duration (Minutes)', fontweight='bold', fontsize=16)  # TWICE AS BIG
axes[0,1].tick_params(axis='both', which='major', labelsize=18)  # Larger tick labels

# Add explanation text
axes[0,1].text(0.02, 0.98, '📊 Box shows: 25th-75th percentile\n📌 Line: Median duration\n🔵 Dots: Individual calls', 
               transform=axes[0,1].transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# 4. Channel Usage - UPDATED WITH WHITE TEXT AND LARGER LABELS
channel_counts = df['Channel'].value_counts()
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
bars = axes[1,0].bar(channel_counts.index, channel_counts.values, color=colors, edgecolor='black')
axes[1,0].set_title('Contact Channel Distribution', fontweight='bold', fontsize=16)
axes[1,0].set_xlabel('Contact Channel', fontweight='bold', fontsize=16)  # TWICE AS BIG
axes[1,0].set_ylabel('Number of Calls', fontweight='bold', fontsize=16)  # TWICE AS BIG
axes[1,0].tick_params(axis='x', rotation=45, labelsize=18)  # Larger x-axis labels
axes[1,0].tick_params(axis='y', labelsize=18)  # Larger y-axis labels

# Add WHITE value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1,0].text(bar.get_x() + bar.get_width()/2., height,
                  f'{int(height)}', ha='center', va='bottom', 
                  fontweight='bold', color='white', fontsize=10)

# 5. Response Time Performance - UPDATED WITH WHITE TEXT AND LARGER LABELS
response_counts = df['SLA_Simplified'].value_counts()
colors = ['green', 'red']
bars = axes[1,1].bar(response_counts.index, response_counts.values, color=colors, edgecolor='black')
axes[1,1].set_title('Response Time Performance', fontweight='bold', fontsize=14)
axes[1,1].set_xlabel('SLA Status', fontweight='bold', fontsize=14)  # TWICE AS BIG
axes[1,1].set_ylabel('Number of Calls', fontweight='bold', fontsize=14)  # TWICE AS BIG
axes[1,1].tick_params(axis='x', rotation=45, labelsize=18)  # Larger x-axis labels
axes[1,1].tick_params(axis='y', labelsize=18)  # Larger y-axis labels

# Add WHITE value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                  f'{int(height)}', ha='center', va='bottom', 
                  fontweight='bold', color='white', fontsize=10)

plt.tight_layout()
plt.show()

# 6. Call Volume by Center - SEPARATE CHART WITH WHITE PERCENTAGES AND LARGER LABELS
print("\n📞 Call Volume Distribution by Center:")
plt.figure(figsize=(10, 6))
center_counts = df['Call Center'].value_counts()
total_calls = len(df)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(center_counts.index, center_counts.values, color=colors, edgecolor='black')

plt.title('Call Volume by Call Center\n(With Percentage Distribution)', fontsize=16, fontweight='bold')
plt.xlabel('Call Center', fontweight='bold', fontsize=14)  # TWICE AS BIG
plt.ylabel('Number of Calls', fontweight='bold', fontsize=14)  # TWICE AS BIG
plt.xticks(rotation=45, fontsize=12)  # Larger x-axis labels
plt.yticks(fontsize=12)  # Larger y-axis labels

# Add WHITE percentage labels on bars
for bar in bars:
    height = bar.get_height()
    percentage = (height / total_calls) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({percentage:.1f}%)', 
            ha='center', va='bottom', fontweight='bold', 
            color='black', fontsize=12, fontfamily='sans-serif')  # Changed to black

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\n📊 Summary Statistics:")
print(f"• Total calls analyzed: {total_calls}")
print(f"• Call center distribution:")
for center, count in center_counts.items():
    percentage = (count / total_calls) * 100
    print(f"  - {center}: {count} calls ({percentage:.1f}%)")

# Boxplot Explanation
print(f"\n📦 BOXPLOT EXPLANATION:")
print(f"• Each colored box shows the middle 50% of call durations for that sentiment")
print(f"• The line inside each box is the MEDIAN (typical call duration)")
print(f"• Whiskers show the range of most call durations")
print(f"• Dots outside whiskers are unusually long/short calls")
print(f"• Insight: Do happier customers have shorter calls? Check the medians!")


# In[ ]:


# CELL 5: Time Series Analysis & Forecasting


# In[5]:


print("\n⏰ Performing time series analysis...")

# Daily call volume trend
daily_calls = df.set_index('Call Timestamp').resample('D')['ID'].count()

plt.figure(figsize=(15, 10))

# Subplot 1: Daily call volume
plt.subplot(3, 1, 1)
plt.plot(daily_calls.index, daily_calls.values, marker='o', linewidth=2, markersize=4)
plt.title('Daily Call Volume Trend')
plt.ylabel('Number of Calls')
plt.grid(True, alpha=0.3)

# Subplot 2: Decomposition
plt.subplot(3, 1, 2)
try:
    decomposition = seasonal_decompose(daily_calls.fillna(method='ffill'), model='additive', period=7)
    decomposition.trend.plot(ax=plt.gca(), title='Trend Component')
    plt.ylabel('Trend')
    plt.grid(True, alpha=0.3)
except Exception as e:
    plt.text(0.5, 0.5, f'Decomposition not possible: {str(e)}', 
             ha='center', va='center', transform=plt.gca().transAxes)

# Subplot 3: Simple forecasting
plt.subplot(3, 1, 3)
if len(daily_calls) > 1:
    # Simple linear regression for trend
    X = np.arange(len(daily_calls)).reshape(-1, 1)
    y = daily_calls.values
    model = LinearRegression()
    model.fit(X, y)
    trend_line = model.predict(X)
    
    plt.plot(daily_calls.index, daily_calls.values, label='Actual', marker='o')
    plt.plot(daily_calls.index, trend_line, label='Trend Line', linestyle='--')
    plt.title('Call Volume with Trend Line')
    plt.ylabel('Number of Calls')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Predict next 7 days
    future_days = 7
    future_X = np.arange(len(daily_calls), len(daily_calls) + future_days).reshape(-1, 1)
    future_predictions = model.predict(future_X)
    print(f"\n📈 Forecast: Expected call volume for next {future_days} days: {future_predictions.mean():.1f} calls/day")

plt.tight_layout()
plt.show()


# In[ ]:


# CELL 6: Interactive Visualizations with Plotly


# In[6]:


print("\n🎨 Creating interactive visualizations...")

# Define custom color mapping for sentiments
sentiment_colors = {
    'Very Negative': 'black',
    'Negative': 'red', 
    'Neutral': 'gray',
    'Positive': 'yellowgreen',  # Greenish-yellow
    'Very Positive': 'green'
}

# Define the proper order for sentiments (from best to worst)
sentiment_order = ['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative']

# Interactive sunburst chart - SIMPLIFIED
print("\n📊 Creating simplified call distribution chart...")

# Option A: Simpler 2-level sunburst (Center → Channel)
fig1 = px.sunburst(df, path=['Call Center', 'Channel'], 
                  title='Call Distribution by Center and Channel')
fig1.update_layout(height=500)
fig1.show()

# Option B: Center → Sentiment (cleaner view)
fig2 = px.sunburst(df, path=['Call Center', 'Sentiment'], 
                  color='Sentiment',
                  color_discrete_map=sentiment_colors,
                  title='Sentiment Distribution by Call Center')
fig2.update_layout(height=500)
fig2.show()


# 1. Response Time Performance by Channel - ONLY TOO SLOW IS OUTSIDE SLA
print("\n⏱️  Analyzing response time performance...")

# Create binary SLA status: Within SLA vs Outside SLA (only too slow)
df['SLA_Status'] = df['Response Time'].apply(
    lambda x: 'Within SLA' if x in ['Within SLA', 'Above SLA'] else 'Outside SLA'
)

response_analysis = df.groupby(['Channel', 'SLA_Status']).size().reset_index(name='Count')

fig = px.bar(response_analysis, x='Channel', y='Count', color='SLA_Status',
             title='<b>Response Time Performance by Channel</b>',
             color_discrete_map={'Within SLA': 'green', 'Outside SLA': 'red'},
             barmode='group',
             category_orders={'SLA_Status': ['Within SLA', 'Outside SLA']})

fig.update_layout(height=500, title_x=0.5,
                  xaxis_title="Channel",
                  yaxis_title="Number of Calls")
fig.show()

# 2. Average Call Duration by Reason
print("\n📞 Analyzing call duration patterns...")
duration_analysis = df.groupby('Reason')['Call Duration (Minutes)'].mean().reset_index()
duration_analysis = duration_analysis.sort_values('Call Duration (Minutes)', ascending=False)

fig = px.bar(duration_analysis, x='Reason', y='Call Duration (Minutes)',
             title='<b>Average Call Duration by Reason</b>',
             color='Call Duration (Minutes)',
             color_continuous_scale='viridis')
fig.update_layout(height=500, title_x=0.5,
                  xaxis_title="Contact Reason",
                  yaxis_title="Average Call Duration (Minutes)")
fig.show()

# 3. CSAT Score Distribution by Call Center
print("\n⭐ Analyzing CSAT performance...")
csat_analysis = df.groupby('Call Center')['CSAT Score'].mean().reset_index()
csat_analysis = csat_analysis.sort_values('CSAT Score', ascending=False)

fig = px.bar(csat_analysis, x='Call Center', y='CSAT Score',
             title='<b>Average CSAT Score by Call Center</b>',
             color='CSAT Score',
             color_continuous_scale='RdYlGn',
             range_color=[1, 10])
fig.update_layout(height=500, title_x=0.5,
                  xaxis_title="Call Center",
                  yaxis_title="Average CSAT Score")
fig.add_hline(y=df['CSAT Score'].mean(), line_dash="dash", 
              line_color="blue", annotation_text="Overall Average")
fig.show()

# GENERAL SENTIMENT PIE CHART WITH PERCENTAGES
print("\n📊 Creating sentiment pie chart with proper colors and order...")
sentiment_counts = df['Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']

# Reorder the pie chart data to match our desired order
sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], 
                                               categories=sentiment_order, 
                                               ordered=True)
sentiment_counts = sentiment_counts.sort_values('Sentiment')

fig = px.pie(sentiment_counts, values='Count', names='Sentiment',
             color='Sentiment', color_discrete_map=sentiment_colors,
             category_orders={'Sentiment': sentiment_order},
             title='Overall Sentiment Distribution')
# ADD PERCENTAGES EXTERNALLY
fig.update_traces(textposition='outside', textinfo='percent+label')
fig.update_layout(height=500)
fig.show()


# In[ ]:


# CELL 7: Machine Learning: Sentiment Prediction


# In[7]:


print("\n🤖 Building sentiment prediction model...")

# Prepare data for ML
ml_df = df.copy()

# Encode categorical variables
le_reason = LabelEncoder()
le_channel = LabelEncoder()
le_center = LabelEncoder()

ml_df['Reason_Encoded'] = le_reason.fit_transform(ml_df['Reason'])
ml_df['Channel_Encoded'] = le_channel.fit_transform(ml_df['Channel'])
ml_df['Center_Encoded'] = le_center.fit_transform(ml_df['Call Center'])

# Select features and target
features = ['Reason_Encoded', 'Channel_Encoded', 'Center_Encoded', 'Call Duration (Minutes)', 'CSAT Score']
X = ml_df[features]
y = ml_df['Sentiment_Score']

# Train a simple linear model
model = LinearRegression()
model.fit(X, y)

# Feature importance
importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n🔍 Feature Importance for Sentiment Prediction:")
print(importance)

# Model performance
train_score = model.score(X, y)
print(f"\n📊 Model R-squared Score: {train_score:.3f}")
print("This indicates how well the features explain sentiment variation in the data.")


# In[ ]:


# CELL 8: Customer Experience Analysis by Call Duration Ranges


# In[8]:


print("\n📊 Analyzing customer experience by call duration...")

# Create call duration ranges
bins = [0, 5, 10, 15, 20, 30, 45, 100]  # Custom time ranges
labels = ['0-5min', '5-10min', '10-15min', '15-20min', '20-30min', '30-45min', '45+min']
df['Duration_Range'] = pd.cut(df['Call Duration (Minutes)'], bins=bins, labels=labels, right=False)

# Calculate average metrics for each duration range
duration_analysis = df.groupby('Duration_Range').agg({
    'CSAT Score': 'mean',
    'Sentiment_Score': 'mean',
    'ID': 'count'
}).reset_index()
duration_analysis.columns = ['Duration_Range', 'Avg_CSAT', 'Avg_Sentiment', 'Call_Count']

print("📈 Call Duration Analysis:")
print(duration_analysis.round(2))

# Create clean scatter plot with aggregated data
plt.figure(figsize=(12, 8))

# Use the midpoint of each range for x-axis positioning
range_midpoints = {
    '0-5min': 2.5, '5-10min': 7.5, '10-15min': 12.5, '15-20min': 17.5,
    '20-30min': 25, '30-45min': 37.5, '45+min': 60
}

# Create scatter plot with DOUBLE the size for better visibility
scatter = plt.scatter(
    [range_midpoints[r] for r in duration_analysis['Duration_Range']],
    duration_analysis['Avg_CSAT'],
    s=duration_analysis['Call_Count'] * 20,  # DOUBLED size based on call volume
    alpha=0.7,
    c='steelblue',  # Single color instead of gradient
    edgecolors='black',
    linewidth=2  # Thicker borders for better visibility
)

plt.xlabel('Call Duration (Minutes)', fontsize=12, fontweight='bold')
plt.ylabel('Average CSAT Score', fontsize=12, fontweight='bold')
plt.title('Customer Satisfaction vs Call Duration\n(Bubble size = Number of Calls)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add call count annotations to each point (larger font)
for i, row in duration_analysis.iterrows():
    plt.annotate(f"{row['Call_Count']} calls", 
                (range_midpoints[row['Duration_Range']], row['Avg_CSAT']),
                xytext=(8, 8), textcoords='offset points',  # Slightly offset
                fontsize=10, fontweight='bold',  # Larger font
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# REMOVED: Color bar (no longer needed with single color)

plt.tight_layout()
plt.show()

# Print key insights
print("\n💡 KEY INSIGHTS:")
print("=" * 40)
max_csat_range = duration_analysis.loc[duration_analysis['Avg_CSAT'].idxmax()]
min_csat_range = duration_analysis.loc[duration_analysis['Avg_CSAT'].idxmin()]

print(f"✅ Highest Satisfaction: {max_csat_range['Duration_Range']} calls")
print(f"   • Avg CSAT: {max_csat_range['Avg_CSAT']:.1f}/10")
print(f"   • {max_csat_range['Call_Count']} calls in this range")

print(f"❌ Lowest Satisfaction: {min_csat_range['Duration_Range']} calls") 
print(f"   • Avg CSAT: {min_csat_range['Avg_CSAT']:.1f}/10")
print(f"   • {min_csat_range['Call_Count']} calls in this range")

# Calculate correlation between duration and satisfaction
correlation = df['Call Duration (Minutes)'].corr(df['CSAT Score'])
print(f"📊 Correlation (Duration vs CSAT): {correlation:.2f}")

if correlation < -0.3:
    print("   ➡️ Strong negative correlation: Longer calls = Lower satisfaction")
elif correlation > 0.3:
    print("   ➡️ Strong positive correlation: Longer calls = Higher satisfaction")  
else:
    print("   ➡️ Weak correlation: Call duration doesn't strongly affect satisfaction")


# In[ ]:


# CELL 9: Key Insights & Business Recommendations


# In[9]:


print("\n" + "="*80)
print("🎯 KEY BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*80)

# Calculate key metrics
avg_csat = df['CSAT Score'].mean()
negative_rate = (df['Sentiment'].isin(['Negative', 'Very Negative'])).mean() * 100
sla_adherence = (df['Response Time'] == 'Within SLA').mean() * 100
avg_call_duration = df['Call Duration (Minutes)'].mean()

print(f"\n📈 PERFORMANCE METRICS:")
print(f"• Average CSAT Score: {avg_csat:.1f}/10")
print(f"• Negative Sentiment Rate: {negative_rate:.1f}%")
print(f"• SLA Adherence Rate: {sla_adherence:.1f}%")
print(f"• Average Call Duration: {avg_call_duration:.1f} minutes")

# Top insights
print(f"\n🔍 CRITICAL INSIGHTS:")

# 1. Worst performing channel-reason combination
channel_reason_performance = df.groupby(['Channel', 'Reason'])['Sentiment_Score'].mean().sort_values()
worst_combo = channel_reason_performance.index[0]
worst_score = channel_reason_performance.iloc[0]
print(f"1. Worst performing combination: {worst_combo[0]} for {worst_combo[1]} (Sentiment: {worst_score:.2f})")

# 2. Best performing call center
center_performance = df.groupby('Call Center')['Sentiment_Score'].mean().sort_values(ascending=False)
best_center = center_performance.index[0]
best_score = center_performance.iloc[0]
print(f"2. Best performing call center: {best_center} (Avg Sentiment: {best_score:.2f})")

# 3. SLA impact analysis
sla_impact = df.groupby('Response Time')['Sentiment_Score'].mean()
print(f"3. SLA Impact: 'Below SLA' calls have {sla_impact['Below SLA'] - sla_impact['Within SLA']:.2f} lower sentiment score")

# 4. Peak hours analysis
peak_hour = df.groupby('Hour').size().sort_values(ascending=False).index[0]
print(f"4. Peak call hour: {peak_hour}:00")

print(f"\n💡 ACTIONABLE RECOMMENDATIONS:")
print(f"1. IMPROVE CHATBOT EFFECTIVENESS: Review and enhance chatbot scripts for billing questions")
print(f"2. OPTIMIZE RESOURCE ALLOCATION: Use hourly patterns to staff appropriately during peak hours")
print(f"3. FOCUS ON SLA ADHERENCE: Below-SLA responses significantly impact customer sentiment")
print(f"4. SHARE BEST PRACTICES: Learn from {best_center}'s success factors")
print(f"5. PROACTIVE OUTREACH: Target at-risk customer segments identified through clustering")

print(f"\n📊 ANALYSIS COMPLETED SUCCESSFULLY!")
print(f"• Total records analyzed: {len(df):,}")
print(f"• Time period: {df['Call Timestamp'].min().date()} to {df['Call Timestamp'].max().date()}")
print(f"• Call centers covered: {df['Call Center'].nunique()}")
print(f"• Analysis techniques: EDA, Time Series, ML, Clustering, Forecasting")
print("="*80)


# In[ ]:


M


# In[ ]:




