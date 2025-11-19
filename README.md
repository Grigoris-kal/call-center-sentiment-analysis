# 📞 Call Center Sentiment Analysis

## 🧠 Overview
This project analyzes customer sentiment from call center transcripts using Python and AI libraries.  
Developed in **Jupyter Notebook via Anaconda**, it explores emotional trends, agent performance, and actionable insights for improving customer service quality.

## 📊 Key Insights

### 💡 Key Performance Insights
**✅ Highest Satisfaction**: 5-10min calls
- Avg CSAT: 6.8/10
- 13 calls in this range

**❌ Lowest Satisfaction**: 45+min calls  
- Avg CSAT: 3.0/10
- 1 call in this range

**📊 Correlation (Duration vs CSAT)**: -0.16
- Weak correlation: Call duration doesn't strongly affect satisfaction

### 📈 Performance Metrics
- **Average CSAT Score**: 5.3/10
- **Negative Sentiment Rate**: 62.9%
- **SLA Adherence Rate**: 75.7%
- **Average Call Duration**: 26.3 minutes

### 🔍 Critical Insights
1. **Worst performing combination**: Chatbot for Billing Question (Sentiment: 2.14)
2. **Best performing call center**: Los Angeles/CA (Avg Sentiment: 2.90)
3. **SLA Impact**: 'Below SLA' calls have 0.32 lower sentiment score
4. **Peak call hour**: 0:00

### 💡 Actionable Recommendations
1. **Improve Chatbot Effectiveness**: Review and enhance chatbot scripts for billing questions
2. **Optimize Resource Allocation**: Use hourly patterns to staff appropriately during peak hours
3. **Focus on SLA Adherence**: Below-SLA responses significantly impact customer sentiment
4. **Share Best Practices**: Learn from Los Angeles/CA's success factors
5. **Proactive Outreach**: Target at-risk customer segments identified through clustering

## 🛠️ Technologies Used
- **Python** (pandas, numpy, matplotlib, seaborn)
- **NLP Libraries**: NLTK, TextBlob, spaCy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, WordCloud
- **Jupyter Notebook** via Anaconda

## 📁 Project Structure