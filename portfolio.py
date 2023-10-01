import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import datetime as dt
import openai  # Import the OpenAI library

# Set Streamlit title
st.title("Investment Portfolio Dashboard")

# Define your OpenAI API key
openai.api_key = "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"

# Get today's date
end = dt.datetime.today().strftime('%Y-%m-%d')

# User input for assets and allocations
assets = st.text_input("Enter assets (comma-separated):", "AAPL, MSFT,ULTA,CMG,META,TSLA,SPGI,MA,INTU,TXHR")
allocations = st.text_input("Enter portfolio allocations (%) (comma-separated):", "10,10,10,10,10,10,10,10,10,10")

# Convert allocations to a list of floats
allocations = [float(alloc.strip()) for alloc in allocations.split(',')]

# Normalize allocations to sum to 1
allocations /= np.sum(allocations)

# User input for start date
start = st.date_input("Pick a starting date:", value=pd.to_datetime('2022-01-01'))

# User input for benchmark symbol
benchmark_symbol = st.text_input("Enter benchmark symbol (e.g., ^GSPC for S&P 500):", "^GSPC")

# Function to get a list of companies for a given sector using GPT-3.5
def get_companies_by_sector(sector):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"List companies operating in the {sector} sector on the US stock market.",
        max_tokens=50
    )
    companies = response.choices[0].text.strip().split(", ") if response.choices else []
    return companies

# Add a dropdown button to select a sector from different sectors
st.sidebar.subheader("Select a Sector")
sectors = ["Finance", "Tech", "Real Estate", "Healthcare", "Consumer Staples", "Energy","Defense"]
selected_sector = st.sidebar.selectbox("Select a sector:", sectors)
sector_companies = get_companies_by_sector(selected_sector)

# Display filtered companies
st.subheader(f"Companies in the {selected_sector} Sector on the US Stock Market")
st.write(sector_companies)

# Download asset data
data = yf.download(assets, start=start, end=end)['Adj Close']

# Calculate returns
ret_df = data.pct_change()

# Calculate percentage returns
percentage_returns = ret_df * 100

# Calculate cumulative returns
cumul_ret = (ret_df + 1).cumprod() - 1

# Calculate portfolio cumulative returns
pf_cumul_ret = (cumul_ret * allocations).sum(axis=1)

# Create a hypothetical growth chart starting with $10,000 for portfolio
initial_investment = 10000
portfolio_growth = initial_investment * (1 + pf_cumul_ret)

# Download benchmark data
benchmark_data = yf.download(benchmark_symbol, start=start, end=end)['Adj Close']

# Calculate benchmark returns
bench_ret = benchmark_data.pct_change()

# Calculate percentage returns for benchmark
benchmark_percentage_returns = bench_ret * 100

# Calculate cumulative returns for benchmark
benchmark_cumul_ret = (bench_ret + 1).cumprod() - 1

# Create a hypothetical growth chart starting with $10,000 for benchmark
benchmark_growth = initial_investment * (1 + benchmark_cumul_ret)
st.sidebar.subheader("Select a Region")
regions = ["Africa", "Europe", "England", "India", "China","Japan", "South Korea"]
selected_region = st.sidebar.selectbox("Select a region:", regions)

# Function to get a list of companies for a given region and sector using GPT-3.5
def get_companies_by_region_and_sector(region, sector):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"List companies operating in the {sector} sector in {region} on the US stock market.",
        max_tokens=60
    )
    companies = response.choices[0].text.strip().split(", ") if response.choices else []
    return companies

# Display filtered companies based on selected region and sector
filtered_companies = get_companies_by_region_and_sector(selected_region, selected_sector)
st.subheader(f"Companies in the {selected_sector} Sector in {selected_region} on the US Stock Market")
st.write(filtered_companies)
# Display portfolio vs. benchmark development
st.subheader("Portfolio vs. Benchmark Development")
growth_df = pd.concat([portfolio_growth, benchmark_growth], axis=1)
growth_df.columns = ['Portfolio Growth', 'Benchmark Growth']
st.line_chart(data=growth_df)

# Display percentage returns in years
years = (cumul_ret.index[-1] - cumul_ret.index[0]).days / 365
portfolio_annual_percentage_return = ((portfolio_growth[-1] / initial_investment) ** (1 / years) - 1) * 100
benchmark_annual_percentage_return = ((benchmark_growth[-1] / initial_investment) ** (1 / years) - 1) * 100

st.subheader("Annual Percentage Returns")
st.write(f"Portfolio Annual Percentage Return: {portfolio_annual_percentage_return:.2f}%")
st.write(f"Benchmark Annual Percentage Return: {benchmark_annual_percentage_return:.2f}%")

# Display portfolio risk
pf_std = np.sqrt(allocations @ ret_df.cov() @ allocations)
st.subheader("Portfolio Risk")
st.write(f"Portfolio Standard Deviation: {pf_std:.4f}")

# Display benchmark risk
bench_risk = bench_ret.std()
st.subheader("Benchmark Risk")
st.write(f"Benchmark Standard Deviation: {bench_risk:.4f}")

# Display portfolio composition
st.subheader("Portfolio Composition")
fig, ax = plt.subplots(facecolor='#121212')
ax.pie(allocations, labels=data.columns, autopct='%1.1f%%', textprops={'color': 'white'})
st.pyplot(fig)

# Add a chatbot to classify companies into sectors
st.subheader("Classify Companies")
company_name = st.text_input("Enter a company name:")
if company_name:
    # Use GPT-3.5 to classify the company into sectors
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use GPT-3.5 engine
        prompt=f"Classify the company {company_name} into sectors.",
        max_tokens=50
    )
    sector_classification = response.choices[0].text.strip()
    st.write(f"The company {company_name} is classified as: {sector_classification}")

# Add a sidebar menu to get quick summaries of companies
st.sidebar.title("Company Summaries")
selected_company = st.sidebar.selectbox("Select a company:", sector_companies)

if selected_company:
    # Use GPT-3.5 to provide a summary of the selected company
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use GPT-3.5 engine
        prompt=f"Provide a brief summary of {selected_company}.",
        max_tokens=200 # Adjust the max tokens as needed
    )
    company_summary = response.choices[0].text.strip()
    st.sidebar.write(company_summary)
    
st.sidebar.title("Company Summaries abroad")
selected_abroad = st.sidebar.selectbox("Select a company:", filtered_companies)
if selected_abroad:
    # Use GPT-3.5 to provide a summary of the selected company
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use GPT-3.5 engine
        prompt=f"Provide a brief summary of {filtered_companies}.",
        max_tokens=200  # Adjust the max tokens as needed
    )
    company_summary1 = response.choices[0].text.strip()
    st.sidebar.write(company_summary1)

# Add the Investment Advice section to the main page
st.subheader("Investment Advice")
investment_expert = st.radio(
    "Select an investment expert for advice:",
    ["Warren Buffett", "Bill Ackman", "Charlie Munger"]
)

investment_advice = None

if investment_expert:
    # Use GPT-3.5 to generate investment advice based on the selected expert
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Provide investment advice from {investment_expert}.",
        max_tokens=50
    )
    investment_advice = response.choices[0].text.strip()

if investment_advice:
    st.write(f"advises: {investment_advice}")



