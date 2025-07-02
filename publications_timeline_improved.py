import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from collections import Counter
import re

def parse_date(date_data):
    """Parse various date formats from the metadata"""
    if not date_data:
        return None
    
    # Handle array format: [year, month] or [year, month, day]
    if isinstance(date_data, list):
        if len(date_data) >= 2:
            year = int(date_data[0])
            month = int(date_data[1])
            day = int(date_data[2]) if len(date_data) > 2 else 1
            try:
                return datetime(year, month, day)
            except ValueError:
                return None
    
    # Handle string formats
    date_str = str(date_data)
    
    # Common date patterns
    patterns = [
        r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
        r'(\d{4})-(\d{1,2})',            # YYYY-MM
        r'(\d{4})',                      # YYYY
        r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
        r'(\d{1,2})-(\d{1,2})-(\d{4})',  # MM-DD-YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            groups = match.groups()
            if len(groups) == 3:  # Full date
                year, month, day = groups
                try:
                    return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
            elif len(groups) == 2:  # Year and month
                if len(groups[0]) == 4:  # YYYY-MM
                    year, month = groups
                else:  # MM-YYYY
                    month, year = groups
                try:
                    return datetime(int(year), int(month), 1)
                except ValueError:
                    continue
            elif len(groups) == 1:  # Just year
                try:
                    return datetime(int(groups[0]), 1, 1)
                except ValueError:
                    continue
    
    return None

def load_and_process_data():
    """Load the fullmerged.json file and extract publication dates"""
    print("Loading fullmerged.json...")
    
    with open('fullmerged.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} publications")
    
    publications = []
    
    for i, pub in enumerate(data):
        if i % 1000 == 0:
            print(f"Processing publication {i}/{len(data)}")
        
        # Extract publication date from metadata
        published_date = None
        if 'metadata' in pub and 'published' in pub['metadata']:
            published_date = parse_date(pub['metadata']['published'])
        
        if published_date:
            publications.append({
                'title': pub.get('title', 'Unknown Title')[:100] + '...' if len(pub.get('title', '')) > 100 else pub.get('title', 'Unknown Title'),
                'published_date': published_date,
                'year': published_date.year,
                'month': published_date.month,
                'year_month': f"{published_date.year}-{published_date.month:02d}"
            })
    
    print(f"Found {len(publications)} publications with valid dates")
    return publications

def create_visualizations(publications):
    """Create various timeline visualizations"""
    
    # Convert to DataFrame
    df = pd.DataFrame(publications)
    
    # 1. Publications per year
    yearly_counts = df['year'].value_counts().sort_index()
    
    fig1 = px.bar(
        x=yearly_counts.index,
        y=yearly_counts.values,
        title='Publications per Year',
        labels={'x': 'Year', 'y': 'Number of Publications'},
        color=yearly_counts.values,
        color_continuous_scale='viridis'
    )
    fig1.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Publications",
        showlegend=False
    )
    fig1.write_html('publications_per_year.html')
    
    # 1.5. Publications per month (across all years)
    monthly_counts = df['month'].value_counts().sort_index()
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    fig1_5 = px.bar(
        x=[month_names[i-1] for i in monthly_counts.index],
        y=monthly_counts.values,
        title='Publications per Month (All Years Combined)',
        labels={'x': 'Month', 'y': 'Number of Publications'},
        color=monthly_counts.values,
        color_continuous_scale='plasma'
    )
    fig1_5.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Publications",
        showlegend=False
    )
    fig1_5.write_html('publications_per_month.html')
    
    # 2. Publications per month (heatmap)
    monthly_data = df.groupby(['year', 'month']).size().reset_index(name='count')
    monthly_pivot = monthly_data.pivot(index='year', columns='month', values='count').fillna(0)
    
    fig2 = px.imshow(
        monthly_pivot,
        title='Publications Heatmap by Year and Month',
        labels=dict(x="Month", y="Year", color="Publications"),
        aspect="auto",
        color_continuous_scale='viridis'
    )
    fig2.update_xaxes(tickmode='array', tickvals=list(range(12)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    fig2.write_html('publications_heatmap.html')
    
    # 3. IMPROVED Timeline visualization - use histogram instead of scatter for better performance
    fig3 = go.Figure()
    
    # Create histogram of publications over time
    fig3.add_trace(go.Histogram(
        x=df['published_date'],
        nbinsx=50,  # Adjust number of bins as needed
        name='Publications',
        marker_color='rgba(55, 83, 109, 0.7)',
        opacity=0.8
    ))
    
    fig3.update_layout(
        title='Publications Distribution Over Time',
        xaxis_title="Publication Date",
        yaxis_title="Number of Publications",
        bargap=0.1,
        showlegend=False
    )
    fig3.write_html('publications_timeline_histogram.html')
    
    # 4. Alternative: Timeline with aggregated data (better for large datasets)
    # Group by month and create a line chart
    monthly_counts = df.groupby('year_month').size().reset_index(name='count')
    monthly_counts['date'] = pd.to_datetime(monthly_counts['year_month'] + '-01')
    monthly_counts = monthly_counts.sort_values('date')
    
    fig4 = px.line(
        monthly_counts,
        x='date',
        y='count',
        title='Publications per Month (Timeline)',
        labels={'date': 'Month', 'count': 'Number of Publications'}
    )
    fig4.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Publications"
    )
    fig4.write_html('publications_monthly_timeline.html')
    
    # 5. Yearly trend with moving average
    yearly_trend = df.groupby('year').size().reset_index(name='count')
    yearly_trend['moving_avg'] = yearly_trend['count'].rolling(window=3, center=True).mean()
    
    fig5 = go.Figure()
    
    fig5.add_trace(go.Scatter(
        x=yearly_trend['year'],
        y=yearly_trend['count'],
        mode='markers+lines',
        name='Publications per Year',
        marker=dict(size=8, color='blue')
    ))
    
    fig5.add_trace(go.Scatter(
        x=yearly_trend['year'],
        y=yearly_trend['moving_avg'],
        mode='lines',
        name='3-Year Moving Average',
        line=dict(color='red', width=3)
    ))
    
    fig5.update_layout(
        title='Publications per Year with Moving Average',
        xaxis_title="Year",
        yaxis_title="Number of Publications",
        showlegend=True
    )
    fig5.write_html('publications_yearly_trend.html')
    
    print("Visualizations created:")
    print("- publications_per_year.html")
    print("- publications_per_month.html")
    print("- publications_heatmap.html") 
    print("- publications_timeline_histogram.html (IMPROVED timeline)")
    print("- publications_monthly_timeline.html")
    print("- publications_yearly_trend.html")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Date range: {df['published_date'].min()} to {df['published_date'].max()}")
    print(f"Total publications: {len(df)}")
    print(f"Years covered: {df['year'].nunique()}")
    print(f"Most active year: {yearly_counts.idxmax()} with {yearly_counts.max()} publications")
    
    # Show sample of the data
    print(f"\nSample publications:")
    sample_df = df.sample(min(5, len(df)))
    for _, row in sample_df.iterrows():
        print(f"- {row['title']} ({row['published_date'].strftime('%Y-%m-%d')})")

if __name__ == "__main__":
    try:
        publications = load_and_process_data()
        if publications:
            create_visualizations(publications)
        else:
            print("No publications with valid dates found!")
    except FileNotFoundError:
        print("Error: fullmerged.json not found in current directory")
    except Exception as e:
        print(f"Error: {e}") 