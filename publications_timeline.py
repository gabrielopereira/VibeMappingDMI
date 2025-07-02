import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from collections import Counter
import re

def parse_date(date_str):
    """Parse various date formats from the metadata"""
    if not date_str:
        return None
    
    # Common date patterns
    patterns = [
        r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
        r'(\d{4})-(\d{1,2})',            # YYYY-MM
        r'(\d{4})',                      # YYYY
        r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
        r'(\d{1,2})-(\d{1,2})-(\d{4})',  # MM-DD-YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(date_str))
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
                'title': pub.get('title', 'Unknown Title'),
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
    
    # 3. Timeline scatter plot
    fig3 = px.scatter(
        df,
        x='published_date',
        y=[1] * len(df),  # All points on same y-level
        title='Publications Timeline',
        labels={'x': 'Publication Date', 'y': ''},
        hover_data=['title'],
        opacity=0.6
    )
    fig3.update_layout(
        xaxis_title="Publication Date",
        yaxis_showticklabels=False,
        yaxis_title=""
    )
    fig3.write_html('publications_timeline.html')
    
    # 4. Monthly trend line
    monthly_counts = df['year_month'].value_counts().sort_index()
    monthly_dates = [datetime.strptime(date, '%Y-%m') for date in monthly_counts.index]
    
    fig4 = px.line(
        x=monthly_dates,
        y=monthly_counts.values,
        title='Publications per Month (Trend)',
        labels={'x': 'Month', 'y': 'Number of Publications'}
    )
    fig4.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Publications"
    )
    fig4.write_html('publications_monthly_trend.html')
    
    print("Visualizations created:")
    print("- publications_per_year.html")
    print("- publications_heatmap.html") 
    print("- publications_timeline.html")
    print("- publications_monthly_trend.html")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Date range: {df['published_date'].min()} to {df['published_date'].max()}")
    print(f"Total publications: {len(df)}")
    print(f"Years covered: {df['year'].nunique()}")
    print(f"Most active year: {yearly_counts.idxmax()} with {yearly_counts.max()} publications")

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