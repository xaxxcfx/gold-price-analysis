# Visualization Module

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_price_trend(df, save_path=None):
    """
    Plot gold price over time.

    Args:
        df: DataFrame with Date and Close columns
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['Date'], df['Close'], linewidth=2, color='gold')
    ax.fill_between(df['Date'], df['Close'], alpha=0.3, color='gold')

    ax.set_title('Gold Price Trend', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_yearly_comparison(df, save_path=None):
    """
    Create box plot comparing price distribution by year.

    Args:
        df: DataFrame with Year and Close columns
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(data=df, x='Year', y='Close', ax=ax)
    ax.set_title('Gold Price Distribution by Year', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def create_interactive_chart(df):
    """
    Create an interactive Plotly chart.

    Args:
        df: DataFrame with Date and Close columns

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Gold Price',
        line=dict(color='gold', width=2)
    ))

    fig.update_layout(
        title='Gold Price Interactive Chart',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig

if __name__ == "__main__":
    print("Visualization module loaded successfully!")
