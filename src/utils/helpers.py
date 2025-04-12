import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    return str(timedelta(seconds=int(seconds)))

def plot_confidence_distribution(df, column):
    """
    Plot confidence distribution for a given column.
    
    Args:
        df (pd.DataFrame): The data
        column (str): The column to plot
        
    Returns:
        plotly figure: The plot
    """
    fig = px.histogram(df, x=column, nbins=20, 
                      title=f'Distribution of {column} Confidence',
                      labels={column: 'Confidence (%)'},
                      color_discrete_sequence=['#FF4B4B'])
    fig.update_layout(bargap=0.1)
    return fig
