import streamlit as st
import pandas as pd
import os
import argparse
import logging
import glob
from typing import Optional
from bokeh.models import (
    LabelSet,
    BoxAnnotation,
    Whisker,
    FactorRange,
    Legend,
    LegendItem,
)
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category10, Viridis256
import re

TITLE = "LLM Profiler Dashboard"

st.set_page_config(layout="wide", page_title=TITLE, page_icon="🚀")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description=TITLE)
parser.add_argument('--directory', '-d', type=str, help='Directory containing the metrics files')
args = parser.parse_args()

# Global variable to store the directory path
METRICS_DIRECTORY = args.directory

# File patterns for automatic loading
FILE_PATTERNS = {
    "ollama_metrics": ["ollama_metrics.csv", "ollama_metrics*.csv"],
    "ollama_score": ["models_score.csv", "*score*.csv"],
    "prometheus_metrics": ["prometheus_metrics.csv", "prometheus_metrics*.csv"],
    "general_info": ["general_info.txt", "general_info*.txt"],
}


def find_file_in_directory(directory: str, file_key: str) -> Optional[str]:
    """Find file in directory based on file patterns"""
    if not directory or not os.path.exists(directory):
        return None

    patterns = FILE_PATTERNS.get(file_key, [])

    for pattern in patterns:
        if "*" in pattern:
            files = glob.glob(os.path.join(directory, pattern))
            if files:
                return files[0]  # Take the first match
        else:
            file_path = os.path.join(directory, pattern)
            if os.path.exists(file_path):
                return file_path

    return None


def load_files_from_directory(directory: str) -> dict:
    """Load all required files from the specified directory"""
    files = {}
    missing_files = []
    found_files = []

    for file_key in FILE_PATTERNS.keys():
        file_path = find_file_in_directory(directory, file_key)
        if file_path:
            files[file_key] = file_path
            found_files.append(f"{file_key}: {os.path.basename(file_path)}")
        else:
            missing_files.append(file_key)

    if found_files:
        st.success(f"✅ Found {len(found_files)} files")

    if missing_files:
        st.error(f"❌ Missing files: {', '.join(missing_files)}")
        return None

    return files


@st.cache_data
def load_dataframe(file_path: str) -> Optional[pd.DataFrame]:
    """Load and cache dataframe from file path"""
    try:
        data = pd.read_csv(file_path, delimiter=";")

        # Validate required columns
        if "timestamp" not in data.columns:
            st.error(f"Missing required 'timestamp' column in {file_path}")
            return None

        # Convert numeric columns safely
        for col in data.columns:
            if col not in ["model", "timestamp"]:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        # Handle timestamp conversion with error checking
        try:
            data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")
        except Exception as e:
            st.warning(f"Timestamp conversion failed, trying alternative formats: {e}")
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

        return data

    except Exception as e:
        st.error(f"Error loading dataframe from {file_path}: {e}")
        logger.error(f"Error loading dataframe from {file_path}: {e}", exc_info=True)
        return None


def display_general_info(file_path: str):
    """Display general information from text file in a compact, formatted way"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse key-value pairs from the content
        info = {}
        for line in content.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                info[key.strip()] = value.strip()

        # Create a compact display with fundamental information
        col1, col2, col3 = st.columns(3)

        with col1:
            # Operating System
            os_name = info.get('PRETTY_NAME', info.get('NAME', 'Unknown OS'))
            st.metric("🖥️ Operating System", os_name)

            # CPU
            cpu_model = info.get('CPU_MODEL', 'Unknown CPU')
            # Shorten CPU name if too long
            if len(cpu_model) > 35:
                cpu_model = cpu_model[:32] + "..."
            st.metric("⚡ CPU", cpu_model)

        with col2:
            # Memory
            memory = info.get('TOTAL_MEMORY', 'Unknown')
            st.metric("🧠 Memory", memory)

            # IP Address
            ip_address = info.get('IP_ADDRESS', 'Unknown')
            st.metric("🌐 IP Address", ip_address)

        with col3:
            # GPU
            gpu_info = info.get('GPU_INFO', 'No GPU info')
            st.metric("🎮 GPU", gpu_info)

        # Show full details in an expandable section if needed
        with st.expander("📋 Full System Details", expanded=False):
            st.text(content)

    except Exception as e:
        st.error(f"Error reading general info file: {e}")


@st.cache_data
def process_model_metrics(ollama_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    """Process and combine model metrics for analysis"""
    # Convert nanoseconds to seconds for duration columns
    duration_cols = [
            'total_duration',
            'load_duration',
            'ttft_duration',
            'prompt_eval_duration',
            'eval_duration',
        ]
    ollama_processed = ollama_df.copy()

    for col in duration_cols:
        if col in ollama_processed.columns:
            ollama_processed[col] = pd.to_numeric(ollama_processed[col], errors='coerce') / 1e9 # Convertir medida de tiempo 

    # Calculate derived metrics
    ollama_processed['response_time'] = (
        ollama_processed['total_duration'] - ollama_processed['load_duration']
    )
    ollama_processed['tokens_per_second'] = (
        ollama_processed['eval_count'] / ollama_processed['eval_duration']
    )
    ollama_processed['prompt_tokens_per_second'] = (
        ollama_processed['prompt_eval_count'] / ollama_processed['prompt_eval_duration']
    )

    # Aggregate by model
    aggregation = {
        'total_duration': ['mean', 'std', 'count'],
        'response_time': ['mean', 'std'],
        'eval_duration': ['mean', 'std'],
        'load_duration': ['mean'],
        'tokens_per_second': ['mean', 'std'],
        'prompt_tokens_per_second': ['mean'],
        'eval_count': ['mean', 'sum'],
        'prompt_eval_count': ['mean'],
    }
    if 'ttft_duration' in ollama_processed.columns:
        aggregation['ttft_duration'] = ['mean', 'std']

    model_stats = ollama_processed.groupby('model').agg(aggregation).round(3)


    # Flatten column names
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns]
    model_stats = model_stats.reset_index()

    # Merge with scores
    if 'Model' in score_df.columns and 'Score' in score_df.columns:
        model_stats = model_stats.merge(score_df, left_on='model', right_on='Model', how='left')

    return model_stats


def create_performance_overview_chart(model_stats: pd.DataFrame):
    """Create the main performance overview scatter plot with family-based grouping"""
    if model_stats.empty:
        st.warning("No model statistics available for performance overview")
        return None

    if 'Score' not in model_stats.columns:
        st.warning("No score data available for performance overview")
        return None

    # Check for required columns
    required_cols = [
        'response_time_mean',
        'tokens_per_second_mean',
        'eval_count_sum',
        'response_time_std',
    ]
    missing_cols = [col for col in required_cols if col not in model_stats.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return None

    # Filter out rows with NaN values
    clean_stats = model_stats.dropna(
        subset=['Score', 'response_time_mean', 'tokens_per_second_mean']
    )

    if clean_stats.empty:
        st.warning("No valid data points for performance overview")
        return None

    # Sort models by family and size
    models_list = clean_stats['model'].tolist()
    sorted_models = sort_models_by_family_and_size(models_list)

    # Get model family groupings and colors
    family_to_models, family_colors = get_model_families_and_colors(sorted_models)

    # Prepare data
    source_data = {
        'model': clean_stats['model'].tolist(),
        'avg_response_time': clean_stats['response_time_mean'].tolist(),
        'score': clean_stats['Score'].tolist(),
        'tokens_per_sec': clean_stats['tokens_per_second_mean'].tolist(),
        'total_tokens': clean_stats['eval_count_sum'].tolist(),
        'consistency': (1 / (clean_stats['response_time_std'] + 0.001)).tolist(),
    }

    # Add family information to tooltips
    model_info_list = []
    colors_list = []
    for model in source_data['model']:
        family, size = parse_model_info(model)
        model_info_list.append(f"{model} ({family}, {size}B)")
        colors_list.append(get_model_color_with_shade(model, family_to_models, family_colors))

    source_data['model_info'] = model_info_list
    source_data['colors'] = colors_list

    source = ColumnDataSource(source_data)

    # Create figure
    p = figure(
        title="🎯 Model Performance Overview: Quality vs Speed (Grouped by Family)",
        x_axis_label="Average Response Time (seconds)",
        y_axis_label="Quality Score",
        width=800,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Create scatter plot with family-based colors
    p.scatter(
        x='avg_response_time',
        y='score',
        size=15,
        color='colors',
        alpha=0.8,
        line_color='white',
        line_width=2,
        source=source,
    )

    # Enhanced hover tool with family information
    hover = HoverTool(
        tooltips=[
            ("Model", "@model_info"),
            ("Quality Score", "@score{0.00}"),
            ("Avg Response Time", "@avg_response_time{0.00}s"),
            ("Tokens/Second", "@tokens_per_sec{0.0}"),
            ("Total Tokens", "@total_tokens{0,0}"),
        ]
    )
    p.add_tools(hover)

    # Add labels for each point
    labels = LabelSet(
        x='avg_response_time',
        y='score',
        text='model',
        x_offset=5,
        y_offset=5,
        source=source,
        text_font_size='9pt',
        text_color='black',
    )
    p.add_layout(labels)

    return p


def create_response_time_distribution(ollama_df: pd.DataFrame):
    """Create box plot showing response time distribution by model with family grouping"""
    # Convert to seconds
    ollama_processed = ollama_df.copy()
    ollama_processed['response_time'] = (
        ollama_processed['total_duration'] - ollama_processed['load_duration']
    ) / 1e9

    df = ollama_processed[["model", "response_time"]].rename(
        columns={"model": "kind", "response_time": "value"}
    )

    models = df.kind.unique()
    sorted_models = sort_models_by_family_and_size(models)
    family_to_models, family_colors = get_model_families_and_colors(sorted_models)

    # Compute quantiles
    grouper = df.groupby("kind")
    qs = grouper.value.quantile([0.25, 0.5, 0.75]).unstack().reset_index()
    qs.columns = ["kind", "q1", "q2", "q3"]

    # Compute IQR outlier bounds
    iqr = qs.q3 - qs.q1
    qs["upper"] = qs.q3 + 1.5 * iqr
    qs["lower"] = qs.q1 - 1.5 * iqr

    # Update the whiskers to actual data points
    for kind, group in grouper:
        qs_idx = qs.query(f"kind=={kind!r}").index[0]
        data = group["value"]

        # The upper whisker is the maximum between q3 and upper
        q3 = qs.loc[qs_idx, "q3"]
        upper = qs.loc[qs_idx, "upper"]
        wiskhi = group[(q3 <= data) & (data <= upper)]["value"]
        qs.loc[qs_idx, "upper"] = q3 if len(wiskhi) == 0 else wiskhi.max()

        # The lower whisker is the minimum between q1 and lower
        q1 = qs.loc[qs_idx, "q1"]
        lower = qs.loc[qs_idx, "lower"]
        wisklo = group[(lower <= data) & (data <= q1)]["value"]
        qs.loc[qs_idx, "lower"] = q1 if len(wisklo) == 0 else wisklo.min()

    color_map = {}
    for model in sorted_models:
        color_map[model] = get_model_color_with_shade(model, family_to_models, family_colors)

    qs['color'] = qs['kind'].map(color_map)

    source = ColumnDataSource(qs)

    plot = figure(
        x_range=sorted_models,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        title="📊 Response Time Distribution by Model (Grouped by Family)",
        y_axis_label="Response Time (seconds)",
        width=800,
        height=400,
    )

    # Whiskers (outlier range)
    whisker = Whisker(base="kind", upper="upper", lower="lower", source=source, line_color="black")
    whisker.upper_head.size = whisker.lower_head.size = 10
    plot.add_layout(whisker)

    # Quantile boxes
    plot.vbar("kind", 0.6, "q2", "q3", source=source, color="color", line_color="black", alpha=0.7)
    plot.vbar("kind", 0.6, "q1", "q2", source=source, color="color", line_color="black", alpha=0.7)

    # Outliers
    df_with_bounds = pd.merge(df, qs[["kind", "lower", "upper"]], on="kind", how="left")
    outliers = df_with_bounds[
        ~df_with_bounds.value.between(df_with_bounds.lower, df_with_bounds.upper)
    ]

    if not outliers.empty:
        outlier_colors = [color_map.get(model, "#888888") for model in outliers["kind"]]
        plot.scatter(
            "kind",
            "value",
            source=ColumnDataSource(outliers.assign(color=outlier_colors)),
            size=6,
            color="color",
            alpha=0.6,
        )

    # Add hover functionality
    hover_data = []
    for _, row in qs.iterrows():
        family, size = parse_model_info(row['kind'])
        hover_info = {
            'x': row['kind'],
            'y': (row['q1'] + row['q3']) / 2,
            'model': row['kind'],
            'family': family,
            'size': f"{size}B",
            'median': f"{row['q2']:.3f}s",
            'q1': f"{row['q1']:.3f}s",
            'q3': f"{row['q3']:.3f}s",
            'iqr': f"{row['q3'] - row['q1']:.3f}s",
            'min': f"{row['lower']:.3f}s",
            'max': f"{row['upper']:.3f}s",
            'outliers_count': len(outliers[outliers['kind'] == row['kind']]),
        }
        hover_data.append(hover_info)

    if hover_data:
        hover_source = ColumnDataSource(pd.DataFrame(hover_data))
        hover_circles = plot.circle(x='x', y='y', size=20, alpha=0, source=hover_source)

        hover = HoverTool(
            renderers=[hover_circles],
            tooltips=[
                ("Model", "@model (@family, @size)"),
                ("Median", "@median"),
                ("Q1 (25%)", "@q1"),
                ("Q3 (75%)", "@q3"),
                ("IQR", "@iqr"),
                ("Min", "@min"),
                ("Max", "@max"),
                ("Outliers", "@outliers_count"),
            ],
        )
        plot.add_tools(hover)

    plot.xaxis.major_label_orientation = 45
    return plot


def create_resource_timeline(prometheus_df: pd.DataFrame, ollama_df: pd.DataFrame):
    """Create resource utilization timeline"""
    p = figure(
        title="📈 Resource Utilization Over Time",
        x_axis_type='datetime',
        y_axis_label="Utilization (%)",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    if 'gpu_utilization' in prometheus_df.columns:
        p.line(
            prometheus_df['timestamp'],
            prometheus_df['gpu_utilization'],
            legend_label="GPU Utilization",
            line_color='red',
            line_width=2,
        )

    if 'cpu' in prometheus_df.columns:
        p.line(
            prometheus_df['timestamp'],
            prometheus_df['cpu'],
            legend_label="CPU Utilization",
            line_color='blue',
            line_width=2,
        )

    # Add model execution periods as shaded regions
    if not ollama_df.empty and 'timestamp' in ollama_df.columns:
        models = ollama_df['model'].unique()
        family_to_models, family_colors = get_model_families_and_colors(models)

        for i, row in ollama_df.iterrows():
            start_time = row['timestamp']
            duration = row['total_duration'] / 1e9  # Convert to seconds
            end_time = start_time + pd.Timedelta(seconds=duration)

            # Get family-based color for this model
            model_color = get_model_color_with_shade(row['model'], family_to_models, family_colors)

            # Add shaded box for model execution period
            box = BoxAnnotation(
                left=start_time.timestamp() * 1000,  # Convert to milliseconds for Bokeh
                right=end_time.timestamp() * 1000,
                fill_alpha=0.2,
                fill_color=model_color,
                line_color=model_color,
                line_alpha=0.5,
            )
            p.add_layout(box)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def create_tokens_per_second_chart(model_stats: pd.DataFrame):
    """Create tokens per second performance chart with family grouping"""
    if model_stats.empty:
        return None

    models = model_stats['model'].tolist()
    sorted_models = sort_models_by_family_and_size(models)

    # Reorder data according to sorted models
    sorted_stats = model_stats.set_index('model').loc[sorted_models].reset_index()
    tokens_per_sec = sorted_stats['tokens_per_second_mean'].tolist()

    # Get family groupings and colors
    family_to_models, family_colors = get_model_families_and_colors(sorted_models)

    p = figure(
        title="🚀 Model Throughput: Tokens per Second (Grouped by Family)",
        x_range=sorted_models,
        y_axis_label="Tokens per Second",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Assign colors to each model
    colors = []
    for model in sorted_models:
        colors.append(get_model_color_with_shade(model, family_to_models, family_colors))

    bars = p.vbar(x=sorted_models, top=tokens_per_sec, width=0.6, color=colors, alpha=0.8)

    # Add value labels on bars
    source = ColumnDataSource(
        dict(x=sorted_models, y=tokens_per_sec, labels=[f"{val:.1f}" for val in tokens_per_sec])
    )
    labels = LabelSet(
        x='x', y='y', text='labels', x_offset=-10, y_offset=5, source=source, text_font_size='10pt'
    )
    p.add_layout(labels)

    p.xaxis.major_label_orientation = 45

    return p


@st.cache_data
def calculate_model_resource_usage(
    ollama_df: pd.DataFrame, prometheus_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate average resource usage per model during execution periods"""
    if ollama_df.empty or prometheus_df.empty:
        st.warning("🔍 Resource correlation: Empty input dataframes")
        return pd.DataFrame()

    # Check for required timestamp columns
    if 'timestamp' not in ollama_df.columns or 'timestamp' not in prometheus_df.columns:
        st.warning("Missing timestamp data for resource correlation")
        return pd.DataFrame()

    model_resources = []
    processed_count = 0
    matched_count = 0

    for _, row in ollama_df.iterrows():
        model = row['model']
        start_time = row['timestamp']

        # Check if we have duration data
        if 'total_duration' not in row or pd.isna(row['total_duration']):
            continue

        processed_count += 1
        duration = row['total_duration'] / 1e9  # Convert to seconds
        end_time = start_time + pd.Timedelta(seconds=duration)

        # Find Prometheus metrics during this model execution
        mask = (prometheus_df['timestamp'] >= start_time) & (prometheus_df['timestamp'] <= end_time)
        execution_metrics = prometheus_df[mask]

        if not execution_metrics.empty:
            matched_count += 1
            # Calculate averages for this execution
            resource_data = {'model': model}

            if 'cpu' in execution_metrics.columns:
                resource_data['avg_cpu_utilization'] = execution_metrics['cpu'].mean()

            if 'gpu_utilization' in execution_metrics.columns:
                resource_data['avg_gpu_utilization'] = execution_metrics['gpu_utilization'].mean()

            if 'memory' in execution_metrics.columns:
                resource_data['avg_memory_utilization'] = execution_metrics['memory'].mean()

            if (
                'gpu_memory_used' in execution_metrics.columns
                and 'gpu_memory_total' in execution_metrics.columns
            ):
                gpu_memory_pct = execution_metrics.apply(
                    lambda r: (r['gpu_memory_used'] / r['gpu_memory_total']) * 100
                    if r['gpu_memory_total'] > 0
                    else 0,
                    axis=1,
                )
                resource_data['avg_gpu_memory_utilization'] = gpu_memory_pct.mean()

            model_resources.append(resource_data)

    if not model_resources:
        st.warning("🔍 No resource correlations found - check timestamp alignment between datasets")
        return pd.DataFrame()

    # Convert to DataFrame and aggregate by model
    resource_df = pd.DataFrame(model_resources)

    # Group by model and calculate overall averages
    model_resource_stats = (
        resource_df.groupby('model')
        .agg({col: 'mean' for col in resource_df.columns if col != 'model'})
        .round(2)
    )

    model_resource_stats = model_resource_stats.reset_index()

    return model_resource_stats


def create_cpu_gpu_utilization_chart(model_resource_stats: pd.DataFrame):
    """Create CPU and GPU utilization chart per model with family grouping"""
    if model_resource_stats.empty:
        st.warning("No resource utilization data available")
        return None

    # Check for required columns
    has_cpu = 'avg_cpu_utilization' in model_resource_stats.columns
    has_gpu = 'avg_gpu_utilization' in model_resource_stats.columns

    if not has_cpu and not has_gpu:
        st.warning("No CPU or GPU utilization data available")
        return None

    models = model_resource_stats['model'].tolist()
    sorted_models = sort_models_by_family_and_size(models)

    # Reorder data according to sorted models
    sorted_stats = model_resource_stats.set_index('model').loc[sorted_models].reset_index()

    # Get family groupings and colors
    family_to_models, family_colors = get_model_families_and_colors(sorted_models)

    # Prepare data for side-by-side bars
    chart_data = []
    for i, model in enumerate(sorted_models):
        stats_row = sorted_stats.iloc[i]
        family, size = parse_model_info(model)
        base_color = get_model_color_with_shade(model, family_to_models, family_colors)

        if has_cpu:
            chart_data.append(
                {
                    'model': model,
                    'resource_type': 'CPU',
                    'utilization': stats_row['avg_cpu_utilization'],
                    'color': base_color,
                    'alpha': 0.9,  # More opaque for CPU
                    'line_color': '#333333',  # Dark border for CPU
                    'line_width': 3,
                    'hatch_pattern': '',  # Solid fill for CPU
                    'family': family,
                    'size': f"{size}B",
                }
            )

        if has_gpu:
            # Make GPU bars with increased saturation and diagonal stripes effect
            gpu_color = base_color
            if base_color.startswith('#'):
                r, g, b = tuple(int(base_color[i : i + 2], 16) for i in (1, 3, 5))
                # Increase saturation for GPU bars
                r = min(255, int(r * 1.3))
                g = min(255, int(g * 1.3))
                b = min(255, int(b * 1.3))
                gpu_color = f"#{r:02x}{g:02x}{b:02x}"

            chart_data.append(
                {
                    'model': model,
                    'resource_type': 'GPU',
                    'utilization': stats_row['avg_gpu_utilization'],
                    'color': gpu_color,
                    'alpha': 0.75,  # Slightly transparent for GPU
                    'line_color': '#666666',  # Medium gray border for GPU
                    'line_width': 2,
                    'hatch_pattern': '///',  # Diagonal stripes for GPU
                    'family': family,
                    'size': f"{size}B",
                }
            )

    chart_df = pd.DataFrame(chart_data)

    # Transform data for x axis BEFORE creating source
    chart_df['x'] = list(zip(chart_df['model'], chart_df['resource_type']))

    source = ColumnDataSource(chart_df)

    # Create factors for grouped bars
    factors = []
    for model in sorted_models:
        if has_cpu and has_gpu:
            factors.extend([(model, 'CPU'), (model, 'GPU')])
        elif has_cpu:
            factors.append((model, 'CPU'))
        elif has_gpu:
            factors.append((model, 'GPU'))

    p = figure(
        title="⚡ Average CPU & GPU Utilization per Model (Grouped by Family)",
        x_range=FactorRange(*factors),
        y_axis_label="Utilization (%)",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Create bars with enhanced visual distinction
    bars = p.vbar(
        x='x',
        top='utilization',
        width=0.8,
        color='color',
        alpha='alpha',
        line_color='line_color',
        line_width='line_width',
        source=source,
    )

    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Model", "@model (@family, @size)"),
            ("Resource", "@resource_type"),
            ("Utilization", "@utilization{0.1f}%"),
        ]
    )
    p.add_tools(hover)

    # Customize x-axis
    p.xaxis.major_label_orientation = 45
    p.xgrid.grid_line_color = None

    # Create dummy glyphs for legend
    if has_cpu:
        cpu_glyph = p.rect(
            x=0,
            y=0,
            width=0,
            height=0,
            color=family_colors[list(family_colors.keys())[0]],
            alpha=0.9,
            line_color='#333333',
            line_width=3,
            visible=False,
        )
    if has_gpu:
        gpu_glyph = p.rect(
            x=0,
            y=0,
            width=0,
            height=0,
            color=family_colors[list(family_colors.keys())[0]],
            alpha=0.75,
            line_color='#666666',
            line_width=2,
            visible=False,
        )

    legend_items = []
    if has_cpu:
        legend_items.append(LegendItem(label="CPU (solid border)", renderers=[cpu_glyph]))
    if has_gpu:
        legend_items.append(LegendItem(label="GPU (thin border)", renderers=[gpu_glyph]))

    legend = Legend(items=legend_items, location="top_right")
    p.add_layout(legend)

    return p


def create_memory_utilization_chart(model_resource_stats: pd.DataFrame):
    """Create memory utilization chart per model with family grouping"""
    if model_resource_stats.empty:
        st.warning("No memory utilization data available")
        return None

    # Check for required columns
    has_ram = 'avg_memory_utilization' in model_resource_stats.columns
    has_gpu_mem = 'avg_gpu_memory_utilization' in model_resource_stats.columns

    if not has_ram and not has_gpu_mem:
        st.warning("No memory utilization data available")
        return None

    models = model_resource_stats['model'].tolist()
    sorted_models = sort_models_by_family_and_size(models)

    # Reorder data according to sorted models
    sorted_stats = model_resource_stats.set_index('model').loc[sorted_models].reset_index()

    # Get family groupings and colors
    family_to_models, family_colors = get_model_families_and_colors(sorted_models)

    # Prepare data for side-by-side bars
    chart_data = []
    for i, model in enumerate(sorted_models):
        stats_row = sorted_stats.iloc[i]
        family, size = parse_model_info(model)
        color = get_model_color_with_shade(model, family_to_models, family_colors)

        if has_ram:
            chart_data.append(
                {
                    'model': model,
                    'memory_type': 'System RAM',
                    'utilization': stats_row['avg_memory_utilization'],
                    'color': color,
                    'alpha': 0.9,  # More opaque for System RAM
                    'line_color': '#333333',  # Dark border for System RAM
                    'line_width': 3,
                    'hatch_pattern': '',  # Solid fill for System RAM
                    'family': family,
                    'size': f"{size}B",
                }
            )

        if has_gpu_mem:
            # Make GPU memory bars with enhanced visual distinction
            gpu_color = color
            if color.startswith('#'):
                r, g, b = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
                # Make GPU memory bars more vibrant
                r = min(255, int(r * 1.25))
                g = min(255, int(g * 1.25))
                b = min(255, int(b * 1.25))
                gpu_color = f"#{r:02x}{g:02x}{b:02x}"

            chart_data.append(
                {
                    'model': model,
                    'memory_type': 'GPU Memory',
                    'utilization': stats_row['avg_gpu_memory_utilization'],
                    'color': gpu_color,
                    'alpha': 0.75,  # Slightly transparent for GPU Memory
                    'line_color': '#666666',  # Medium gray border for GPU Memory
                    'line_width': 2,
                    'hatch_pattern': '\\\\\\',  # Reverse diagonal stripes for GPU Memory
                    'family': family,
                    'size': f"{size}B",
                }
            )

    chart_df = pd.DataFrame(chart_data)

    # Transform data for x axis BEFORE creating source
    chart_df['x'] = list(zip(chart_df['model'], chart_df['memory_type']))

    source = ColumnDataSource(chart_df)

    # Create factors for grouped bars
    factors = []
    for model in sorted_models:
        if has_ram and has_gpu_mem:
            factors.extend([(model, 'System RAM'), (model, 'GPU Memory')])
        elif has_ram:
            factors.append((model, 'System RAM'))
        elif has_gpu_mem:
            factors.append((model, 'GPU Memory'))

    p = figure(
        title="💾 Average Memory Utilization per Model (Grouped by Family)",
        x_range=FactorRange(*factors),
        y_axis_label="Memory Utilization (%)",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Create bars with enhanced visual distinction
    bars = p.vbar(
        x='x',
        top='utilization',
        width=0.8,
        color='color',
        alpha='alpha',
        line_color='line_color',
        line_width='line_width',
        source=source,
    )

    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Model", "@model (@family, @size)"),
            ("Memory Type", "@memory_type"),
            ("Utilization", "@utilization{0.1f}%"),
        ]
    )
    p.add_tools(hover)

    # Customize x-axis
    p.xaxis.major_label_orientation = 45
    p.xgrid.grid_line_color = None

    # Create dummy glyphs for legend
    if has_ram:
        ram_glyph = p.rect(
            x=0,
            y=0,
            width=0,
            height=0,
            color=family_colors[list(family_colors.keys())[0]],
            alpha=0.9,
            line_color='#333333',
            line_width=3,
            visible=False,
        )
    if has_gpu_mem:
        gpu_mem_glyph = p.rect(
            x=0,
            y=0,
            width=0,
            height=0,
            color=family_colors[list(family_colors.keys())[0]],
            alpha=0.75,
            line_color='#666666',
            line_width=2,
            visible=False,
        )

    legend_items = []
    if has_ram:
        legend_items.append(LegendItem(label="System RAM (solid border)", renderers=[ram_glyph]))
    if has_gpu_mem:
        legend_items.append(LegendItem(label="GPU Memory (thin border)", renderers=[gpu_mem_glyph]))

    legend = Legend(items=legend_items, location="top_right")
    p.add_layout(legend)

    return p


def parse_model_info(model_name: str) -> tuple[str, float]:
    """Parse model name to extract family and parameter size.

    Returns:
        tuple: (family_name, parameter_size_in_billions)

    """
    model_name = model_name.lower().strip()

    # Common patterns for parameter sizes
    size_patterns = [
        r'(\d+\.?\d*)b\b',  # e.g., "7b", "13b", "0.5b"
        r'(\d+\.?\d*)-?billion',  # e.g., "7-billion", "13billion"
        r'(\d+\.?\d*)t\b',  # e.g., "1t" (trillion parameters, convert to billions)
    ]

    parameter_size = None
    for pattern in size_patterns:
        match = re.search(pattern, model_name)
        if match:
            size = float(match.group(1))
            if 't' in pattern:  # trillion parameters
                size *= 1000  # convert to billions
            parameter_size = size
            break

    # Extract family name (everything before the colon or size indicator)
    family_patterns = [
        r'^([^:]+):',  # e.g., "gemma3:12b" -> "gemma3"
        r'^([a-zA-Z-]+)',  # e.g., "orca-mini" -> "orca-mini"
    ]

    family_name = model_name
    for pattern in family_patterns:
        match = re.search(pattern, model_name)
        if match:
            family_name = match.group(1)
            break

    # Handle special cases and estimate sizes for known models
    size_estimates = {
        'orca-mini': 3.0,  # Typically around 3B
        'orca': 13.0,  # Standard Orca is usually 13B
        'vicuna': 7.0,  # Common Vicuna size
        'alpaca': 7.0,  # Common Alpaca size
        'codellama': 7.0,  # Default CodeLlama size
        'llama': 7.0,  # Default Llama size
    }

    if parameter_size is None:
        # Try to estimate based on family name
        for family_key, estimated_size in size_estimates.items():
            if family_key in family_name:
                parameter_size = estimated_size
                break

        # If still no size, default to 1.0B
        if parameter_size is None:
            parameter_size = 1.0

    return family_name, parameter_size


def get_model_families_and_colors(models: list) -> tuple[dict, dict]:
    """Get family groupings and assign colors to model families.

    Returns:
        tuple: (family_to_models_dict, family_to_color_dict)

    """
    # Parse all models
    model_info = {}
    families = set()

    for model in models:
        family, size = parse_model_info(model)
        model_info[model] = (family, size)
        families.add(family)

    # Assign base colors to families
    family_colors = {}
    base_colors = Category10[10] if len(families) <= 10 else Viridis256[:: int(256 / len(families))]

    for i, family in enumerate(sorted(families)):
        family_colors[family] = base_colors[i % len(base_colors)]

    # Group models by family
    family_to_models = {}
    for model in models:
        family, size = model_info[model]
        if family not in family_to_models:
            family_to_models[family] = []
        family_to_models[family].append((model, size))

    # Sort models within each family by parameter size
    for family in family_to_models:
        family_to_models[family].sort(key=lambda x: x[1])  # Sort by parameter size

    return family_to_models, family_colors


def sort_models_by_family_and_size(models: list) -> list:
    """Sort models by family name and then by parameter size within each family."""
    model_info = [(model, *parse_model_info(model)) for model in models]
    # Sort by family name first, then by parameter size
    model_info.sort(key=lambda x: (x[1], x[2]))
    return [model[0] for model in model_info]


def get_model_color_with_shade(model: str, family_to_models: dict, family_colors: dict) -> str:
    """Get a color for a specific model based on its family and position within the family.
    Models in the same family get different shades of the same base color.
    """
    family, size = parse_model_info(model)
    base_color = family_colors.get(family, "#888888")

    if family not in family_to_models:
        return base_color

    # Get models in this family sorted by size
    family_models = family_to_models[family]
    model_index = next((i for i, (m, s) in enumerate(family_models) if m == model), 0)

    # Create shades by adjusting the brightness
    num_models = len(family_models)
    if num_models == 1:
        return base_color

    # Convert hex to RGB for manipulation
    base_color = base_color.lstrip('#')
    r, g, b = tuple(int(base_color[i : i + 2], 16) for i in (0, 2, 4))

    # Create darker/lighter shades
    # Smallest model gets lightest shade, largest gets darkest
    shade_factor = 0.3 + 0.7 * (model_index / (num_models - 1))  # Range from 0.3 to 1.0

    r = int(r * shade_factor)
    g = int(g * shade_factor)
    b = int(b * shade_factor)

    return f"#{r:02x}{g:02x}{b:02x}"


def create_memory_usage_chart(prometheus_df: pd.DataFrame, ollama_df: pd.DataFrame):
    """Create memory usage timeline showing both system RAM and GPU memory"""
    p = figure(
        title="💾 Memory Usage Over Time",
        x_axis_type='datetime',
        y_axis_label="Memory Usage (%)",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # System RAM usage (percentage)
    if 'memory' in prometheus_df.columns:
        p.line(
            prometheus_df['timestamp'],
            prometheus_df['memory'],
            legend_label="System RAM (%)",
            line_color='blue',
            line_width=2,
        )

    # GPU Memory usage (convert to percentage)
    if 'gpu_memory_used' in prometheus_df.columns and 'gpu_memory_total' in prometheus_df.columns:
        # Calculate GPU memory percentage, handling division by zero
        gpu_memory_pct = prometheus_df.apply(
            lambda row: (row['gpu_memory_used'] / row['gpu_memory_total']) * 100
            if row['gpu_memory_total'] > 0
            else 0,
            axis=1,
        )
        p.line(
            prometheus_df['timestamp'],
            gpu_memory_pct,
            legend_label="GPU Memory (%)",
            line_color='red',
            line_width=2,
        )

        # Add GPU memory in GB as additional info
        gpu_memory_gb = prometheus_df['gpu_memory_used'] / (1024**3)
        max_gpu_gb = prometheus_df['gpu_memory_total'].max() / (1024**3)

        # Create hover tool with additional GPU memory info
        hover_source = ColumnDataSource(
            data=dict(
                timestamp=prometheus_df['timestamp'],
                ram_pct=prometheus_df['memory']
                if 'memory' in prometheus_df.columns
                else [0] * len(prometheus_df),
                gpu_pct=gpu_memory_pct,
                gpu_gb=gpu_memory_gb,
                gpu_total_gb=[max_gpu_gb] * len(prometheus_df),
            )
        )

        # Add invisible circles for detailed hover information
        hover_circles = p.circle(x='timestamp', y='gpu_pct', size=8, alpha=0, source=hover_source)

        hover = HoverTool(
            renderers=[hover_circles],
            tooltips=[
                ("Time", "@timestamp{%F %T}"),
                ("System RAM", "@ram_pct{0.1f}%"),
                ("GPU Memory", "@gpu_pct{0.1f}%"),
                ("GPU Memory", "@gpu_gb{0.1f} / @gpu_total_gb{0.1f} GB"),
            ],
            formatters={'@timestamp': 'datetime'},
        )
        p.add_tools(hover)

    # Add model execution periods as shaded regions
    if not ollama_df.empty and 'timestamp' in ollama_df.columns:
        models = ollama_df['model'].unique()
        family_to_models, family_colors = get_model_families_and_colors(models)

        for i, row in ollama_df.iterrows():
            start_time = row['timestamp']
            duration = row['total_duration'] / 1e9  # Convert to seconds
            end_time = start_time + pd.Timedelta(seconds=duration)

            # Get family-based color for this model
            model_color = get_model_color_with_shade(row['model'], family_to_models, family_colors)

            # Add shaded box for model execution period
            box = BoxAnnotation(
                left=start_time.timestamp() * 1000,  # Convert to milliseconds for Bokeh
                right=end_time.timestamp() * 1000,
                fill_alpha=0.15,
                fill_color=model_color,
                line_color=model_color,
                line_alpha=0.3,
            )
            p.add_layout(box)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def create_disk_io_timeline(prometheus_df: pd.DataFrame, ollama_df: pd.DataFrame):
    """Create disk I/O timeline showing read/write activity"""
    p = figure(
        title="💽 Disk I/O Activity Over Time",
        x_axis_type='datetime',
        y_axis_label="I/O Operations per Second",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Calculate I/O operations per second by taking differences
    if (
        'disk_reads_completed' in prometheus_df.columns
        and 'disk_writes_completed' in prometheus_df.columns
    ):
        # Sort by timestamp to ensure proper calculation
        sorted_df = prometheus_df.sort_values('timestamp').copy()

        # Calculate time differences in seconds
        time_diffs = sorted_df['timestamp'].diff().dt.total_seconds()

        # Calculate I/O rates (operations per second)
        read_ops_diff = sorted_df['disk_reads_completed'].diff()
        write_ops_diff = sorted_df['disk_writes_completed'].diff()

        # Avoid division by zero
        time_diffs = time_diffs.fillna(1).replace(0, 1)

        # Calculate rates and ensure non-negative values (filter out counter resets)
        read_ops_per_sec = (read_ops_diff / time_diffs).fillna(0)
        write_ops_per_sec = (write_ops_diff / time_diffs).fillna(0)

        # Set negative rates to 0 (counter resets or data inconsistencies)
        read_ops_per_sec = read_ops_per_sec.clip(lower=0)
        write_ops_per_sec = write_ops_per_sec.clip(lower=0)

        # Plot I/O operations per second
        p.line(
            sorted_df['timestamp'],
            read_ops_per_sec,
            legend_label="Disk Reads/sec",
            line_color='blue',
            line_width=2,
        )

        p.line(
            sorted_df['timestamp'],
            write_ops_per_sec,
            legend_label="Disk Writes/sec",
            line_color='red',
            line_width=2,
        )

    # Add disk busy time as percentage if available
    if 'disk_busy_time' in prometheus_df.columns:
        # Add disk busy time line (assuming it's already a percentage)
        p.line(
            prometheus_df['timestamp'],
            prometheus_df['disk_busy_time'],
            legend_label="Disk Busy %",
            line_color='orange',
            line_width=2,
            line_dash='dashed',
        )

    # Add model execution periods as shaded regions
    if not ollama_df.empty and 'timestamp' in ollama_df.columns:
        models = ollama_df['model'].unique()
        family_to_models, family_colors = get_model_families_and_colors(models)

        for i, row in ollama_df.iterrows():
            start_time = row['timestamp']
            duration = row['total_duration'] / 1e9  # Convert to seconds
            end_time = start_time + pd.Timedelta(seconds=duration)

            # Get family-based color for this model
            model_color = get_model_color_with_shade(row['model'], family_to_models, family_colors)

            # Add shaded box for model execution period
            box = BoxAnnotation(
                left=start_time.timestamp() * 1000,  # Convert to milliseconds for Bokeh
                right=end_time.timestamp() * 1000,
                fill_alpha=0.15,
                fill_color=model_color,
                line_color=model_color,
                line_alpha=0.3,
            )
            p.add_layout(box)

    # Add hover tool with disk I/O information
    if len(prometheus_df) > 1:
        hover_source = ColumnDataSource(
            data=dict(
                timestamp=prometheus_df['timestamp'],
                disk_reads=prometheus_df.get('disk_reads_completed', [0] * len(prometheus_df)),
                disk_writes=prometheus_df.get('disk_writes_completed', [0] * len(prometheus_df)),
                disk_busy=prometheus_df.get('disk_busy_time', [0] * len(prometheus_df)),
                disk_usage_gb=prometheus_df.get('disk_used_bytes', [0] * len(prometheus_df))
                / (1024**3),
            )
        )

        # Add invisible circles for detailed hover information
        hover_circles = p.circle(x='timestamp', y=0, size=8, alpha=0, source=hover_source)

        hover = HoverTool(
            renderers=[hover_circles],
            tooltips=[
                ("Time", "@timestamp{%F %T}"),
                ("Total Reads", "@disk_reads{0,0}"),
                ("Total Writes", "@disk_writes{0,0}"),
                ("Disk Busy", "@disk_busy{0.1f}%"),
                ("Disk Usage", "@disk_usage_gb{0.1f} GB"),
            ],
            formatters={'@timestamp': 'datetime'},
        )
        p.add_tools(hover)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


@st.cache_data
def calculate_disk_metrics_per_model(
    ollama_df: pd.DataFrame, prometheus_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate average disk I/O metrics per model during execution periods"""
    if ollama_df.empty or prometheus_df.empty:
        st.warning("🔍 Disk correlation: Empty input dataframes")
        return pd.DataFrame()

    # Check for required columns
    required_cols = ['disk_reads_completed', 'disk_writes_completed', 'disk_busy_time']
    missing_cols = [col for col in required_cols if col not in prometheus_df.columns]

    if missing_cols:
        st.warning(f"Missing disk metrics columns: {missing_cols}")
        return pd.DataFrame()

    if 'timestamp' not in ollama_df.columns or 'timestamp' not in prometheus_df.columns:
        st.warning("Missing timestamp data for disk correlation")
        return pd.DataFrame()

    model_disk_metrics = []
    processed_count = 0
    matched_count = 0

    # Sort prometheus data by timestamp for proper I/O rate calculation
    prometheus_sorted = prometheus_df.sort_values('timestamp').copy()

    for _, row in ollama_df.iterrows():
        model = row['model']
        start_time = row['timestamp']

        # Check if we have duration data
        if 'total_duration' not in row or pd.isna(row['total_duration']):
            continue

        processed_count += 1
        duration = row['total_duration'] / 1e9  # Convert to seconds
        end_time = start_time + pd.Timedelta(seconds=duration)

        # Find Prometheus metrics during this model execution
        mask = (prometheus_sorted['timestamp'] >= start_time) & (
            prometheus_sorted['timestamp'] <= end_time
        )
        execution_metrics = prometheus_sorted[mask]

        if len(execution_metrics) > 1:  # Need at least 2 points to calculate rates
            matched_count += 1

            # Calculate I/O rates during execution
            time_diffs = execution_metrics['timestamp'].diff().dt.total_seconds().fillna(1)
            time_diffs = time_diffs.replace(0, 1)  # Avoid division by zero

            read_ops_diff = execution_metrics['disk_reads_completed'].diff().fillna(0)
            write_ops_diff = execution_metrics['disk_writes_completed'].diff().fillna(0)

            # Filter out negative differences (counter resets or data inconsistencies)
            # Only consider positive differences for rate calculation
            valid_read_diffs = read_ops_diff[read_ops_diff >= 0]
            valid_write_diffs = write_ops_diff[write_ops_diff >= 0]
            valid_time_diffs_read = time_diffs[read_ops_diff >= 0]
            valid_time_diffs_write = time_diffs[write_ops_diff >= 0]

            # Calculate rates only from valid positive differences
            if len(valid_read_diffs) > 0 and len(valid_time_diffs_read) > 0:
                avg_read_rate = (valid_read_diffs / valid_time_diffs_read).mean()
            else:
                avg_read_rate = 0.0

            if len(valid_write_diffs) > 0 and len(valid_time_diffs_write) > 0:
                avg_write_rate = (valid_write_diffs / valid_time_diffs_write).mean()
            else:
                avg_write_rate = 0.0

            # Ensure rates are non-negative (additional safety check)
            avg_read_rate = max(0.0, avg_read_rate)
            avg_write_rate = max(0.0, avg_write_rate)

            avg_busy_time = execution_metrics['disk_busy_time'].mean()

            # Calculate total I/O during execution
            total_reads = (
                execution_metrics['disk_reads_completed'].iloc[-1]
                - execution_metrics['disk_reads_completed'].iloc[0]
            )
            total_writes = (
                execution_metrics['disk_writes_completed'].iloc[-1]
                - execution_metrics['disk_writes_completed'].iloc[0]
            )

            disk_data = {
                'model': model,
                'avg_disk_read_rate': avg_read_rate,
                'avg_disk_write_rate': avg_write_rate,
                'avg_disk_busy_pct': avg_busy_time,
                'total_disk_reads': total_reads,
                'total_disk_writes': total_writes,
                'execution_duration': duration,
            }

            model_disk_metrics.append(disk_data)

    if not model_disk_metrics:
        st.warning("🔍 No disk correlations found - check timestamp alignment between datasets")
        return pd.DataFrame()

    # Convert to DataFrame and aggregate by model
    disk_df = pd.DataFrame(model_disk_metrics)

    # Group by model and calculate overall averages
    model_disk_stats = (
        disk_df.groupby('model')
        .agg({col: 'mean' for col in disk_df.columns if col != 'model'})
        .round(2)
    )

    model_disk_stats = model_disk_stats.reset_index()

    return model_disk_stats


def create_disk_utilization_per_model_chart(model_disk_stats: pd.DataFrame):
    """Create disk utilization chart per model with family grouping"""
    if model_disk_stats.empty:
        st.warning("No disk utilization data available")
        return None

    models = model_disk_stats['model'].tolist()
    sorted_models = sort_models_by_family_and_size(models)

    # Reorder data according to sorted models
    sorted_stats = model_disk_stats.set_index('model').loc[sorted_models].reset_index()

    # Get family groupings and colors
    family_to_models, family_colors = get_model_families_and_colors(sorted_models)

    p = figure(
        title="💽 Average Disk Activity per Model (Grouped by Family)",
        x_range=sorted_models,
        y_axis_label="Operations per Second",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Prepare data for grouped bars
    chart_data = []
    for i, model in enumerate(sorted_models):
        stats_row = sorted_stats.iloc[i]
        family, size = parse_model_info(model)
        base_color = get_model_color_with_shade(model, family_to_models, family_colors)

        # Read operations
        read_rate = max(0, stats_row.get('avg_disk_read_rate', 0))  # Ensure non-negative
        chart_data.append(
            {
                'model': model,
                'operation_type': 'Reads',
                'rate': read_rate,
                'color': base_color,
                'alpha': 0.9,
                'line_color': '#333333',
                'line_width': 3,
                'family': family,
                'size': f"{size}B",
            }
        )

        # Write operations (with slightly different shade)
        write_color = base_color
        if base_color.startswith('#'):
            r, g, b = tuple(int(base_color[i : i + 2], 16) for i in (1, 3, 5))
            r = min(255, int(r * 1.2))
            g = min(255, int(g * 1.2))
            b = min(255, int(b * 1.2))
            write_color = f"#{r:02x}{g:02x}{b:02x}"

        write_rate = max(0, stats_row.get('avg_disk_write_rate', 0))  # Ensure non-negative
        chart_data.append(
            {
                'model': model,
                'operation_type': 'Writes',
                'rate': write_rate,
                'color': write_color,
                'alpha': 0.75,
                'line_color': '#666666',
                'line_width': 2,
                'family': family,
                'size': f"{size}B",
            }
        )

    chart_df = pd.DataFrame(chart_data)
    chart_df['x'] = list(zip(chart_df['model'], chart_df['operation_type']))

    source = ColumnDataSource(chart_df)

    # Create factors for grouped bars
    factors = []
    for model in sorted_models:
        factors.extend([(model, 'Reads'), (model, 'Writes')])

    p.x_range = FactorRange(*factors)

    # Create bars
    bars = p.vbar(
        x='x',
        top='rate',
        width=0.8,
        color='color',
        alpha='alpha',
        line_color='line_color',
        line_width='line_width',
        source=source,
    )

    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Model", "@model (@family, @size)"),
            ("Operation", "@operation_type"),
            ("Rate", "@rate{0.1f} ops/sec"),
        ]
    )
    p.add_tools(hover)

    # Customize x-axis
    p.xaxis.major_label_orientation = 45
    p.xgrid.grid_line_color = None

    # Create legend
    read_glyph = p.rect(
        x=0,
        y=0,
        width=0,
        height=0,
        color=family_colors[list(family_colors.keys())[0]],
        alpha=0.9,
        line_color='#333333',
        line_width=3,
        visible=False,
    )
    write_glyph = p.rect(
        x=0,
        y=0,
        width=0,
        height=0,
        color=family_colors[list(family_colors.keys())[0]],
        alpha=0.75,
        line_color='#666666',
        line_width=2,
        visible=False,
    )

    legend_items = [
        LegendItem(label="Reads (solid border)", renderers=[read_glyph]),
        LegendItem(label="Writes (thin border)", renderers=[write_glyph]),
    ]

    legend = Legend(items=legend_items, location="top_right")
    p.add_layout(legend)

    return p


def create_gpu_power_timeline(prometheus_df: pd.DataFrame, ollama_df: pd.DataFrame):
    """Create GPU power usage timeline"""
    p = figure(
        title="⚡ GPU Power Usage Over Time",
        x_axis_type='datetime',
        y_axis_label="Power Usage (W)",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    if 'gpu_power_usage' in prometheus_df.columns:
        # Convert from milliwatts to watts if values are in mW range
        power_values = prometheus_df['gpu_power_usage'].copy()

        # Check if values are likely in milliwatts (typical GPU power is 50-400W, so mW would be 50000-400000)
        if power_values.max() > 1000:
            power_values = power_values / 1000  # Convert mW to W

        p.line(
            prometheus_df['timestamp'],
            power_values,
            legend_label="GPU Power (W)",
            line_color='orange',
            line_width=2,
        )

        # Add GPU temperature as secondary info if available
        if 'gpu_temperature' in prometheus_df.columns:
            # Create a second y-axis for temperature
            temp_values = prometheus_df['gpu_temperature']

            # Scale temperature to fit nicely with power values for visual purposes
            # We'll add it as a separate line but keep the tooltip informative
            p.line(
                prometheus_df['timestamp'],
                temp_values * 3,  # Scale factor to make temperature visible alongside power
                legend_label="GPU Temp (°C × 3)",
                line_color='red',
                line_width=1,
                line_dash='dashed',
                alpha=0.7,
            )

    # Add model execution periods as shaded regions
    if not ollama_df.empty and 'timestamp' in ollama_df.columns:
        models = ollama_df['model'].unique()
        family_to_models, family_colors = get_model_families_and_colors(models)

        for i, row in ollama_df.iterrows():
            start_time = row['timestamp']
            duration = row['total_duration'] / 1e9  # Convert to seconds
            end_time = start_time + pd.Timedelta(seconds=duration)

            # Get family-based color for this model
            model_color = get_model_color_with_shade(row['model'], family_to_models, family_colors)

            # Add shaded box for model execution period
            box = BoxAnnotation(
                left=start_time.timestamp() * 1000,  # Convert to milliseconds for Bokeh
                right=end_time.timestamp() * 1000,
                fill_alpha=0.15,
                fill_color=model_color,
                line_color=model_color,
                line_alpha=0.3,
            )
            p.add_layout(box)

    # Add hover tool with power information
    if 'gpu_power_usage' in prometheus_df.columns:
        power_values = prometheus_df['gpu_power_usage'].copy()
        if power_values.max() > 1000:
            power_values = power_values / 1000

        hover_source = ColumnDataSource(
            data=dict(
                timestamp=prometheus_df['timestamp'],
                gpu_power_w=power_values,
                gpu_power_raw=prometheus_df['gpu_power_usage'],
                gpu_utilization=prometheus_df.get('gpu_utilization', [0] * len(prometheus_df)),
                gpu_temp=prometheus_df.get('gpu_temperature', [0] * len(prometheus_df)),
            )
        )

        # Add invisible circles for detailed hover information
        hover_circles = p.circle(
            x='timestamp', y='gpu_power_w', size=8, alpha=0, source=hover_source
        )

        hover = HoverTool(
            renderers=[hover_circles],
            tooltips=[
                ("Time", "@timestamp{%F %T}"),
                ("GPU Power", "@gpu_power_w{0.1f} W"),
                ("GPU Utilization", "@gpu_utilization{0.0f}%"),
                ("GPU Temperature", "@gpu_temp{0.0f}°C"),
            ],
            formatters={'@timestamp': 'datetime'},
        )
        p.add_tools(hover)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


@st.cache_data
def calculate_gpu_power_per_model(
    ollama_df: pd.DataFrame, prometheus_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate average GPU power usage per model during execution periods"""
    if ollama_df.empty or prometheus_df.empty:
        st.warning("🔍 GPU power correlation: Empty input dataframes")
        return pd.DataFrame()

    # Check for required columns
    if 'gpu_power_usage' not in prometheus_df.columns:
        st.warning("Missing gpu_power_usage column in prometheus data")
        return pd.DataFrame()

    if 'timestamp' not in ollama_df.columns or 'timestamp' not in prometheus_df.columns:
        st.warning("Missing timestamp data for GPU power correlation")
        return pd.DataFrame()

    model_power_metrics = []
    processed_count = 0
    matched_count = 0

    for _, row in ollama_df.iterrows():
        model = row['model']
        start_time = row['timestamp']

        # Check if we have duration data
        if 'total_duration' not in row or pd.isna(row['total_duration']):
            continue

        processed_count += 1
        duration = row['total_duration'] / 1e9  # Convert to seconds
        end_time = start_time + pd.Timedelta(seconds=duration)

        # Find Prometheus metrics during this model execution
        mask = (prometheus_df['timestamp'] >= start_time) & (prometheus_df['timestamp'] <= end_time)
        execution_metrics = prometheus_df[mask]

        if len(execution_metrics) > 0:
            matched_count += 1

            # Calculate power metrics during execution
            power_values = execution_metrics['gpu_power_usage']

            # Convert from milliwatts to watts if necessary
            if power_values.max() > 1000:
                power_values = power_values / 1000

            avg_power = power_values.mean()
            max_power = power_values.max()
            min_power = power_values.min()

            # Calculate energy consumption (average power × duration)
            energy_wh = (avg_power * duration) / 3600  # Convert to watt-hours

            power_data = {
                'model': model,
                'avg_gpu_power_w': avg_power,
                'max_gpu_power_w': max_power,
                'min_gpu_power_w': min_power,
                'energy_consumption_wh': energy_wh,
                'execution_duration': duration,
            }

            # Add temperature data if available
            if 'gpu_temperature' in execution_metrics.columns:
                power_data['avg_gpu_temp'] = execution_metrics['gpu_temperature'].mean()
                power_data['max_gpu_temp'] = execution_metrics['gpu_temperature'].max()

            model_power_metrics.append(power_data)

    if not model_power_metrics:
        st.warning(
            "🔍 No GPU power correlations found - check timestamp alignment between datasets"
        )
        return pd.DataFrame()

    # Convert to DataFrame and aggregate by model
    power_df = pd.DataFrame(model_power_metrics)

    # Group by model and calculate overall averages
    model_power_stats = (
        power_df.groupby('model')
        .agg({col: 'mean' for col in power_df.columns if col != 'model'})
        .round(2)
    )

    model_power_stats = model_power_stats.reset_index()

    return model_power_stats


def create_gpu_power_per_model_chart(model_power_stats: pd.DataFrame):
    """Create GPU power usage chart per model with family grouping"""
    if model_power_stats.empty:
        st.warning("No GPU power usage data available")
        return None

    models = model_power_stats['model'].tolist()
    sorted_models = sort_models_by_family_and_size(models)

    # Reorder data according to sorted models
    sorted_stats = model_power_stats.set_index('model').loc[sorted_models].reset_index()

    # Get family groupings and colors
    family_to_models, family_colors = get_model_families_and_colors(sorted_models)

    p = figure(
        title="⚡ Average GPU Power Usage per Model (Grouped by Family)",
        x_range=sorted_models,
        y_axis_label="Power Usage (W)",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Assign colors to each model
    colors = []
    for model in sorted_models:
        colors.append(get_model_color_with_shade(model, family_to_models, family_colors))

    avg_power = sorted_stats['avg_gpu_power_w'].tolist()

    bars = p.vbar(x=sorted_models, top=avg_power, width=0.6, color=colors, alpha=0.8)

    # Add value labels on bars
    source = ColumnDataSource(
        dict(
            x=sorted_models,
            y=avg_power,
            labels=[f"{val:.1f}W" for val in avg_power],
            energy=[f"{val:.2f}Wh" for val in sorted_stats['energy_consumption_wh'].tolist()],
            max_power=[f"{val:.1f}W" for val in sorted_stats['max_gpu_power_w'].tolist()],
            temp=[
                f"{val:.0f}°C"
                for val in sorted_stats.get('avg_gpu_temp', [0] * len(sorted_stats)).tolist()
            ]
            if 'avg_gpu_temp' in sorted_stats.columns
            else ["N/A"] * len(sorted_stats),
        )
    )

    labels = LabelSet(
        x='x', y='y', text='labels', x_offset=-15, y_offset=5, source=source, text_font_size='9pt'
    )
    p.add_layout(labels)

    # Enhanced hover tool
    hover = HoverTool(
        tooltips=[
            ("Model", "@x"),
            ("Avg Power", "@labels"),
            ("Max Power", "@max_power"),
            ("Energy/Execution", "@energy"),
            ("Avg Temperature", "@temp"),
        ]
    )
    p.add_tools(hover)

    p.xaxis.major_label_orientation = 45

    return p


def main():
    """Main application function"""
    st.title(f"🚀 {TITLE}")

    # Initialize session state for directory
    if 'current_directory' not in st.session_state:
        st.session_state.current_directory = METRICS_DIRECTORY

    # Directory selection section - compact layout
    col1, col2, col3 = st.columns([2, 3, 1])

    with col1:
        st.markdown("**📁 Data Directory:**")

    with col2:
        new_directory = st.text_input(
            "Directory path:",
            value=st.session_state.current_directory or "",
            placeholder="/path/to/metrics/directory",
            label_visibility="collapsed",
        )

    with col3:
        if st.button("Load", type="primary"):
            if new_directory and os.path.exists(new_directory):
                st.session_state.current_directory = new_directory
                st.rerun()
            elif new_directory:
                st.error(f"Directory does not exist: {new_directory}")
            else:
                st.error("Please enter a valid directory path")

    # If no directory is set, stop here
    if not st.session_state.current_directory:
        st.warning("Please select a directory containing the metrics files.")
        st.stop()

    # Load files from directory
    with st.spinner("Loading data files..."):
        files = load_files_from_directory(st.session_state.current_directory)

    if not files:
        st.error("Failed to load required files. Please check the directory and file formats.")
        st.stop()

    # Display file loading status in one compact line
    file_names = [os.path.basename(path) for path in files.values()]
    st.info(f"📋 **Loaded files:** {' • '.join(file_names)}")

    # Display general information
    st.header("📊 System Information")
    display_general_info(files['general_info'])

    # Load dataframes
    st.header("📈 Data Loading")

    try:
        with st.spinner("Processing data files..."):
            ollama_df = load_dataframe(files['ollama_metrics'])
            prometheus_df = load_dataframe(files['prometheus_metrics'])
            score_df = pd.read_csv(files['ollama_score'], delimiter=";")

        if ollama_df is not None and prometheus_df is not None:
            st.success("✅ All data files loaded successfully!")

            # Display basic data info
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Ollama Records", len(ollama_df))
                if 'model' in ollama_df.columns:
                    st.metric("Models", ollama_df['model'].nunique())

            with col2:
                st.metric("Prometheus Records", len(prometheus_df))
                if 'timestamp' in prometheus_df.columns:
                    time_range = prometheus_df['timestamp'].max() - prometheus_df['timestamp'].min()
                    # Convert to human readable format
                    total_seconds = int(time_range.total_seconds())
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60

                    if hours > 0:
                        time_display = f"{hours}h {minutes}m {seconds}s"
                    elif minutes > 0:
                        time_display = f"{minutes}m {seconds}s"
                    else:
                        time_display = f"{seconds}s"

                    st.metric("Time Range", time_display)

            with col3:
                st.metric("Score Records", len(score_df))
                if 'Score' in score_df.columns:
                    st.metric("Avg Score", f"{score_df['Score'].mean():.2f}")

            # Process model metrics
            with st.spinner("Processing model metrics..."):
                model_stats = process_model_metrics(ollama_df, score_df)
                model_resource_stats = calculate_model_resource_usage(ollama_df, prometheus_df)
                model_disk_stats = calculate_disk_metrics_per_model(ollama_df, prometheus_df)
                model_power_stats = calculate_gpu_power_per_model(ollama_df, prometheus_df)

            # 1. Performance Overview (Quality vs Speed)
            st.subheader("🎯 Performance Overview: Quality vs Speed")
            st.markdown(
                "*This chart shows the critical trade-off between model quality (score) and response speed.*"
            )

            perf_chart = create_performance_overview_chart(model_stats)
            if perf_chart:
                st.bokeh_chart(perf_chart, use_container_width=True)

            # 2. Response Time Distribution
            st.subheader("📊 Response Time Consistency")
            st.markdown(
                "*Box plots showing response time distribution for each model. Smaller boxes indicate more consistent performance.*"
            )

            dist_chart = create_response_time_distribution(ollama_df)
            if dist_chart:
                st.bokeh_chart(dist_chart, use_container_width=True)

            # 3. Tokens per Second Performance
            st.subheader("🚀 Model Throughput")
            st.markdown(
                "*Higher tokens/second means faster text generation - critical for production deployments.*"
            )

            tokens_chart = create_tokens_per_second_chart(model_stats)
            if tokens_chart:
                st.bokeh_chart(tokens_chart, use_container_width=True)

            # 4. Resource Utilization Timeline
            st.subheader("📈 System Resource Usage")
            st.markdown(
                "*Timeline showing GPU and CPU utilization during model execution. Vertical lines indicate model runs.*"
            )

            resource_chart = create_resource_timeline(prometheus_df, ollama_df)
            if resource_chart:
                st.bokeh_chart(resource_chart, use_container_width=True)

            # 5. Memory Usage Over Time
            st.subheader("💾 Memory Usage Over Time")
            st.markdown(
                "*Tracks system RAM and GPU memory usage over time, with model execution periods.*"
            )

            memory_chart = create_memory_usage_chart(prometheus_df, ollama_df)
            if memory_chart:
                st.bokeh_chart(memory_chart, use_container_width=True)

            # 6. Disk I/O Activity Timeline
            st.subheader("💽 Disk I/O Activity Over Time")
            st.markdown(
                "*Shows disk read/write operations per second and disk busy percentage during model execution.*"
            )

            disk_io_chart = create_disk_io_timeline(prometheus_df, ollama_df)
            if disk_io_chart:
                st.bokeh_chart(disk_io_chart, use_container_width=True)

            # 7. GPU Power Usage Over Time
            st.subheader("⚡ GPU Power Usage Over Time")
            st.markdown(
                "*Shows GPU power consumption in watts and temperature during model execution.*"
            )

            gpu_power_chart = create_gpu_power_timeline(prometheus_df, ollama_df)
            if gpu_power_chart:
                st.bokeh_chart(gpu_power_chart, use_container_width=True)

            # 8. Average CPU & GPU Utilization per Model
            st.subheader("⚡ Average Resource Utilization per Model")
            st.markdown(
                "*Shows average CPU and GPU utilization during each model's execution periods.*"
            )
            st.info(
                "🎨 **Visual Guide:** CPU bars have thick dark borders and are more opaque, while GPU bars have thinner gray borders and are slightly transparent. Colors are grouped by model family."
            )

            cpu_gpu_chart = create_cpu_gpu_utilization_chart(model_resource_stats)
            if cpu_gpu_chart:
                st.bokeh_chart(cpu_gpu_chart, use_container_width=True)

            # 9. Average Memory Utilization per Model
            st.subheader("💾 Average Memory Utilization per Model")
            st.markdown(
                "*Shows average system RAM and GPU memory usage during each model's execution periods.*"
            )
            st.info(
                "🎨 **Visual Guide:** System RAM bars have thick dark borders and are more opaque, while GPU Memory bars have thinner gray borders and are slightly transparent. Colors are grouped by model family."
            )

            memory_util_chart = create_memory_utilization_chart(model_resource_stats)
            if memory_util_chart:
                st.bokeh_chart(memory_util_chart, use_container_width=True)

            # 10. Average Disk Activity per Model
            st.subheader("💽 Average Disk Activity per Model")
            st.markdown(
                "*Shows average disk read and write operations per second during each model's execution periods.*"
            )
            st.info(
                "🎨 **Visual Guide:** Read bars have thick dark borders and are more opaque, while Write bars have thinner gray borders and are slightly transparent. Colors are grouped by model family."
            )

            disk_activity_chart = create_disk_utilization_per_model_chart(model_disk_stats)
            if disk_activity_chart:
                st.bokeh_chart(disk_activity_chart, use_container_width=True)

            # 11. Average GPU Power Usage per Model
            st.subheader("⚡ Average GPU Power Usage per Model")
            st.markdown(
                "*Shows average GPU power consumption in watts and energy usage during each model's execution periods.*"
            )

            gpu_power_model_chart = create_gpu_power_per_model_chart(model_power_stats)
            if gpu_power_model_chart:
                st.bokeh_chart(gpu_power_model_chart, use_container_width=True)

            # 12. Model Statistics Summary
            st.subheader("📋 Model Performance Summary")

            # Create a summary table with family information
            if not model_stats.empty:
                # Add family and size information
                model_families = []
                model_sizes = []
                for model in model_stats['model']:
                    family, size = parse_model_info(model)
                    model_families.append(family)
                    model_sizes.append(f"{size}B")

                summary_data = {
                    'Model': model_stats['model'],
                    'Family': model_families,
                    'Size': model_sizes,
                    'Quality Score': model_stats.get('Score', [0] * len(model_stats)),
                    'Avg Response Time (s)': model_stats['response_time_mean'],
                    'Tokens/Second': model_stats['tokens_per_second_mean'],
                    'Total Requests': model_stats['total_duration_count'],
                    'Consistency (1/std)': (
                        1 / (model_stats['response_time_std'] + 0.001)
                    ).round(2),
                }
                if 'ttft_duration_mean' in model_stats.columns:
                    summary_data['Avg TTFT (s)'] = model_stats['ttft_duration_mean']

                summary_df = pd.DataFrame(summary_data)

                # Add disk metrics if available
                if not model_disk_stats.empty:
                    disk_summary = model_disk_stats[
                        ['model', 'avg_disk_read_rate', 'avg_disk_write_rate', 'avg_disk_busy_pct']
                    ].copy()
                    disk_summary.columns = [
                        'Model',
                        'Avg Disk Reads/sec',
                        'Avg Disk Writes/sec',
                        'Avg Disk Busy %',
                    ]
                    summary_df = summary_df.merge(disk_summary, on='Model', how='left')

                # Add GPU power metrics if available
                if not model_power_stats.empty:
                    power_summary = model_power_stats[
                        ['model', 'avg_gpu_power_w', 'max_gpu_power_w', 'energy_consumption_wh']
                    ].copy()
                    power_summary.columns = [
                        'Model',
                        'Avg GPU Power (W)',
                        'Max GPU Power (W)',
                        'Energy/Execution (Wh)',
                    ]
                    summary_df = summary_df.merge(power_summary, on='Model', how='left')

                # Sort by family and size
                sorted_models = sort_models_by_family_and_size(model_stats['model'].tolist())
                summary_df = summary_df.set_index('Model').loc[sorted_models].reset_index()

                # Format the dataframe for better display
                summary_df['Quality Score'] = summary_df['Quality Score'].round(2)
                summary_df['Avg Response Time (s)'] = summary_df['Avg Response Time (s)'].round(2)
                summary_df['Tokens/Second'] = summary_df['Tokens/Second'].round(1)

                if 'Avg TTFT (s)' in summary_df.columns:
                    summary_df['Avg TTFT (s)'] = summary_df['Avg TTFT (s)'].round(3)

                if 'Avg Disk Reads/sec' in summary_df.columns:
                    summary_df['Avg Disk Reads/sec'] = summary_df['Avg Disk Reads/sec'].round(2)
                    summary_df['Avg Disk Writes/sec'] = summary_df['Avg Disk Writes/sec'].round(2)
                    summary_df['Avg Disk Busy %'] = summary_df['Avg Disk Busy %'].round(1)

                if 'Avg GPU Power (W)' in summary_df.columns:
                    summary_df['Avg GPU Power (W)'] = summary_df['Avg GPU Power (W)'].round(2)
                    summary_df['Max GPU Power (W)'] = summary_df['Max GPU Power (W)'].round(2)
                    summary_df['Energy/Execution (Wh)'] = summary_df['Energy/Execution (Wh)'].round(
                        2
                    )

                st.dataframe(summary_df, use_container_width=True)

                # Key insights
                st.subheader("💡 Key Insights")

                if len(summary_df) > 0:
                    best_quality = summary_df.loc[summary_df['Quality Score'].idxmax()]
                    fastest = summary_df.loc[summary_df['Tokens/Second'].idxmax()]
                    most_consistent = summary_df.loc[summary_df['Consistency (1/std)'].idxmax()]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "🏆 Highest Quality",
                            best_quality['Model'],
                            f"Score: {best_quality['Quality Score']}",
                        )

                    with col2:
                        st.metric(
                            "⚡ Fastest Throughput",
                            fastest['Model'],
                            f"{fastest['Tokens/Second']:.1f} tok/s",
                        )

                    with col3:
                        st.metric(
                            "🎯 Most Consistent",
                            most_consistent['Model'],
                            f"Consistency: {most_consistent['Consistency (1/std)']:.1f}",
                        )

                    # Add disk-specific insights if available
                    if (
                        'Avg Disk Reads/sec' in summary_df.columns
                        and not summary_df['Avg Disk Reads/sec'].isna().all()
                    ):
                        col4, col5 = st.columns(2)

                        with col4:
                            most_disk_reads = summary_df.loc[
                                summary_df['Avg Disk Reads/sec'].idxmax()
                            ]
                            st.metric(
                                "💽 Most Disk Reads",
                                most_disk_reads['Model'],
                                f"{most_disk_reads['Avg Disk Reads/sec']:.1f} reads/s",
                            )

                        with col5:
                            most_disk_writes = summary_df.loc[
                                summary_df['Avg Disk Writes/sec'].idxmax()
                            ]
                            st.metric(
                                "✏️ Most Disk Writes",
                                most_disk_writes['Model'],
                                f"{most_disk_writes['Avg Disk Writes/sec']:.1f} writes/s",
                            )

                    # Add GPU power-specific insights if available
                    if (
                        'Avg GPU Power (W)' in summary_df.columns
                        and not summary_df['Avg GPU Power (W)'].isna().all()
                    ):
                        col6, col7 = st.columns(2)

                        with col6:
                            highest_power = summary_df.loc[summary_df['Avg GPU Power (W)'].idxmax()]
                            st.metric(
                                "⚡ Highest GPU Power",
                                highest_power['Model'],
                                f"{highest_power['Avg GPU Power (W)']:.1f} W",
                            )

                        with col7:
                            most_energy = summary_df.loc[
                                summary_df['Energy/Execution (Wh)'].idxmax()
                            ]
                            st.metric(
                                "🔋 Most Energy per Execution",
                                most_energy['Model'],
                                f"{most_energy['Energy/Execution (Wh)']:.2f} Wh",
                            )

            # Display data preview
            with st.expander("🔍 Data Preview", expanded=False):
                tab1, tab2, tab3 = st.tabs(["Ollama Data", "Prometheus Data", "Score Data"])

                with tab1:
                    st.dataframe(ollama_df.head(), use_container_width=True)

                with tab2:
                    st.dataframe(prometheus_df.head(), use_container_width=True)

                with tab3:
                    st.dataframe(score_df, use_container_width=True)

        else:
            st.error("Failed to load data files. Please check the file formats and content.")

    except Exception as e:
        st.error(f"An error occurred while processing data: {e}")
        logger.error(f"Data processing error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
