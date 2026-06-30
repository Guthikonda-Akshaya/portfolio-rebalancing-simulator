"""Portfolio Rebalancing Simulator.

A single-page Streamlit application for comparing buy-and-hold,
calendar rebalancing, and threshold rebalancing strategies using
historical stock and ETF price data.
"""

# =============================================================================
# 1. Imports
# =============================================================================

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


# =============================================================================
# 2. Helper Functions
# =============================================================================

INITIAL_INVESTMENT = 10_000
RISK_FREE_RATE = 0.0
TRADING_DAYS_PER_YEAR = 252

CRISIS_PERIODS = {
    "2008 Financial Crisis": (date(2007, 10, 1), date(2009, 3, 31)),
    "COVID Crash (2020)": (date(2020, 1, 1), date(2020, 6, 30)),
    "2022 Inflation Bear Market": (date(2022, 1, 1), date(2022, 12, 31)),
    "Dot-com Crash": (date(2000, 3, 1), date(2002, 10, 31)),
    "Custom Range": (date(2018, 1, 1), date.today()),
}


def parse_tickers(ticker_text):
    """Parse comma-separated ticker symbols into a clean uppercase list."""
    return [
        ticker.strip().upper()
        for ticker in ticker_text.split(",")
        if ticker.strip()
    ]


def parse_weight_input(weight_text):
    """Parse weights from comma-separated values or ticker-weight lines."""
    cleaned_text = weight_text.replace("\n", ",").replace(";", ",")
    raw_parts = [part.strip() for part in cleaned_text.split(",") if part.strip()]
    weights = []

    for part in raw_parts:
        normalized_part = (
            part.replace(":", " ")
            .replace("=", " ")
            .replace("%", "")
            .strip()
        )
        tokens = normalized_part.split()
        weight_candidate = tokens[-1] if tokens else normalized_part
        weights.append(float(weight_candidate))

    return weights


def validate_portfolio_inputs(tickers, weights):
    """Validate tickers and target weights before downloading data."""
    errors = []

    if not tickers:
        errors.append("Please enter at least one ticker symbol.")

    if len(tickers) != len(set(tickers)):
        errors.append("Duplicate tickers found. Please enter each ticker once.")

    if len(weights) != len(tickers):
        errors.append(
            "Please enter exactly one target weight for each ticker symbol."
        )

    if weights and any(weight < 0 for weight in weights):
        errors.append("Target weights cannot be negative.")

    if weights and abs(sum(weights) - 100.0) > 0.01:
        errors.append("Target weights must sum to 100%.")

    return errors


def format_currency(value):
    """Format a numeric value as US dollars."""
    return f"${value:,.2f}"


def format_percent(value):
    """Format a decimal return as a percentage."""
    return f"{value * 100:,.2f}%"


def get_frequency_rule(frequency_label):
    """Convert a sidebar frequency label into a pandas period rule."""
    frequency_map = {
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "Y",
    }
    return frequency_map[frequency_label]


def normalize_price_frame(raw_data, tickers):
    """Extract adjusted close prices, falling back to close prices if needed."""
    if raw_data.empty:
        return pd.DataFrame()

    if isinstance(raw_data.columns, pd.MultiIndex):
        first_level = raw_data.columns.get_level_values(0)
        if "Adj Close" in first_level:
            prices = raw_data["Adj Close"].copy()
        elif "Close" in first_level:
            prices = raw_data["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        if "Adj Close" in raw_data.columns:
            prices = raw_data["Adj Close"].copy()
        elif "Close" in raw_data.columns:
            prices = raw_data["Close"].copy()
        else:
            return pd.DataFrame()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.reindex(columns=tickers)
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices


def create_metric_table(metrics_by_strategy):
    """Create a strategy comparison table from calculated metrics."""
    metric_labels = {
        "final_value": "Final Value",
        "total_return": "Total Return",
        "annual_return": "Annual Return",
        "annual_volatility": "Volatility",
        "sharpe_ratio": "Sharpe Ratio",
        "max_drawdown": "Max Drawdown",
        "rebalance_count": "Number of Rebalances",
    }

    table_data = {}
    for strategy_name, metrics in metrics_by_strategy.items():
        table_data[strategy_name] = {
            "Final Value": format_currency(metrics["final_value"]),
            "Total Return": format_percent(metrics["total_return"]),
            "Annual Return": format_percent(metrics["annual_return"]),
            "Volatility": format_percent(metrics["annual_volatility"]),
            "Sharpe Ratio": f"{metrics['sharpe_ratio']:.2f}",
            "Max Drawdown": format_percent(metrics["max_drawdown"]),
            "Number of Rebalances": str(int(metrics["rebalance_count"])),
        }

    comparison_table = pd.DataFrame(table_data)
    comparison_table.insert(0, "Metric", list(metric_labels.values()))
    return comparison_table


def create_allocation_snapshot_table(simulation_results):
    """Create a table of the latest allocation for each strategy."""
    rows = []

    for strategy_name, result in simulation_results.items():
        final_weights = result["weights"].iloc[-1]
        row = {"Strategy": strategy_name}
        row.update(
            {
                ticker: format_percent(weight)
                for ticker, weight in final_weights.items()
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def style_application():
    """Apply lightweight custom styling to the Streamlit interface."""
    st.set_page_config(
        page_title="Portfolio Rebalancing Simulator",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .kpi-card {
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 8px;
            padding: 1rem;
            background: rgba(250, 250, 250, 0.75);
        }
        .kpi-title {
            font-size: 0.95rem;
            color: #4b5563;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }
        .kpi-value {
            font-size: 1.45rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .kpi-subtext {
            font-size: 0.9rem;
            color: #374151;
            line-height: 1.45;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.35rem 0.75rem;
            margin-top: 0.65rem;
            font-size: 0.86rem;
        }
        .kpi-label {
            color: #6b7280;
            display: block;
            font-size: 0.76rem;
            font-weight: 600;
        }
        .kpi-metric {
            color: #111827;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 3. Data Fetching
# =============================================================================


@st.cache_data(show_spinner=False)
def fetch_price_data(tickers, start_date, end_date):
    """Download historical adjusted closing prices from yfinance."""
    try:
        download_end_date = end_date + timedelta(days=1)
        raw_data = yf.download(
            tickers=tickers,
            start=start_date,
            end=download_end_date,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=True,
        )
    except Exception as error:
        return pd.DataFrame(), tickers, str(error)

    prices = normalize_price_frame(raw_data, tickers)

    if prices.empty:
        return pd.DataFrame(), tickers, "No price data was returned."

    invalid_tickers = [
        ticker
        for ticker in tickers
        if ticker not in prices.columns or prices[ticker].dropna().empty
    ]

    if invalid_tickers:
        return prices, invalid_tickers, ""

    prices = prices.dropna(how="all")
    prices = prices.ffill().bfill()
    prices = prices.dropna()

    if prices.empty:
        return (
            pd.DataFrame(),
            tickers,
            "Downloaded data contained no complete price history.",
        )

    return prices, [], ""


# =============================================================================
# 4. Portfolio Simulation
# =============================================================================


def should_calendar_rebalance(current_date, previous_date, frequency_rule):
    """Return True when a date is the first trading day of a new period."""
    if previous_date is None:
        return False

    current_period = current_date.to_period(frequency_rule)
    previous_period = previous_date.to_period(frequency_rule)
    return current_period != previous_period


def initialize_shares(first_prices, target_weights, initial_investment):
    """Create the starting share position for a target allocation."""
    target_dollars = initial_investment * target_weights
    return target_dollars / first_prices


def simulate_portfolio(
    prices,
    target_weights,
    strategy_name,
    frequency_rule="M",
    tolerance_band=0.05,
):
    """Simulate a portfolio strategy and return values, weights, and events."""
    shares = initialize_shares(
        prices.iloc[0],
        target_weights,
        INITIAL_INVESTMENT,
    )

    portfolio_values = []
    weight_history = []
    rebalance_dates = []
    previous_date = None

    for current_date, current_prices in prices.iterrows():
        current_asset_values = shares * current_prices
        current_total_value = current_asset_values.sum()
        current_weights = current_asset_values / current_total_value

        rebalance_now = False

        if strategy_name == "Calendar":
            rebalance_now = should_calendar_rebalance(
                current_date,
                previous_date,
                frequency_rule,
            )
        elif strategy_name == "Threshold":
            weight_drift = (current_weights - target_weights).abs()
            rebalance_now = bool((weight_drift > tolerance_band).any())

        if rebalance_now:
            shares = (current_total_value * target_weights) / current_prices
            current_asset_values = shares * current_prices
            current_total_value = current_asset_values.sum()
            current_weights = current_asset_values / current_total_value
            rebalance_dates.append(current_date)

        portfolio_values.append(current_total_value)
        weight_history.append(current_weights)
        previous_date = current_date

    value_series = pd.Series(
        portfolio_values,
        index=prices.index,
        name=strategy_name,
    )
    weight_frame = pd.DataFrame(weight_history, index=prices.index)
    weight_frame.columns = prices.columns

    return {
        "values": value_series,
        "weights": weight_frame,
        "rebalance_dates": rebalance_dates,
    }


def run_all_simulations(prices, target_weights, frequency_rule, tolerance_band):
    """Run all supported portfolio strategies."""
    return {
        "Buy & Hold": simulate_portfolio(
            prices,
            target_weights,
            "Buy & Hold",
            frequency_rule,
            tolerance_band,
        ),
        "Calendar": simulate_portfolio(
            prices,
            target_weights,
            "Calendar",
            frequency_rule,
            tolerance_band,
        ),
        "Threshold": simulate_portfolio(
            prices,
            target_weights,
            "Threshold",
            frequency_rule,
            tolerance_band,
        ),
    }


# =============================================================================
# 5. Performance Metrics
# =============================================================================


def calculate_drawdown(portfolio_values):
    """Calculate the drawdown series for a portfolio value history."""
    running_max = portfolio_values.cummax()
    return (portfolio_values / running_max) - 1.0


def calculate_performance_metrics(portfolio_values, rebalance_count):
    """Calculate standard performance metrics for a strategy."""
    daily_returns = portfolio_values.pct_change().dropna()
    final_value = portfolio_values.iloc[-1]
    total_return = (final_value / portfolio_values.iloc[0]) - 1.0

    elapsed_days = max((portfolio_values.index[-1] - portfolio_values.index[0]).days, 1)
    elapsed_years = elapsed_days / 365.25
    annual_return = (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0

    annual_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if annual_volatility == 0 or np.isnan(annual_volatility):
        sharpe_ratio = 0.0
    else:
        excess_daily_return = daily_returns.mean() - (
            RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        )
        sharpe_ratio = (
            excess_daily_return
            / daily_returns.std()
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

    max_drawdown = calculate_drawdown(portfolio_values).min()

    return {
        "final_value": final_value,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "rebalance_count": rebalance_count,
    }


def calculate_all_metrics(simulation_results):
    """Calculate performance metrics for every strategy."""
    metrics_by_strategy = {}

    for strategy_name, result in simulation_results.items():
        metrics_by_strategy[strategy_name] = calculate_performance_metrics(
            result["values"],
            len(result["rebalance_dates"]),
        )

    return metrics_by_strategy


def analyze_market_regimes(benchmark_prices, simulation_results):
    """Identify bull and bear market days using SPY versus its 200-day SMA."""
    benchmark = benchmark_prices.dropna().copy()
    if len(benchmark) < 200:
        return pd.DataFrame()

    benchmark_sma = benchmark.rolling(window=200).mean()
    aligned_index = benchmark_sma.dropna().index
    benchmark = benchmark.loc[aligned_index]
    benchmark_sma = benchmark_sma.loc[aligned_index]

    regime_series = pd.Series(
        np.where(benchmark > benchmark_sma, "Bull Market", "Bear Market"),
        index=aligned_index,
        name="Market Regime",
    )

    strategy_returns = pd.DataFrame(
        {
            strategy_name: result["values"].pct_change()
            for strategy_name, result in simulation_results.items()
        }
    ).reindex(regime_series.index)

    rows = []
    for regime_name in ["Bull Market", "Bear Market"]:
        regime_dates = regime_series[regime_series == regime_name].index
        if regime_dates.empty:
            continue

        compounded_returns = (
            (1.0 + strategy_returns.loc[regime_dates].fillna(0.0)).prod() - 1.0
        )
        best_strategy = compounded_returns.idxmax()
        rows.append(
            {
                "Market Regime": regime_name,
                "Best Performing Strategy": best_strategy,
                "Return": format_percent(compounded_returns[best_strategy]),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 6. Plotting Functions
# =============================================================================


def plot_equity_curve(simulation_results):
    """Create an equity curve chart with calendar and threshold markers."""
    figure = go.Figure()

    for strategy_name, result in simulation_results.items():
        figure.add_trace(
            go.Scatter(
                x=result["values"].index,
                y=result["values"],
                mode="lines",
                name=strategy_name,
            )
        )

    marker_styles = {
        "Calendar": {
            "symbol": "circle-open",
            "color": "#2563eb",
            "name": "Calendar Rebalance",
        },
        "Threshold": {
            "symbol": "diamond-open",
            "color": "#dc2626",
            "name": "Threshold Rebalance",
        },
    }

    for strategy_name, marker_style in marker_styles.items():
        result = simulation_results[strategy_name]
        rebalance_dates = result["rebalance_dates"]

        if not rebalance_dates:
            continue

        figure.add_trace(
            go.Scatter(
                x=rebalance_dates,
                y=result["values"].loc[rebalance_dates],
                mode="markers",
                name=marker_style["name"],
                marker=dict(
                    size=6,
                    symbol=marker_style["symbol"],
                    color=marker_style["color"],
                    line=dict(width=1.5),
                ),
                opacity=0.75,
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    "Portfolio Value: $%{y:,.2f}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    figure.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        template="plotly_white",
        legend_title_text="Strategy",
    )
    return figure


def plot_risk_return(metrics_by_strategy):
    """Create a risk versus return scatter plot from calculated metrics."""
    display_names = {
        "Buy & Hold": "Buy & Hold",
        "Calendar": "Calendar Rebalancing",
        "Threshold": "Threshold Rebalancing",
    }
    risk_return_data = pd.DataFrame(
        [
            {
                "Strategy": display_names.get(strategy_name, strategy_name),
                "Annual Return": metrics["annual_return"],
                "Annualized Volatility": metrics["annual_volatility"],
                "Sharpe Ratio": metrics["sharpe_ratio"],
            }
            for strategy_name, metrics in metrics_by_strategy.items()
        ]
    )

    figure = go.Figure(
        data=[
            go.Scatter(
                x=risk_return_data["Annualized Volatility"],
                y=risk_return_data["Annual Return"],
                mode="markers+text",
                text=risk_return_data["Strategy"],
                textposition="top center",
                marker=dict(
                    size=13,
                    color=risk_return_data["Sharpe Ratio"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Sharpe"),
                    line=dict(color="white", width=1.5),
                ),
                customdata=risk_return_data[["Strategy", "Sharpe Ratio"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Volatility: %{x:.2%}<br>"
                    "Annual Return: %{y:.2%}<br>"
                    "Sharpe Ratio: %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        title="Risk vs Return",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        template="plotly_white",
        showlegend=False,
    )
    return figure


def plot_weight_drift(weight_frame, title):
    """Create a stacked area chart of portfolio weights over time."""
    figure = go.Figure()

    for ticker in weight_frame.columns:
        figure.add_trace(
            go.Scatter(
                x=weight_frame.index,
                y=weight_frame[ticker],
                mode="lines",
                stackgroup="one",
                name=ticker,
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>%{y:.2%}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Weight",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        template="plotly_white",
        legend_title_text="Asset",
    )
    return figure


def plot_allocation_pie(final_weights, strategy_name):
    """Create a pie chart for a strategy's final portfolio allocation."""
    figure = go.Figure(
        data=[
            go.Pie(
                labels=final_weights.index,
                values=final_weights.values,
                hole=0.38,
                textinfo="label+percent",
            )
        ]
    )
    figure.update_layout(
        title=f"{strategy_name} Final Allocation",
        template="plotly_white",
        showlegend=False,
        margin=dict(t=55, b=10, l=10, r=10),
    )
    return figure


def plot_drawdown(simulation_results):
    """Create an interactive drawdown comparison chart."""
    figure = go.Figure()

    for strategy_name, result in simulation_results.items():
        drawdown = calculate_drawdown(result["values"])
        figure.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode="lines",
                name=strategy_name,
            )
        )

    figure.update_layout(
        title="Drawdown Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        template="plotly_white",
        legend_title_text="Strategy",
    )
    return figure


def plot_regime_background(benchmark_prices):
    """Create a simple SPY chart with its 200-day SMA for regime context."""
    benchmark = benchmark_prices.dropna().copy()
    moving_average = benchmark.rolling(window=200).mean()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=benchmark.index,
            y=benchmark,
            mode="lines",
            name="SPY",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=moving_average.index,
            y=moving_average,
            mode="lines",
            name="200-Day SMA",
        )
    )
    figure.update_layout(
        title="SPY Market Regime Reference",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
    )
    return figure


def plot_weight_drift_comparison(simulation_results):
    """Create side-by-side stacked area charts for weight drift comparison."""
    buy_hold_weights = simulation_results["Buy & Hold"]["weights"]
    calendar_weights = simulation_results["Calendar"]["weights"]

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Buy & Hold", "Calendar Rebalanced"),
        shared_yaxes=True,
    )

    for ticker in buy_hold_weights.columns:
        figure.add_trace(
            go.Scatter(
                x=buy_hold_weights.index,
                y=buy_hold_weights[ticker],
                mode="lines",
                stackgroup="buy_hold",
                name=ticker,
                legendgroup=ticker,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=calendar_weights.index,
                y=calendar_weights[ticker],
                mode="lines",
                stackgroup="calendar",
                name=ticker,
                legendgroup=ticker,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    figure.update_layout(
        title="Portfolio Weight Drift",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        template="plotly_white",
        legend_title_text="Asset",
    )
    figure.update_yaxes(title_text="Portfolio Weight", row=1, col=1)
    figure.update_xaxes(title_text="Date")
    return figure


# =============================================================================
# 7. Streamlit UI
# =============================================================================


def render_header():
    """Render the application title and summary."""
    st.title("Portfolio Rebalancing Simulator")
    st.write(
        "Compare buy-and-hold, calendar rebalancing, and threshold rebalancing "
        "using historical market data. The simulator uses a $10,000 starting "
        "portfolio, daily adjusted closing prices, and excludes taxes, slippage, "
        "and transaction costs."
    )


def render_strategy_expander():
    """Render concise explanations of the rebalancing strategies."""
    with st.expander("About the Rebalancing Strategies"):
        st.markdown(
            """
            **Buy & Hold**  
            Invest once according to the target allocation and never rebalance.
            Asset weights naturally drift over time as prices change.

            **Calendar Rebalancing**  
            Restore the portfolio to the target allocation at fixed intervals
            (monthly, quarterly, or yearly), regardless of market movements.

            **Threshold Rebalancing**  
            Only rebalance when any asset's allocation deviates beyond the
            user-selected tolerance band from its target weight.
            """
        )


def render_sidebar():
    """Render sidebar controls and return validated user selections."""
    st.sidebar.header("Portfolio")

    ticker_text = st.sidebar.text_input(
        "Ticker symbols",
        value="AAPL,MSFT,SPY,GLD",
        help="Enter comma-separated stock or ETF symbols.",
    )
    tickers = parse_tickers(ticker_text)

    default_weights = "30,30,25,15"
    weight_text = st.sidebar.text_area(
        "Target weights (%)",
        value=default_weights,
        help=(
            "Enter one weight per asset in the same order as the tickers. "
            "Examples: 30,30,25,15 or AAPL 30% on separate lines."
        ),
    )

    try:
        raw_weights = parse_weight_input(weight_text)
    except ValueError:
        raw_weights = []
        st.sidebar.error("Weights must be numeric percentages.")

    validation_errors = validate_portfolio_inputs(tickers, raw_weights)

    if tickers and raw_weights and len(tickers) == len(raw_weights):
        preview_frame = pd.DataFrame(
            {
                "Ticker": tickers,
                "Target Weight": [f"{weight:.2f}%" for weight in raw_weights],
            }
        )
        st.sidebar.dataframe(preview_frame, hide_index=True, width="stretch")

    st.sidebar.header("Time Period")
    selected_period = st.sidebar.selectbox(
        "Preset crisis periods",
        list(CRISIS_PERIODS.keys()),
        index=4,
    )

    preset_start, preset_end = CRISIS_PERIODS[selected_period]

    if selected_period == "Custom Range":
        start_date = st.sidebar.date_input("Custom start date", value=preset_start)
        end_date = st.sidebar.date_input("Custom end date", value=preset_end)
    else:
        start_date = preset_start
        end_date = preset_end
        st.sidebar.caption(
            f"Selected range: {start_date:%b %d, %Y} to {end_date:%b %d, %Y}"
        )

    if end_date <= start_date:
        validation_errors.append("End date must be after start date.")

    st.sidebar.header("Rebalancing Settings")
    calendar_frequency = st.sidebar.selectbox(
        "Calendar rebalancing frequency",
        ["Monthly", "Quarterly", "Yearly"],
        index=1,
    )
    tolerance_percentage = st.sidebar.slider(
        "Tolerance band",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Rebalance when any asset drifts this many percentage points from target.",
    )

    sidebar_inputs = {
        "tickers": tickers,
        "weights": np.array(raw_weights, dtype=float) / 100.0
        if raw_weights
        else np.array([]),
        "start_date": start_date,
        "end_date": end_date,
        "calendar_frequency": calendar_frequency,
        "tolerance_band": tolerance_percentage / 100.0,
        "validation_errors": validation_errors,
    }
    return sidebar_inputs


def render_kpi_cards(metrics_by_strategy):
    """Render KPI cards for the three strategies."""
    st.subheader("Strategy Snapshot")
    columns = st.columns(3)

    for column, (strategy_name, metrics) in zip(
        columns,
        metrics_by_strategy.items(),
    ):
        card_html = f"""
        <div class="kpi-card">
            <div class="kpi-title">{strategy_name}</div>
            <div class="kpi-value">{format_currency(metrics["final_value"])}</div>
            <div class="kpi-subtext">
                <div class="kpi-grid">
                    <div>
                        <span class="kpi-label">Total Return</span>
                        <span class="kpi-metric">
                            {format_percent(metrics["total_return"])}
                        </span>
                    </div>
                    <div>
                        <span class="kpi-label">Annual Return</span>
                        <span class="kpi-metric">
                            {format_percent(metrics["annual_return"])}
                        </span>
                    </div>
                    <div>
                        <span class="kpi-label">Max Drawdown</span>
                        <span class="kpi-metric">
                            {format_percent(metrics["max_drawdown"])}
                        </span>
                    </div>
                    <div>
                        <span class="kpi-label">Rebalances</span>
                        <span class="kpi-metric">
                            {int(metrics["rebalance_count"])}
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """
        column.markdown(card_html, unsafe_allow_html=True)


def render_allocation_snapshot_table(simulation_results):
    """Render the latest portfolio allocation table for every strategy."""
    st.subheader("Latest Portfolio Allocation")
    allocation_table = create_allocation_snapshot_table(simulation_results)
    st.dataframe(allocation_table, hide_index=True, width="stretch")


def render_allocation_pies(simulation_results):
    """Render final allocation pie charts."""
    st.subheader("Asset Allocation Snapshot")
    columns = st.columns(3)

    for column, (strategy_name, result) in zip(columns, simulation_results.items()):
        final_weights = result["weights"].iloc[-1]
        column.plotly_chart(
            plot_allocation_pie(final_weights, strategy_name),
            width="stretch",
        )


def render_download_section(value_history, metrics_table):
    """Render CSV download buttons for key outputs."""
    st.subheader("Download Results")
    columns = st.columns(2)

    columns[0].download_button(
        label="Download portfolio value history",
        data=value_history.to_csv(index=True).encode("utf-8"),
        file_name="portfolio_value_history.csv",
        mime="text/csv",
        width="stretch",
    )
    columns[1].download_button(
        label="Download performance metrics",
        data=metrics_table.to_csv(index=False).encode("utf-8"),
        file_name="performance_metrics.csv",
        mime="text/csv",
        width="stretch",
    )


def render_footer():
    """Render a minimal educational footer."""
    st.markdown("---")
    st.caption("Data Source: Yahoo Finance")
    st.caption(
        "This simulator is for educational purposes only. Transaction costs, "
        "taxes, and slippage are ignored."
    )


# =============================================================================
# 8. Main Execution
# =============================================================================


def main():
    """Run the Streamlit application."""
    style_application()
    render_header()
    render_strategy_expander()

    sidebar_inputs = render_sidebar()
    validation_errors = sidebar_inputs["validation_errors"]

    if validation_errors:
        for error in validation_errors:
            st.error(error)
        st.stop()

    tickers = sidebar_inputs["tickers"]
    target_weights = pd.Series(sidebar_inputs["weights"], index=tickers)
    start_date = sidebar_inputs["start_date"]
    end_date = sidebar_inputs["end_date"]
    frequency_rule = get_frequency_rule(sidebar_inputs["calendar_frequency"])
    tolerance_band = sidebar_inputs["tolerance_band"]

    with st.spinner("Downloading historical price data..."):
        prices, invalid_tickers, data_error = fetch_price_data(
            tickers,
            start_date,
            end_date,
        )

    if data_error:
        st.error(f"Unable to fetch price data: {data_error}")
        st.stop()

    if invalid_tickers:
        st.error(
            "The following tickers could not be validated or had no usable "
            f"price data: {', '.join(invalid_tickers)}"
        )
        st.stop()

    if len(prices) < 2:
        st.error("The selected period does not contain enough trading days.")
        st.stop()

    st.caption(
        f"Using {len(prices):,} trading days from "
        f"{prices.index.min():%b %d, %Y} to {prices.index.max():%b %d, %Y}."
    )

    simulation_results = run_all_simulations(
        prices,
        target_weights,
        frequency_rule,
        tolerance_band,
    )
    metrics_by_strategy = calculate_all_metrics(simulation_results)
    metrics_table = create_metric_table(metrics_by_strategy)
    value_history = pd.DataFrame(
        {
            strategy_name: result["values"]
            for strategy_name, result in simulation_results.items()
        }
    )

    render_kpi_cards(metrics_by_strategy)

    st.subheader("Performance Comparison")
    st.dataframe(metrics_table, hide_index=True, width="stretch")

    st.subheader("Risk vs Return")
    st.plotly_chart(plot_risk_return(metrics_by_strategy), width="stretch")

    st.subheader("Equity Curve")
    st.plotly_chart(plot_equity_curve(simulation_results), width="stretch")

    st.subheader("Portfolio Weight Drift")
    st.plotly_chart(
        plot_weight_drift_comparison(simulation_results),
        width="stretch",
    )
    render_allocation_snapshot_table(simulation_results)

    render_allocation_pies(simulation_results)

    st.subheader("Drawdown Chart")
    st.plotly_chart(plot_drawdown(simulation_results), width="stretch")

    st.subheader("Market Regime Analysis")
    if "SPY" in prices.columns:
        benchmark_prices = prices["SPY"]
    else:
        with st.spinner("Downloading SPY benchmark data for regime analysis..."):
            spy_prices, spy_invalid, spy_error = fetch_price_data(
                ["SPY"],
                start_date,
                end_date,
            )

        if spy_error or spy_invalid or spy_prices.empty:
            benchmark_prices = pd.Series(dtype=float)
        else:
            benchmark_prices = spy_prices["SPY"]

    regime_table = analyze_market_regimes(benchmark_prices, simulation_results)
    if regime_table.empty:
        st.warning(
            "Market regime analysis requires at least 200 trading days of SPY data."
        )
    else:
        st.dataframe(regime_table, hide_index=True, width="stretch")
        st.plotly_chart(
            plot_regime_background(benchmark_prices),
            width="stretch",
        )

    render_download_section(value_history, metrics_table)
    render_footer()


if __name__ == "__main__":
    main()
