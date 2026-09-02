import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="F1 Data Analysis Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏎️ Formula 1 Data Analysis Dashboard")
st.markdown("### Comprehensive Analysis of F1 Seasons 2024-2026")
st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("📊 Navigation")

analysis_type = st.sidebar.selectbox(
    "Select Analysis",
    [
        "🏆 Championship Overview",
        "📈 Driver Performance",
        "🏎️ Constructor Analysis",
        "🔧 Reliability & DNFs",
        "🏁 Race Analysis",
        "📊 Points Distribution",
        "🌍 Track Analysis",
        "💡 Key Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "**Data Source:**\n"
    "F1 Dataset 2024-2026\n\n"
    f"**Report Generated:**\n"
    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
)


# ---------------------------------------------------------
# LOAD / GENERATE DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():

    np.random.seed(42)

    drivers = [
        "Verstappen",
        "Hamilton",
        "Leclerc",
        "Norris",
        "Piastri",
        "Sainz",
        "Perez",
        "Russell",
        "Alonso",
        "Stroll",
        "Gasly",
        "Ocon",
        "Albon",
        "Zhou",
        "Bottas"
    ]

    constructors = [
        "Red Bull",
        "Ferrari",
        "Mercedes",
        "McLaren",
        "Aston Martin",
        "Alpine",
        "Williams",
        "AlphaTauri",
        "Haas",
        "Alfa Romeo"
    ]

    circuits = [
        "Silverstone",
        "Monza",
        "Monaco",
        "Suzuka",
        "Singapore",
        "Melbourne",
        "Montreal",
        "Interlagos",
        "Austin",
        "Bahrain",
        "Jeddah",
        "Miami",
        "Imola",
        "Baku",
        "Spa",
        "Zandvoort",
        "Mexico City",
        "Las Vegas",
        "Abu Dhabi",
        "Losail"
    ]

    seasons = [2024, 2025, 2026]

    n_races = 22

    # IMPORTANT:
    # We only have 15 drivers, therefore we cannot sample 20
    # unique drivers with replace=False.
    n_entries = min(20, len(drivers))

    # -----------------------------------------------------
    # RACE RESULTS
    # -----------------------------------------------------

    data = []

    for season in seasons:

        for race in range(1, n_races + 1):

            # Select unique drivers
            race_drivers = np.random.choice(
                drivers,
                n_entries,
                replace=False
            )

            # Select constructors
            race_constructors = np.random.choice(
                constructors,
                n_entries,
                replace=True
            )

            # One circuit per race
            race_circuit = np.random.choice(circuits)

            # Create unique finishing positions
            race_positions = np.random.permutation(
                np.arange(1, n_entries + 1)
            )

            # Create grid positions
            grid_positions = np.random.permutation(
                np.arange(1, n_entries + 1)
            )

            # Create fastest lap ranks
            fastest_lap_ranks = np.random.permutation(
                np.arange(1, n_entries + 1)
            )

            for i, driver in enumerate(race_drivers):

                position = int(race_positions[i])
                grid_pos = int(grid_positions[i])
                fastest_lap_rank = int(fastest_lap_ranks[i])

                # F1 points system
                points_system = [
                    25, 18, 15, 12, 10,
                    8, 6, 4, 2, 1
                ]

                if position <= len(points_system):
                    points = points_system[position - 1]
                else:
                    points = 0

                # DNF probability
                if np.random.random() > 0.12:
                    status = "Finished"
                else:
                    status = np.random.choice(
                        [
                            "Accident",
                            "Engine",
                            "Gearbox"
                        ]
                    )

                data.append({
                    "season": season,
                    "round": race,
                    "circuit": race_circuit,
                    "driver": driver,
                    "constructor": race_constructors[i],
                    "position": position,
                    "grid_pos": grid_pos,
                    "points": points,
                    "fastest_lap_rank": fastest_lap_rank,
                    "status": status
                })

    race_results = pd.DataFrame(data)

    # -----------------------------------------------------
    # PIT STOP DATA
    # -----------------------------------------------------

    pit_data = []

    for season in seasons:

        for driver in drivers[:10]:

            for _ in range(np.random.randint(2, 4)):

                pit_data.append({
                    "season": season,
                    "driver": driver,
                    "constructor": np.random.choice(constructors),
                    "lap": np.random.randint(10, 50),
                    "duration_s": np.random.uniform(2.0, 4.0)
                })

    pit_stops = pd.DataFrame(pit_data)

    # -----------------------------------------------------
    # DRIVER STANDINGS
    # -----------------------------------------------------

    driver_standings = (
        race_results
        .groupby(["season", "driver"])
        .agg(
            points=("points", "sum"),
            position=("position", "mean"),
            constructor=("constructor", "first")
        )
        .reset_index()
    )

    # Correct wins calculation
    driver_wins = (
        race_results[race_results["position"] == 1]
        .groupby(["season", "driver"])
        .size()
        .reset_index(name="wins")
    )

    driver_standings = driver_standings.merge(
        driver_wins,
        on=["season", "driver"],
        how="left"
    )

    driver_standings["wins"] = (
        driver_standings["wins"]
        .fillna(0)
        .astype(int)
    )

    # Championship ranking
    driver_standings["position_rank"] = (
        driver_standings
        .groupby("season")["points"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    # -----------------------------------------------------
    # CONSTRUCTOR STANDINGS
    # -----------------------------------------------------

    constructor_standings = (
        race_results
        .groupby(["season", "constructor"])
        .agg(
            points=("points", "sum"),
            position=("position", "mean")
        )
        .reset_index()
    )

    constructor_wins = (
        race_results[race_results["position"] == 1]
        .groupby(["season", "constructor"])
        .size()
        .reset_index(name="wins")
    )

    constructor_standings = constructor_standings.merge(
        constructor_wins,
        on=["season", "constructor"],
        how="left"
    )

    constructor_standings["wins"] = (
        constructor_standings["wins"]
        .fillna(0)
        .astype(int)
    )

    return (
        race_results,
        pit_stops,
        driver_standings,
        constructor_standings,
        circuits
    )


# Load data
(
    race_results,
    pit_stops,
    driver_standings,
    constructor_standings,
    circuits
) = load_data()


# ---------------------------------------------------------
# CHAMPIONSHIP OVERVIEW
# ---------------------------------------------------------
def championship_overview():

    st.header("🏆 Championship Overview")

    col1, col2 = st.columns(2)

    # Driver champions
    with col1:

        st.subheader("Driver Champions")

        champions = driver_standings[
            driver_standings["position_rank"] == 1
        ]

        for _, row in champions.iterrows():

            st.metric(
                label=f"{int(row['season'])} Season",
                value=row["driver"],
                delta=(
                    f"{int(row['points'])} points, "
                    f"{int(row['wins'])} wins"
                )
            )

    # Constructor champions
    with col2:

        st.subheader("Constructor Champions")

        idx = (
            constructor_standings
            .groupby("season")["points"]
            .idxmax()
        )

        const_champs = constructor_standings.loc[idx]

        for _, row in const_champs.iterrows():

            st.metric(
                label=f"{int(row['season'])} Season",
                value=row["constructor"],
                delta=f"{int(row['points'])} points"
            )


# ---------------------------------------------------------
# DRIVER PERFORMANCE
# ---------------------------------------------------------
def driver_performance():

    st.header("📈 Driver Performance Analysis")

    season = st.selectbox(
        "Select Season",
        [2024, 2025, 2026]
    )

    season_data = driver_standings[
        driver_standings["season"] == season
    ]

    top_drivers = season_data.nlargest(
        10,
        "points"
    )

    col1, col2 = st.columns(2)

    # Chart
    with col1:

        st.subheader("Top 10 Drivers by Points")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            top_drivers["driver"],
            top_drivers["points"]
        )

        ax.set_xlabel("Points")
        ax.set_title(
            f"{season} Season - Driver Standings"
        )

        ax.invert_yaxis()
        ax.grid(
            True,
            alpha=0.3,
            axis="x"
        )

        st.pyplot(fig)

    # Metrics
    with col2:

        st.subheader("Performance Metrics")

        selected_driver = st.selectbox(
            "Select Driver",
            top_drivers["driver"].tolist()
        )

        # IMPORTANT:
        # Filter by driver AND selected season
        driver_data = race_results[
            (race_results["driver"] == selected_driver) &
            (race_results["season"] == season)
        ]

        if len(driver_data) > 0:

            avg_pos = driver_data["position"].mean()

            wins = (
                driver_data["position"] == 1
            ).sum()

            podiums = (
                driver_data["position"] <= 3
            ).sum()

            points = driver_data["points"].sum()

            metrics = {
                "Average Position": f"{avg_pos:.2f}",
                "Wins": wins,
                "Podiums": podiums,
                "Total Points": points,
                "Races": len(driver_data)
            }

            for key, value in metrics.items():

                st.metric(
                    key,
                    value
                )


# ---------------------------------------------------------
# CONSTRUCTOR ANALYSIS
# ---------------------------------------------------------
def constructor_analysis():

    st.header("🏎️ Constructor Performance Analysis")

    season = st.selectbox(
        "Select Season",
        [2024, 2025, 2026],
        key="constructor_season"
    )

    season_data = constructor_standings[
        constructor_standings["season"] == season
    ]

    season_data = season_data.sort_values(
        "points",
        ascending=False
    )

    col1, col2 = st.columns(2)

    # Chart
    with col1:

        st.subheader("Constructor Standings")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            season_data["constructor"],
            season_data["points"]
        )

        ax.set_xlabel("Points")
        ax.set_title(
            f"{season} Season - Constructor Standings"
        )

        ax.invert_yaxis()
        ax.grid(
            True,
            alpha=0.3,
            axis="x"
        )

        st.pyplot(fig)

    # Metrics
    with col2:

        st.subheader("Performance Metrics")

        selected_constructor = st.selectbox(
            "Select Constructor",
            season_data["constructor"].tolist()
        )

        const_data = race_results[
            (race_results["constructor"] == selected_constructor) &
            (race_results["season"] == season)
        ]

        if len(const_data) > 0:

            avg_pos = const_data["position"].mean()

            wins = (
                const_data["position"] == 1
            ).sum()

            podiums = (
                const_data["position"] <= 3
            ).sum()

            points = const_data["points"].sum()

            metrics = {
                "Average Position": f"{avg_pos:.2f}",
                "Wins": wins,
                "Podiums": podiums,
                "Total Points": points,
                "Entries": len(const_data)
            }

            for key, value in metrics.items():

                st.metric(
                    key,
                    value
                )


# ---------------------------------------------------------
# RELIABILITY
# ---------------------------------------------------------
def reliability_analysis():

    st.header("🔧 Reliability & DNF Analysis")

    dnf_data = race_results[
        race_results["status"] != "Finished"
    ]

    col1, col2 = st.columns(2)

    # DNF Rate
    with col1:

        st.subheader("DNF Rate by Season")

        dnf_by_season = (
            race_results
            .groupby("season")
            .apply(
                lambda x:
                (x["status"] != "Finished").mean() * 100
            )
            .reset_index(
                name="dnf_rate"
            )
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            dnf_by_season["season"].astype(str),
            dnf_by_season["dnf_rate"]
        )

        ax.set_ylabel("DNF Rate (%)")
        ax.set_title("DNF Rate by Season")

        ax.grid(
            True,
            alpha=0.3,
            axis="y"
        )

        for bar, val in zip(
            bars,
            dnf_by_season["dnf_rate"]
        ):

            ax.text(
                bar.get_x() +
                bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center"
            )

        st.pyplot(fig)

    # DNF Causes
    with col2:

        st.subheader("DNF Causes")

        if len(dnf_data) > 0:

            dnf_causes = (
                dnf_data["status"]
                .value_counts()
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            ax.pie(
                dnf_causes.values,
                labels=dnf_causes.index,
                autopct="%1.1f%%"
            )

            ax.set_title(
                "DNF Causes"
            )

            st.pyplot(fig)


# ---------------------------------------------------------
# RACE ANALYSIS
# ---------------------------------------------------------
def race_analysis():

    st.header("🏁 Race Analysis")

    col1, col2 = st.columns(2)

    # Winners
    with col1:

        st.subheader("Winners by Driver")

        winners = race_results[
            race_results["position"] == 1
        ]

        winners = (
            winners["driver"]
            .value_counts()
            .head(10)
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            winners.index,
            winners.values
        )

        ax.set_xlabel("Wins")
        ax.set_title(
            "Most Race Wins (All Seasons)"
        )

        ax.invert_yaxis()

        ax.grid(
            True,
            alpha=0.3,
            axis="x"
        )

        st.pyplot(fig)

    # Pole positions
    with col2:

        st.subheader("Pole Positions")

        poles = race_results[
            race_results["grid_pos"] == 1
        ]

        poles = (
            poles["driver"]
            .value_counts()
            .head(10)
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            poles.index,
            poles.values
        )

        ax.set_xlabel("Pole Positions")
        ax.set_title(
            "Most Pole Positions (All Seasons)"
        )

        ax.invert_yaxis()

        ax.grid(
            True,
            alpha=0.3,
            axis="x"
        )

        st.pyplot(fig)


# ---------------------------------------------------------
# POINTS DISTRIBUTION
# ---------------------------------------------------------
def points_distribution():

    st.header("📊 Points Distribution Analysis")

    col1, col2 = st.columns(2)

    # Points by position
    with col1:

        st.subheader(
            "Points Distribution by Position"
        )

        points_by_pos = (
            race_results
            .groupby("position")["points"]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.bar(
            points_by_pos["position"],
            points_by_pos["points"]
        )

        ax.set_xlabel(
            "Finishing Position"
        )

        ax.set_ylabel(
            "Average Points"
        )

        ax.set_title(
            "Average Points by Finishing Position"
        )

        ax.grid(
            True,
            alpha=0.3,
            axis="y"
        )

        st.pyplot(fig)

    # Points concentration
    with col2:

        st.subheader(
            "Points Concentration"
        )

        top_drivers = (
            race_results
            .groupby("driver")["points"]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.pie(
            top_drivers.values,
            labels=top_drivers.index,
            autopct="%1.1f%%"
        )

        ax.set_title(
            "Points Distribution Among Top 10 Drivers"
        )

        st.pyplot(fig)


# ---------------------------------------------------------
# TRACK ANALYSIS
# ---------------------------------------------------------
def track_analysis():

    st.header("🌍 Track Performance Analysis")

    track_performance = (
        race_results
        .groupby("circuit")
        .agg(
            average_points=("points", "mean"),
            average_position=("position", "mean")
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    # Best circuits
    with col1:

        st.subheader(
            "Best Performing Circuits"
        )

        top_circuits = (
            track_performance
            .nlargest(
                10,
                "average_points"
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            top_circuits["circuit"],
            top_circuits["average_points"]
        )

        ax.set_xlabel(
            "Average Points"
        )

        ax.set_title(
            "Top 10 Circuits by Average Points"
        )

        ax.invert_yaxis()

        ax.grid(
            True,
            alpha=0.3,
            axis="x"
        )

        st.pyplot(fig)

    # Challenging circuits
    with col2:

        st.subheader(
            "Circuit Difficulty"
        )

        challenging = (
            track_performance
            .nlargest(
                10,
                "average_position"
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            challenging["circuit"],
            challenging["average_position"]
        )

        ax.set_xlabel(
            "Average Finishing Position"
        )

        ax.set_title(
            "10 Most Challenging Circuits"
        )

        ax.invert_yaxis()

        ax.grid(
            True,
            alpha=0.3,
            axis="x"
        )

        st.pyplot(fig)


# ---------------------------------------------------------
# KEY INSIGHTS
# ---------------------------------------------------------
def key_insights():

    st.header("💡 Key Insights")

    # Correct total race count
    total_races = (
        race_results[
            ["season", "round"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    total_drivers = (
        race_results["driver"]
        .nunique()
    )

    total_constructors = (
        race_results["constructor"]
        .nunique()
    )

    total_points = (
        race_results["points"]
        .sum()
    )

    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Races",
            total_races
        )

    with col2:
        st.metric(
            "Total Drivers",
            total_drivers
        )

    with col3:
        st.metric(
            "Total Constructors",
            total_constructors
        )

    with col4:
        st.metric(
            "Total Points",
            f"{total_points:.0f}"
        )

    st.markdown("---")

    # -----------------------------------------------------
    # TOP PERFORMERS
    # -----------------------------------------------------

    winner_data = race_results[
        race_results["position"] == 1
    ]

    most_wins_driver = (
        winner_data["driver"]
        .value_counts()
        .idxmax()
    )

    most_wins_count = (
        winner_data["driver"]
        .value_counts()
        .max()
    )

    most_wins_const = (
        winner_data["constructor"]
        .value_counts()
        .idxmax()
    )

    most_wins_const_count = (
        winner_data["constructor"]
        .value_counts()
        .max()
    )

    consistency = (
        race_results
        .groupby("driver")["position"]
        .std()
        .sort_values()
    )

    most_consistent = (
        consistency.index[0]
        if len(consistency) > 0
        else "N/A"
    )

    avg_position = (
        race_results
        .groupby("driver")["position"]
        .mean()
        .sort_values()
    )

    best_avg_driver = (
        avg_position.index[0]
        if len(avg_position) > 0
        else "N/A"
    )

    best_avg_value = (
        avg_position.iloc[0]
        if len(avg_position) > 0
        else 0
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🏆 Top Performers")

        st.write(
            f"**Most Wins:** "
            f"{most_wins_driver} "
            f"({most_wins_count} wins)"
        )

        st.write(
            f"**Most Wins (Constructor):** "
            f"{most_wins_const} "
            f"({most_wins_const_count} wins)"
        )

        st.write(
            f"**Best Average Position:** "
            f"{best_avg_driver} "
            f"({best_avg_value:.2f})"
        )

        st.write(
            f"**Most Consistent:** "
            f"{most_consistent}"
        )

    # -----------------------------------------------------
    # SEASON STATISTICS
    # -----------------------------------------------------

    with col2:

        st.subheader("📊 Season Statistics")

        for season in [2024, 2025, 2026]:

            season_data = race_results[
                race_results["season"] == season
            ]

            season_points = (
                season_data["points"]
                .sum()
            )

            season_races = (
                season_data["round"]
                .nunique()
            )

            season_drivers = (
                season_data["driver"]
                .nunique()
            )

            st.write(
                f"**{season} Season:**"
            )

            st.write(
                f"• {season_races} races, "
                f"{season_drivers} drivers"
            )

            st.write(
                f"• {season_points:.0f} total points"
            )

            st.write(
                f"• "
                f"Avg {season_points / season_races:.1f} "
                f"points per race"
            )

    # -----------------------------------------------------
    # RELIABILITY
    # -----------------------------------------------------

    dnf_rate = (
        (
            race_results["status"] != "Finished"
        ).mean()
        * 100
    )

    st.info(
        f"🔧 Overall Reliability: "
        f"{100 - dnf_rate:.1f}% finish rate"
    )

    # -----------------------------------------------------
    # MOST COMPETITIVE SEASON
    # -----------------------------------------------------

    season_std = (
        race_results
        .groupby("season")["position"]
        .std()
    )

    most_competitive = (
        season_std.idxmin()
    )

    st.success(
        f"🏁 Most Competitive Season: "
        f"{most_competitive}"
    )


# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------

if analysis_type == "🏆 Championship Overview":
    championship_overview()

elif analysis_type == "📈 Driver Performance":
    driver_performance()

elif analysis_type == "🏎️ Constructor Analysis":
    constructor_analysis()

elif analysis_type == "🔧 Reliability & DNFs":
    reliability_analysis()

elif analysis_type == "🏁 Race Analysis":
    race_analysis()

elif analysis_type == "📊 Points Distribution":
    points_distribution()

elif analysis_type == "🌍 Track Analysis":
    track_analysis()

elif analysis_type == "💡 Key Insights":
    key_insights()


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🏎️ Formula 1 Data Analysis Dashboard | Built with Streamlit</p>
        <p>Data: 2024-2026 Seasons | Analysis includes race results,
        driver standings, and constructor performance</p>
    </div>
    """,
    unsafe_allow_html=True
)