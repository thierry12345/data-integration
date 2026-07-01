import io
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Data Quality Dashboard",
    layout="wide",
)

DEFAULT_FILES = {
    "characteristics": "caract.csv",
    "places": "lieux.csv",
    "vehicles": "vehicules.csv",
    "users": "usagers.csv",
}

# Mapping exact des clés primaires par fichier selon vos spécifications
PRIMARY_KEYS = {
    "characteristics": "Num_Acc",
    "vehicles": "id_vehicule",
    "users": "id_usager",
    "places": None  # Indique qu'il n'y a pas de PK unique sur une seule colonne
}

CONDITIONAL_NA_COLUMNS = {
    ("users", "locp"): "only relevant if catu = 3 (pedestrian)",
    ("users", "actp"): "only relevant if catu = 3 (pedestrian)",
    ("users", "etatp"): "only relevant if catu = 3 (pedestrian)",
    ("users", "secu2"): "secondary safety equipment (optional)",
    ("users", "secu3"): "tertiary safety equipment (optional)",
    ("vehicles", "occutc"): "only relevant for public transport vehicles",
}

EXPECTED_DOMAINS = {
    "characteristics": {"lum": set(range(1, 6)), "agg": {1, 2}},
    "users": {"catu": {1, 2, 3}, "grav": {1, 2, 3, 4}, "sexe": {-1, 1, 2}},
}

@st.cache_data(show_spinner="Loading regulatory datasets...")
def load_csv(file_or_path) -> pd.DataFrame:
    return pd.read_csv(file_or_path, sep=";", low_memory=False)

def verify_and_load_data() -> dict:
    dfs = {}
    missing_files = []
    for name, default_filename in DEFAULT_FILES.items():
        local_path = Path(__file__).parent / default_filename
        if not local_path.exists():
            missing_files.append(default_filename)
        else:
            dfs[name] = load_csv(local_path)
            
    if missing_files:
        st.error(
            f"**Critical Error: Initialization failed.**\n\n"
            f"The following source files were not found in the root directory: "
            f"`{', '.join(missing_files)}`.\n\nPlease add them to activate the dashboard."
        )
        st.stop()
    return dfs

def create_gauge(title, value, color="#19d3a2"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 70], 'color': '#ffccd5'},
                {'range': [70, 90], 'color': '#ffeaa7'},
                {'range': [90, 100], 'color': '#e3faf2'}
            ],
        }
    ))
    fig.update_layout(height=160, margin=dict(l=15, r=15, t=35, b=15))
    return fig

def missing_metrics(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in df.columns:
        col = df[c]
        n_nan = int(col.isna().sum())
        if col.dtype == object or str(col.dtype).startswith("str"):
            s = col.astype(str).str.strip()
            n_neg1 = int((s == "-1").sum())
            n_na_str = int((s.str.upper() == "N/A").sum())
            n_empty = int((s == "").sum())
        else:
            n_neg1 = int((col == -1).sum())
            n_na_str = 0
            n_empty = 0
        total_missing = n_nan + n_neg1 + n_na_str + n_empty
        conditional = CONDITIONAL_NA_COLUMNS.get((table_name, c))
        rows.append({
            "table": table_name,
            "column": c,
            "total_missing": total_missing,
            "pct_missing": round(100 * total_missing / n, 2) if n else 0,
            "nature": "Structural (Not applicable)" if conditional else "True Gap (Missing)",
        })
    return pd.DataFrame(rows)

def duplicate_and_cardinality_report(dfs: dict) -> pd.DataFrame:
    rows = []
    for name, df in dfs.items():
        n_full_dup = int(df.duplicated().sum())
        if "Num_Acc" in df.columns:
            n_acc = df["Num_Acc"].nunique()
            n_key_dup = int((df["Num_Acc"].value_counts() > 1).sum())
        else:
            n_acc, n_key_dup = 0, 0
        rows.append({
            "Table": name,
            "Total Rows": len(df),
            "Unique Accidents (Num_Acc)": n_acc,
            "Multi-row Accidents": n_key_dup,
            "Strict Duplicates": n_full_dup,
        })
    return pd.DataFrame(rows)

def primary_key_uniqueness_report(dfs: dict) -> tuple:
    """Génère l'audit des clés primaires spécifiques et isole les doublons de PK."""
    rows = []
    pk_duplicates_dict = {}
    
    for name, df in dfs.items():
        pk_col = PRIMARY_KEYS.get(name)
        
        if pk_col and pk_col in df.columns:
            # Conversion propre pour éviter les problèmes d'espaces insécables (ex: id_usager avec espaces)
            series_pk = df[pk_col].astype(str).str.replace('\xa0', '').str.strip()
            
            total_keys = len(series_pk)
            unique_keys = series_pk.nunique()
            duplicate_keys_count = total_keys - unique_keys
            pct_unique = round(100 * (unique_keys / total_keys), 2) if total_keys else 100.0
            
            if duplicate_keys_count > 0:
                status = "PRIMARY KEY VIOLATION (Duplicates Found)"
                # Extraction des lignes ayant une PK dupliquée
                duplicate_ids = series_pk[series_pk.duplicated()].unique()
                pk_duplicates_dict[name] = df[series_pk.isin(duplicate_ids)].sort_values(by=pk_col)
            else:
                status = "Valid Uniqueness (Clean PK)"
        elif pk_col is None:
            total_keys, unique_keys, duplicate_keys_count, pct_unique = len(df), 0, 0, 0.0
            status = "ℹ️ No Single PK Defined (Composite/Relation Table)"
        else:
            total_keys, unique_keys, duplicate_keys_count, pct_unique = 0, 0, 0, 0.0
            status = f"Configured PK Column '{pk_col}' Not Found"
            
        rows.append({
            "Table Name": name,
            "Assigned Primary Key": pk_col if pk_col else "None",
            "Total Records": total_keys,
            "Distinct Keys": unique_keys,
            "Duplicate PK Count": duplicate_keys_count,
            "Uniqueness Rate (%)": pct_unique,
            "PK Integrity Status": status
        })
    return pd.DataFrame(rows), pk_duplicates_dict

def referential_integrity_check(dfs: dict, all_dfs: dict) -> pd.DataFrame:
    if "characteristics" not in all_dfs:
        return pd.DataFrame()
    base_ids = set(all_dfs["characteristics"]["Num_Acc"])
    rows = []
    for name, df in dfs.items():
        if name == "characteristics" or "Num_Acc" not in df.columns:
            continue
        current_ids = set(df["Num_Acc"])
        orphans = len(current_ids - base_ids)
        missing_in_child = len(base_ids - current_ids)
        rows.append({
            "Table": name,
            "Orphan Accidents (Invalid keys)": orphans,
            "Accidents missing from this table": missing_in_child,
            "Status": "Compliant" if not orphans else "Referential Anomaly",
        })
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# Pipeline Initialization & Sidebar
# --------------------------------------------------------------------------- #
all_raw_data = verify_and_load_data()

st.sidebar.title("🔍 Audit Scope Configuration")
options_scope = ["All Tables"] + list(all_raw_data.keys())
selected_scope = st.sidebar.selectbox(
    "Select Target DataFrame Filter:",
    options=options_scope,
    index=0
)

if selected_scope == "All Tables":
    dfs = all_raw_data
else:
    dfs = {selected_scope: all_raw_data[selected_scope]}

st.sidebar.markdown("---")
st.sidebar.subheader("Active Scope Inventory")
for name, df in dfs.items():
    st.sidebar.caption(f"📊 **{name}**: {len(df):,} records")

# --------------------------------------------------------------------------- #
# GLOBAL CALCULATIONS FOR GAUGES
# --------------------------------------------------------------------------- #
# 1. COMPLETENESS SCORE
rows_missing = []
for name, df in dfs.items():
    for c in df.columns:
        n_nan = df[c].isna().sum()
        n_neg1 = ((df[c] == -1) | (df[c] == "-1")).sum() if df[c].dtype in [object, int, float] else 0
        if not CONDITIONAL_NA_COLUMNS.get((name, c)):
            rows_missing.append((n_nan + n_neg1, len(df)))
total_missing_cells = sum(r[0] for r in rows_missing)
total_cells = sum(r[1] for r in rows_missing)
completeness_score = round(100 * (1 - (total_missing_cells / total_cells)), 1) if total_cells else 100.0

# 2. UNIQUENESS SCORE
total_rows = sum(len(df) for df in dfs.values())
total_duplicates = sum(df.duplicated().sum() for df in dfs.values())
uniqueness_score = round(100 * (1 - (total_duplicates / total_rows)), 1) if total_rows else 100.0

# 3. ACCURACY SCORE (Geographic including Saint-Martin & Saint-Pierre)
accuracy_score = 100.0
out_of_bounds_df = pd.DataFrame()

if "characteristics" in dfs:
    caract_df = dfs["characteristics"].copy()
    caract_df["lat_f"] = caract_df["lat"].astype(str).str.replace(",", ".").astype(float)
    caract_df["long_f"] = caract_df["long"].astype(str).str.replace(",", ".").astype(float)
    
    is_zero = (caract_df["lat_f"] == 0) & (caract_df["long_f"] == 0)
    
    is_metropole = (caract_df["lat_f"].between(41.0, 51.5)) & (caract_df["long_f"].between(-5.0, 10.0))
    is_antilles = (caract_df["lat_f"].between(14.0, 18.5)) & (caract_df["long_f"].between(-63.5, -60.0))
    is_guyane = (caract_df["lat_f"].between(2.0, 6.0)) & (caract_df["long_f"].between(-55.0, -51.0))
    is_reunion_mayotte = (caract_df["lat_f"].between(-22.5, -12.0)) & (caract_df["long_f"].between(43.0, 56.5))
    is_polynesie = (caract_df["lat_f"].between(-28.0, -7.0)) & (caract_df["long_f"].between(-155.0, -134.0))
    is_caledonie_wallis = (caract_df["lat_f"].between(-23.0, -13.0)) & (caract_df["long_f"].between(157.0, 179.0))
    is_saint_pierre = (caract_df["lat_f"].between(46.7, 47.1)) & (caract_df["long_f"].between(-56.6, -56.1))
    
    is_valid_geo = (is_metropole | is_antilles | is_guyane | is_reunion_mayotte | is_polynesie | is_caledonie_wallis | is_saint_pierre) & ~is_zero
    
    accuracy_score = round(100 * (is_valid_geo.sum() / len(caract_df)), 1) if len(caract_df) else 100.0
    out_of_bounds_df = caract_df[~is_valid_geo][["Num_Acc", "dep", "com", "adr", "lat", "long"]]

# 4. VALIDITY SCORE
total_domain_checked = 0
total_domain_errors = 0
for table_name, columns_dict in EXPECTED_DOMAINS.items():
    if table_name in dfs:
        for col_name, expected_set in columns_dict.items():
            if col_name in dfs[table_name].columns:
                series = dfs[table_name][col_name].dropna()
                total_domain_checked += len(series)
                invalid_mask = ~series.isin(expected_set) & (series != -1) & (series != "-1")
                total_domain_errors += invalid_mask.sum()
validity_score = round(100 * (1 - (total_domain_errors / total_domain_checked)), 1) if total_domain_checked else 100.0

# 5. TIMELINESS SCORE
timeliness_score = 100.0
if "characteristics" in dfs:
    year_series = dfs["characteristics"]["an"]
    correct_year = (year_series == 2024).sum()
    timeliness_score = round(100 * (correct_year / len(year_series)), 1) if len(year_series) else 100.0

# --------------------------------------------------------------------------- #
# UI RENDERING - UNIFIED SCROLLING SINGLE PAGE (NO TABS)
# --------------------------------------------------------------------------- #
st.title("🚦 Data Quality Production Dashboard")
st.subheader("Data Quality Dimensions (SLA Framework)")

# Section 1: Gauges Grid Layout
g1, g2, g3, g4, g5 = st.columns(5)
with g1:
    st.plotly_chart(create_gauge("Accuracy (GPS Boundaries)", accuracy_score, "#19d3a2"), width="stretch")
with g2:
    st.plotly_chart(create_gauge("Validity (Domain Constraints)", validity_score, "#636efa"), width="stretch")
with g3:
    st.plotly_chart(create_gauge("Completeness (True Gaps)", completeness_score, "#ab63fa"), width="stretch")
with g4:
    st.plotly_chart(create_gauge("Uniqueness (Strict Duplicates)", uniqueness_score, "#00cc96"), width="stretch")
with g5:
    st.plotly_chart(create_gauge("Timeliness (Year Context)", timeliness_score, "#ffa15a"), width="stretch")

# SLA Metrics Computational Descriptions
with st.expander("SLA Calculation Logic Descriptions"):
    st.markdown("""
    - **Accuracy (GPS):** Measures geospatial telemetry integrity. Evaluates the % of records in `characteristics` located inside authorized bounding boxes (Mainland France, French West Indies/Saint-Martin, French Guiana, Réunion, Mayotte, French Polynesia, New Caledonia, and Saint Pierre & Miquelon) while penalizing unmasked null coordinates `(0,0)`.
    - **Validity (Domain):** Assesses conformity against structural data dictionary constraints. Verifies that values belong completely to mandated sets (e.g., `lum` inside $[1, 5]$, `agg` must be $1$ or $2$).
    - **Completeness:** Computes the percentage of fully populated non-null cells across vital dimensions, ignoring conditional columns that are structurally non-applicable.
    - **Uniqueness:** Scans and outputs the percentage of distinct tuples. Computes redundant rows matching existing records at $100\\%$ across all attributes.
    - **Timeliness:** Validates production timeline compliance. Tests if the active pipeline records strictly correspond to the historical dataset delivery millésime (`an == 2024`).
    """)

st.markdown("---")

# --------------------------------------------------------------------------- #
# SECTION 2: PRIMARY KEY UNIQUENESS AUDIT & ISOLATED DUPLICATES
# --------------------------------------------------------------------------- #
st.header("1. Primary Key Structural Uniqueness & Violation Logs")
pk_report_df, pk_duplicates_found = primary_key_uniqueness_report(dfs)

st.write("**Primary Key Consistency Registry:**")
st.dataframe(pk_report_df, width="stretch", hide_index=True)

# Affichage automatique des lignes en doublon si des PK invalides existent
if pk_duplicates_found:
    st.error("**Critical Anomalies: Duplicate Primary Key rows detected!**")
    for table_name, dups_df in pk_duplicates_found.items():
        st.write(f"**Isolated Outlier Rows inside `{table_name}` table (matching duplicated `{PRIMARY_KEYS[table_name]}`):**")
        st.dataframe(dups_df.head(200), width="stretch", hide_index=True)
else:
    st.success("Primary Key Integrity Check passed. No duplicate PK instances found within active scope.")

st.markdown("---")

# --------------------------------------------------------------------------- #
# SECTION 3: COMPLETENESS & GAPS ANALYSIS
# --------------------------------------------------------------------------- #
st.header("2. Completeness & Missing Values Distribution")
all_missing_df = pd.concat([missing_metrics(df, name) for name, df in dfs.items()], ignore_index=True)
col_c1, col_c2 = st.columns([2, 1])

with col_c1:
    if not all_missing_df[all_missing_df["pct_missing"] > 0].empty:
        fig_missing_bar = px.bar(
            all_missing_df[all_missing_df["pct_missing"] > 0].sort_values("pct_missing", ascending=False).head(15),
            x="pct_missing", y="column", color="nature", facet_col="table", orientation="h",
            title="Top 15 Variables Impacting Pipeline Completeness (%)",
            color_discrete_map={"True Gap (Missing)": "#ef553b", "Structural (Not applicable)": "#636efa"}
        )
        st.plotly_chart(fig_missing_bar, width="stretch")

with col_c2:
    critical_alerts = all_missing_df[all_missing_df["pct_missing"] > 10].sort_values("pct_missing", ascending=False)
    st.write("**⚠️ Critical Fields Missingness (>10%)**")
    if not critical_alerts.empty:
        st.dataframe(critical_alerts[["table", "column", "pct_missing", "nature"]], width="stretch", hide_index=True)
    else:
        st.success("No single column exceeds 10% missingness threshold within current scope.")

st.markdown("---")

# --------------------------------------------------------------------------- #
# SECTION 4: CARDINALITIES & REFERENTIAL INTEGRITY
# --------------------------------------------------------------------------- #
st.header("3. Cross-Table Volume Duplications & Orphans Check")
dup_metrics = duplicate_and_cardinality_report(dfs)
col_i1, col_i2 = st.columns(2)

with col_i1:
    fig_card = px.bar(dup_metrics, x="Table", y=["Total Rows", "Unique Accidents (Num_Acc)"], barmode="group", title="Table Volumes Hierarchy Comparison")
    st.plotly_chart(fig_card, width="stretch")

with col_i2:
    st.write("**📝 Table Volume Registries**")
    st.dataframe(dup_metrics, width="stretch", hide_index=True)

ref_df = referential_integrity_check(dfs, all_raw_data)
if not ref_df.empty:
    st.write("**🔗 Referential Integrity Violation Reports**")
    st.dataframe(ref_df, width="stretch", hide_index=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# SECTION 5: GEOSPATIAL & BUSINESS RULES DIAGNOSTICS
# --------------------------------------------------------------------------- #
st.header("4. Geospatial & Business Rules Diagnostics")
col_v1, col_v2 = st.columns(2)

with col_v1:
    st.write("**Isolated Critical Out-of-Bounds Logs**")
    if not out_of_bounds_df.empty:
        st.error(f"Alert: {len(out_of_bounds_df):,} records detected outside allowed national bounding boxes.")
        st.dataframe(out_of_bounds_df.head(100), width="stretch", hide_index=True)
        
        # Map Processing Block
        map_df = out_of_bounds_df.dropna(subset=["lat", "long"]).copy()
        map_df["latitude"] = pd.to_numeric(map_df["lat"].astype(str).str.replace(",", "."), errors="coerce")
        map_df["longitude"] = pd.to_numeric(map_df["long"].astype(str).str.replace(",", "."), errors="coerce")
        final_map_data = map_df.dropna(subset=["latitude", "longitude"])
        final_map_data = final_map_data[final_map_data["latitude"].between(-90.0, 90.0) & final_map_data["longitude"].between(-180.0, 180.0)]
        
        if not final_map_data.empty:
            st.map(final_map_data[["latitude", "longitude"]].astype(float))
    else:
        st.success("Compliance confirmed. All accident coordinates map inside authorized French Territories.")
        
with col_v2:
    st.write("**Categorical Schema Domain Integrity Logs**")
    anomalies_cat = []
    for table_name, columns_dict in EXPECTED_DOMAINS.items():
        if table_name in dfs:
            for col_name, expected_set in columns_dict.items():
                if col_name in dfs[table_name].columns:
                    current_uniques = set(dfs[table_name][col_name].dropna().unique())
                    out_of_domain = current_uniques - expected_set - {-1, "-1"}
                    anomalies_cat.append({
                        "Table": table_name, "Column": col_name,
                        "Detected Violations": list(out_of_domain) if out_of_domain else "None",
                        "Status": "✅ Compliant" if not out_of_domain else "Domain Breach"
                    })
    if anomalies_cat:
        st.dataframe(pd.DataFrame(anomalies_cat), width="stretch", hide_index=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# SECTION 6: CONTINUOUS OUTLIERS & EXTREMES DETECTION
# --------------------------------------------------------------------------- #
st.header("5. Statistical Outliers and Continuous Extremes Audit")
o_col1, o_col2 = st.columns(2)

with o_col1:
    st.write("** Authorized Max Speed (`vma`) Distribution Analysis**")
    if "places" in dfs:
        places_df = dfs["places"].copy()
        places_df["vma_clean"] = pd.to_numeric(places_df["vma"], errors='coerce')
        
        vma_outliers = places_df[(places_df["vma_clean"] > 130) | (places_df["vma_clean"] < 10)]
        fig_vma = px.box(places_df.dropna(subset=["vma_clean"]), y="vma_clean", title="VMA Variable Boxplot")
        st.plotly_chart(fig_vma, width="stretch")
        
        if not vma_outliers.empty:
            st.warning(f"Found {len(vma_outliers):,} records violating standard speed legal boundaries.")
            st.dataframe(vma_outliers[["Num_Acc", "catr", "voie", "vma"]].head(100), width="stretch", hide_index=True)
    else:
        st.info("Activate `places` dataset in scope selector to process speed metric outliers.")
        
with o_col2:
    st.write("**👤 Users Age Outliers Detection **")
    if "users" in dfs:
        users_df = dfs["users"].copy()
        users_df["an_nais_clean"] = pd.to_numeric(users_df["an_nais"], errors='coerce')
        users_df["age"] = 2024 - users_df["an_nais_clean"]
        
        age_outliers = users_df[(users_df["age"] > 100) | (users_df["age"] < 0)]
        fig_age = px.box(users_df.dropna(subset=["age"]), y="age", title="Recorded Users Age Boxplot")
        st.plotly_chart(fig_age, width="stretch")
        
        if not age_outliers.empty:
            st.warning(f"Found {len(age_outliers):,} user profiles with extreme or impossible age rows.")
            st.dataframe(age_outliers[["Num_Acc", "id_usager", "catu", "age"]].head(100), width="stretch", hide_index=True)
    else:
        st.info("Activate `users` dataset in scope selector to calculate users demographic outliers.")