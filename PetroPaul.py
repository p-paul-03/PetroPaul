# ============================================================
#  PetroPaul | Rock Physics & Well Log Analysis Platform
#  IIT (ISM) Dhanbad
#  Requirements: streamlit, lasio, pandas, numpy, plotly
#  Install: pip install streamlit lasio pandas numpy plotly
# ============================================================

import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import lasio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter 
import requests, base64


# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="PetroPaul",
    layout="wide",
    page_icon="🪨"
)

# ─── GLOBAL CSS ──────────────────────────────────────────────
st.markdown("""
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #f0eef8;
    }
    /* Sidebar text colors */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stMarkdown {
        color: #3a3a5c !important;
    }
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: white;
        border: 1px solid #ddd;
        border-radius: 10px;
        color: #3a3a5c;
        width: 100%;
        text-align: left;
        padding: 10px 16px;
        font-size: 0.88rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #e8e6f5;
        border-color: #9b8ec4;
        color: #1a1a3c;
    }
    /* Remove top padding */
    .block-container { padding-top: 1.2rem !important; }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid #f0f0f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-size: 0.88rem;
        font-weight: 500;
        color: #666;
    }
    .stTabs [aria-selected="true"] {
        background: #f0eef8 !important;
        color: #3a3a5c !important;
        border-bottom: 2px solid #9b8ec4 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────
CURVE_DEFAULTS = {
    "GR"  : {"color": "#2ca02c", "scale": "linear"},
    "RHOB": {"color": "#d62728", "scale": "linear"},
    "NPHI": {"color": "#1f77b4", "scale": "linear"},
    "DT"  : {"color": "#9467bd", "scale": "linear"},
    "DTS" : {"color": "#e377c2", "scale": "linear"},
}

COLOR_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
    "#e377c2", "#8c564b", "#17becf", "#bcbd22",
    "#ff7f0e", "#7f7f7f"
]

# ─── LOADERS ─────────────────────────────────────────────────
@st.cache_data
def load_las(raw_bytes: bytes):
    text = raw_bytes.decode("utf-8", errors="ignore")
    las  = lasio.read(io.StringIO(text), ignore_header_errors=True)
    df   = las.df().reset_index()
    df.columns = [c.upper().strip() for c in df.columns]
    return df, las

@st.cache_data
def load_csv(raw_bytes: bytes):
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = [c.upper().strip() for c in df.columns]
    return df, None

def detect_depth_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if any(kw in c for kw in ["DEPTH", "DEPT", "MD", "TVD"]):
            return c
    return df.columns[0]

# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:

    # ── File Upload (top — immediately visible) ───────────────
    st.markdown(
        '<div style="padding: 14px 4px 6px 4px; font-family: Georgia, serif; '
        'font-size: 0.82rem; font-weight: 600; color: #3a3a5c; '
        'letter-spacing: 0.5px;">Upload Well Log</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "LAS or CSV file",
        type=["las", "csv"],
        help="Accepts LAS 2.0 or a CSV with a header row. Max 200 MB."
    )

    st.divider()

    # ── App Branding ──────────────────────────────────────────
    st.markdown("""
        <div style="padding: 6px 4px 10px 4px; text-align: center;">
            <div style="font-family: Georgia, serif; font-size: 1.35rem;
                        font-variant: small-caps; font-weight: 700;
                        color: #3a3a5c; letter-spacing: 2px;">
                PetroPaul
            </div>
            <div style="font-size: 0.70rem; color: #9b8ec4;
                        letter-spacing: 1px; margin-top: 3px;">
                Rock Physics &middot; Well Log Analysis
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Info Expanders ────────────────────────────────────────
    with st.expander("About PetroPaul", expanded=False):
        st.markdown("""
        **PetroPaul** is an interactive well log analysis platform for
        petrophysicists and geoscientists.

        Developed at **IIT (ISM) Dhanbad** as part of applied geophysics research.
        """)

    with st.expander("Supported Formats", expanded=False):
        st.markdown("""
        | Format | Details |
        |--------|---------|
        | LAS 2.0 | Standard well log ASCII |
        | CSV | Header row required |

        Max file size: 200 MB.
        """)

    with st.expander("Contact & Credits", expanded=False):
        st.markdown("""
        **Developer:** Pratyan Paul  
        **Institution:** IIT (ISM) Dhanbad  
        **Stack:** Python · Streamlit · Plotly · LASio · NumPy · SciPy
        """)

# ─── LANDING PAGE ────────────────────────────────────────────
if uploaded is None:

    IMAGE_URL    = "https://www.petrosync.com/blog/wp-content/uploads/2024/02/well-log-interpretation.png"
    IMAGE_HEIGHT = "520px"
    IMAGE_WIDTH  = "100%"

    try:
        img_bytes = requests.get(IMAGE_URL, timeout=5).content
        img_b64   = base64.b64encode(img_bytes).decode()
        img_src   = f"data:image/jpeg;base64,{img_b64}"
    except Exception:
        img_src   = IMAGE_URL

    st.markdown(f"""<style>
        .landing-hero {{
            background: #ffffff;
            border-radius: 16px;
            border: 1px solid #e8e8e8;
            overflow: hidden;
            margin-bottom: 24px;
            box-shadow: 0 2px 16px rgba(0,0,0,0.06);
        }}
        .hero-img-wrap {{ position: relative; }}
        .hero-img-wrap img {{
            width: {IMAGE_WIDTH};
            height: {IMAGE_HEIGHT};
            object-fit: cover;
            display: block;
        }}
        .hero-overlay-text {{
            position: absolute;
            top: 0; right: 0;
            padding: 28px 36px;
            text-align: right;
        }}
        .overlay-appname {{
            font-family: 'Georgia', serif;
            font-size: 3rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 3px;
            font-variant: small-caps;
            margin-bottom: 6px;
            text-shadow: 0 1px 6px rgba(0,0,0,0.6);
        }}
        .overlay-tagline {{
            font-size: 0.85rem;
            color: rgba(255,255,255,0.75);
            letter-spacing: 1px;
            text-shadow: 0 1px 4px rgba(0,0,0,0.5);
        }}
        .overlay-inst {{
            font-size: 0.78rem;
            color: rgba(255,255,255,0.55);
            letter-spacing: 1px;
            margin-top: 3px;
            text-shadow: 0 1px 4px rgba(0,0,0,0.5);
        }}
        .workflow-row {{
            display: flex;
            border-top: 1px solid #f0f0f0;
        }}
        .wf-step {{
            flex: 1;
            text-align: center;
            padding: 20px 10px;
            border-right: 1px solid #f0f0f0;
        }}
        .wf-step:last-child {{ border-right: none; }}
        .wf-title {{
            font-size: 0.88rem;
            font-weight: 600;
            color: #444;
        }}
        .landing-footer {{
            font-size: 0.73rem;
            color: #ccc;
            text-align: center;
            padding: 12px 0 4px 0;
        }}
    </style>""", unsafe_allow_html=True)

    st.markdown(f"""
        <div class="landing-hero">
            <div class="hero-img-wrap">
                <img src="{img_src}" alt="Well Log" />
                <div class="hero-overlay-text">
                    <div class="overlay-appname">PetroPaul</div>
                    <div class="overlay-tagline">Rock Physics &amp; Well Log Analysis</div>
                    <div class="overlay-inst">IIT (ISM) Dhanbad</div>
                </div>
            </div>
            <div class="workflow-row">
                <div class="wf-step"><div class="wf-title">📤 Upload</div></div>
                <div class="wf-step"><div class="wf-title">🔍 Quality Control</div></div>
                <div class="wf-step"><div class="wf-title">📈 Multi-Track Visualization</div></div>
                <div class="wf-step"><div class="wf-title">🔵 Crossplot Analysis</div></div>
                <div class="wf-step"><div class="wf-title">⚗️ Formation Evaluation</div></div>
            </div>
        </div>
        <div class="landing-footer">
            Built with Python · Streamlit · Plotly · LASio
            &nbsp;·&nbsp; Pratyan Paul &nbsp;·&nbsp; IIT (ISM) Dhanbad
        </div>
    """, unsafe_allow_html=True)

    st.stop()   # ← stops here until file is uploaded

# ─── FILE LOADED — everything below runs only after upload ───
else:
    raw = uploaded.read()

    if uploaded.name.lower().endswith(".las"):
        df_raw, las_obj = load_las(raw)
    else:
        df_raw, las_obj = load_csv(raw)

    depth_col = detect_depth_col(df_raw)
    df_raw    = df_raw.replace(-9999.25, np.nan).copy()

    # ── Reset session state on new file ──────────────────────
    file_id = uploaded.name + str(uploaded.size)
    if st.session_state.get("current_file_id") != file_id:
        st.session_state["current_file_id"]     = file_id
        st.session_state["depth_top"]           = float(df_raw[depth_col].min())
        st.session_state["depth_bot"]           = float(df_raw[depth_col].max())
        st.session_state["depth_raw"]           = df_raw[depth_col].copy()
        st.session_state["depth_unit_label"]    = ""
        st.session_state["raw_curves_snapshot"] = df_raw.drop(columns=[depth_col]).copy()
        st.session_state.pop("depth_conv_factor", None)
        st.session_state.pop("depth_converted",   None)
        st.session_state.pop("df", None)
        st.session_state.pop("curve_settings", None)
        st.session_state.pop("track_settings", None)

    if "depth_top" not in st.session_state:
        st.session_state["depth_top"] = float(df_raw[depth_col].min())
    if "depth_bot" not in st.session_state:
        st.session_state["depth_bot"] = float(df_raw[depth_col].max())
    if "depth_raw" not in st.session_state:
        st.session_state["depth_raw"] = df_raw[depth_col].copy()

    if "depth_conv_factor" in st.session_state:
        df_raw[depth_col] = (
            st.session_state["depth_raw"] * st.session_state["depth_conv_factor"]
        ).round(4)
    if "raw_curves_snapshot" not in st.session_state:
        st.session_state["raw_curves_snapshot"] = df_raw.drop(columns=[depth_col]).copy()

    d_min = float(df_raw[depth_col].min())
    d_max = float(df_raw[depth_col].max())

    st.session_state["depth_top"] = max(d_min, min(st.session_state["depth_top"], d_max))
    st.session_state["depth_bot"] = max(d_min, min(st.session_state["depth_bot"], d_max))

    # ── SIDEBAR — depth controls only, no nav, no upload ─────
    with st.sidebar:

        unit_label = st.session_state.get("depth_unit_label", "") or "m"
        st.markdown(f"#### Depth Interval ({unit_label})")

        depth_range = st.slider(
            "Select range",
            min_value=d_min, max_value=d_max,
            value=(st.session_state["depth_top"], st.session_state["depth_bot"]),
            step=0.5
        )
        interval = abs(depth_range[1] - depth_range[0])
        st.caption(f"Interval: {interval:.1f} {unit_label}")

        st.session_state["depth_top"] = depth_range[0]
        st.session_state["depth_bot"] = depth_range[1]

        col_a, col_b = st.columns(2)
        with col_a:
            depth_top = st.number_input(
                "Top", min_value=d_min, max_value=d_max,
                value=st.session_state["depth_top"],
                step=0.5, format="%.1f"
            )
        with col_b:
            depth_bot = st.number_input(
                "Bottom", min_value=d_min, max_value=d_max,
                value=st.session_state["depth_bot"],
                step=0.5, format="%.1f"
            )

        if depth_top != st.session_state["depth_top"] or depth_bot != st.session_state["depth_bot"]:
            st.session_state["depth_top"] = depth_top
            st.session_state["depth_bot"] = depth_bot
            st.rerun()

        st.divider()
        st.markdown("#### Depth Conversion")

        UNIT_OPTIONS = ["m", "ft", "km", "cm"]
        TO_METERS    = {"m": 1.0, "ft": 0.3048, "km": 1000.0, "cm": 0.01}
        current_unit = st.session_state.get("depth_unit_label", "") or UNIT_OPTIONS[0]

        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.markdown("**From**")
            st.info(current_unit)
        with col_u2:
            cur_idx    = UNIT_OPTIONS.index(current_unit) if current_unit in UNIT_OPTIONS else 0
            to_default = (cur_idx + 1) % len(UNIT_OPTIONS)
            unit_to    = st.selectbox("To", options=UNIT_OPTIONS, index=to_default, key="unit_to")

        if st.button("Apply Conversion", key="apply_depth_conv", use_container_width=True):
            if current_unit == unit_to:
                st.info("Same unit — no conversion needed.")
            else:
                current_factor = st.session_state.get("depth_conv_factor", 1.0)
                step_factor    = TO_METERS[current_unit] / TO_METERS[unit_to]
                new_factor     = current_factor * step_factor
                st.session_state["depth_conv_factor"] = new_factor
                st.session_state["depth_unit_label"]  = unit_to
                st.session_state["depth_top"] = float(st.session_state["depth_raw"].min() * new_factor)
                st.session_state["depth_bot"] = float(st.session_state["depth_raw"].max() * new_factor)
                st.session_state.pop("df", None)
                st.success(f"Converted: {current_unit} to {unit_to}")
                st.rerun()

    # ── Depth filter ─────────────────────────────────────────
    final_top = min(st.session_state["depth_top"], st.session_state["depth_bot"])
    final_bot = max(st.session_state["depth_top"], st.session_state["depth_bot"])

    df_filtered = df_raw[
        (df_raw[depth_col] >= final_top) &
        (df_raw[depth_col] <= final_bot)
    ].reset_index(drop=True)

    # ── Sync session state ────────────────────────────────────
    if "df" in st.session_state:
        prev_df    = st.session_state["df"]
        base_cols  = list(df_raw.columns)
        extra_cols = [c for c in prev_df.columns if c not in base_cols]
        new_df     = df_filtered.copy()

        for col in base_cols:
            if col == depth_col:
                continue
            if col in prev_df.columns:
                reindexed = prev_df[col].reindex(df_filtered.index)
                raw_vals  = df_raw.loc[df_filtered.index, col]
                if not reindexed.reset_index(drop=True).equals(raw_vals.reset_index(drop=True)):
                    new_df[col] = reindexed.values

        for col in extra_cols:
            new_df[col] = prev_df[col].reindex(df_filtered.index)

        st.session_state["df"] = new_df
    else:
        st.session_state["df"] = df_filtered.copy()

    df         = st.session_state["df"]
    all_curves = [c for c in df.columns if c != depth_col]

    # ─── TABS ─────────────────────────────────────────────────
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            position: sticky;
            top: 0;
            z-index: 999;
            background: white;
            padding-top: 6px;
            gap: 4px;
            border-bottom: 2px solid #f0f0f0;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
            font-size: 0.88rem;
            font-weight: 500;
            color: #666;
        }
        .stTabs [aria-selected="true"] {
            background: #f0eef8 !important;
            color: #3a3a5c !important;
            border-bottom: 2px solid #9b8ec4 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Description & Statistics",
        "Quality Control",
        "Plot",
        "Crossplot",
        "Petrophysical Property",
        "Elastic Properties",
        "Rock Physics"
    ])


    # ════════════════════════════════════════════════════════
    #  TAB 1 — DESCRIPTION & STATISTICS
    # ════════════════════════════════════════════════════════
    with tab1:

        # ── Well Information (LAS header) ─────────────────────
        with st.expander("Well Information", expanded=False):
            if las_obj is not None:
                well_rows = []
                for item in las_obj.well:
                    val = str(item.value).strip()
                    if val and val.lower() not in ["", "none", "nan", "--"]:
                        well_rows.append({
                            "Parameter" : item.descr if item.descr.strip() else item.mnemonic,
                            "Mnemonic"  : item.mnemonic,
                            "Value"     : val,
                            "Unit"      : item.unit if item.unit.strip() else "—",
                        })
                if well_rows:
                    st.dataframe(
                        pd.DataFrame(well_rows),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No well header entries found in this LAS file.")
            else:
                st.info("Well header is only available for LAS files.")

        # ── Curve Information ─────────────────────────────────
        with st.expander("Curve Information", expanded=False):
            if las_obj is not None:
                curve_info = []
                for curve in las_obj.curves:
                    curve_info.append({
                        "Mnemonic"   : curve.mnemonic,
                        "Unit"       : curve.unit,
                        "Description": curve.descr,
                    })
                st.dataframe(
                    pd.DataFrame(curve_info),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Curve metadata is only available for LAS files.")
                st.write("Columns found:", all_curves)

        # ── Statistical Summary ───────────────────────────────
        with st.expander("Statistical Summary", expanded=False):

            all_cols    = [depth_col] + all_curves
            summary_df  = df[all_cols].describe().T.round(4)
            summary_df.index.name = "Curve"

            # NaN count and percent
            nan_count   = df[all_cols].isna().sum().rename("NaN count")
            summary_df  = pd.concat([summary_df, nan_count], axis=1)

            st.dataframe(
                summary_df.reset_index(),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "NaN %": st.column_config.ProgressColumn(
                        "NaN %",
                        help="Percentage of missing values",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    ),
                    "mean": st.column_config.NumberColumn("mean", format="%.4f"),
                    "std" : st.column_config.NumberColumn("std",  format="%.4f"),
                    "min" : st.column_config.NumberColumn("min",  format="%.4f"),
                    "max" : st.column_config.NumberColumn("max",  format="%.4f"),
                }
            )

    # ════════════════════════════════════════════════════════
    #  TAB 2 — QUALITY CONTROL
    # ════════════════════════════════════════════════════════
    with tab2:

        cur_df   = st.session_state["df"]
        all_cols = [c for c in cur_df.columns if c != depth_col]

        st.markdown("All modifications here update the working dataset.")

        # ════════════════════════════════════════════════
        #  HISTOGRAM
        # ════════════════════════════════════════════════
        with st.expander("Histogram", expanded=False):

            c1, c2 = st.columns([3, 1])
            with c1:
                hist_cols = st.multiselect(
                    "Select curves to plot",
                    options=all_cols,
                    default=[],
                    key="hist_cols",
                )
            with c2:
                show_kde = st.checkbox(
                    "Show KDE curve",
                    value=False,
                    key="hist_kde",
                    help="A Kernel Density Estimate (KDE) curve is a smoothed, "
                         "continuous representation of a dataset's distribution."
                )

            if hist_cols:
                st.markdown("**Bins per curve**")
                bin_cols     = st.columns(len(hist_cols))
                bins_per_col = {}
                for i, col in enumerate(hist_cols):
                    with bin_cols[i]:
                        bins_per_col[col] = st.number_input(
                            col,
                            min_value=5,
                            max_value=500,
                            value=50,
                            step=5,
                            key=f"hist_bins_{col}",
                        )

            st.markdown("---")

            if not hist_cols:
                st.info("Select at least one curve above to plot.")
            else:
                for row_start in range(0, len(hist_cols), 2):
                    row_cols_list = hist_cols[row_start : row_start + 2]
                    plot_cols     = st.columns(2)

                    for i, col in enumerate(row_cols_list):
                        data  = cur_df[col].dropna()
                        nbins = bins_per_col[col]

                        with plot_cols[i]:
                            if data.empty:
                                st.warning(f"{col} — all NaN, nothing to plot.")
                                continue

                            data_range = data.max() - data.min()
                            if data_range == 0:
                                st.warning(
                                    f"{col} — all values are identical "
                                    f"({data.iloc[0]:.4f}), cannot plot histogram."
                                )
                                continue

                            bin_width = data_range / nbins

                            fig = go.Figure()

                            fig.add_trace(go.Histogram(
                                x=data,
                                xbins=dict(
                                    start=data.min(),
                                    end=data.max(),
                                    size=bin_width,
                                ),
                                name="Frequency",
                                marker_color="steelblue",
                                opacity=0.75,
                                yaxis="y1",
                            ))

                            if show_kde and len(data) > 5:
                                kde     = gaussian_kde(data)
                                x_range = np.linspace(data.min(), data.max(), 300)
                                fig.add_trace(go.Scatter(
                                    x=x_range,
                                    y=kde(x_range),
                                    mode="lines",
                                    name="KDE",
                                    line=dict(color="crimson", width=2),
                                    yaxis="y2",
                                ))

                            layout_kwargs = dict(
                                title=dict(text=col, font=dict(size=13)),
                                xaxis=dict(title=col),
                                yaxis=dict(title="Frequency", side="left"),
                                height=320,
                                margin=dict(l=50, r=50, t=40, b=40),
                                legend=dict(orientation="h", y=1.08, x=0),
                                bargap=0.05,
                                showlegend=show_kde,
                            )

                            if show_kde and len(data) > 5:
                                layout_kwargs["yaxis2"] = dict(
                                    title="Density",
                                    overlaying="y",
                                    side="right",
                                    showgrid=False,
                                )

                            fig.update_layout(**layout_kwargs)
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"hist_chart_{col}"
                            )

                            st.caption(
                                f"n={len(data)}  |  "
                                f"mean={data.mean():.3f}  |  "
                                f"std={data.std():.3f}  |  "
                                f"NaN={cur_df[col].isna().sum()}"
                            )

        # ════════════════════════════════════════════════
        #  ROLLING AVERAGE (SMOOTHING)
        # ════════════════════════════════════════════════
        with st.expander("Rolling Average", expanded=False):

            c1, c2 = st.columns([3, 1])
            with c1:
                ra_cols = st.multiselect(
                    "Select curves",
                    options=all_cols,
                    default=[],
                    key="ra_cols",
                    help="Smoothed values replace the original column in the dataframe."
                )
            with c2:
                ra_method = st.selectbox(
                    "Method",
                    options=["mean", "median", "savitzky-golay"],
                    index=0,
                    key="ra_method",
                    help=(
                        "Mean: averages values in the window. "
                        "Good for general smoothing. Sensitive to spikes.\n\n"
                        "Median: takes the middle value. Better for logs "
                        "with sharp spikes or outliers.\n\n"
                        "Savitzky-Golay: fits a polynomial to each window. "
                        "Best for smoothing while preserving formation peaks and troughs. "
                        "Window size must be odd and >= 3."
                    )
                )

            window_per_col = {}
            if ra_cols:
                st.markdown("**Window size per curve**")
                win_cols = st.columns(len(ra_cols))
                for i, col in enumerate(ra_cols):
                    with win_cols[i]:
                        window_per_col[col] = st.number_input(
                            col,
                            min_value=0,
                            value=5,
                            step=1,
                            key=f"ra_win_{col}",
                            help="0 = skip this curve"
                        )

            if ra_cols:
                st.markdown("---")
                st.markdown("**Preview**")

                raw_snap_preview = st.session_state.get("raw_curves_snapshot", None)

                if raw_snap_preview is None:
                    st.warning(
                        "Raw snapshot not available — "
                        "preview shows current (possibly already smoothed) data."
                    )

                COLS_PER_ROW = 4
                chunks = [
                    ra_cols[i : i + COLS_PER_ROW]
                    for i in range(0, len(ra_cols), COLS_PER_ROW)
                ]

                for chunk in chunks:
                    row_ui = st.columns(COLS_PER_ROW)

                    for idx, col in enumerate(chunk):
                        w        = window_per_col.get(col, 5)
                        is_first = (idx == 0)

                        if raw_snap_preview is not None and col in raw_snap_preview.columns:
                            original_line = raw_snap_preview[col].reindex(cur_df.index)
                        else:
                            original_line = cur_df[col]

                        if w > 1:
                            if ra_method == "mean":
                                smoothed = original_line.rolling(
                                    window=w, center=True, min_periods=1
                                ).mean()
                            elif ra_method == "median":
                                smoothed = original_line.rolling(
                                    window=w, center=True, min_periods=1
                                ).median()
                            elif ra_method == "savitzky-golay":
                                filled  = original_line.ffill().bfill()
                                win_sg  = w if w % 2 == 1 else w + 1
                                win_sg  = max(win_sg, 3)
                                poly    = min(2, win_sg - 1)
                                smoothed = pd.Series(
                                    savgol_filter(
                                        filled.values,
                                        window_length=win_sg,
                                        polyorder=poly
                                    ),
                                    index=original_line.index
                                )
                                smoothed[original_line.isna()] = np.nan
                        else:
                            smoothed = original_line

                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=original_line,
                            y=cur_df[depth_col],
                            mode="lines",
                            name="Raw",
                            line=dict(color="steelblue", width=1, dash="dot"),
                            opacity=0.5,
                        ))
                        fig.add_trace(go.Scatter(
                            x=smoothed,
                            y=cur_df[depth_col],
                            mode="lines",
                            name=f"w={w}",
                            line=dict(color="crimson", width=1.5),
                        ))

                        fig.update_layout(
                            xaxis=dict(
                                title=dict(text=col, font=dict(size=11), standoff=8),
                                side="top",
                                tickfont=dict(size=9),
                                automargin=True,
                                showgrid=True,
                                gridcolor="lightgrey",
                            ),
                            yaxis=dict(
                                title=dict(
                                    text=depth_col if is_first else "",
                                    font=dict(size=10),
                                ),
                                autorange="reversed",
                                showticklabels=is_first,
                                tickfont=dict(size=9),
                                showgrid=True,
                                gridcolor="lightgrey",
                            ),
                            height=500,
                            margin=dict(
                                l=60 if is_first else 10,
                                r=8, t=65, b=10,
                            ),
                            legend=dict(
                                orientation="h",
                                y=1.20, x=0,
                                font=dict(size=9),
                                bgcolor="rgba(255,255,255,0.8)",
                            ),
                            showlegend=True,
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                        )

                        with row_ui[idx]:
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"ra_chart_{col}"
                            )

            st.markdown("---")

            # ── Action buttons ────────────────────────────────────
            btn1, btn2 = st.columns([1, 1])

            with btn1:
                if st.button("Apply", key="apply_ra", type="primary", use_container_width=True):
                    if not ra_cols:
                        st.warning("Select a curve first.")
                    else:
                        work_df  = st.session_state["df"].copy()
                        raw_snap = st.session_state.get("raw_curves_snapshot", None)

                        for col in ra_cols:
                            w = window_per_col.get(col, 5)
                            if w <= 1:
                                continue

                            source = raw_snap[col].reindex(work_df.index) \
                                     if (raw_snap is not None and col in raw_snap.columns) \
                                     else work_df[col]

                            if ra_method == "mean":
                                work_df[col] = source.rolling(
                                    window=w, center=True, min_periods=1
                                ).mean().round(6)
                            elif ra_method == "median":
                                work_df[col] = source.rolling(
                                    window=w, center=True, min_periods=1
                                ).median().round(6)
                            elif ra_method == "savitzky-golay":
                                filled  = source.ffill().bfill()
                                win_sg  = w if w % 2 == 1 else w + 1
                                win_sg  = max(win_sg, 3)
                                poly    = min(2, win_sg - 1)
                                sg_vals = savgol_filter(
                                    filled.values,
                                    window_length=win_sg,
                                    polyorder=poly
                                )
                                result  = pd.Series(sg_vals, index=source.index).round(6)
                                result[source.isna()] = np.nan
                                work_df[col] = result

                        for col in ra_cols:
                            if col in st.session_state.get("curve_settings", {}):
                                updated_data = work_df[col].dropna()
                                if not updated_data.empty:
                                    st.session_state["curve_settings"][col]["x_min"] = float(updated_data.min())
                                    st.session_state["curve_settings"][col]["x_max"] = float(updated_data.max())

                        st.session_state["df"] = work_df
                        st.session_state["ra_success"] = {
                            "cols"   : ra_cols,
                            "windows": window_per_col,
                            "method" : ra_method,
                        }
                        st.session_state.pop("ra_reset", None)
                        st.rerun()

            with btn2:
                if st.button("Reset to Raw", key="reset_ra", use_container_width=True):
                    if not ra_cols:
                        st.warning("Select curves to reset.")
                    else:
                        raw_snap = st.session_state.get("raw_curves_snapshot", None)
                        if raw_snap is None:
                            st.error("Raw snapshot not found — cannot reset.")
                        else:
                            work_df = st.session_state["df"].copy()
                            for col in ra_cols:
                                if col in raw_snap.columns:
                                    work_df[col] = raw_snap[col].reindex(work_df.index)

                            for col in ra_cols:
                                if col in st.session_state.get("curve_settings", {}):
                                    raw_data = raw_snap[col].dropna()
                                    if not raw_data.empty:
                                        st.session_state["curve_settings"][col]["x_min"] = float(raw_data.min())
                                        st.session_state["curve_settings"][col]["x_max"] = float(raw_data.max())

                            st.session_state["df"] = work_df
                            st.session_state["ra_reset"] = ra_cols
                            st.session_state.pop("ra_success", None)
                            st.rerun()

            # ── Status messages ───────────────────────────────────
            if st.session_state.get("ra_success"):
                r            = st.session_state["ra_success"]
                applied_info = "  |  ".join(
                    [f"{c}: w={r['windows'].get(c, '?')}" for c in r["cols"]]
                )
                st.success(f"{r['method'].capitalize()} smoothing applied")
                st.info(applied_info + "  |  Values updated in dataframe")

            if st.session_state.get("ra_reset"):
                reset_cols = ", ".join(st.session_state["ra_reset"])
                st.success(f"Reset to raw: {reset_cols}")

        # ════════════════════════════════════════════════
        #  MISSING DATA HANDLING
        # ════════════════════════════════════════════════
        with st.expander("Missing Data Handling", expanded=False):

            cur_df   = st.session_state["df"]
            all_cols = [c for c in cur_df.columns if c != depth_col]

            # Show NaN summary first
            nan_summary = cur_df[all_cols].isna().sum()
            nan_summary = nan_summary[nan_summary > 0].reset_index()
            nan_summary.columns = ["Curve", "NaN Count"]
            nan_summary["NaN %"] = (
                nan_summary["NaN Count"] / len(cur_df) * 100
            ).round(2)

            if nan_summary.empty:
                st.success("No missing values found in the current dataset.")
            else:
                st.markdown("**Curves with missing values**")
                st.dataframe(nan_summary, use_container_width=True, hide_index=True)

                st.markdown("---")

                c1, c2 = st.columns([3, 1])
                with c1:
                    nan_cols = st.multiselect(
                        "Select curves to fill",
                        options=nan_summary["Curve"].tolist(),
                        default=[],
                        key="nan_cols"
                    )
                with c2:
                    nan_method = st.selectbox(
                        "Method",
                        options=[
                            "linear interpolation",
                            "spline interpolation",
                            "forward fill",
                            "backward fill",
                            "fill with mean",
                            "fill with median",
                        ],
                        index=0,
                        key="nan_method",
                        help=(
                            "Linear: straight line between known values. Good for short gaps.\n\n"
                            "Spline: smooth curve fit. Better for gradual trends.\n\n"
                            "Forward fill: repeats last valid value downward.\n\n"
                            "Backward fill: repeats next valid value upward.\n\n"
                            "Fill with mean/median: uses interval statistics."
                        )
                    )

                # Max gap limit
                if nan_method in ["linear interpolation", "spline interpolation"]:
                    max_gap = st.number_input(
                        "Max gap to fill (samples) — 0 = fill all gaps",
                        min_value=0,
                        value=10,
                        step=1,
                        key="nan_max_gap",
                        help="Gaps larger than this will not be filled. "
                             "Prevents interpolating across long unreliable intervals."
                    )
                else:
                    max_gap = 0

                st.markdown("---")

                btn_a, btn_b = st.columns([1, 1])

                with btn_a:
                    if st.button("Apply", key="apply_nan", type="primary", use_container_width=True):
                        if not nan_cols:
                            st.warning("Select at least one curve.")
                        else:
                            work_df = st.session_state["df"].copy()

                            for col in nan_cols:
                                s = work_df[col].copy()
                                lim = max_gap if max_gap > 0 else None

                                if nan_method == "linear interpolation":
                                    work_df[col] = s.interpolate(
                                        method="linear", limit=lim,
                                        limit_direction="both"
                                    )
                                elif nan_method == "spline interpolation":
                                    work_df[col] = s.interpolate(
                                        method="spline", order=3, limit=lim,
                                        limit_direction="both"
                                    )
                                elif nan_method == "forward fill":
                                    work_df[col] = s.ffill()
                                elif nan_method == "backward fill":
                                    work_df[col] = s.bfill()
                                elif nan_method == "fill with mean":
                                    work_df[col] = s.fillna(s.mean())
                                elif nan_method == "fill with median":
                                    work_df[col] = s.fillna(s.median())

                            st.session_state["df"] = work_df
                            st.session_state["nan_success"] = {
                                "cols"  : nan_cols,
                                "method": nan_method,
                            }
                            st.rerun()

                with btn_b:
                    if st.button("Reset to Raw", key="reset_nan", use_container_width=True):
                        if not nan_cols:
                            st.warning("Select curves to reset.")
                        else:
                            raw_snap = st.session_state.get("raw_curves_snapshot", None)
                            if raw_snap is None:
                                st.error("Raw snapshot not found — cannot reset.")
                            else:
                                work_df = st.session_state["df"].copy()
                                for col in nan_cols:
                                    if col in raw_snap.columns:
                                        work_df[col] = raw_snap[col].reindex(work_df.index)
                                st.session_state["df"] = work_df
                                st.session_state.pop("nan_success", None)
                                st.rerun()

                if st.session_state.get("nan_success"):
                    r = st.session_state["nan_success"]
                    st.success(
                        f"{r['method'].capitalize()} applied to: "
                        f"{', '.join(r['cols'])}"
                    )


    # ════════════════════════════════════════════════════════
    #  TAB 3 — PLOT  (professional rewrite)
    # ════════════════════════════════════════════════════════
    with tab3:

        # ── Inject compact professional CSS ──────────────────
        st.markdown("""
        <style>
        .track-badge {
            display: inline-block;
            background: #f0f4ff;
            border: 1px solid #d0d8f0;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 13px;
            font-weight: 600;
            color: #2c3e6b;
            margin-bottom: 6px;
            letter-spacing: 0.3px;
        }
        .curve-chip {
            display: inline-block;
            background: #e8f0fe;
            border-radius: 10px;
            padding: 1px 7px;
            font-size: 11px;
            color: #3a56a0;
            margin-left: 6px;
        }
        .empty-track {
            border: 1.5px dashed #dce0ea;
            border-radius: 8px;
            padding: 14px 10px;
            text-align: center;
            color: #aab0c4;
            font-size: 12px;
            margin-top: 6px;
        }
        div[data-testid="stNumberInput"] label {
            font-size: 12px !important;
            color: #555 !important;
            margin-bottom: 0px !important;
        }
        div[data-testid="stSegmentedControl"] {
            margin-top: -4px;
        }
        </style>
        """, unsafe_allow_html=True)

        # ── Always read from session_state (includes QC changes) ──
        plot_df       = st.session_state["df"]
        plot_all_cols = [c for c in plot_df.columns if c != depth_col]

        # ── Depth unit label ──────────────────────────────────
        depth_unit_label = st.session_state.get("depth_unit_label", "") or "m"
        y_axis_title     = f"{depth_col} ({depth_unit_label})"

        # ── Top controls row: tracks + plot size ─────────────
        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])

        with ctrl1:
            n_tracks = st.number_input(
                "Number of tracks",
                min_value=1,
                max_value=10,
                value=min(3, len(plot_all_cols)),
                step=1,
                help="Each track is an independent vertical panel in the log plot"
            )
        with ctrl2:
            plot_height = st.number_input(
                "Plot height (px)",
                min_value=400,
                max_value=4000,
                value=1000,
                step=100,
                help="Total height of the log plot in pixels"
            )
        with ctrl3:
            plot_width = st.number_input(
                "Plot width (px)",
                min_value=400,
                max_value=3000,
                value=1200,
                step=100,
                help="Set the total width of the log plot in pixels"
            )

        st.markdown("---")

        # ── Initialize session state ──────────────────────────
        if "track_settings" not in st.session_state:
            st.session_state["track_settings"] = {}
        if "curve_settings" not in st.session_state:
            st.session_state["curve_settings"] = {}

        # ── Per-track configuration UI ────────────────────────
        track_cols = st.columns(n_tracks)

        for t in range(1, n_tracks + 1):
            tkey = f"track_{t}"

            # Always init with empty curves — user selects manually
            if tkey not in st.session_state["track_settings"]:
                st.session_state["track_settings"][tkey] = {"curves": []}

            # Remove stale curve names
            st.session_state["track_settings"][tkey]["curves"] = [
                c for c in st.session_state["track_settings"][tkey]["curves"]
                if c in plot_all_cols
            ]

            with track_cols[t - 1]:
                n_c  = len(st.session_state["track_settings"][tkey]["curves"])
                chip = f'<span class="curve-chip">{n_c} curve{"s" if n_c != 1 else ""}</span>'
                st.markdown(
                    f'<div class="track-badge">Track {t}{chip}</div>',
                    unsafe_allow_html=True
                )

                chosen_curves = st.multiselect(
                    "Curves",
                    options=plot_all_cols,
                    default=st.session_state["track_settings"][tkey]["curves"],
                    key=f"curves_{t}",
                    label_visibility="collapsed",
                    placeholder=f"Select curves for Track {t}…",
                )
                st.session_state["track_settings"][tkey]["curves"] = chosen_curves

                if not chosen_curves:
                    st.markdown(
                        '<div class="empty-track">No curves selected</div>',
                        unsafe_allow_html=True
                    )

                # ── Per-curve settings popovers ───────────────
                for ci, curve in enumerate(chosen_curves):

                    if curve not in st.session_state["curve_settings"]:
                        xdata = plot_df[curve].dropna()
                        defs  = CURVE_DEFAULTS.get(
                            curve,
                            {"color": COLOR_PALETTE[ci % len(COLOR_PALETTE)], "scale": "linear"}
                        )
                        st.session_state["curve_settings"][curve] = {
                            "color"     : defs["color"],
                            "scale"     : defs["scale"],
                            "x_min"     : float(xdata.min()) if not xdata.empty else 0.0,
                            "x_max"     : float(xdata.max()) if not xdata.empty else 1.0,
                            "direction" : "forward",
                            "line_width": 1.5,
                            "line_dash" : "solid",
                        }

                    cs = st.session_state["curve_settings"][curve]
                    cs.setdefault("line_width", 1.5)
                    cs.setdefault("line_dash", "solid")

                    with st.popover(f"⬤ {curve}", use_container_width=True):

                        st.markdown(f"**{curve}** — axis settings")
                        st.divider()

                        # Row 1: color + line width
                        c1, c2 = st.columns([1, 1.5])
                        with c1:
                            new_color = st.color_picker(
                                "Color",
                                value=cs["color"],
                                key=f"color_{curve}_t{t}"
                            )
                        with c2:
                            new_width = st.slider(
                                "Line width",
                                min_value=0.5,
                                max_value=4.0,
                                value=float(cs.get("line_width", 1.5)),
                                step=0.5,
                                key=f"lw_{curve}_t{t}"
                            )

                        # Row 2: line style
                        dash_opts = ["solid", "dash", "dot", "dashdot"]
                        new_dash = st.selectbox(
                            "Line style",
                            options=dash_opts,
                            index=dash_opts.index(cs.get("line_dash", "solid")),
                            key=f"dash_{curve}_t{t}"
                        )

                        st.markdown("---")

                        # Row 3: scale + direction
                        c3, c4 = st.columns(2)
                        with c3:
                            new_scale = st.segmented_control(
                                "Scale",
                                options=["linear", "log"],
                                default=cs["scale"],
                                key=f"scale_{curve}_t{t}"
                            )
                        with c4:
                            new_dir = st.segmented_control(
                                "Direction",
                                options=["forward", "reverse"],
                                default=cs.get("direction", "forward"),
                                key=f"dir_{curve}_t{t}"
                            )

                        st.markdown("---")

                        # Row 4: axis range
                        xdata    = plot_df[curve].dropna()
                        data_min = float(xdata.min()) if not xdata.empty else 0.0
                        data_max = float(xdata.max()) if not xdata.empty else 1.0

                        st.caption(f"Data range: **{data_min:.3f}** → **{data_max:.3f}**")

                        c5, c6 = st.columns(2)
                        with c5:
                            new_xmin = st.number_input(
                                "Min",
                                value=float(cs["x_min"]),
                                format="%.3f",
                                key=f"xmin_{curve}_t{t}"
                            )
                        with c6:
                            new_xmax = st.number_input(
                                "Max",
                                value=float(cs["x_max"]),
                                format="%.3f",
                                key=f"xmax_{curve}_t{t}"
                            )

                        st.markdown("---")

                        # Apply + Reset
                        ba, bb = st.columns(2)
                        with ba:
                            if st.button("Apply", key=f"apply_{curve}_t{t}", type="primary", use_container_width=True):
                                st.session_state["curve_settings"][curve] = {
                                    "color"     : new_color,
                                    "scale"     : new_scale or "linear",
                                    "x_min"     : new_xmin,
                                    "x_max"     : new_xmax,
                                    "direction" : new_dir or "forward",
                                    "line_width": new_width,
                                    "line_dash" : new_dash,
                                }
                                st.rerun()
                        with bb:
                            if st.button("Reset", key=f"reset_{curve}_t{t}", use_container_width=True):
                                st.session_state["curve_settings"].pop(curve, None)
                                st.rerun()

        st.markdown("---")

        # ── Collect active tracks ─────────────────────────────
        active_tracks = [
            t for t in range(1, n_tracks + 1)
            if st.session_state["track_settings"][f"track_{t}"]["curves"]
        ]

        if not active_tracks:
            st.info("💡 Assign at least one curve to a track above to generate the log plot.")

        else:
            depth    = st.session_state["df"][depth_col].values
            n_active = len(active_tracks)

            # ── Assign global axis numbers ────────────────────
            primary_xaxis = {}
            overlay_xaxis = {}
            axis_counter  = 0

            for idx, t in enumerate(active_tracks):
                axis_counter += 1
                primary_xaxis[t] = axis_counter
                overlay_xaxis[t] = {}
                curves = st.session_state["track_settings"][f"track_{t}"]["curves"]
                for ci in range(1, len(curves)):
                    axis_counter += 1
                    overlay_xaxis[t][ci] = axis_counter

            def ax_ref(num):
                return "x" if num == 1 else f"x{num}"

            def ax_key(num):
                return "xaxis" if num == 1 else f"xaxis{num}"

            # ── Column domains ────────────────────────────────
            col_width   = 1.0 / n_active
            h_gap       = 0.03
            y_domain_lo = 0.0
            y_domain_hi = 0.82

            col_domains = []
            for i in range(n_active):
                x0 = i * col_width + (h_gap / 2 if i > 0 else 0)
                x1 = (i + 1) * col_width - (h_gap / 2 if i < n_active - 1 else 0)
                col_domains.append((x0, x1))

            # ── Dynamic top margin ────────────────────────────
            max_overlay = max(
                len(st.session_state["track_settings"][f"track_{t}"]["curves"])
                for t in active_tracks
            )
            top_margin = max(60, 40 + max_overlay * 38)

            # ── Build figure ──────────────────────────────────
            fig = go.Figure()

            fig.update_layout(
                yaxis=dict(
                    domain=[y_domain_lo, y_domain_hi],
                    autorange="reversed",
                    title_text=y_axis_title,
                    title_font=dict(size=12, color="#444"),
                    showgrid=True,
                    gridcolor="#eeeeee",
                    gridwidth=0.5,
                    showline=True,
                    linecolor="#bbbbbb",
                    linewidth=1,
                    tickfont=dict(size=10, color="#555"),
                    ticks="outside",
                    tickcolor="#bbbbbb",
                    anchor="x",
                )
            )

            for idx, t in enumerate(active_tracks):
                track_cfg = st.session_state["track_settings"][f"track_{t}"]
                curves    = track_cfg["curves"]
                x0, x1   = col_domains[idx]
                p_num     = primary_xaxis[t]

                for ci, curve in enumerate(curves):
                    cs = st.session_state["curve_settings"].get(
                        curve,
                        {
                            "color"     : COLOR_PALETTE[ci % len(COLOR_PALETTE)],
                            "scale"     : "linear",
                            "x_min"     : float(plot_df[curve].dropna().min()),
                            "x_max"     : float(plot_df[curve].dropna().max()),
                            "direction" : "forward",
                            "line_width": 1.5,
                            "line_dash" : "solid",
                        }
                    )

                    y_vals = st.session_state["df"][curve].values

                    x_range = (
                        [cs["x_max"], cs["x_min"]]
                        if cs.get("direction") == "reverse"
                        else [cs["x_min"], cs["x_max"]]
                    )

                    axis_num = p_num if ci == 0 else overlay_xaxis[t][ci]

                    fig.add_trace(
                        go.Scatter(
                            x=y_vals,
                            y=depth,
                            mode="lines",
                            name=curve,
                            line=dict(
                                color=cs["color"],
                                width=cs.get("line_width", 1.5),
                                dash=cs.get("line_dash", "solid"),
                            ),
                            xaxis=ax_ref(axis_num),
                            yaxis="y",
                            hovertemplate=(
                                f"<b>{curve}</b>: %{{x:.3f}}<br>"
                                f"Depth: %{{y:.2f}} {depth_unit_label}"
                                f"<extra></extra>"
                            )
                        )
                    )

                    shared_axis = dict(
                        domain=[x0, x1],
                        side="top",
                        range=x_range,
                        type=cs["scale"],
                        showgrid=False,
                        showline=True,
                        linecolor=cs["color"],
                        linewidth=1.5,
                        tickcolor=cs["color"],
                        tickfont=dict(color=cs["color"], size=10),
                        title_text=curve,
                        title_font=dict(color=cs["color"], size=11),
                        ticks="outside",
                        ticklen=4,
                        mirror=False,
                        zeroline=False,
                        nticks=5,
                    )

                    if ci == 0:
                        fig.update_layout(**{
                            ax_key(p_num): dict(
                                **shared_axis,
                                anchor="y",
                                position=y_domain_hi,
                            )
                        })
                    else:
                        ov_num     = overlay_xaxis[t][ci]
                        position_y = y_domain_hi + ci * 0.07
                        fig.update_layout(**{
                            ax_key(ov_num): dict(
                                **shared_axis,
                                anchor="free",
                                overlaying=ax_ref(p_num),
                                position=position_y,
                            )
                        })

            # ── Track border shapes ───────────────────────────
            track_borders = []
            for i, t in enumerate(active_tracks):
                x0, x1 = col_domains[i]
                track_borders.append(dict(
                    type="rect",
                    xref="paper", yref="paper",
                    x0=x0, y0=y_domain_lo,
                    x1=x1, y1=y_domain_hi,
                    line=dict(color="#cccccc", width=1.0),
                    fillcolor="rgba(0,0,0,0)",
                ))

            # ── Apply height/width ────────────────────────────
            fig.update_layout(
                height=plot_height,
                plot_bgcolor="white",
                paper_bgcolor="#fafafa",
                showlegend=False,
                margin=dict(l=75, r=40, t=top_margin, b=10),
                shapes=track_borders,
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor="#cccccc",
                    font=dict(size=12, color="#333"),
                ),
            )

            # ── Render with or without custom width ──────────
            st.plotly_chart(fig, use_container_width=False, width=plot_width)

    
    # ════════════════════════════════════════════════════════
    #  TAB 4 — CROSSPLOT
    # ════════════════════════════════════════════════════════
    with tab4:

        st.markdown("### Crossplot")
        st.markdown("---")

        col_nx, _ = st.columns([2, 6])
        with col_nx:
            n_xplots = st.number_input(
                "Number of crossplots",
                min_value=1,
                max_value=6,          # increased to 6 since 3 per row now
                value=1,
                step=1,
                key="n_xplots"
            )

        st.markdown("---")

        # ── Initialize all crossplot session states ───────────
        for xp in range(1, n_xplots + 1):
            xp_key = f"xplot_settings_{xp}"
            if xp_key not in st.session_state:
                st.session_state[xp_key] = {
                    "x_curve"    : None,
                    "y_curve"    : None,
                    "c_curve"    : "None",
                    "x_scale"    : "linear",
                    "y_scale"    : "linear",
                    "x_dir"      : "forward",
                    "y_dir"      : "forward",
                    "x_min"      : 0.0,
                    "x_max"      : 1.0,
                    "y_min"      : 0.0,
                    "y_max"      : 1.0,
                    "marker_size": 4,
                    "opacity"    : 0.7,
                    "colorscale" : "Viridis",
                }

        # ── Layout: 3 crossplots per row ──────────────────────
        xp_list = list(range(1, n_xplots + 1))
        rows    = [xp_list[i:i+3] for i in range(0, len(xp_list), 3)]  # 3 per row

        for row in rows:
            row_padded              = row + [None] * (3 - len(row))     # pad to 3
            col_a, col_b, col_c     = st.columns(3)                     # 3 columns
            ui_cols                 = [col_a, col_b, col_c]

            for ui_col, xp in zip(ui_cols, row_padded):
                if xp is None:
                    continue

                xp_key = f"xplot_settings_{xp}"
                xps    = st.session_state[xp_key]

                with ui_col:
                    # ── Collapsed expander = compact by default ──
                    with st.expander(f"⚙ Crossplot {xp} — Configure", expanded=False):

                        st.markdown("**X Axis**")
                        x_curve = st.selectbox(
                            "Curve",
                            options=[None] + all_curves,
                            index=([None] + all_curves).index(xps["x_curve"])
                                if xps["x_curve"] in all_curves else 0,
                            format_func=lambda v: "— select —" if v is None else v,
                            key=f"xplot_x_curve_{xp}"
                        )
                        cx1, cx2 = st.columns(2)
                        with cx1:
                            new_x_scale = st.segmented_control(
                                "Scale", ["linear", "log"],
                                default=xps["x_scale"],
                                key=f"xplot_x_scale_{xp}"
                            )
                        with cx2:
                            new_x_dir = st.segmented_control(
                                "Direction", ["forward", "reverse"],
                                default=xps["x_dir"],
                                key=f"xplot_x_dir_{xp}"
                            )
                        if x_curve:
                            xdata      = df[x_curve].dropna()
                            x_data_min = float(xdata.min()) if not xdata.empty else 0.0
                            x_data_max = float(xdata.max()) if not xdata.empty else 1.0
                            st.caption(f"Range: **{x_data_min:.3f}** → **{x_data_max:.3f}**")
                            rx1, rx2 = st.columns(2)
                            with rx1:
                                x_min = st.number_input("Min", value=xps["x_min"] if xps["x_curve"]==x_curve else x_data_min, format="%.3f", key=f"xplot_x_min_{xp}")
                            with rx2:
                                x_max = st.number_input("Max", value=xps["x_max"] if xps["x_curve"]==x_curve else x_data_max, format="%.3f", key=f"xplot_x_max_{xp}")
                        else:
                            x_min, x_max = 0.0, 1.0

                        st.markdown("---")
                        st.markdown("**Y Axis**")
                        y_curve = st.selectbox(
                            "Curve",
                            options=[None] + all_curves,
                            index=([None] + all_curves).index(xps["y_curve"])
                                if xps["y_curve"] in all_curves else 0,
                            format_func=lambda v: "— select —" if v is None else v,
                            key=f"xplot_y_curve_{xp}"
                        )
                        cy1, cy2 = st.columns(2)
                        with cy1:
                            new_y_scale = st.segmented_control(
                                "Scale", ["linear", "log"],
                                default=xps["y_scale"],
                                key=f"xplot_y_scale_{xp}"
                            )
                        with cy2:
                            new_y_dir = st.segmented_control(
                                "Direction", ["forward", "reverse"],
                                default=xps["y_dir"],
                                key=f"xplot_y_dir_{xp}"
                            )
                        if y_curve:
                            ydata      = df[y_curve].dropna()
                            y_data_min = float(ydata.min()) if not ydata.empty else 0.0
                            y_data_max = float(ydata.max()) if not ydata.empty else 1.0
                            st.caption(f"Range: **{y_data_min:.3f}** → **{y_data_max:.3f}**")
                            ry1, ry2 = st.columns(2)
                            with ry1:
                                y_min = st.number_input("Min", value=xps["y_min"] if xps["y_curve"]==y_curve else y_data_min, format="%.3f", key=f"xplot_y_min_{xp}")
                            with ry2:
                                y_max = st.number_input("Max", value=xps["y_max"] if xps["y_curve"]==y_curve else y_data_max, format="%.3f", key=f"xplot_y_max_{xp}")
                        else:
                            y_min, y_max = 0.0, 1.0

                        st.markdown("---")
                        st.markdown("**Color By (optional)**")
                        color_options = ["None"] + all_curves
                        c_curve = st.selectbox(
                            "Color curve",
                            options=color_options,
                            index=color_options.index(xps["c_curve"])
                                if xps["c_curve"] in color_options else 0,
                            key=f"xplot_c_curve_{xp}"
                        )
                        color_scale = st.selectbox(
                            "Color scale",
                            options=["Viridis","Plasma","Inferno","Magma",
                                    "Cividis","Turbo","RdBu","Spectral","Hot","Jet"],
                            index=["Viridis","Plasma","Inferno","Magma",
                                "Cividis","Turbo","RdBu","Spectral","Hot","Jet"]
                                .index(xps.get("colorscale","Viridis")),
                            key=f"xplot_colorscale_{xp}"
                        )

                        st.markdown("---")
                        st.markdown("**Marker**")
                        cm1, cm2 = st.columns(2)
                        with cm1:
                            marker_size = st.slider("Size", 1, 15, int(xps.get("marker_size",4)), key=f"xplot_marker_size_{xp}")
                        with cm2:
                            opacity = st.slider("Opacity", 0.1, 1.0, float(xps.get("opacity",0.7)), step=0.05, key=f"xplot_opacity_{xp}")

                        st.markdown("---")
                        ba, bb = st.columns(2)
                        with ba:
                            if st.button("Apply", key=f"xplot_apply_{xp}", type="primary", use_container_width=True):
                                st.session_state[xp_key].update({
                                    "x_curve": x_curve, "y_curve": y_curve,
                                    "c_curve": c_curve,
                                    "x_scale": new_x_scale or "linear",
                                    "y_scale": new_y_scale or "linear",
                                    "x_dir"  : new_x_dir  or "forward",
                                    "y_dir"  : new_y_dir  or "forward",
                                    "x_min": x_min, "x_max": x_max,
                                    "y_min": y_min, "y_max": y_max,
                                    "marker_size": marker_size,
                                    "opacity": opacity,
                                    "colorscale": color_scale,
                                })
                                st.rerun()
                        with bb:
                            if st.button("Reset", key=f"xplot_reset_{xp}", use_container_width=True):
                                st.session_state.pop(xp_key, None)
                                st.rerun()

                    # ── Plot ──────────────────────────────────
                    saved = st.session_state[xp_key]

                    if saved["x_curve"] is None or saved["y_curve"] is None:
                        st.markdown(
                            """<div style="border:1.5px dashed #dce0ea; border-radius:8px;
                            padding:24px; text-align:center; color:#aab0c4; font-size:12px;">
                            📊 Configure above and click Apply
                            </div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        cols_needed  = [depth_col, saved["x_curve"], saved["y_curve"]]
                        if saved["c_curve"] != "None":
                            cols_needed.append(saved["c_curve"])

                        valid_curves = all(c in df.columns for c in cols_needed if c is not None)

                        if not valid_curves:
                            st.warning("Curve not found in dataset.")
                        else:
                            plot_df = df[cols_needed].dropna(subset=[saved["x_curve"], saved["y_curve"]])

                            if plot_df.empty:
                                st.warning("No valid data points after removing NaN.")
                            else:
                                x_range = ([saved["x_max"], saved["x_min"]] if saved["x_dir"]=="reverse" else [saved["x_min"], saved["x_max"]])
                                y_range = ([saved["y_max"], saved["y_min"]] if saved["y_dir"]=="reverse" else [saved["y_min"], saved["y_max"]])

                                if saved["c_curve"] != "None":
                                    marker_dict = dict(
                                        size=saved["marker_size"], opacity=saved["opacity"],
                                        color=plot_df[saved["c_curve"]],
                                        colorscale=saved["colorscale"], showscale=True,
                                        colorbar=dict(title=saved["c_curve"], thickness=10, len=0.6)
                                    )
                                else:
                                    marker_dict = dict(size=saved["marker_size"], opacity=saved["opacity"], color="#1f77b4")

                                fig_xp = go.Figure()
                                fig_xp.add_trace(go.Scatter(
                                    x=plot_df[saved["x_curve"]], y=plot_df[saved["y_curve"]],
                                    mode="markers", marker=marker_dict,
                                    text=plot_df[depth_col].round(1),
                                    hovertemplate=(
                                        f"<b>{saved['x_curve']}</b>: %{{x:.3f}}<br>"
                                        f"<b>{saved['y_curve']}</b>: %{{y:.3f}}<br>"
                                        f"Depth: %{{text}} {depth_unit_label}<extra></extra>"
                                    )
                                ))
                                fig_xp.update_xaxes(
                                    title_text=saved["x_curve"], type=saved["x_scale"],
                                    range=x_range, showgrid=True, gridcolor="#eeeeee",
                                    showline=True, linecolor="#aaaaaa",
                                    ticks="outside", tickcolor="#aaaaaa", mirror=True,
                                )
                                fig_xp.update_yaxes(
                                    title_text=saved["y_curve"], type=saved["y_scale"],
                                    range=y_range, showgrid=True, gridcolor="#eeeeee",
                                    showline=True, linecolor="#aaaaaa",
                                    ticks="outside", tickcolor="#aaaaaa", mirror=True,
                                )
                                fig_xp.update_layout(
                                title=dict(
                                    text=f"{saved['x_curve']} vs {saved['y_curve']}",
                                    x=0.5, xanchor="center",
                                    font=dict(size=13, color="#333333")
                                ),
                                height=420,
                                width=420,
                                plot_bgcolor="white", paper_bgcolor="#fafafa",
                                margin=dict(l=50, r=20, t=45, b=45),
                                showlegend=False,
                                hoverlabel=dict(bgcolor="white", bordercolor="#cccccc", font=dict(size=11)),
                            )
                            st.plotly_chart(fig_xp, use_container_width=False, key=f"xplot_fig_{xp}")

            st.markdown("---")

    # ════════════════════════════════════════════════════════
    #  TAB 5 — PETROPHYSICAL CALCULATOR
    # ════════════════════════════════════════════════════════
    with tab5:

        # ── CSS for professional look ─────────────────────────
        st.markdown("""
        <style>
        .petro-section-header {
            background: linear-gradient(90deg, #f0f4ff 0%, #fafbff 100%);
            border-left: 3px solid #3a56a0;
            border-radius: 0 6px 6px 0;
            padding: 8px 14px;
            margin-bottom: 12px;
            font-size: 15px;
            font-weight: 600;
            color: #2c3e6b;
            letter-spacing: 0.2px;
        }
        .formula-box {
            background: #f8f9ff;
            border: 1px solid #e0e4f0;
            border-radius: 8px;
            padding: 10px 16px;
            margin: 8px 0;
        }
        .method-desc {
            background: #fafafa;
            border-left: 3px solid #d0d8f0;
            border-radius: 0 6px 6px 0;
            padding: 8px 12px;
            font-size: 12px;
            color: #555;
            line-height: 1.6;
            margin-top: 6px;
        }
        .empty-select {
            border: 1.5px dashed #dce0ea;
            border-radius: 8px;
            padding: 18px;
            text-align: center;
            color: #aab0c4;
            font-size: 12px;
            margin: 8px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        calc_df   = st.session_state["df"].copy()
        calc_cols = [c for c in st.session_state["df"].columns if c != depth_col]

        GR_ALIASES = ["GR","GAMMA","GAMMARAY","GAMMA_RAY","CGR","SGR","HRGR","TGR"]

        def detect_curve(columns, aliases):
            cols = [c.upper() for c in columns]
            for alias in aliases:
                if alias in cols:
                    return columns[cols.index(alias)]
            for i, c in enumerate(cols):
                for alias in aliases:
                    if c.startswith(alias) or alias in c:
                        return columns[i]
            return None

        # ════════════════════════════════════════════════════
        #  VOLUME OF SHALE
        # ════════════════════════════════════════════════════
        with st.expander("Volume of Shale", expanded=True):

            st.markdown(
                '<div class="petro-section-header">Gamma Ray Index (IGR)</div>',
                unsafe_allow_html=True
            )

            with st.container(border=True):
                st.latex(r"I_{GR} = \frac{GR_{log} - GR_{min}}{GR_{max} - GR_{min}}")

            # ── GR curve selector — NO default ───────────────
            gr_col = st.selectbox(
                "Select GR log",
                options=[None] + calc_cols,
                index=0,
                format_func=lambda v: "— select GR curve —" if v is None else v,
                help="Select the Gamma Ray log from the uploaded dataset"
            )

            if gr_col is None:
                st.markdown(
                    '<div class="empty-select">👆 Select a GR curve above to continue</div>',
                    unsafe_allow_html=True
                )

            else:
                gr_data = calc_df[gr_col].dropna()

                st.markdown("---")

                # ── GR min/max mode ───────────────────────────
                mode = st.segmented_control(
                    "GR min & max determination",
                    options=["Percentile", "Manual"],
                    default="Percentile",
                    key="vsh_gr_mode"
                )

                pct_min, pct_max = 5, 95

                if mode == "Percentile":
                    c1, c2 = st.columns(2)
                    with c1:
                        pct_min = st.slider(
                            "GR min percentile (clean sand)",
                            min_value=0, max_value=50,
                            value=5, step=1,
                            help="Low percentile represents clean sand / shale-free zone"
                        )
                    with c2:
                        pct_max = st.slider(
                            "GR max percentile (pure shale)",
                            min_value=50, max_value=100,
                            value=95, step=1,
                            help="High percentile represents pure shale zone"
                        )

                    gr_min = float(np.percentile(gr_data, pct_min))
                    gr_max = float(np.percentile(gr_data, pct_max))

                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        gr_min = st.number_input(
                            "GR min (clean sand)",
                            value=float(round(gr_data.min(), 2)),
                            format="%.2f"
                        )
                    with c2:
                        gr_max = st.number_input(
                            "GR max (pure shale)",
                            value=float(round(gr_data.max(), 2)),
                            format="%.2f"
                        )
                    pct_min, pct_max = None, None

                # ── Metrics row ───────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric("GR min", f"{gr_min:.2f} API")
                m2.metric("GR max", f"{gr_max:.2f} API")
                m3.metric("Data range", f"{gr_data.min():.2f} → {gr_data.max():.2f}")

                # ── GR histogram ──────────────────────────────
                with st.expander("GR Distribution", expanded=False):
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=gr_data, nbinsx=50,
                        marker_color="#2ca02c",
                        opacity=0.75, name="GR"
                    ))
                    fig_hist.add_vline(
                        x=gr_min, line_dash="dash", line_color="#1f77b4",
                        annotation_text=f"P{pct_min}={gr_min:.1f}" if pct_min else f"GR min={gr_min:.1f}",
                        annotation_position="top right"
                    )
                    fig_hist.add_vline(
                        x=float(gr_data.mean()), line_dash="dot", line_color="#ff7f0e",
                        annotation_text=f"Mean={gr_data.mean():.1f}",
                        annotation_position="top right"
                    )
                    fig_hist.add_vline(
                        x=gr_max, line_dash="dash", line_color="#d62728",
                        annotation_text=f"P{pct_max}={gr_max:.1f}" if pct_max else f"GR max={gr_max:.1f}",
                        annotation_position="top left"
                    )
                    fig_hist.update_layout(
                        height=260,
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis_title="GR (API)", yaxis_title="Count",
                        showlegend=False,
                        margin=dict(l=50, r=30, t=20, b=45),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("---")

                # ── STEP 2: VSH Method ────────────────────────
                st.markdown(
                    '<div class="petro-section-header">Volume of Shale (VSH)</div>',
                    unsafe_allow_html=True
                )

                methods = {
                    "Linear"              : r"V_{sh} = I_{GR}",
                    "Larionov — Younger Rocks"  : r"V_{sh} = 0.083 \left(2^{3.7 \cdot I_{GR}} - 1\right)",
                    "Larionov — Older Rocks"    : r"V_{sh} = 0.33 \left(2^{2 \cdot I_{GR}} - 1\right)",
                    "Steiber"             : r"V_{sh} = \frac{I_{GR}}{3 - 2 \cdot I_{GR}}",
                    "Clavier"             : r"V_{sh} = 1.7 - \sqrt{3.38 - \left(I_{GR} + 0.7\right)^2}",
                }

                method_desc = {
                    "Linear"             : "The Linear method assumes a direct, 1:1 relationship between the Gamma Ray Index and the volume of shale. This model represents the most conservative approach in petrophysical analysis, as it does not account for the non-linear response of radioactive minerals or the effects of compaction. While it is often used as a baseline, it consistently overestimates shale content in most geological settings, particularly in younger, unconsolidated formations.",
                    "Larionov — Younger Rocks" : "Specifically engineered for Cenozoic-age formations, the Larionov Tertiary model provides a significant non-linear correction to the Gamma Ray Index to account for the high porosity and low density of young, unconsolidated rocks. It yields the most optimistic results of all standard models, producing the lowest shale volume estimates. This approach is essential for preventing the over-classification of potential reservoir sands as shale in younger basins.",
                    "Larionov — Older Rocks"   : "The Larionov model for older rocks is tailored for Paleozoic and Mesozoic formations where geological time and pressure have resulted in higher degrees of rock consolidation and lower intrinsic porosity. This method serves as an intermediate estimate between the aggressive Tertiary correction and the raw Linear index. It is widely used in global basins where the target reservoirs are deeply buried and fully lithified.",
                    "Steiber"            : "Originally calibrated against core data and log responses in the Gulf Coast, the Steiber method utilizes a hyperbolic relationship. This model is highly regarded in the industry for providing a moderate reduction of the shale volume without the computational complexity of exponential models. It is particularly effective in deltaic and marine depositional environments where the distribution of clay and silt requires a balanced, non-linear adjustment.",
                    "Clavier"            : "The Clavier model was developed as an evolution of the Larionov concepts, grounded in the dual-water theory of shaly-sand interpretation. This method provides a nuanced compromise between the Tertiary and Older Rock models. It is frequently employed as a safe standard in multi-well studies because it effectively handles the transition between different degrees of rock compaction while maintaining physical consistency.",
                }

                # Method selector in columns for compact layout
                sel_col, _ = st.columns([3, 1])
                with sel_col:
                    selected_method = st.selectbox(
                        "VSH method",
                        options=list(methods.keys()),
                        index=None,
                        placeholder="— select a method —",
                        help="Each method applies a different correction to the GR Index"
                    )

                if selected_method is None:
                    st.markdown(
                        '<div class="empty-select">👆 Select a VSH method above to continue</div>',
                        unsafe_allow_html=True
                    )

                else:
                    # Formula + description
                    with st.container(border=True):
                        st.latex(methods[selected_method])

                    st.markdown(
                        f'<div class="method-desc">ℹ️ {method_desc[selected_method]}</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown("---")

                    # ── Apply button ──────────────────────────
                    ba, bb = st.columns([2, 6])
                    with ba:
                        apply_vsh = st.button(
                            "▶ Calculate & Save",
                            type="primary",
                            use_container_width=True,
                            key="apply_vsh"
                        )

                    if apply_vsh:
                        if gr_max == gr_min:
                            st.error("GR min and GR max cannot be equal — check your inputs.")
                        else:
                            work_df = st.session_state["df"].copy()

                            igr = (work_df[gr_col] - gr_min) / (gr_max - gr_min)
                            igr = igr.clip(0, 1)

                            if selected_method == "Linear":
                                vsh = igr.copy()
                            elif selected_method == "Larionov — Younger Rocks":
                                vsh = 0.083 * (2 ** (3.7 * igr) - 1)
                            elif selected_method == "Larionov — Older Rocks":
                                vsh = 0.33 * (2 ** (2 * igr) - 1)
                            elif selected_method == "Steiber":
                                vsh = igr / (3 - 2 * igr)
                            elif selected_method == "Clavier":
                                inner = (3.38 - (igr + 0.7) ** 2).clip(0, None)
                                vsh   = 1.7 - np.sqrt(inner)
                            else:
                                st.error(f"Unknown method: {selected_method}")
                                st.stop()

                            vsh = pd.Series(vsh, index=work_df.index).clip(0, 1)

                            work_df["IGR"] = igr.round(4)
                            work_df["VSH"] = vsh.round(4)
                            st.session_state["df"]          = work_df
                            st.session_state["vsh_success"] = True
                            st.rerun()

                    if st.session_state.get("vsh_success"):
                        sv1, sv2, sv3 = st.columns(3)
                        vsh_vals = st.session_state["df"]["VSH"].dropna()
                        sv1.metric("VSH mean",  f"{vsh_vals.mean():.3f}")
                        sv2.metric("VSH min",   f"{vsh_vals.min():.3f}")
                        sv3.metric("VSH max",   f"{vsh_vals.max():.3f}")
                        st.success(f"✅ IGR and VSH calculated using **{selected_method}** method and saved to dataframe")
                        st.session_state["vsh_success"] = False
                    


        # ════════════════════════════════════════════════════
        #  DENSITY POROSITY
        # ════════════════════════════════════════════════════
        with st.expander("Density Porosity", expanded=False):

            # ── Formula ───────────────────────────────────────
            with st.container(border=True):
                st.latex(r"\phi_D = \frac{\rho_{ma} - \rho_b}{\rho_{ma} - \rho_{fl}}")

            st.markdown("---")

            # ── RHOB curve selector — NO default ─────────────
            RHOB_ALIASES = ["RHOB","DEN","DENS","DENSITY","RHOZ","RHOG"]

            rhob_col = st.selectbox(
                "Select density log",
                options=[None] + calc_cols,
                index=0,
                format_func=lambda v: "— select RHOB curve —" if v is None else v,
                help="Bulk density log (RHOB). Expected unit: g/cc (range 1.8–3.0)",
                key="dphi_rhob_col"
            )

            if rhob_col is None:
                st.markdown(
                    '<div class="empty-select">👆 Select a density log above to continue</div>',
                    unsafe_allow_html=True
                )

            else:
                rhob_data = st.session_state["df"][rhob_col].dropna()

                # ── Unit warning ──────────────────────────────
                if rhob_data.mean() > 10:
                    st.warning(
                        f"⚠️ '{rhob_col}' values appear to be in **kg/m³** "
                        f"(mean = {rhob_data.mean():.1f}). "
                        f"Expected range: **1.8 – 3.0 g/cc**. "
                        f"Verify units or divide by 1000."
                    )

                # ── Data stats row ────────────────────────────
                d1, d2, d3 = st.columns(3)
                d1.metric("Min", f"{rhob_data.min():.3f} g/cc")
                d2.metric("Max", f"{rhob_data.max():.3f} g/cc")
                d3.metric("Mean", f"{rhob_data.mean():.3f} g/cc")

                st.markdown("---")

                # ── Matrix and fluid density inputs ──────────
                st.markdown(
                    '<div class="petro-section-header">Rock & Fluid Parameters</div>',
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)
                with c1:
                    rho_ma = st.number_input(
                        "Matrix density ρma (g/cc)",
                        value=2.650,
                        format="%.3f",
                        help="Sandstone = 2.65 | Limestone = 2.71 | Dolomite = 2.87",
                        key="dphi_rho_ma"
                    )
                    st.caption("🪨 Sandstone=2.65 · Limestone=2.71 · Dolomite=2.87")

                with c2:
                    rho_fl = st.number_input(
                        "Fluid density ρfl (g/cc)",
                        value=1.000,
                        format="%.3f",
                        help="Fresh water=1.0 | Salt water=1.1 | Oil=0.8 | Gas=0.3",
                        key="dphi_rho_fl"
                    )
                    st.caption("💧 Fresh water=1.0 · Salt water=1.1 · Oil=0.8 · Gas=0.3")

                st.markdown("---")

                # ── Already calculated notice ─────────────────
                if "DPHI" in st.session_state["df"].columns:
                    dphi_existing = st.session_state["df"]["DPHI"].dropna()
                    ex1, ex2, ex3 = st.columns(3)
                    ex1.metric("DPHI mean", f"{dphi_existing.mean():.3f}")
                    ex2.metric("DPHI min",  f"{dphi_existing.min():.3f}")
                    ex3.metric("DPHI max",  f"{dphi_existing.max():.3f}")
                    st.info("ℹ️ DPHI already exists in the dataframe. Click **Calculate & Save** to overwrite.")

                # ── Apply button ──────────────────────────────
                ba, _ = st.columns([2, 6])
                with ba:
                    apply_dphi = st.button(
                        "▶ Calculate & Save",
                        key="apply_phid",
                        type="primary",
                        use_container_width=True
                    )

                if apply_dphi:
                    if rho_ma == rho_fl:
                        st.error("❌ Matrix density and fluid density cannot be equal.")
                    else:
                        work_df = st.session_state["df"].copy()

                        phid = (rho_ma - work_df[rhob_col]) / (rho_ma - rho_fl)
                        phid = pd.Series(phid, index=work_df.index)

                        work_df["DPHI"] = phid.round(4)
                        st.session_state["df"]           = work_df
                        st.session_state["phid_success"] = True
                        st.rerun()

                if st.session_state.get("phid_success"):
                    dphi_new = st.session_state["df"]["DPHI"].dropna()
                    r1, r2, r3 = st.columns(3)
                    r1.metric("DPHI mean", f"{dphi_new.mean():.3f}")
                    r2.metric("DPHI min",  f"{dphi_new.min():.3f}")
                    r3.metric("DPHI max",  f"{dphi_new.max():.3f}")
                    st.success(
                        f"✅ DPHI calculated (ρma={rho_ma:.3f}, ρfl={rho_fl:.3f}) "
                        f"and saved to dataframe"
                    )
                    st.session_state["phid_success"] = False


        # ════════════════════════════════════════════════════
        #  TOTAL POROSITY
        # ════════════════════════════════════════════════════
        with st.expander("Total Porosity", expanded=False):

            # ── Formula display ───────────────────────────────
            fc1, fc2 = st.columns(2)
            with fc1:
                with st.container(border=True):
                    st.markdown("**Oil / Brine zone**")
                    st.latex(r"\phi_T = \frac{\phi_D + \phi_N}{2}")
            with fc2:
                with st.container(border=True):
                    st.markdown("**Gas bearing zone**")
                    st.latex(r"\phi_T = \sqrt{\frac{\phi_D^2 + \phi_N^2}{2}}")

            st.markdown("---")

            # ── Initialize session state ──────────────────────
            if "tc_track_settings" not in st.session_state:
                st.session_state["tc_track_settings"] = {
                    "track_1": {"curves": []},
                    "track_2": {"curves": []},
                    "track_3": {"curves": []},
                }
            if "tc_curve_settings" not in st.session_state:
                st.session_state["tc_curve_settings"] = {}
            if "tphi_markers" not in st.session_state:
                st.session_state["tphi_markers"] = []

            TC_TRACK_LABELS = {
                "track_1": ("Track 1", "GR"),
                "track_2": ("Track 2", "Shallow + Deep Resistivity"),
                "track_3": ("Track 3", "RHOB + NPHI"),
            }

            # ════════════════════════════════════════════════
            #  TRIPLE COMBO SETUP
            # ════════════════════════════════════════════════
            st.markdown(
                '<div class="petro-section-header">Triple Combo Setup</div>',
                unsafe_allow_html=True
            )

            # ── Plot size controls ────────────────────────────
            ps1, ps2 = st.columns(2)
            with ps1:
                tc_height = st.number_input(
                    "Plot height (px)", min_value=400, max_value=4000,
                    value=700, step=100, key="tc_plot_height"
                )
            with ps2:
                tc_width = st.number_input(
                    "Plot width (px)", min_value=400, max_value=3000,
                    value=900, step=100, key="tc_plot_width"
                )

            st.markdown("")

            # ── Track columns ─────────────────────────────────
            tc_col1, tc_col2, tc_col3 = st.columns(3)
            tc_track_cols = [tc_col1, tc_col2, tc_col3]

            for ti, (tkey, (tlabel, thint)) in enumerate(TC_TRACK_LABELS.items()):

                st.session_state["tc_track_settings"][tkey]["curves"] = [
                    c for c in st.session_state["tc_track_settings"][tkey]["curves"]
                    if c in calc_cols
                ]

                with tc_track_cols[ti]:
                    n_c  = len(st.session_state["tc_track_settings"][tkey]["curves"])
                    chip = f'<span class="curve-chip">{n_c} curve{"s" if n_c!=1 else ""}</span>'
                    st.markdown(
                        f'<div class="track-badge">{tlabel}{chip}</div>',
                        unsafe_allow_html=True
                    )
                    st.caption(thint)

                    chosen = st.multiselect(
                        "Curves",
                        options=calc_cols,
                        default=st.session_state["tc_track_settings"][tkey]["curves"],
                        key=f"tc_curves_{tkey}",
                        label_visibility="collapsed",
                        placeholder="Select curves…",
                    )
                    st.session_state["tc_track_settings"][tkey]["curves"] = chosen

                    if not chosen:
                        st.markdown(
                            '<div class="empty-track">No curves selected</div>',
                            unsafe_allow_html=True
                        )

                    for ci, curve in enumerate(chosen):
                        if curve not in st.session_state["tc_curve_settings"]:
                            xdata = st.session_state["df"][curve].dropna()
                            defs  = CURVE_DEFAULTS.get(
                                curve,
                                {"color": COLOR_PALETTE[ci % len(COLOR_PALETTE)], "scale": "linear"}
                            )
                            st.session_state["tc_curve_settings"][curve] = {
                                "color"     : defs["color"],
                                "scale"     : defs.get("scale", "linear"),
                                "x_min"     : float(xdata.min()) if not xdata.empty else 0.0,
                                "x_max"     : float(xdata.max()) if not xdata.empty else 1.0,
                                "direction" : "forward",
                                "line_width": 1.5,
                                "line_dash" : "solid",
                            }

                        cs = st.session_state["tc_curve_settings"][curve]
                        cs.setdefault("line_width", 1.5)
                        cs.setdefault("line_dash",  "solid")

                        with st.popover(f"⬤ {curve}", use_container_width=True):
                            st.markdown(f"**{curve}** — axis settings")
                            st.divider()

                            p1, p2 = st.columns([1, 1.5])
                            with p1:
                                new_color = st.color_picker(
                                    "Color", value=cs["color"],
                                    key=f"tc_color_{curve}_{tkey}"
                                )
                            with p2:
                                new_width = st.slider(
                                    "Line width", 0.5, 4.0,
                                    float(cs.get("line_width", 1.5)), 0.5,
                                    key=f"tc_lw_{curve}_{tkey}"
                                )

                            dash_opts = ["solid", "dash", "dot", "dashdot"]
                            new_dash  = st.selectbox(
                                "Line style", dash_opts,
                                index=dash_opts.index(cs.get("line_dash", "solid")),
                                key=f"tc_dash_{curve}_{tkey}"
                            )
                            st.markdown("---")

                            p3, p4 = st.columns(2)
                            with p3:
                                new_scale = st.segmented_control(
                                    "Scale", ["linear", "log"],
                                    default=cs["scale"],
                                    key=f"tc_scale_{curve}_{tkey}"
                                )
                            with p4:
                                new_dir = st.segmented_control(
                                    "Direction", ["forward", "reverse"],
                                    default=cs.get("direction", "forward"),
                                    key=f"tc_dir_{curve}_{tkey}"
                                )
                            st.markdown("---")

                            xdata    = st.session_state["df"][curve].dropna()
                            data_min = float(xdata.min()) if not xdata.empty else 0.0
                            data_max = float(xdata.max()) if not xdata.empty else 1.0
                            st.caption(f"Data range: **{data_min:.3f}** → **{data_max:.3f}**")

                            p5, p6 = st.columns(2)
                            with p5:
                                new_xmin = st.number_input(
                                    "Min", value=float(cs["x_min"]),
                                    format="%.3f", key=f"tc_xmin_{curve}_{tkey}"
                                )
                            with p6:
                                new_xmax = st.number_input(
                                    "Max", value=float(cs["x_max"]),
                                    format="%.3f", key=f"tc_xmax_{curve}_{tkey}"
                                )
                            st.markdown("---")

                            ba2, bb2 = st.columns(2)
                            with ba2:
                                if st.button("Apply", key=f"tc_apply_{curve}_{tkey}",
                                            type="primary", use_container_width=True):
                                    st.session_state["tc_curve_settings"][curve] = {
                                        "color"     : new_color,
                                        "scale"     : new_scale or "linear",
                                        "x_min"     : new_xmin,
                                        "x_max"     : new_xmax,
                                        "direction" : new_dir or "forward",
                                        "line_width": new_width,
                                        "line_dash" : new_dash,
                                    }
                                    st.rerun()
                            with bb2:
                                if st.button("Reset", key=f"tc_reset_{curve}_{tkey}",
                                            use_container_width=True):
                                    st.session_state["tc_curve_settings"].pop(curve, None)
                                    st.rerun()

            # ── After track loop: check what is selected ──────
            # BUG FIX: moved outside the for loop — correct indentation level
            all_tc_curves = [
                c for tkey in TC_TRACK_LABELS
                for c in st.session_state["tc_track_settings"][tkey]["curves"]
            ]

            if not all_tc_curves:
                st.markdown(
                    '<div class="empty-select">Select curves in the tracks above to continue</div>',
                    unsafe_allow_html=True
                )
            else:
                # ── Porosity curve selectors ──────────────────
                # BUG FIX: always define dphi_col / nphi_col here so they are
                # in scope for the guard check further below
                st.markdown("---")
                st.markdown(
                    '<div class="petro-section-header">Porosity Curves</div>',
                    unsafe_allow_html=True
                )
                st.caption("Select the porosity curves (0–1 fraction) to use in the PHIT formula.")

                cc1, cc2 = st.columns(2)
                with cc1:
                    dphi_opts = list(dict.fromkeys(
                        [c for c in ["DPHI"] if c in st.session_state["df"].columns] + calc_cols
                    ))
                    dphi_col = st.selectbox(
                        "Density Porosity (DPHI)",
                        options=[None] + dphi_opts,
                        index=0,
                        format_func=lambda v: "— select DPHI —" if v is None else v,
                        key="tphi_dphi_final"
                    )
                with cc2:
                    nphi_col = st.selectbox(
                        "Neutron Porosity (NPHI)",
                        options=[None] + calc_cols,
                        index=0,
                        format_func=lambda v: "— select NPHI —" if v is None else v,
                        key="tphi_nphi_final"
                    )

                st.markdown("---")

                # ════════════════════════════════════════════
                #  TRIPLE COMBO PLOT
                # ════════════════════════════════════════════
                # BUG FIX: moved outside track loop — correct indentation level
                st.markdown(
                    '<div class="petro-section-header">Triple Combo Plot</div>',
                    unsafe_allow_html=True
                )
                st.caption("Hover over the plot to read depth values, then enter them in Zone Selection.")

                df_tc    = st.session_state["df"]
                tc_depth = df_tc[depth_col].values
                y_lo, y_hi = 0.0, 0.88

                active_tc_tracks = [
                    tkey for tkey in TC_TRACK_LABELS
                    if st.session_state["tc_track_settings"][tkey]["curves"]
                ]
                n_tc = len(active_tc_tracks)

                if n_tc == 0:
                    st.info("No curves assigned to any track yet.")
                else:
                    col_w    = 1.0 / n_tc
                    h_gap    = 0.03
                    col_doms = []
                    for i in range(n_tc):
                        x0 = i * col_w + (h_gap / 2 if i > 0 else 0)
                        x1 = (i + 1) * col_w - (h_gap / 2 if i < n_tc - 1 else 0)
                        col_doms.append((x0, x1))

                    fig_tc = go.Figure()
                    fig_tc.update_layout(
                        yaxis=dict(
                            domain=[y_lo, y_hi],
                            autorange="reversed",
                            title_text=y_axis_title,
                            title_font=dict(size=11, color="#444"),
                            showgrid=True, gridcolor="#eeeeee", gridwidth=0.5,
                            showline=True, linecolor="#bbbbbb",
                            tickfont=dict(size=10, color="#555"),
                            ticks="outside", tickcolor="#bbbbbb",
                            anchor="x",
                        )
                    )

                    axis_counter = 0
                    prim_nums    = {}

                    for ti, tkey in enumerate(active_tc_tracks):
                        x0, x1  = col_doms[ti]
                        curves   = st.session_state["tc_track_settings"][tkey]["curves"]
                        prim_num = None

                        for ci, curve in enumerate(curves):
                            axis_counter += 1
                            anum = axis_counter
                            aref = "x" if anum == 1 else f"x{anum}"
                            akey = "xaxis" if anum == 1 else f"xaxis{anum}"

                            cs = st.session_state["tc_curve_settings"].get(curve, {
                                "color"     : COLOR_PALETTE[ci % len(COLOR_PALETTE)],
                                "scale"     : "linear",
                                "x_min"     : float(np.nanmin(df_tc[curve].values)),
                                "x_max"     : float(np.nanmax(df_tc[curve].values)),
                                "direction" : "forward",
                                "line_width": 1.5,
                                "line_dash" : "solid",
                            })

                            x_range = (
                                [cs["x_max"], cs["x_min"]]
                                if cs.get("direction") == "reverse"
                                else [cs["x_min"], cs["x_max"]]
                            )

                            fig_tc.add_trace(go.Scatter(
                                x=df_tc[curve].values,
                                y=tc_depth,
                                mode="lines", name=curve,
                                line=dict(
                                    color=cs["color"],
                                    width=cs.get("line_width", 1.5),
                                    dash=cs.get("line_dash", "solid"),
                                ),
                                xaxis=aref, yaxis="y",
                                hovertemplate=(
                                    f"<b>{curve}</b>: %{{x:.3f}}<br>"
                                    f"<b>Depth: %{{y:.2f}} {depth_unit_label}</b>"
                                    "<extra></extra>"
                                )
                            ))

                            shared = dict(
                                domain=[x0, x1],
                                side="top",
                                range=x_range,
                                type=cs.get("scale", "linear"),
                                showgrid=False, showline=True,
                                linecolor=cs["color"], linewidth=1.5,
                                tickcolor=cs["color"],
                                tickfont=dict(color=cs["color"], size=9),
                                title_text=curve,
                                title_font=dict(color=cs["color"], size=10),
                                ticks="outside", ticklen=4, nticks=5,
                                zeroline=False, mirror=False,
                            )

                            if ci == 0:
                                prim_num = anum
                                fig_tc.update_layout(**{
                                    akey: dict(**shared, anchor="y", position=y_hi)
                                })
                            else:
                                fig_tc.update_layout(**{
                                    akey: dict(
                                        **shared,
                                        anchor="free",
                                        overlaying="x" if prim_num == 1 else f"x{prim_num}",
                                        position=y_hi + ci * 0.06,
                                    )
                                })

                        prim_nums[tkey] = prim_num

                    # ── Marker lines ──────────────────────────
                    shapes = []
                    for m in st.session_state["tphi_markers"]:
                        shapes.append(dict(
                            type="line", xref="paper", yref="y",
                            x0=0, x1=1, y0=m, y1=m,
                            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                        ))

                    for i in range(n_tc):
                        x0, x1 = col_doms[i]
                        shapes.append(dict(
                            type="rect", xref="paper", yref="paper",
                            x0=x0, y0=y_lo, x1=x1, y1=y_hi,
                            line=dict(color="#cccccc", width=1.0),
                            fillcolor="rgba(0,0,0,0)",
                        ))

                    max_ov = max(
                        len(st.session_state["tc_track_settings"][tkey]["curves"])
                        for tkey in active_tc_tracks
                    )
                    top_margin = max(60, 40 + max_ov * 36)

                    fig_tc.update_layout(
                        height=tc_height,
                        plot_bgcolor="white", paper_bgcolor="#fafafa",
                        showlegend=False,
                        margin=dict(l=75, r=30, t=top_margin, b=10),
                        shapes=shapes,
                        hoverlabel=dict(
                            bgcolor="white", bordercolor="#ccc",
                            font=dict(size=11, color="#333")
                        ),
                    )

                    st.plotly_chart(fig_tc, use_container_width=False,
                                    width=tc_width, key="tphi_triple_combo")

                # ════════════════════════════════════════════
                #  ZONE SELECTION
                # ════════════════════════════════════════════
                st.markdown("---")
                st.markdown(
                    '<div class="petro-section-header">Zone Selection</div>',
                    unsafe_allow_html=True
                )

                depth_vals = st.session_state["df"][depth_col].dropna()
                well_top   = float(depth_vals.min())
                well_bot   = float(depth_vals.max())

                st.caption(
                    f"Well: **{well_top:.1f} – {well_bot:.1f} {depth_unit_label}**  ·  "
                    "Read depths from the hover tooltip above."
                )

                b1, b2, b3 = st.columns([5, 1.4, 1.4])
                with b1:
                    bulk_input = st.text_input(
                        "Boundary depths (comma-separated)",
                        placeholder=f"e.g.  {well_top+50:.0f}, {well_top+150:.0f}, {well_top+280:.0f}",
                        key="tphi_bulk_input",
                        help="Enter one or more boundary depths separated by commas, then click Set"
                    )
                with b2:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                    if st.button("Set", key="tphi_bulk_add",
                                use_container_width=True, type="primary"):
                        try:
                            parsed  = [round(float(x.strip()), 1) for x in bulk_input.split(",") if x.strip()]
                            valid   = [d for d in parsed if well_top < d < well_bot]
                            invalid = [d for d in parsed if d not in valid]
                            if invalid:
                                st.warning(f"Skipped out-of-range: {', '.join(str(d) for d in invalid)}")
                            if valid:
                                existing = set(st.session_state["tphi_markers"])
                                existing.update(valid)
                                st.session_state["tphi_markers"] = sorted(existing)
                                st.rerun()
                        except ValueError:
                            st.error("Invalid input — enter numbers separated by commas.")
                with b3:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                    if st.button("Clear All", key="tphi_clear",
                                use_container_width=True):
                        st.session_state["tphi_markers"] = []
                        st.rerun()

                if st.session_state["tphi_markers"]:
                    sorted_m   = sorted(st.session_state["tphi_markers"])
                    pills_html = "".join([
                        f'<span style="background:#fff3e0; border:1px solid #ff9800; '
                        f'border-radius:12px; padding:3px 10px; margin:2px 4px; '
                        f'font-size:12px; color:#e65100; display:inline-block;">'
                        f'{m:.1f} {depth_unit_label}</span>'
                        for m in sorted_m
                    ])
                    st.markdown(pills_html, unsafe_allow_html=True)
                    rl1, _ = st.columns([1.5, 8])
                    with rl1:
                        if st.button("Remove Last", key="tphi_remove_last",
                                    use_container_width=True):
                            st.session_state["tphi_markers"].pop()
                            st.rerun()
                    st.caption(f"{len(sorted_m)} marker(s) → **{len(sorted_m)+1} zone(s)**")
                else:
                    st.caption("No markers set — entire well treated as one zone.")

                # ════════════════════════════════════════════
                #  FORMULA PER ZONE
                # ════════════════════════════════════════════
                st.markdown("---")
                st.markdown(
                    '<div class="petro-section-header">Formula per Zone</div>',
                    unsafe_allow_html=True
                )

                sorted_markers = sorted(st.session_state.get("tphi_markers", []))
                boundaries     = [well_top] + sorted_markers + [well_bot]
                zones          = [{"top": boundaries[i], "bot": boundaries[i+1]}
                                for i in range(len(boundaries) - 1)]

                zone_models = []
                for zi, zone in enumerate(zones):
                    zc1, zc2 = st.columns([4, 2])
                    with zc1:
                        st.markdown(
                            f'<div style="background:#f0f4ff; border-radius:6px; '
                            f'padding:7px 14px; font-size:13px; color:#2c3e6b; margin:3px 0;">'
                            f'<b>Zone {zi+1}</b> &nbsp;│&nbsp; '
                            f'{zone["top"]:.1f} → {zone["bot"]:.1f} {depth_unit_label}'
                            f'<span style="color:#aaa; font-size:11px; margin-left:8px;">'
                            f'({zone["bot"]-zone["top"]:.1f} {depth_unit_label})'
                            f'</span></div>',
                            unsafe_allow_html=True
                        )
                    with zc2:
                        model = st.segmented_control(
                            "Model",
                            options=["Oil/Brine", "Gas"],
                            default="Oil/Brine",
                            key=f"tphi_zone_model_{zi}",
                            label_visibility="collapsed"
                        )
                    zone_models.append({
                        "top"  : zone["top"],
                        "bot"  : zone["bot"],
                        "model": model or "Oil/Brine"
                    })

                st.markdown("---")

                if "PHIT" in st.session_state["df"].columns:
                    ex = st.session_state["df"]["PHIT"].dropna()
                    e1, e2, e3 = st.columns(3)
                    e1.metric("PHIT mean", f"{ex.mean():.3f}")
                    e2.metric("PHIT min",  f"{ex.min():.3f}")
                    e3.metric("PHIT max",  f"{ex.max():.3f}")
                    st.info("PHIT already exists. Click Calculate & Save to overwrite.")

                if dphi_col is None or nphi_col is None:
                    st.warning("Select both Density Porosity and Neutron Porosity curves above.")
                else:
                    ba, _ = st.columns([2, 6])
                    with ba:
                        apply_tphi = st.button(
                            "▶ Calculate & Save",
                            key="apply_tphi",
                            type="primary",
                            use_container_width=True
                        )

                    if apply_tphi:
                        tphi_df  = st.session_state["df"].copy()
                        dphi_ser = tphi_df[dphi_col]
                        nphi_ser = tphi_df[nphi_col]
                        tphi_df["PHIT"] = np.nan

                        for z in zone_models:
                            mask = (
                                (tphi_df[depth_col] >= z["top"]) &
                                (tphi_df[depth_col] <= z["bot"])
                            )
                            if z["model"] == "Oil/Brine":
                                tphi_df.loc[mask, "PHIT"] = (
                                    (dphi_ser[mask] + nphi_ser[mask]) / 2
                                ).round(4)
                            else:
                                tphi_df.loc[mask, "PHIT"] = (
                                    np.sqrt(
                                        (dphi_ser[mask]**2 + nphi_ser[mask]**2) / 2
                                    )
                                ).round(4)

                        st.session_state["df"] = tphi_df
                        phit_vals = tphi_df["PHIT"].dropna()
                        st.session_state["phit_success"] = {
                            "min"  : round(float(phit_vals.min()),  3),
                            "max"  : round(float(phit_vals.max()),  3),
                            "mean" : round(float(phit_vals.mean()), 3),
                            "std"  : round(float(phit_vals.std()),  3),
                            "zones": len(zone_models),
                        }
                        st.rerun()

                    if st.session_state.get("phit_success"):
                        stats = st.session_state["phit_success"]
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("PHIT min",  f"{stats['min']:.3f}")
                        r2.metric("PHIT max",  f"{stats['max']:.3f}")
                        r3.metric("PHIT mean", f"{stats['mean']:.3f}")
                        r4.metric("PHIT std",  f"{stats['std']:.3f}")
                        st.success(
                            f"PHIT calculated across {stats['zones']} zone(s) "
                            f"using {dphi_col} + {nphi_col} — saved to dataframe"
                        )
                        st.session_state["phit_success"] = None
        

        # ════════════════════════════════════════════════════
        #  EFFECTIVE POROSITY
        # ════════════════════════════════════════════════════
        with st.expander("Effective Porosity", expanded=False):

            # ── Formulas ──────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                with st.container(border=True):
                    st.markdown("**From Total Porosity**")
                    st.latex(r"\phi_E = \phi_T - (V_{sh} \cdot \phi_{T,sh})")
            with c2:
                with st.container(border=True):
                    st.markdown("**From Density Porosity**")
                    st.latex(r"\phi_{E,D} = \phi_D - (V_{sh} \cdot \phi_{D,sh})")
            with c3:
                with st.container(border=True):
                    st.markdown("**From Neutron Porosity**")
                    st.latex(r"\phi_{E,N} = \phi_N - (V_{sh} \cdot \phi_{N,sh})")

            st.markdown("---")

            # ── VSH input ─────────────────────────────────────
            st.markdown(
                '<div class="petro-section-header">Shale Reference</div>',
                unsafe_allow_html=True
            )

            cur_df = st.session_state["df"]

            sv1, sv2 = st.columns(2)
            with sv1:
                vsh_options = list(dict.fromkeys(
                    [c for c in ["VSH"] if c in cur_df.columns]
                    + [c for c in cur_df.columns if c != depth_col]
                ))
                vsh_col = st.selectbox(
                    "Volume of Shale (VSH)",
                    options=vsh_options,
                )
            with sv2:
                vsh_thresh = st.number_input(
                    "Pure shale threshold",
                    min_value=0.0, max_value=1.0,
                    value=0.8, step=0.05, format="%.2f",
                    help="Rows where VSH exceeds this value are used to compute apparent shale porosity (φsh)"
                )

            st.markdown("---")

            # ── Porosity log selectors ─────────────────────────
            st.markdown(
                '<div class="petro-section-header">Porosity Logs to Correct</div>',
                unsafe_allow_html=True
            )
            st.caption("Each selected log produces a shale-corrected output column (e.g. PHITe, DPHIe).")

            # ── PHIT ──────────────────────────────────────────
            use_phit = st.checkbox(
                "Correct Total Porosity (PHIT)",
                value=("PHIT" in cur_df.columns)
            )
            if use_phit:
                phit_options = list(dict.fromkeys(
                    [c for c in ["PHIT"] if c in cur_df.columns]
                    + [c for c in cur_df.columns if c != depth_col]
                ))
                pt1, pt2 = st.columns(2)
                with pt1:
                    phit_col = st.selectbox("PHIT log", options=phit_options, key="phie_phit_col")
                with pt2:
                    phit_sh_data = cur_df[cur_df[vsh_col] > vsh_thresh][phit_col].dropna()
                    phi_tsh      = float(phit_sh_data.mean()) if not phit_sh_data.empty else 0.0
                    st.metric(f"φT,sh  (VSH > {vsh_thresh})", f"{phi_tsh:.4f}")
            else:
                phit_col = None

            # ── DPHI ──────────────────────────────────────────
            use_dphi = st.checkbox(
                "Correct Density Porosity (DPHI)",
                value=("DPHI" in cur_df.columns)
            )
            if use_dphi:
                dphi_options = list(dict.fromkeys(
                    [c for c in ["DPHI"] if c in cur_df.columns]
                    + [c for c in cur_df.columns if c != depth_col]
                ))
                pd1, pd2 = st.columns(2)
                with pd1:
                    dphi_col_e = st.selectbox("DPHI log", options=dphi_options, key="phie_dphi_col")
                with pd2:
                    dphi_sh_data = cur_df[cur_df[vsh_col] > vsh_thresh][dphi_col_e].dropna()
                    phi_dsh      = float(dphi_sh_data.mean()) if not dphi_sh_data.empty else 0.0
                    st.metric(f"φD,sh  (VSH > {vsh_thresh})", f"{phi_dsh:.4f}")
            else:
                dphi_col_e = None

            # ── NPHI ──────────────────────────────────────────
            use_nphi = st.checkbox("Correct Neutron Porosity (NPHI)", value=False)
            if use_nphi:
                NPHI_ALIASES = ["NPHI", "NEUT", "NEUTRON", "PHIN", "TNPH", "CNCF", "NPOR"]
                nphi_auto_e  = detect_curve(
                    [c for c in cur_df.columns if c != depth_col], NPHI_ALIASES
                )
                nphi_options = list(dict.fromkeys(
                    ([nphi_auto_e] if nphi_auto_e else [])
                    + [c for c in cur_df.columns if c != depth_col]
                ))
                pn1, pn2 = st.columns(2)
                with pn1:
                    nphi_col_e = st.selectbox("NPHI log", options=nphi_options, key="phie_nphi_col")
                with pn2:
                    nphi_sh_data = cur_df[cur_df[vsh_col] > vsh_thresh][nphi_col_e].dropna()
                    phi_nsh      = float(nphi_sh_data.mean()) if not nphi_sh_data.empty else 0.0
                    st.metric(f"φN,sh  (VSH > {vsh_thresh})", f"{phi_nsh:.4f}")
            else:
                nphi_col_e = None

            st.markdown("---")

            # ── Existing columns notice ────────────────────────
            existing_phie = [
                c for c in cur_df.columns
                if c.endswith("e") and c != depth_col
            ]
            if existing_phie:
                st.info(f"{', '.join(existing_phie)} already in dataframe — will be overwritten.")

            # ── Apply ─────────────────────────────────────────
            ba, _ = st.columns([2, 6])
            with ba:
                apply_phie = st.button(
                    "▶ Calculate & Save",
                    key="apply_phie",
                    type="primary",
                    use_container_width=True
                )

            if apply_phie:
                if not use_phit and not use_dphi and not use_nphi:
                    st.warning("Select at least one porosity log to correct.")
                else:
                    phie_df = st.session_state["df"].copy()
                    vsh_ser = phie_df[vsh_col]

                    computed_cols  = []
                    computed_stats = {}

                    if use_phit:
                        phi_tsh_ser = phie_df[phie_df[vsh_col] > vsh_thresh][phit_col].dropna().mean()
                        phie_df[f"{phit_col}e"] = (
                            phie_df[phit_col] - (vsh_ser * phi_tsh_ser)
                        ).round(4)
                        computed_cols.append(f"{phit_col}e")
                        computed_stats[f"{phit_col}e"] = phi_tsh_ser

                    if use_dphi:
                        phi_dsh_ser = phie_df[phie_df[vsh_col] > vsh_thresh][dphi_col_e].dropna().mean()
                        phie_df[f"{dphi_col_e}e"] = (
                            phie_df[dphi_col_e] - (vsh_ser * phi_dsh_ser)
                        ).round(4)
                        computed_cols.append(f"{dphi_col_e}e")
                        computed_stats[f"{dphi_col_e}e"] = phi_dsh_ser

                    if use_nphi:
                        phi_nsh_ser = phie_df[phie_df[vsh_col] > vsh_thresh][nphi_col_e].dropna().mean()
                        phie_df[f"{nphi_col_e}e"] = (
                            phie_df[nphi_col_e] - (vsh_ser * phi_nsh_ser)
                        ).round(4)
                        computed_cols.append(f"{nphi_col_e}e")
                        computed_stats[f"{nphi_col_e}e"] = phi_nsh_ser

                    st.session_state["df"]                   = phie_df
                    st.session_state["phie_success"]         = True
                    st.session_state["phie_computed_cols"]   = computed_cols
                    st.session_state["phie_computed_stats"]  = computed_stats
                    st.rerun()

            # ── Results after rerun ────────────────────────────
            if st.session_state.get("phie_success"):
                computed_cols  = st.session_state.get("phie_computed_cols", [])
                computed_stats = st.session_state.get("phie_computed_stats", {})

                for col_name in computed_cols:
                    phi_sh_val = computed_stats.get(col_name, 0.0)
                    st.success(
                        f"{col_name} calculated — φsh = {phi_sh_val:.4f} — saved to dataframe"
                    )
                    if col_name in st.session_state["df"].columns:
                        vals = st.session_state["df"][col_name].dropna()
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric(f"{col_name} min",  f"{vals.min():.3f}")
                        r2.metric(f"{col_name} max",  f"{vals.max():.3f}")
                        r3.metric(f"{col_name} mean", f"{vals.mean():.3f}")
                        r4.metric(f"{col_name} std",  f"{vals.std():.3f}")

                st.session_state["phie_success"] = False



        # ════════════════════════════════════════════════════
        #  WATER SATURATION
        # ════════════════════════════════════════════════════
        with st.expander("Water Saturation", expanded=False):

            # ── Model selector with formula ───────────────────
            st.markdown(
                '<div class="petro-section-header">Model</div>',
                unsafe_allow_html=True
            )

            mc1, mc2 = st.columns([1, 3])
            with mc1:
                model = st.radio(
                    "Model",
                    options=["Archie", "Simandoux"],
                    label_visibility="collapsed"
                )
            with mc2:
                with st.container(border=True):
                    if model == "Archie":
                        st.markdown("**Archie** — clean (shale-free) formations")
                        st.latex(
                            r"S_w = \left(\frac{a \cdot R_w}{\phi^m \cdot R_t}\right)^{\frac{1}{n}}"
                        )
                    else:
                        st.markdown("**Simandoux** — shaly formations")
                        st.latex(
                            r"S_w = \frac{a \cdot R_w}{2 \cdot \phi^m} "
                            r"\left[\sqrt{\left(\frac{V_{sh}}{R_{sh}}\right)^2 + "
                            r"\frac{4 \cdot \phi^m}{a \cdot R_w \cdot R_t}} "
                            r"- \frac{V_{sh}}{R_{sh}}\right]"
                        )

            st.markdown("---")

            # ── Log selectors ─────────────────────────────────
            st.markdown(
                '<div class="petro-section-header">Input Logs</div>',
                unsafe_allow_html=True
            )

            cur_df   = st.session_state["df"]
            all_cols = [c for c in cur_df.columns if c != depth_col]

            RT_ALIASES = ["RT", "LLD", "ILD", "RDEP", "RD", "RESD", "AT90", "M2RX", "RLA5", "RLA4"]
            rt_auto    = detect_curve(all_cols, RT_ALIASES)
            rt_index   = all_cols.index(rt_auto) if rt_auto in all_cols else 0

            phi_options = list(dict.fromkeys(
                [c for c in ["PHIT", "PHIE", "DPHI", "NPHI"] if c in cur_df.columns] + all_cols
            ))

            if model == "Archie":
                lc1, lc2 = st.columns(2)
                with lc1:
                    rt_col  = st.selectbox("Resistivity log (Rt)", options=all_cols,
                                        index=rt_index, key="sw_rt_col")
                with lc2:
                    phi_col = st.selectbox("Porosity log (φ)", options=phi_options,
                                        key="sw_phi_col_archie")
            else:
                lc1, lc2, lc3 = st.columns(3)
                with lc1:
                    rt_col  = st.selectbox("Resistivity log (Rt)", options=all_cols,
                                        index=rt_index, key="sw_rt_col")
                with lc2:
                    phi_col = st.selectbox("Porosity log (φ)", options=phi_options,
                                        key="sw_phi_col_sim")
                with lc3:
                    vsh_options = list(dict.fromkeys(
                        [c for c in ["VSH"] if c in cur_df.columns] + all_cols
                    ))
                    vsh_col_sw = st.selectbox("Volume of Shale (VSH)", options=vsh_options,
                                            key="sw_vsh_col")

            st.markdown("---")

            # ── Parameters ────────────────────────────────────
            st.markdown(
                '<div class="petro-section-header">Parameters</div>',
                unsafe_allow_html=True
            )

            if model == "Archie":
                pc1, pc2, pc3, pc4 = st.columns(4)
                with pc1:
                    rw = st.number_input("Rw (Ω·m)", value=0.5, format="%.4f",
                                        help="Formation water resistivity")
                with pc2:
                    a  = st.number_input("a — Tortuosity factor", value=1.0, format="%.3f")
                with pc3:
                    m  = st.number_input("m — Cementation exponent", value=2.0, format="%.3f")
                with pc4:
                    n  = st.number_input("n — Saturation exponent", value=2.0, format="%.3f")
            else:
                pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                with pc1:
                    rw  = st.number_input("Rw (Ω·m)", value=0.5, format="%.4f",
                                        help="Formation water resistivity")
                with pc2:
                    rsh = st.number_input("Rsh (Ω·m)", value=4.0, format="%.3f",
                                        help="Shale resistivity")
                with pc3:
                    a   = st.number_input("a — Tortuosity factor", value=1.0, format="%.3f")
                with pc4:
                    m   = st.number_input("m — Cementation exponent", value=2.0, format="%.3f")
                with pc5:
                    n   = st.number_input("n — Saturation exponent", value=2.0, format="%.3f")

            st.markdown("---")

            # ── Existing column notice ─────────────────────────
            sw_col_name = "SW_A" if model == "Archie" else "SW_S"
            if sw_col_name in st.session_state["df"].columns:
                ex_sw = st.session_state["df"][sw_col_name].dropna()
                e1, e2, e3 = st.columns(3)
                e1.metric(f"{sw_col_name} mean", f"{ex_sw.mean():.3f}")
                e2.metric(f"{sw_col_name} min",  f"{ex_sw.min():.3f}")
                e3.metric(f"{sw_col_name} max",  f"{ex_sw.max():.3f}")
                st.info(f"{sw_col_name} already exists. Click Calculate & Save to overwrite.")

            # ── Apply ─────────────────────────────────────────
            ba, _ = st.columns([2, 6])
            with ba:
                apply_sw = st.button(
                    "▶ Calculate & Save",
                    key="apply_sw",
                    type="primary",
                    use_container_width=True
                )

            if apply_sw:
                sw_df = st.session_state["df"].copy()
                phi   = sw_df[phi_col].replace(0, np.nan)
                rt    = sw_df[rt_col].replace(0, np.nan)

                if model == "Archie":
                    sw      = ((a * rw) / (phi ** m * rt)) ** (1 / n)
                    sw      = pd.Series(sw, index=sw_df.index).clip(0, 1)
                    sw_df["SW_A"] = sw.round(4)
                    out_col = "SW_A"
                else:
                    vsh   = sw_df[vsh_col_sw]
                    term1 = (vsh / rsh) ** 2
                    term2 = (4 * phi ** m) / (a * rw * rt)
                    sw    = (a * rw) / (2 * phi ** m) * (
                        np.sqrt(term1 + term2) - (vsh / rsh)
                    )
                    sw      = pd.Series(sw, index=sw_df.index).clip(0, 1)
                    sw_df["SW_S"] = sw.round(4)
                    out_col = "SW_S"

                st.session_state["df"] = sw_df
                sw_vals = sw_df[out_col].dropna()
                st.session_state["sw_success"] = {
                    "col"  : out_col,
                    "model": model,
                    "rt"   : rt_col,
                    "phi"  : phi_col,
                    "rw"   : rw,
                    "a"    : a, "m": m, "n": n,
                    "min"  : float(sw_vals.min()),
                    "max"  : float(sw_vals.max()),
                    "mean" : float(sw_vals.mean()),
                    "std"  : float(sw_vals.std()),
                }
                st.rerun()

            # ── Results after rerun ────────────────────────────
            if st.session_state.get("sw_success"):
                r = st.session_state["sw_success"]

                r1, r2, r3, r4 = st.columns(4)
                r1.metric(f"{r['col']} min",  f"{r['min']:.3f}")
                r2.metric(f"{r['col']} max",  f"{r['max']:.3f}")
                r3.metric(f"{r['col']} mean", f"{r['mean']:.3f}")
                r4.metric(f"{r['col']} std",  f"{r['std']:.3f}")

                st.success(
                    f"{r['col']} saved — "
                    f"Model: {r['model']}  |  Rt: {r['rt']}  |  φ: {r['phi']}  |  "
                    f"Rw: {r['rw']}  |  a: {r['a']}  |  m: {r['m']}  |  n: {r['n']}"
                )
                st.session_state["sw_success"] = None


        # ════════════════════════════════════════════════════
        #  NET-TO-GROSS (NTG)
        # ════════════════════════════════════════════════════
        with st.expander("Net to Gross (NTG)", expanded=False):

            # ── Definition & formula ──────────────────────────
            fc1, fc2 = st.columns([2, 1])
            with fc1:
                with st.container(border=True):
                    st.markdown(
                        "The **Net-to-Gross ratio** quantifies the proportion of a reservoir "
                        "interval that meets minimum quality standards for hydrocarbon production."
                    )
                    st.markdown(
                        "- **Net Pay:** Cumulative thickness of layers satisfying all cutoffs\n"
                        "- **Gross:** Total vertical thickness of the evaluated interval"
                    )
            with fc2:
                with st.container(border=True):
                    st.latex(
                        r"\text{NTG} = \frac{\sum \Delta h_{\text{net}}}{\sum \Delta h_{\text{gross}}}"
                    )

            st.markdown("---")

            # ── Log selectors ─────────────────────────────────
            st.markdown(
                '<div class="petro-section-header">Input Logs</div>',
                unsafe_allow_html=True
            )

            cur_df   = st.session_state["df"]
            all_cols = [c for c in cur_df.columns if c != depth_col]

            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                vsh_options = list(dict.fromkeys(
                    [c for c in ["VSH"] if c in cur_df.columns] + all_cols
                ))
                ntg_vsh_col = st.selectbox("Volume of Shale (VSH)", options=vsh_options,
                                        key="ntg_vsh_col")
            with lc2:
                phi_options = list(dict.fromkeys(
                    [c for c in ["PHIE", "PHIT", "DPHI", "NPHI"] if c in cur_df.columns] + all_cols
                ))
                ntg_phi_col = st.selectbox("Porosity (φ)", options=phi_options,
                                        key="ntg_phi_col")
            with lc3:
                sw_options = list(dict.fromkeys(
                    [c for c in ["SW_A", "SW_S"] if c in cur_df.columns] + all_cols
                ))
                ntg_sw_col = st.selectbox("Water Saturation (SW)", options=sw_options,
                                        key="ntg_sw_col")

            st.markdown("---")

            # ── Cutoffs & depth step ───────────────────────────
            st.markdown(
                '<div class="petro-section-header">Cutoffs & Depth Step</div>',
                unsafe_allow_html=True
            )

            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                vsh_cutoff = st.number_input(
                    "VSH cutoff (max)", value=0.40,
                    min_value=0.0, max_value=1.0, format="%.3f",
                    help="Gross zone: VSH < this value"
                )
            with cc2:
                phi_cutoff = st.number_input(
                    "Porosity cutoff (min)", value=0.05,
                    min_value=0.0, max_value=1.0, format="%.3f",
                    help="Net pay: φ > this value"
                )
            with cc3:
                sw_cutoff = st.number_input(
                    "Sw cutoff (max)", value=0.70,
                    min_value=0.0, max_value=1.0, format="%.3f",
                    help="Net pay: SW < this value"
                )
            with cc4:
                delta_h = st.number_input(
                    "Depth step Δh (m)", value=0.1524,
                    min_value=0.0001, format="%.4f",
                    help="Sample interval — 0.1524 m = 0.5 ft"
                )

            st.markdown("---")

            # ── Existing column notice ─────────────────────────
            if "NTG_FLAG" in st.session_state["df"].columns:
                existing_net   = int(st.session_state["df"]["NTG_FLAG"].sum())
                existing_gross = int((st.session_state["df"][ntg_vsh_col] < vsh_cutoff).sum())
                existing_ntg   = (existing_net * delta_h) / (existing_gross * delta_h) \
                                if existing_gross > 0 else 0.0
                en1, en2, en3 = st.columns(3)
                en1.metric("Gross Thickness (m)", f"{existing_gross * delta_h:.2f}")
                en2.metric("Net Thickness (m)",   f"{existing_net   * delta_h:.2f}")
                en3.metric("NTG",                 f"{existing_ntg:.4f}")
                st.info("NTG_FLAG already exists. Click Calculate & Save to overwrite.")

            # ── Apply ─────────────────────────────────────────
            ba, _ = st.columns([2, 6])
            with ba:
                apply_ntg = st.button(
                    "▶ Calculate & Save",
                    key="apply_ntg",
                    type="primary",
                    use_container_width=True
                )

            if apply_ntg:
                st.session_state["ntg_success"] = None

                ntg_df = st.session_state["df"].copy()

                gross_mask = ntg_df[ntg_vsh_col] < vsh_cutoff
                net_mask   = (
                    (ntg_df[ntg_vsh_col] < vsh_cutoff) &
                    (ntg_df[ntg_phi_col] > phi_cutoff)  &
                    (ntg_df[ntg_sw_col]  < sw_cutoff)
                )

                ntg_df["NTG_FLAG"] = net_mask.astype(int)
                st.session_state["df"] = ntg_df

                gross_count     = int(gross_mask.sum())
                net_count       = int(net_mask.sum())
                gross_thickness = gross_count * delta_h
                net_thickness   = net_count   * delta_h
                ntg_ratio       = net_thickness / gross_thickness if gross_thickness > 0 else 0.0

                st.session_state["ntg_success"] = {
                    "net_count"      : net_count,
                    "gross_count"    : gross_count,
                    "net_thickness"  : net_thickness,
                    "gross_thickness": gross_thickness,
                    "ntg"            : ntg_ratio,
                    "vsh_col"        : ntg_vsh_col,
                    "phi_col"        : ntg_phi_col,
                    "sw_col"         : ntg_sw_col,
                    "vsh_cut"        : vsh_cutoff,
                    "phi_cut"        : phi_cutoff,
                    "sw_cut"         : sw_cutoff,
                    "delta_h"        : delta_h,
                }
                st.rerun()

            # ── Results after rerun ────────────────────────────
            if st.session_state.get("ntg_success"):
                r = st.session_state["ntg_success"]

                r1, r2, r3 = st.columns(3)
                r1.metric("Gross Thickness (m)", f"{r['gross_thickness']:.2f}")
                r2.metric("Net Thickness (m)",   f"{r['net_thickness']:.2f}")
                r3.metric("NTG",                 f"{r['ntg']:.4f}")

                st.success(
                    f"NTG_FLAG saved — "
                    f"VSH < {r['vsh_cut']}  |  φ > {r['phi_cut']}  |  SW < {r['sw_cut']}  |  "
                    f"Δh = {r['delta_h']} m"
                )
                st.session_state["ntg_success"] = None



    # ════════════════════════════════════════════════════════
    #  TAB 6 — Elastic Properties
    # ════════════════════════════════════════════════════════
    with tab6:

        # ════════════════════════════════════════════════════
        #  EXPANDER 1 — COMPRESSIONAL VELOCITY (Vp)
        # ════════════════════════════════════════════════════
        with st.expander("Compressional Velocity", expanded=True):

            # ── Initialize session state ──────────────────────
            if "vp_success" not in st.session_state:
                st.session_state["vp_success"] = None
            if "vp_source" not in st.session_state:
                st.session_state["vp_source"]  = None   # "DTC" | "Wyllie" | "RHG" | "Gardner" | "ML"

            cur_df   = st.session_state["df"]
            all_cols = [c for c in cur_df.columns if c != depth_col]

            DT_ALIASES = ["DT", "DTC", "DTCO", "AC", "DT4P", "DTHM", "SLOWNESS"]
            dt_auto    = detect_curve(all_cols, DT_ALIASES)

            # ── DTC availability toggle ───────────────────────
            dtc_available = st.radio(
                "Is a compressional sonic log available?",
                options=["Yes", "No"],
                index=0 if dt_auto else 1,
                horizontal=True,
                key="ep_dtc_available"
            )

            st.markdown("---")

            # ════════════════════════════════════════════════
            #  BRANCH A — DTC AVAILABLE
            # ════════════════════════════════════════════════
            if dtc_available == "Yes":

                st.markdown(
                    '<div class="petro-section-header">DTC → Vp Conversion</div>',
                    unsafe_allow_html=True
                )

                with st.container(border=True):
                    st.markdown("**Conversion formula**")
                    st.latex(r"V_P = \frac{C}{DT}")
                    st.caption(
                        "Where C is the unit conversion constant — "
                        "1,000,000 for µs/m (→ m/s), 3,280,840 for µs/ft (→ m/s), etc."
                    )

                st.markdown("")

                ba1, ba2, ba3 = st.columns([2, 1.5, 1.5])
                with ba1:
                    dt_options = list(dict.fromkeys(
                        ([dt_auto] if dt_auto else []) + all_cols
                    ))
                    dt_col = st.selectbox(
                        "DTC log",
                        options=[None] + dt_options,
                        index=(1 if dt_auto else 0),
                        format_func=lambda v: "— select DTC —" if v is None else v,
                        key="ep_dt_col"
                    )

                # Unit conversion map: input unit → factor to get m/s
                DT_UNIT_MAP = {
                    "µs/ft" : 3.280840e5,
                    "µs/m " : 1.000000e6,
                    "ms/m " : 1.000000e3,
                    "ms/ft" : 3.280840e2,
                    "s/m " : 1.000000e0,
                    "s/ft" : 3.280840e-1,
                    "m/s"  : None,
                    "km/s "  : None,
                    "ft/s "  : None,
                }

                with ba2:
                    dt_unit_label = st.selectbox(
                        "Input unit",
                        options=list(DT_UNIT_MAP.keys()),
                        index=0,
                        key="ep_dt_unit"
                    )
                with ba3:
                    vp_out_unit = st.selectbox(
                        "Output unit",
                        options=["m/s", "km/s", "ft/s"],
                        index=0,
                        key="ep_vp_out_unit"
                    )

                # ── Sanity check & preview ────────────────────
                if dt_col:
                    dt_data = cur_df[dt_col].dropna()
                    pv1, pv2, pv3 = st.columns(3)
                    pv1.metric("DTC min",  f"{dt_data.min():.2f}")
                    pv2.metric("DTC max",  f"{dt_data.max():.2f}")
                    pv3.metric("DTC mean", f"{dt_data.mean():.2f}")

                    # Expected range check for slowness units
                    EXPECTED_RANGES = {
                        "µs/ft  →  m/s" : (40,   250),
                        "µs/m   →  m/s" : (130,  820),
                        "ms/m   →  m/s" : (0.13, 0.82),
                        "ms/ft  →  m/s" : (0.04, 0.25),
                        "s/m    →  m/s" : (1.3e-4, 8.2e-4),
                        "s/ft   →  m/s" : (4.0e-5, 2.5e-4),
                    }
                    if dt_unit_label in EXPECTED_RANGES:
                        lo, hi = EXPECTED_RANGES[dt_unit_label]
                        if dt_data.min() < lo or dt_data.max() > hi:
                            st.warning(
                                f"DTC values outside expected range for "
                                f"{dt_unit_label.split()[0]} ({lo} – {hi}). "
                                "Verify unit selection."
                            )

                st.markdown("---")

                # ── Apply ─────────────────────────────────────
                if dt_col is None:
                    st.warning("Select a DTC log to proceed.")
                else:
                    cb1, _ = st.columns([2, 6])
                    with cb1:
                        apply_vp_dtc = st.button(
                            "▶ Convert & Save",
                            key="apply_vp_dtc",
                            type="primary",
                            use_container_width=True
                        )

                    if apply_vp_dtc:
                        vp_df    = st.session_state["df"].copy()
                        dt_series = vp_df[dt_col].copy().replace(0, np.nan)

                        factor = DT_UNIT_MAP[dt_unit_label]

                        if factor is not None:
                            # Slowness → velocity in m/s
                            vp_ms = factor / dt_series
                        else:
                            # Input is already velocity — normalize
                            if "km/s" in dt_unit_label:
                                vp_ms = dt_series * 1000.0
                            elif "ft/s" in dt_unit_label:
                                vp_ms = dt_series * 0.3048
                            else:
                                vp_ms = dt_series   # already m/s

                        # Convert to desired output unit
                        if vp_out_unit == "km/s":
                            vp_out = (vp_ms / 1000.0).round(4)
                        elif vp_out_unit == "ft/s":
                            vp_out = (vp_ms / 0.3048).round(4)
                        else:
                            vp_out = vp_ms.round(4)

                        # vp_df["VP"] = vp_out
                        # st.session_state["df"]       = vp_df

                        if "VP" in vp_df.columns:
                            vp_df.drop(columns=["VP"], inplace=True)
                        vp_df["VP"] = vp_out
                        st.session_state["df"]       = vp_df

                        st.session_state["vp_source"] = "DTC"

                        vp_vals = vp_df["VP"].dropna()
                        st.session_state["vp_success"] = {
                            "col"    : "VP",
                            "source" : "DTC",
                            "unit"   : vp_out_unit,
                            "min"    : float(vp_vals.min()),
                            "max"    : float(vp_vals.max()),
                            "mean"   : float(vp_vals.mean()),
                        }
                        st.rerun()

            # ════════════════════════════════════════════════
            #  BRANCH B — DTC NOT AVAILABLE
            # ════════════════════════════════════════════════
            else:
                st.markdown(
                    '<div class="petro-section-header">Vp Estimation Method</div>',
                    unsafe_allow_html=True
                )

                method = st.radio(
                    "Estimation method",
                    options=[
                        "Wyllie Time Average",
                        "Raymer-Hunt-Gardner",
                        "Gardner from RHOB",
                        "ML from Conventional Logs",
                    ],
                    label_visibility="collapsed",
                    key="ep_vp_method"
                )

                st.markdown("---")

                # ─────────────────────────────────────────────
                #  METHOD 1 — WYLLIE TIME AVERAGE
                # ─────────────────────────────────────────────
                if method == "Wyllie Time Average":

                    with st.container(border=True):
                        st.markdown("**Wyllie Time Average**")
                        st.latex(
                            r"V_P = \frac{1}{\dfrac{\phi}{V_{fl}} + \dfrac{1-\phi}{V_{ma}}}"
                        )
                        st.caption(
                            "Suitable for consolidated, fluid-saturated formations. "
                            "Overestimates Vp at high porosities."
                        )

                    st.markdown("")

                    with st.expander("Reference values"):
                        st.markdown("**Formation / Mineral**")
                        st.dataframe(pd.DataFrame([
                            {"Formation / Mineral": "Sandstone (compact/quartz)", "Δtma (µs/ft)": "51.3–55.6", "Vma (ft/s)": "18,000–19,500", "Vma (m/s)": "5,490–5,950"},
                            {"Formation / Mineral": "Limestone (calcite)",        "Δtma (µs/ft)": "43.5–47.6", "Vma (ft/s)": "21,000–23,000", "Vma (m/s)": "6,400–7,010"},
                            {"Formation / Mineral": "Dolomite",                   "Δtma (µs/ft)": "38.5–43.5", "Vma (ft/s)": "23,000–26,000", "Vma (m/s)": "7,010–7,920"},
                            {"Formation / Mineral": "Anhydrite",                  "Δtma (µs/ft)": "50.0",      "Vma (ft/s)": "20,000",        "Vma (m/s)": "6,096"},
                            {"Formation / Mineral": "Halite (Salt)",              "Δtma (µs/ft)": "66.7",      "Vma (ft/s)": "15,000",        "Vma (m/s)": "4,572"},
                            {"Formation / Mineral": "Shale",                      "Δtma (µs/ft)": "60–170",    "Vma (ft/s)": "5,880–16,660",  "Vma (m/s)": "1,790–5,805"},
                            {"Formation / Mineral": "Bituminous Coal",            "Δtma (µs/ft)": "100–140",   "Vma (ft/s)": "7,140–10,000",  "Vma (m/s)": "2,180–3,050"},
                            {"Formation / Mineral": "Lignite",                    "Δtma (µs/ft)": "140–180",   "Vma (ft/s)": "5,560–7,140",   "Vma (m/s)": "1,690–2,180"},
                            {"Formation / Mineral": "Casing (steel)",             "Δtma (µs/ft)": "57.1",      "Vma (ft/s)": "17,500",        "Vma (m/s)": "5,334"},
                        ]), use_container_width=True, hide_index=True)

                        st.markdown("**Fluid**")
                        st.dataframe(pd.DataFrame([
                            {"Fluid Type": "Water (200,000 ppm NaCl, 15 psi)", "Δtf (µs/ft)": "180.5", "Vf (ft/s)": "5,540", "Vf (m/s)": "1,690"},
                            {"Fluid Type": "Water (150,000 ppm NaCl, 15 psi)", "Δtf (µs/ft)": "186.0", "Vf (ft/s)": "5,380", "Vf (m/s)": "1,640"},
                            {"Fluid Type": "Water (100,000 ppm NaCl, 15 psi)", "Δtf (µs/ft)": "192.3", "Vf (ft/s)": "5,200", "Vf (m/s)": "1,580"},
                            {"Fluid Type": "Freshwater (mud filtrate)",        "Δtf (µs/ft)": "~189",  "Vf (ft/s)": "5,290", "Vf (m/s)": "1,610"},
                            {"Fluid Type": "Oil",                              "Δtf (µs/ft)": "238",   "Vf (ft/s)": "4,200", "Vf (m/s)": "1,280"},
                            {"Fluid Type": "Methane/Gas (15 psi)",             "Δtf (µs/ft)": "626",   "Vf (ft/s)": "1,600", "Vf (m/s)": "490"},
                        ]), use_container_width=True, hide_index=True)

                    st.markdown("")

                    wc1, _ = st.columns(2)
                    with wc1:
                        phi_options = list(dict.fromkeys(
                            [c for c in ["PHIE", "PHIT", "DPHI", "NPHI"] if c in cur_df.columns]
                            + all_cols
                        ))
                        wyllie_phi = st.selectbox(
                            "Porosity log (φ)",
                            options=[None] + phi_options,
                            format_func=lambda v: "Select porosity" if v is None else v,
                            key="ep_wyllie_phi"
                        )

                    wp1, wp2, wp3 = st.columns(3)
                    with wp1:
                        w_vma = st.number_input(
                            "V_matrix (m/s)",
                            value=None,
                            format="%.1f",
                            placeholder="e.g. 5490.0",
                            key="ep_wyllie_vma"
                        )
                    with wp2:
                        w_vfl = st.number_input(
                            "V_fluid (m/s)",
                            value=None,
                            format="%.1f",
                            placeholder="e.g. 1500.0",
                            key="ep_wyllie_vfl"
                        )
                    with wp3:
                        wyllie_out = st.selectbox(
                            "Output unit",
                            options=["m/s", "km/s", "ft/s"],
                            key="ep_wyllie_out"
                        )

                    st.markdown("---")

                    if wyllie_phi is None or w_vma is None or w_vfl is None:
                        st.warning("Select a porosity log and enter matrix and fluid velocities.")
                    else:
                        wbtn, _ = st.columns([2, 6])
                        with wbtn:
                            apply_wyllie = st.button(
                                "Calculate & Save",
                                key="apply_wyllie",
                                type="primary",
                                use_container_width=True
                            )

                        if apply_wyllie:
                            w_df  = st.session_state["df"].copy()
                            phi   = w_df[wyllie_phi].replace(0, np.nan).clip(0.001, 0.999)
                            vp_ms = 1.0 / ((phi / w_vfl) + ((1 - phi) / w_vma))

                            if wyllie_out == "km/s":
                                vp_out = (vp_ms / 1000.0).round(4)
                            elif wyllie_out == "ft/s":
                                vp_out = (vp_ms / 0.3048).round(4)
                            else:
                                vp_out = vp_ms.round(4)

                            # w_df["VP_EST"] = vp_out
                            # st.session_state["df"]        = w_df

                            if "VP_EST" in w_df.columns:
                                w_df.drop(columns=["VP_EST"], inplace=True)
                            w_df["VP_EST"] = vp_out
                            st.session_state["df"]        = w_df

                            st.session_state["vp_source"] = "Wyllie"

                            vp_vals = w_df["VP_EST"].dropna()
                            st.session_state["vp_success"] = {
                                "col"    : "VP_EST",
                                "source" : "Wyllie Time Average",
                                "unit"   : wyllie_out,
                                "min"    : float(vp_vals.min()),
                                "max"    : float(vp_vals.max()),
                                "mean"   : float(vp_vals.mean()),
                            }
                            st.rerun()


                # ─────────────────────────────────────────────
                #  METHOD 2 — RAYMER-HUNT-GARDNER
                # ─────────────────────────────────────────────
                elif method == "Raymer-Hunt-Gardner":

                    with st.container(border=True):
                        st.markdown("**Raymer-Hunt-Gardner**")
                        st.latex(
                            r"V_P = (1 - \phi)^2 \cdot V_{ma} + \phi \cdot V_{fl}"
                        )
                        st.caption(
                            "More accurate than Wyllie at higher porosities (φ > 0.2). "
                            "Suitable for unconsolidated or poorly cemented sands."
                        )

                    st.markdown("")

                    with st.expander("Reference values"):
                        st.markdown("**Formation / Mineral**")
                        st.dataframe(pd.DataFrame([
                            {"Formation / Mineral": "Sandstone (compact/quartz)", "Δtma (µs/ft)": "51.3–55.6", "Vma (ft/s)": "18,000–19,500", "Vma (m/s)": "5,490–5,950"},
                            {"Formation / Mineral": "Limestone (calcite)",        "Δtma (µs/ft)": "43.5–47.6", "Vma (ft/s)": "21,000–23,000", "Vma (m/s)": "6,400–7,010"},
                            {"Formation / Mineral": "Dolomite",                   "Δtma (µs/ft)": "38.5–43.5", "Vma (ft/s)": "23,000–26,000", "Vma (m/s)": "7,010–7,920"},
                            {"Formation / Mineral": "Anhydrite",                  "Δtma (µs/ft)": "50.0",      "Vma (ft/s)": "20,000",        "Vma (m/s)": "6,096"},
                            {"Formation / Mineral": "Halite (Salt)",              "Δtma (µs/ft)": "66.7",      "Vma (ft/s)": "15,000",        "Vma (m/s)": "4,572"},
                            {"Formation / Mineral": "Shale",                      "Δtma (µs/ft)": "60–170",    "Vma (ft/s)": "5,880–16,660",  "Vma (m/s)": "1,790–5,805"},
                            {"Formation / Mineral": "Bituminous Coal",            "Δtma (µs/ft)": "100–140",   "Vma (ft/s)": "7,140–10,000",  "Vma (m/s)": "2,180–3,050"},
                            {"Formation / Mineral": "Lignite",                    "Δtma (µs/ft)": "140–180",   "Vma (ft/s)": "5,560–7,140",   "Vma (m/s)": "1,690–2,180"},
                            {"Formation / Mineral": "Casing (steel)",             "Δtma (µs/ft)": "57.1",      "Vma (ft/s)": "17,500",        "Vma (m/s)": "5,334"},
                        ]), use_container_width=True, hide_index=True)

                        st.markdown("**Fluid**")
                        st.dataframe(pd.DataFrame([
                            {"Fluid Type": "Water (200,000 ppm NaCl, 15 psi)", "Δtf (µs/ft)": "180.5", "Vf (ft/s)": "5,540", "Vf (m/s)": "1,690"},
                            {"Fluid Type": "Water (150,000 ppm NaCl, 15 psi)", "Δtf (µs/ft)": "186.0", "Vf (ft/s)": "5,380", "Vf (m/s)": "1,640"},
                            {"Fluid Type": "Water (100,000 ppm NaCl, 15 psi)", "Δtf (µs/ft)": "192.3", "Vf (ft/s)": "5,200", "Vf (m/s)": "1,580"},
                            {"Fluid Type": "Freshwater (mud filtrate)",        "Δtf (µs/ft)": "~189",  "Vf (ft/s)": "5,290", "Vf (m/s)": "1,610"},
                            {"Fluid Type": "Oil",                              "Δtf (µs/ft)": "238",   "Vf (ft/s)": "4,200", "Vf (m/s)": "1,280"},
                            {"Fluid Type": "Methane/Gas (15 psi)",             "Δtf (µs/ft)": "626",   "Vf (ft/s)": "1,600", "Vf (m/s)": "490"},
                        ]), use_container_width=True, hide_index=True)

                    st.markdown("")

                    rc1, _ = st.columns(2)
                    with rc1:
                        phi_options = list(dict.fromkeys(
                            [c for c in ["PHIE", "PHIT", "DPHI", "NPHI"] if c in cur_df.columns]
                            + all_cols
                        ))
                        rhg_phi = st.selectbox(
                            "Porosity log (φ)",
                            options=[None] + phi_options,
                            format_func=lambda v: "Select porosity" if v is None else v,
                            key="ep_rhg_phi"
                        )

                    rp1, rp2, rp3 = st.columns(3)
                    with rp1:
                        r_vma = st.number_input(
                            "V_matrix (m/s)",
                            value=None,
                            format="%.1f",
                            placeholder="e.g. 5490.0",
                            key="ep_rhg_vma"
                        )
                    with rp2:
                        r_vfl = st.number_input(
                            "V_fluid (m/s)",
                            value=None,
                            format="%.1f",
                            placeholder="e.g. 1500.0",
                            key="ep_rhg_vfl"
                        )
                    with rp3:
                        rhg_out = st.selectbox(
                            "Output unit",
                            options=["m/s", "km/s", "ft/s"],
                            key="ep_rhg_out"
                        )

                    st.markdown("---")

                    if rhg_phi is None or r_vma is None or r_vfl is None:
                        st.warning("Select a porosity log and enter matrix and fluid velocities.")
                    else:
                        rbtn, _ = st.columns([2, 6])
                        with rbtn:
                            apply_rhg = st.button(
                                "Calculate & Save",
                                key="apply_rhg",
                                type="primary",
                                use_container_width=True
                            )

                        if apply_rhg:
                            r_df  = st.session_state["df"].copy()
                            phi   = r_df[rhg_phi].replace(0, np.nan).clip(0.001, 0.999)
                            vp_ms = ((1 - phi) ** 2) * r_vma + phi * r_vfl

                            if rhg_out == "km/s":
                                vp_out = (vp_ms / 1000.0).round(4)
                            elif rhg_out == "ft/s":
                                vp_out = (vp_ms / 0.3048).round(4)
                            else:
                                vp_out = vp_ms.round(4)

                            # r_df["VP_EST"] = vp_out
                            # st.session_state["df"]        = r_df

                            if "VP_EST" in r_df.columns:
                                r_df.drop(columns=["VP_EST"], inplace=True)
                            r_df["VP_EST"] = vp_out
                            st.session_state["df"]        = r_df

                            st.session_state["vp_source"] = "RHG"

                            vp_vals = r_df["VP_EST"].dropna()
                            st.session_state["vp_success"] = {
                                "col"    : "VP_EST",
                                "source" : "Raymer-Hunt-Gardner",
                                "unit"   : rhg_out,
                                "min"    : float(vp_vals.min()),
                                "max"    : float(vp_vals.max()),
                                "mean"   : float(vp_vals.mean()),
                            }
                            st.rerun()

                # ─────────────────────────────────────────────
                #  METHOD 3 — GARDNER FROM RHOB
                # ─────────────────────────────────────────────
                elif method == "Gardner from RHOB":

                    with st.container(border=True):
                        st.markdown("**Gardner's Relation — inverted for Vp**")
                        st.latex(r"\rho_b = \alpha \cdot V_P^{\,\beta}")
                        st.latex(r"V_P = \left(\frac{\rho_b}{\alpha}\right)^{1/\beta}")
                        st.caption(
                            "Gardner (1974). α and β are lithology-dependent empirical constants. "
                            "Basin-specific values improve accuracy over generic defaults."
                        )

                    st.markdown("")

                    with st.expander("Reference values — Gardner constants by lithology"):
                        st.dataframe(pd.DataFrame([
                            {"Lithology": "General sedimentary", "α (m/s)": 0.3100, "β": 0.25},
                            {"Lithology": "Shale",               "α (m/s)": 0.3125, "β": 0.25},
                            {"Lithology": "Sandstone",           "α (m/s)": 0.2920, "β": 0.25},
                            {"Lithology": "Limestone",           "α (m/s)": 0.3150, "β": 0.25},
                            {"Lithology": "Dolomite",            "α (m/s)": 0.3400, "β": 0.25},
                        ]), use_container_width=True, hide_index=True)
                        st.caption(
                            "α values above assume Vp in m/s and ρ in g/cc. "
                            "If Vp is in ft/s, use α = 0.23 (general sedimentary)."
                        )

                    st.markdown("")

                    RHO_ALIASES = ["RHOB", "RHOZ", "DEN", "ZDEN", "HRHO", "BDCF"]
                    rho_auto    = detect_curve(all_cols, RHO_ALIASES)
                    rho_options = list(dict.fromkeys(
                        ([rho_auto] if rho_auto else []) + all_cols
                    ))

                    gc1, gc2 = st.columns(2)
                    with gc1:
                        gard_rho = st.selectbox(
                            "RHOB log",
                            options=[None] + rho_options,
                            index=(1 if rho_auto else 0),
                            format_func=lambda v: "Select RHOB" if v is None else v,
                            key="ep_gard_rho"
                        )
                    with gc2:
                        gard_out = st.selectbox(
                            "Output unit",
                            options=["m/s", "km/s", "ft/s"],
                            key="ep_gard_out"
                        )

                    gp1, gp2 = st.columns(2)
                    with gp1:
                        gard_alpha = st.number_input(
                            "α (alpha)",
                            value=None,
                            format="%.4f",
                            placeholder="e.g. 0.3100",
                            key="ep_gard_alpha"
                        )
                    with gp2:
                        gard_beta = st.number_input(
                            "β (beta)",
                            value=None,
                            format="%.4f",
                            placeholder="e.g. 0.25",
                            key="ep_gard_beta"
                        )

                    if gard_rho:
                        rho_data = cur_df[gard_rho].dropna()
                        gm1, gm2, gm3 = st.columns(3)
                        gm1.metric("RHOB min",  f"{rho_data.min():.3f} g/cc")
                        gm2.metric("RHOB max",  f"{rho_data.max():.3f} g/cc")
                        gm3.metric("RHOB mean", f"{rho_data.mean():.3f} g/cc")

                        if rho_data.min() < 1.5 or rho_data.max() > 3.2:
                            st.warning(
                                "RHOB values outside expected range (1.5–3.2 g/cc). "
                                "Verify log or check for bad hole intervals."
                            )

                    st.markdown("---")

                    if gard_rho is None or gard_alpha is None or gard_beta is None:
                        st.warning("Select a RHOB log and enter α and β values.")
                    else:
                        gbtn, _ = st.columns([2, 6])
                        with gbtn:
                            apply_gardner = st.button(
                                "Calculate & Save",
                                key="apply_gardner",
                                type="primary",
                                use_container_width=True
                            )

                        if apply_gardner:
                            g_df  = st.session_state["df"].copy()
                            rho   = g_df[gard_rho].replace(0, np.nan)
                            vp_ms = (rho / gard_alpha) ** (1.0 / gard_beta)

                            if gard_out == "km/s":
                                vp_out = (vp_ms / 1000.0).round(4)
                            elif gard_out == "ft/s":
                                vp_out = (vp_ms / 0.3048).round(4)
                            else:
                                vp_out = vp_ms.round(4)

                            # g_df["VP_EST"] = vp_out
                            # st.session_state["df"]        = g_df

                            if "VP_EST" in g_df.columns:
                                g_df.drop(columns=["VP_EST"], inplace=True)
                            g_df["VP_EST"] = vp_out
                            st.session_state["df"]        = g_df

                            st.session_state["vp_source"] = "Gardner"

                            vp_vals = g_df["VP_EST"].dropna()
                            st.session_state["vp_success"] = {
                                "col"    : "VP_EST",
                                "source" : "Gardner from RHOB",
                                "unit"   : gard_out,
                                "min"    : float(vp_vals.min()),
                                "max"    : float(vp_vals.max()),
                                "mean"   : float(vp_vals.mean()),
                            }
                            st.rerun()

                # ─────────────────────────────────────────────
                #  METHOD 4 — ML FROM CONVENTIONAL LOGS
                # ─────────────────────────────────────────────
                elif method == "ML from Conventional Logs":

                    with st.container(border=True):
                        st.markdown("**ML-Based DTC Prediction**")
                        st.caption(
                            "Predicts DTC for the current well using a model trained on a "
                            "separate well from the same field that has measured DTC. "
                            "Only conventional logs (GR, RHOB, NPHI, resistivity, etc.) "
                            "are used as features — DTC is the prediction target."
                        )

                    st.markdown("")

                    # ── Training well upload ──────────────────────────
                    st.markdown(
                        '<div class="petro-section-header">Training Well</div>',
                        unsafe_allow_html=True
                    )

                    train_las_file = st.file_uploader(
                        "Upload a LAS file from a nearby well with measured DTC",
                        type=["las", "LAS"],
                        key="ep_ml_train_las"
                    )

                    train_df = None
                    if train_las_file:
                        try:
                            import io
                            import lasio

                            file_content = train_las_file.read()
                            train_las    = lasio.read(io.StringIO(file_content.decode("utf-8", errors="replace")))
                            train_df     = train_las.df().reset_index()
                            train_df.columns = [c.upper() for c in train_df.columns]

                            DT_ALIASES_ML = ["DT", "DTC", "DTCO", "AC", "DT4P", "DTHM", "SLOWNESS"]
                            dtc_present   = any(c in train_df.columns for c in DT_ALIASES_ML)

                            tc1, tc2, tc3 = st.columns(3)
                            tc1.metric("Rows",        len(train_df))
                            tc2.metric("Curves",      len(train_df.columns))
                            tc3.metric("DTC present", "Yes" if dtc_present else "No")

                        except Exception as e:
                            st.error(f"Failed to read training LAS file: {e}")
                            train_df = None

                    if train_df is not None:

                        # ── Target & depth column ─────────────────────────
                        st.markdown(
                            '<div class="petro-section-header">Feature & Target Configuration</div>',
                            unsafe_allow_html=True
                        )

                        train_all_cols = list(train_df.columns)
                        dt_target_auto = detect_curve(train_all_cols, DT_ALIASES_ML)

                        fc1, fc2 = st.columns(2)
                        with fc1:
                            ml_target = st.selectbox(
                                "Target log (DTC — training well only)",
                                options=train_all_cols,
                                index=(
                                    train_all_cols.index(dt_target_auto)
                                    if dt_target_auto in train_all_cols else 0
                                ),
                                key="ep_ml_target"
                            )
                        with fc2:
                            ml_depth_col_train = st.selectbox(
                                "Depth column (training well)",
                                options=train_all_cols,
                                index=0,
                                key="ep_ml_depth_train"
                            )

                        # Verify target has data
                        if train_df[ml_target].notna().sum() == 0:
                            st.error(
                                f"{ml_target} has no valid values in the training well. "
                                "Select a different target log or upload a different LAS file."
                            )
                            st.stop()

                        # ── Feature selection ─────────────────────────────
                        ALWAYS_EXCLUDE = {
                            ml_target, ml_depth_col_train,
                            "DT2", "DTSM", "DTS", "DTSH", "DTSV",
                            "SPHI", "PR", "VPVS", "VP", "VS",
                            "VP_EST", "AI", "SI", "DTCO_ML"
                        }

                        candidate_train_features = [
                            c for c in train_df.columns if c not in ALWAYS_EXCLUDE
                        ]
                        current_well_cols  = list(st.session_state["df"].columns)
                        shared_by_name     = [c for c in candidate_train_features if c in current_well_cols]
                        missing_in_current = [c for c in candidate_train_features if c not in current_well_cols]

                        st.markdown("##### Feature Log Pairing")
                        st.caption(
                            "Select one log from the **training well** and pair it with the "
                            "corresponding log in the **current well**. "
                            "Use this to handle mnemonic mismatches (e.g. HCGR ↔ EGR)."
                        )

                        # Column headers
                        hc1, hc2, hc3 = st.columns([5, 5, 1])
                        hc1.caption("**Training well log**")
                        hc2.caption("**Current well log (paired)**")

                        # ── Dynamic feature-pairing table ────────────────
                        if "ep_ml_feature_pairs" not in st.session_state:
                            st.session_state["ep_ml_feature_pairs"] = [
                                {"train": c, "current": c} for c in shared_by_name
                            ]

                        pairs     = st.session_state["ep_ml_feature_pairs"]
                        to_remove = []

                        for i, pair in enumerate(pairs):
                            pc1, pc2, pc3 = st.columns([5, 5, 1])
                            with pc1:
                                new_train = st.selectbox(
                                    f"Training log #{i+1}",
                                    options=candidate_train_features,
                                    index=(
                                        candidate_train_features.index(pair["train"])
                                        if pair["train"] in candidate_train_features else 0
                                    ),
                                    key=f"ep_ml_pair_train_{i}",
                                    label_visibility="collapsed"
                                )
                            with pc2:
                                new_current = st.selectbox(
                                    f"Current well log #{i+1}",
                                    options=current_well_cols,
                                    index=(
                                        current_well_cols.index(pair["current"])
                                        if pair["current"] in current_well_cols else 0
                                    ),
                                    key=f"ep_ml_pair_current_{i}",
                                    label_visibility="collapsed"
                                )
                            with pc3:
                                if st.button("✕", key=f"ep_ml_pair_remove_{i}", help="Remove this pair"):
                                    to_remove.append(i)

                            pairs[i] = {"train": new_train, "current": new_current}

                        for idx in sorted(to_remove, reverse=True):
                            pairs.pop(idx)

                        if st.button("＋ Add feature pair", key="ep_ml_add_pair"):
                            pairs.append({
                                "train":   candidate_train_features[0] if candidate_train_features else "",
                                "current": current_well_cols[0]        if current_well_cols        else "",
                            })
                            st.rerun()

                        if missing_in_current:
                            with st.expander(
                                f"Logs in training well not auto-matched ({len(missing_in_current)})",
                                expanded=False
                            ):
                                st.caption(", ".join(missing_in_current))
                                st.caption(
                                    "Add them manually above and pair with the correct mnemonic "
                                    "from the current well."
                                )

                        # Validate pairs
                        ml_pairs          = [p for p in pairs if p["train"] and p["current"]]
                        train_feat_list   = [p["train"]   for p in ml_pairs]
                        current_feat_list = [p["current"] for p in ml_pairs]
                        duplicate_train   = len(train_feat_list)   != len(set(train_feat_list))
                        duplicate_current = len(current_feat_list) != len(set(current_feat_list))

                        # ── Options ───────────────────────────────────────
                        res_aliases  = ["AT10","AT20","AT30","AT60","AT90","RT","LLD","ILD","RDEP"]
                        res_in_feats = [c for c in train_feat_list if c in res_aliases]

                        oc1, oc2 = st.columns(2)
                        with oc1:
                            use_log_res = st.checkbox(
                                f"Log-transform resistivity"
                                f"{' (' + ', '.join(res_in_feats) + ')' if res_in_feats else ' (none detected)'}",
                                value=bool(res_in_feats),
                                key="ep_ml_log_res"
                            )
                        with oc2:
                            include_depth = st.checkbox(
                                "Include depth as a feature",
                                value=False,
                                key="ep_ml_include_depth"
                            )

                        st.markdown("---")

                        # ── Model configuration ───────────────────────────
                        st.markdown(
                            '<div class="petro-section-header">Model Configuration</div>',
                            unsafe_allow_html=True
                        )

                        try:
                            from xgboost import XGBRegressor
                            HAS_XGB = True
                        except ImportError:
                            HAS_XGB = False

                        MODEL_OPTIONS = ["Random Forest", "Gradient Boosting"]
                        if HAS_XGB:
                            MODEL_OPTIONS.append("XGBoost")

                        mc1, mc2 = st.columns(2)
                        with mc1:
                            selected_models = st.multiselect(
                                "Models to train",
                                options=MODEL_OPTIONS,
                                default=MODEL_OPTIONS,
                                key="ep_ml_models"
                            )
                        with mc2:
                            n_estimators = st.number_input(
                                "Estimators per model",
                                min_value=50, max_value=1000,
                                value=300, step=50,
                                key="ep_ml_n_estimators"
                            )

                        mp1, mp2, mp3 = st.columns(3)
                        with mp1:
                            test_size = st.slider(
                                "Test split",
                                min_value=0.1, max_value=0.4,
                                value=0.2, step=0.05,
                                key="ep_ml_test_size"
                            )
                        with mp2:
                            cv_folds = st.number_input(
                                "CV folds",
                                min_value=3, max_value=10,
                                value=5, step=1,
                                key="ep_ml_cv_folds"
                            )
                        with mp3:
                            ml_vp_out = st.selectbox(
                                "Vp output unit",
                                options=["m/s", "km/s", "ft/s"],
                                key="ep_ml_vp_out"
                            )

                        ml_dt_unit = st.radio(
                            "DTC unit (training well target)",
                            options=["µs/ft", "µs/m"],
                            horizontal=True,
                            key="ep_ml_dt_unit"
                        )

                        st.markdown("---")

                        # ── Validation & Train button ─────────────────────
                        if not ml_pairs:
                            st.warning("Add at least one feature pair to continue.")
                        elif not selected_models:
                            st.warning("Select at least one model.")
                        elif duplicate_train:
                            st.error("Duplicate training-well logs detected. Each training log should appear only once.")
                        elif duplicate_current:
                            st.error("Duplicate current-well logs detected. Each current-well log should appear only once.")
                        else:
                            mbtn, _ = st.columns([2, 6])
                            with mbtn:
                                apply_ml = st.button(
                                    "Train & Predict",
                                    key="apply_vp_ml",
                                    type="primary",
                                    use_container_width=True
                                )

                            if apply_ml:
                                from sklearn.ensemble import (
                                    RandomForestRegressor, GradientBoostingRegressor
                                )
                                from sklearn.model_selection import (
                                    train_test_split, KFold, cross_val_score
                                )
                                from sklearn.preprocessing import RobustScaler
                                from sklearn.metrics import (
                                    r2_score, mean_squared_error, mean_absolute_error
                                )
                                from sklearn.pipeline import Pipeline
                                from sklearn.impute import SimpleImputer

                                with st.spinner("Training models…"):

                                    work_df            = train_df.copy()
                                    final_train_cols   = list(train_feat_list)
                                    final_current_cols = list(current_feat_list)

                                    # Log-transform resistivity
                                    if use_log_res:
                                        new_train_cols   = []
                                        new_current_cols = []
                                        for tc, cc in zip(final_train_cols, final_current_cols):
                                            if tc in res_aliases:
                                                log_tc = f"LOG_{tc}"
                                                log_cc = f"LOG_{cc}"
                                                work_df[log_tc] = np.log10(
                                                    work_df[tc].replace(0, np.nan)
                                                )
                                                new_train_cols.append(log_tc)
                                                new_current_cols.append(log_cc)
                                            else:
                                                new_train_cols.append(tc)
                                                new_current_cols.append(cc)
                                        final_train_cols   = new_train_cols
                                        final_current_cols = new_current_cols

                                    # Include depth
                                    if include_depth and ml_depth_col_train in work_df.columns:
                                        final_train_cols.append(ml_depth_col_train)
                                        current_depth_candidates = [
                                            c for c in current_well_cols
                                            if c in ["DEPTH", "DEPT", "MD", "TVDSS", "TVD"]
                                        ]
                                        final_current_cols.append(
                                            current_depth_candidates[0]
                                            if current_depth_candidates
                                            else ml_depth_col_train
                                        )

                                    # Build training matrix
                                    df_train_ml   = work_df[work_df[ml_target].notna()].copy()
                                    valid_train   = [c for c in final_train_cols   if c in df_train_ml.columns]
                                    valid_current = [c for c in final_current_cols if c in st.session_state["df"].columns]
                                    n_valid       = min(len(valid_train), len(valid_current))
                                    valid_train   = valid_train[:n_valid]
                                    valid_current = valid_current[:n_valid]

                                    X = df_train_ml[valid_train].values
                                    y = df_train_ml[ml_target].values

                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=test_size, random_state=42
                                    )

                                    # Build model pipelines
                                    model_defs = {}
                                    if "Random Forest" in selected_models:
                                        model_defs["Random Forest"] = Pipeline([
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("scaler",  RobustScaler()),
                                            ("model",   RandomForestRegressor(
                                                n_estimators=n_estimators,
                                                max_depth=12,
                                                min_samples_leaf=3,
                                                n_jobs=-1,
                                                random_state=42
                                            ))
                                        ])
                                    if "Gradient Boosting" in selected_models:
                                        model_defs["Gradient Boosting"] = Pipeline([
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("scaler",  RobustScaler()),
                                            ("model",   GradientBoostingRegressor(
                                                n_estimators=n_estimators,
                                                learning_rate=0.05,
                                                max_depth=5,
                                                random_state=42
                                            ))
                                        ])
                                    if HAS_XGB and "XGBoost" in selected_models:
                                        model_defs["XGBoost"] = Pipeline([
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("scaler",  RobustScaler()),
                                            ("model",   XGBRegressor(
                                                n_estimators=n_estimators,
                                                learning_rate=0.05,
                                                max_depth=6,
                                                random_state=42,
                                                verbosity=0
                                            ))
                                        ])

                                    # Train & evaluate all models
                                    results_ml = {}
                                    for name, pipe in model_defs.items():
                                        pipe.fit(X_train, y_train)
                                        y_pred_test = pipe.predict(X_test)
                                        results_ml[name] = {
                                            "pipe"        : pipe,
                                            "r2"          : r2_score(y_test, y_pred_test),
                                            "rmse"        : float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                                            "mae"         : float(mean_absolute_error(y_test, y_pred_test)),
                                            "y_pred_test" : y_pred_test,
                                            "y_test"      : y_test,
                                        }

                                    best_name = max(results_ml, key=lambda k: results_ml[k]["r2"])
                                    best_pipe = results_ml[best_name]["pipe"]

                                    # Cross-validation on best model
                                    kf        = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                                    cv_scores = cross_val_score(best_pipe, X, y, cv=kf, scoring="r2")
                                    cv_mean   = float(cv_scores.mean())
                                    cv_std    = float(cv_scores.std())

                                    # Predict on current well
                                    pred_df = st.session_state["df"].copy()

                                    if use_log_res:
                                        for tc, cc in zip(valid_train, valid_current):
                                            if tc.startswith("LOG_"):
                                                orig = cc.replace("LOG_", "") if cc.startswith("LOG_") else cc
                                                if orig in pred_df.columns:
                                                    pred_df[cc] = np.log10(
                                                        pred_df[orig].replace(0, np.nan)
                                                    )

                                    X_pred   = pred_df[[c for c in valid_current if c in pred_df.columns]].values
                                    dtc_pred = best_pipe.predict(X_pred)

                                    # DTC → Vp conversion
                                    dtc_series = pd.Series(dtc_pred, index=pred_df.index).replace(0, np.nan)
                                    vp_ms = (
                                        3.280840e5 / dtc_series
                                        if ml_dt_unit == "µs/ft"
                                        else 1.000000e6 / dtc_series
                                    )

                                    if ml_vp_out == "km/s":
                                        vp_out = (vp_ms / 1000.0).round(4)
                                    elif ml_vp_out == "ft/s":
                                        vp_out = (vp_ms / 0.3048).round(4)
                                    else:
                                        vp_out = vp_ms.round(4)

                                    # pred_df["DTCO_ML"] = dtc_series.round(4)
                                    # pred_df["VP_EST"]  = vp_out

                                    for _c in ["DTCO_ML", "VP_EST"]:
                                        if _c in pred_df.columns:
                                            pred_df.drop(columns=[_c], inplace=True)
                                    pred_df["DTCO_ML"] = dtc_series.round(4)
                                    pred_df["VP_EST"]  = vp_out

                                    st.session_state["df"]        = pred_df
                                    st.session_state["vp_source"] = "ML"

                                    vp_vals          = pred_df["VP_EST"].dropna()
                                    best_model_obj   = best_pipe.named_steps["model"]
                                    feat_importances = (
                                        best_model_obj.feature_importances_
                                        if hasattr(best_model_obj, "feature_importances_")
                                        else None
                                    )

                                    # Build readable feature labels
                                    feature_labels = []
                                    for tc, cc in zip(valid_train, valid_current):
                                        if tc == cc:
                                            feature_labels.append(tc)
                                        else:
                                            feature_labels.append(f"{tc} (→{cc})")

                                    st.session_state["vp_success"] = {
                                        "col"    : "VP_EST",
                                        "source" : f"ML — {best_name}",
                                        "unit"   : ml_vp_out,
                                        "min"    : float(vp_vals.min()),
                                        "max"    : float(vp_vals.max()),
                                        "mean"   : float(vp_vals.mean()),
                                    }
                                    st.session_state["vp_ml_diagnostics"] = {
                                        "results" : {
                                            k: {
                                                "r2"          : v["r2"],
                                                "rmse"        : v["rmse"],
                                                "mae"         : v["mae"],
                                                "y_pred_test" : v["y_pred_test"].tolist(),
                                                "y_test"      : v["y_test"].tolist(),
                                            }
                                            for k, v in results_ml.items()
                                        },
                                        "best_name"        : best_name,
                                        "cv_mean"          : cv_mean,
                                        "cv_std"           : cv_std,
                                        "cv_folds"         : cv_folds,
                                        "feature_labels"   : feature_labels,
                                        "feat_importances" : feat_importances.tolist()
                                                            if feat_importances is not None else None,
                                        "dt_unit"          : ml_dt_unit,
                                    }
                                st.rerun()

                    # ── Diagnostics (persists after rerun) ───────────────
                    if st.session_state.get("vp_ml_diagnostics"):
                        diag = st.session_state["vp_ml_diagnostics"]

                        st.markdown(
                            '<div class="petro-section-header">Model Diagnostics</div>',
                            unsafe_allow_html=True
                        )

                        # Metrics table
                        metrics_rows = [
                            {
                                "Model" : name,
                                "R²"    : f"{res['r2']:.4f}",
                                "RMSE"  : f"{res['rmse']:.3f}",
                                "MAE"   : f"{res['mae']:.3f}",
                            }
                            for name, res in diag["results"].items()
                        ]
                        st.dataframe(
                            pd.DataFrame(metrics_rows).set_index("Model"),
                            use_container_width=True
                        )
                        st.caption(
                            f"Best model: **{diag['best_name']}**   |   "
                            f"{diag['cv_folds']}-Fold CV R²: "
                            f"{diag['cv_mean']:.4f} ± {diag['cv_std']:.4f}"
                        )

                        # Actual vs Predicted crossplot
                        best_res   = diag["results"][diag["best_name"]]
                        y_test_arr = np.array(best_res["y_test"])
                        y_pred_arr = np.array(best_res["y_pred_test"])
                        lims = [
                            min(y_test_arr.min(), y_pred_arr.min()),
                            max(y_test_arr.max(), y_pred_arr.max()),
                        ]

                        fig_ml = go.Figure()
                        fig_ml.add_trace(go.Scatter(
                            x=y_test_arr, y=y_pred_arr,
                            mode="markers",
                            marker=dict(size=4, color="#01696f", opacity=0.5),
                            name="Test points"
                        ))
                        fig_ml.add_trace(go.Scatter(
                            x=lims, y=lims,
                            mode="lines",
                            line=dict(color="#cc3333", width=1.5, dash="dash"),
                            name="1:1 line"
                        ))
                        fig_ml.update_layout(
                            height=380,
                            xaxis_title=f"Actual DTC ({diag['dt_unit']})",
                            yaxis_title=f"Predicted DTC ({diag['dt_unit']})",
                            title=dict(
                                text=f"{diag['best_name']} — Actual vs Predicted",
                                font=dict(size=12)
                            ),
                            plot_bgcolor="white", paper_bgcolor="white",
                            margin=dict(l=50, r=20, t=50, b=40),
                            showlegend=True,
                            legend=dict(font=dict(size=10)),
                        )
                        fig_ml.update_xaxes(showgrid=True, gridcolor="#eeeeee")
                        fig_ml.update_yaxes(showgrid=True, gridcolor="#eeeeee")
                        st.plotly_chart(fig_ml, use_container_width=True)

                        # Feature importance chart
                        if diag.get("feat_importances") is not None:
                            fi_df = pd.DataFrame({
                                "Feature"   : diag["feature_labels"],
                                "Importance": diag["feat_importances"],
                            }).sort_values("Importance", ascending=True)

                            fig_fi = go.Figure(go.Bar(
                                x=fi_df["Importance"],
                                y=fi_df["Feature"],
                                orientation="h",
                                marker=dict(color="#01696f"),
                            ))
                            fig_fi.update_layout(
                                height=max(300, len(fi_df) * 26),
                                title=dict(text="Feature Importance", font=dict(size=12)),
                                xaxis_title="Importance",
                                plot_bgcolor="white", paper_bgcolor="white",
                                margin=dict(l=140, r=20, t=50, b=40),
                            )
                            fig_fi.update_xaxes(showgrid=True, gridcolor="#eeeeee")
                            st.plotly_chart(fig_fi, use_container_width=True)

            # ════════════════════════════════════════════════
            #  RESULT SUMMARY (all branches)
            # ════════════════════════════════════════════════
            if st.session_state.get("vp_success"):
                r = st.session_state["vp_success"]

                st.markdown(
                    '<div class="petro-section-header">Computed Vp</div>',
                    unsafe_allow_html=True
                )

                rm1, rm2, rm3 = st.columns(3)
                rm1.metric(f"VP min  ({r['unit']})",  f"{r['min']:.1f}")
                rm2.metric(f"VP max  ({r['unit']})",  f"{r['max']:.1f}")
                rm3.metric(f"VP mean ({r['unit']})",  f"{r['mean']:.1f}")

                st.success(
                    f"{r['col']} saved — source: {r['source']}  |  unit: {r['unit']}"
                )
                st.session_state["vp_success"] = None


        # ════════════════════════════════════════════════════
        #  EXPANDER 2 — SHEAR VELOCITY (Vs)
        # ════════════════════════════════════════════════════
        with st.expander("Shear Velocity", expanded=True):

            # ── Initialize session state ──────────────────────
            if "vs_success" not in st.session_state:
                st.session_state["vs_success"] = None
            if "vs_source" not in st.session_state:
                st.session_state["vs_source"]  = None   # "DTS" | "Castagna" | "GC" | "Brocher" | "ML"

            cur_df   = st.session_state["df"]
            all_cols = [c for c in cur_df.columns if c != depth_col]

            DTS_ALIASES = ["DTS", "DTSM", "DTSH", "DTSV", "DT2", "DTST", "SLOWNESS_S"]
            dts_auto    = detect_curve(all_cols, DTS_ALIASES)

            # ── DTS availability toggle ───────────────────────
            dts_available = st.radio(
                "Is a shear sonic log available?",
                options=["Yes", "No"],
                index=0 if dts_auto else 1,
                horizontal=True,
                key="ep_dts_available"
            )

            st.markdown("---")

            # ════════════════════════════════════════════════
            #  BRANCH A — DTS AVAILABLE
            # ════════════════════════════════════════════════
            if dts_available == "Yes":

                st.markdown(
                    '<div class="petro-section-header">DTS → Vs Conversion</div>',
                    unsafe_allow_html=True
                )

                with st.container(border=True):
                    st.markdown("**Conversion formula**")
                    st.latex(r"V_S = \frac{C}{DTS}")
                    st.caption(
                        "Where C is the unit conversion constant — "
                        "1,000,000 for µs/m (→ m/s), 3,280,840 for µs/ft (→ m/s), etc."
                    )

                st.markdown("")

                ba1, ba2, ba3 = st.columns([2, 1.5, 1.5])
                with ba1:
                    dts_options = list(dict.fromkeys(
                        ([dts_auto] if dts_auto else []) + all_cols
                    ))
                    dts_col = st.selectbox(
                        "DTS log",
                        options=[None] + dts_options,
                        index=(1 if dts_auto else 0),
                        format_func=lambda v: "— select DTS —" if v is None else v,
                        key="ep_dts_col"
                    )

                DTS_UNIT_MAP = {
                    "µs/ft" : 3.280840e5,
                    "µs/m " : 1.000000e6,
                    "ms/m " : 1.000000e3,
                    "ms/ft" : 3.280840e2,
                    "s/m  " : 1.000000e0,
                    "s/ft " : 3.280840e-1,
                    "m/s"   : None,
                    "km/s " : None,
                    "ft/s " : None,
                }

                with ba2:
                    dts_unit_label = st.selectbox(
                        "Input unit",
                        options=list(DTS_UNIT_MAP.keys()),
                        index=0,
                        key="ep_dts_unit"
                    )
                with ba3:
                    vs_out_unit = st.selectbox(
                        "Output unit",
                        options=["m/s", "km/s", "ft/s"],
                        index=0,
                        key="ep_vs_out_unit"
                    )

                # ── Sanity check & preview ────────────────────
                if dts_col:
                    dts_data = cur_df[dts_col].dropna()
                    pv1, pv2, pv3 = st.columns(3)
                    pv1.metric("DTS min",  f"{dts_data.min():.2f}")
                    pv2.metric("DTS max",  f"{dts_data.max():.2f}")
                    pv3.metric("DTS mean", f"{dts_data.mean():.2f}")

                    EXPECTED_RANGES_DTS = {
                        "µs/ft" : (60,   450),
                        "µs/m " : (200, 1475),
                        "ms/m " : (0.2,  1.5),
                        "ms/ft" : (0.06, 0.45),
                        "s/m  " : (2.0e-4, 1.5e-3),
                        "s/ft " : (6.0e-5, 4.5e-4),
                    }
                    if dts_unit_label in EXPECTED_RANGES_DTS:
                        lo, hi = EXPECTED_RANGES_DTS[dts_unit_label]
                        if dts_data.min() < lo or dts_data.max() > hi:
                            st.warning(
                                f"DTS values outside expected range for "
                                f"{dts_unit_label.strip()} ({lo} – {hi}). "
                                "Verify unit selection."
                            )

                st.markdown("---")

                # ── Apply ─────────────────────────────────────
                if dts_col is None:
                    st.warning("Select a DTS log to proceed.")
                else:
                    cb1, _ = st.columns([2, 6])
                    with cb1:
                        apply_vs_dts = st.button(
                            "▶ Convert & Save",
                            key="apply_vs_dts",
                            type="primary",
                            use_container_width=True
                        )

                    if apply_vs_dts:
                        vs_df      = st.session_state["df"].copy()
                        dts_series = vs_df[dts_col].copy().replace(0, np.nan)

                        factor = DTS_UNIT_MAP[dts_unit_label]

                        if factor is not None:
                            vs_ms = factor / dts_series
                        else:
                            if "km/s" in dts_unit_label:
                                vs_ms = dts_series * 1000.0
                            elif "ft/s" in dts_unit_label:
                                vs_ms = dts_series * 0.3048
                            else:
                                vs_ms = dts_series

                        if vs_out_unit == "km/s":
                            vs_out = (vs_ms / 1000.0).round(4)
                        elif vs_out_unit == "ft/s":
                            vs_out = (vs_ms / 0.3048).round(4)
                        else:
                            vs_out = vs_ms.round(4)

                        # vs_df["VS"] = vs_out
                        # st.session_state["df"]        = vs_df

                        if "VS" in vs_df.columns:
                            vs_df.drop(columns=["VS"], inplace=True)
                        vs_df["VS"] = vs_out
                        st.session_state["df"]        = vs_df

                        st.session_state["vs_source"] = "DTS"

                        vs_vals = vs_df["VS"].dropna()
                        st.session_state["vs_success"] = {
                            "col"    : "VS",
                            "source" : "DTS",
                            "unit"   : vs_out_unit,
                            "min"    : float(vs_vals.min()),
                            "max"    : float(vs_vals.max()),
                            "mean"   : float(vs_vals.mean()),
                        }
                        st.rerun()

            # ════════════════════════════════════════════════
            #  BRANCH B — DTS NOT AVAILABLE → ESTIMATION
            # ════════════════════════════════════════════════
            else:

                st.markdown(
                    '<div class="petro-section-header">Vs Estimation Method</div>',
                    unsafe_allow_html=True
                )

                method_vs = st.radio(
                    "Select estimation method",
                    options=[
                        "Castagna Mudrock Line",
                        "Greenberg-Castagna",
                        "Brocher Empirical",
                        "ML from Conventional Logs",
                    ],
                    key="ep_vs_method",
                    label_visibility="collapsed"
                )

                st.markdown("")

                # ── Get Vp column (needed by Castagna, GC, Brocher) ──
                VP_ALIASES  = ["VP", "VP_EST", "VEL", "VPCO"]
                vp_auto_col = detect_curve(all_cols, VP_ALIASES)

                # ─────────────────────────────────────────────
                #  METHOD 1 — CASTAGNA MUDROCK LINE
                # ─────────────────────────────────────────────
                if method_vs == "Castagna Mudrock Line":

                    with st.container(border=True):
                        st.markdown("**Castagna Mudrock Line**")
                        st.latex(r"V_S = 0.8621 \cdot V_P - 1172 \quad \text{(m/s)}")
                        st.caption(
                            "Empirical relation for water-saturated clastic rocks. "
                            "Valid when Vp is in m/s. Overestimates Vs in carbonates."
                        )

                    st.markdown("")

                    cm1, cm2 = st.columns(2)
                    with cm1:
                        vp_col_cast = st.selectbox(
                            "Vp log (m/s)",
                            options=[None] + all_cols,
                            index=(all_cols.index(vp_auto_col) + 1 if vp_auto_col in all_cols else 0),
                            format_func=lambda v: "— select Vp —" if v is None else v,
                            key="ep_vs_cast_vp"
                        )
                    with cm2:
                        vs_cast_out_unit = st.selectbox(
                            "Output unit",
                            options=["m/s", "km/s", "ft/s"],
                            key="ep_vs_cast_unit"
                        )

                    if vp_col_cast:
                        vp_preview = cur_df[vp_col_cast].dropna()
                        pc1, pc2, pc3 = st.columns(3)
                        pc1.metric("Vp min",  f"{vp_preview.min():.1f}")
                        pc2.metric("Vp max",  f"{vp_preview.max():.1f}")
                        pc3.metric("Vp mean", f"{vp_preview.mean():.1f}")

                    st.markdown("---")

                    if vp_col_cast is None:
                        st.warning("Select a Vp log to proceed.")
                    else:
                        ab1, _ = st.columns([2, 6])
                        with ab1:
                            apply_cast = st.button(
                                "▶ Estimate & Save",
                                key="apply_vs_castagna",
                                type="primary",
                                use_container_width=True
                            )

                        if apply_cast:
                            vs_df    = st.session_state["df"].copy()
                            vp_series = vs_df[vp_col_cast].copy().replace(0, np.nan)

                            vs_ms = 0.8621 * vp_series - 1172.0
                            vs_ms = vs_ms.clip(lower=0)   # physical floor

                            if vs_cast_out_unit == "km/s":
                                vs_out = (vs_ms / 1000.0).round(4)
                            elif vs_cast_out_unit == "ft/s":
                                vs_out = (vs_ms / 0.3048).round(4)
                            else:
                                vs_out = vs_ms.round(4)

                            # vs_df["VS_EST"]             = vs_out
                            # st.session_state["df"]       = vs_df

                            if "VS_EST" in vs_df.columns:
                                vs_df.drop(columns=["VS_EST"], inplace=True)
                            vs_df["VS_EST"]             = vs_out
                            st.session_state["df"]       = vs_df

                            st.session_state["vs_source"] = "Castagna"

                            vs_vals = vs_df["VS_EST"].dropna()
                            st.session_state["vs_success"] = {
                                "col"    : "VS_EST",
                                "source" : "Castagna Mudrock Line",
                                "unit"   : vs_cast_out_unit,
                                "min"    : float(vs_vals.min()),
                                "max"    : float(vs_vals.max()),
                                "mean"   : float(vs_vals.mean()),
                            }
                            st.rerun()

                # ─────────────────────────────────────────────
                #  METHOD 2 — GREENBERG-CASTAGNA
                # ─────────────────────────────────────────────
                elif method_vs == "Greenberg-Castagna":

                    # Coefficients in ASCENDING power order: a[0] + a[1]*Vp + a[2]*Vp^2
                    # All equations in km/s
                    GC_COEFFS = {
                        "Sandstone"  : {
                            "coeffs" : [-0.85588, 0.80416],
                            "latex"  : r"V_S = 0.80416\,V_P - 0.85588",
                            "note"   : "Clean to slightly shaly sandstone.",
                        },
                        "Limestone"  : {
                            "coeffs" : [-1.03049, 1.01677, -0.05508],
                            "latex"  : r"V_S = -0.05508\,V_P^2 + 1.01677\,V_P - 1.03049",
                            "note"   : "Carbonate limestone — quadratic relation.",
                        },
                        "Dolomite"   : {
                            "coeffs" : [-0.07775, 0.58321],
                            "latex"  : r"V_S = 0.58321\,V_P - 0.07775",
                            "note"   : "Tight dolomite formation.",
                        },
                        "Shale"      : {
                            "coeffs" : [-0.86735, 0.76969],
                            "latex"  : r"V_S = 0.76969\,V_P - 0.86735",
                            "note"   : "Pure shale / clay-rich formation.",
                        },
                    }

                    # ── Formula card — show all equations ────────
                    with st.container(border=True):
                        st.markdown("**Greenberg-Castagna Vp–Vs Relations**")
                        st.caption("All equations in **km/s**. Vp input will be converted internally.")
                        for litho, entry in GC_COEFFS.items():
                            c1, c2 = st.columns([2, 5])
                            c1.markdown(f"**{litho}**")
                            c2.latex(entry["latex"])

                    st.markdown("")

                    # ── Controls ─────────────────────────────────
                    gc1, gc2, gc3 = st.columns(3)
                    with gc1:
                        vp_col_gc = st.selectbox(
                            "Vp log",
                            options=[None] + all_cols,
                            index=(all_cols.index(vp_auto_col) + 1 if vp_auto_col in all_cols else 0),
                            format_func=lambda v: "— select Vp —" if v is None else v,
                            key="ep_vs_gc_vp"
                        )
                    with gc2:
                        gc_vp_unit = st.selectbox(
                            "Vp input unit",
                            options=["m/s", "km/s", "ft/s"],
                            key="ep_vs_gc_vp_unit"
                        )
                    with gc3:
                        vs_gc_out_unit = st.selectbox(
                            "Vs output unit",
                            options=["m/s", "km/s", "ft/s"],
                            key="ep_vs_gc_out_unit"
                        )

                    # ── Lithology selector with equation preview ──
                    gc_lithology = st.radio(
                        "Select lithology",
                        options=list(GC_COEFFS.keys()),
                        horizontal=True,
                        key="ep_vs_gc_litho"
                    )

                    selected_gc = GC_COEFFS[gc_lithology]
                    with st.container(border=True):
                        rc1, rc2 = st.columns([1, 3])
                        rc1.markdown(f"**{gc_lithology}**")
                        rc2.latex(selected_gc["latex"])
                        st.caption(selected_gc["note"])

                    # ── Vp preview metrics ────────────────────────
                    if vp_col_gc:
                        vp_preview = cur_df[vp_col_gc].dropna()
                        pc1, pc2, pc3 = st.columns(3)
                        pc1.metric("Vp min",  f"{vp_preview.min():.3f}")
                        pc2.metric("Vp max",  f"{vp_preview.max():.3f}")
                        pc3.metric("Vp mean", f"{vp_preview.mean():.3f}")

                    st.markdown("---")

                    # ── Apply ─────────────────────────────────────
                    if vp_col_gc is None:
                        st.warning("Select a Vp log to proceed.")
                    else:
                        ab2, _ = st.columns([2, 6])
                        with ab2:
                            apply_gc = st.button(
                                "▶ Estimate & Save",
                                key="apply_vs_gc",
                                type="primary",
                                use_container_width=True
                            )

                        if apply_gc:
                            vs_df     = st.session_state["df"].copy()
                            vp_series = vs_df[vp_col_gc].copy().replace(0, np.nan)

                            # Normalize Vp to km/s
                            if gc_vp_unit == "m/s":
                                vp_kms = vp_series / 1000.0
                            elif gc_vp_unit == "ft/s":
                                vp_kms = vp_series * 0.0003048
                            else:
                                vp_kms = vp_series

                            # Evaluate polynomial — coeffs in ascending power order
                            # vs_kms = a[0] + a[1]*Vp + a[2]*Vp^2
                            a      = selected_gc["coeffs"]
                            vs_kms = sum(a[i] * vp_kms**i for i in range(len(a)))
                            vs_kms = vs_kms.clip(lower=0)

                            # Convert to output unit
                            vs_ms = vs_kms * 1000.0
                            if vs_gc_out_unit == "km/s":
                                vs_out = vs_kms.round(4)
                            elif vs_gc_out_unit == "ft/s":
                                vs_out = (vs_ms / 0.3048).round(4)
                            else:
                                vs_out = vs_ms.round(4)

                            # vs_df["VS_EST"]              = vs_out
                            # st.session_state["df"]        = vs_df
                            # st.session_state["vs_source"] = "GC"

                            if "VS_EST" in vs_df.columns:
                                vs_df.drop(columns=["VS_EST"], inplace=True)
                            vs_df["VS_EST"]              = vs_out
                            st.session_state["df"]        = vs_df
                            st.session_state["vs_source"] = "GC"


                            vs_vals = vs_df["VS_EST"].dropna()
                            st.session_state["vs_success"] = {
                                "col"    : "VS_EST",
                                "source" : f"Greenberg-Castagna ({gc_lithology})",
                                "unit"   : vs_gc_out_unit,
                                "min"    : float(vs_vals.min()),
                                "max"    : float(vs_vals.max()),
                                "mean"   : float(vs_vals.mean()),
                            }
                            st.rerun()

                # ─────────────────────────────────────────────
                #  METHOD 3 — BROCHER EMPIRICAL
                # ─────────────────────────────────────────────
                elif method_vs == "Brocher Empirical":

                    with st.container(border=True):
                        st.markdown("**Brocher (2005) Empirical Relation**")
                        st.latex(
                            r"V_S = 0.7858 - 1.2344\,V_P + 0.7949\,V_P^2 "
                            r"- 0.1238\,V_P^3 + 0.0064\,V_P^4 \quad \text{(km/s)}"
                        )
                        st.caption(
                            "Broad empirical polynomial fit across mixed crustal lithologies. "
                            "Vp must be in km/s (converted internally). Valid for Vp 1.5 – 8.5 km/s."
                        )

                    st.markdown("")

                    br1, br2, br3 = st.columns(3)
                    with br1:
                        vp_col_br = st.selectbox(
                            "Vp log",
                            options=[None] + all_cols,
                            index=(all_cols.index(vp_auto_col) + 1 if vp_auto_col in all_cols else 0),
                            format_func=lambda v: "— select Vp —" if v is None else v,
                            key="ep_vs_br_vp"
                        )
                    with br2:
                        br_vp_unit = st.selectbox(
                            "Vp input unit",
                            options=["m/s", "km/s", "ft/s"],
                            key="ep_vs_br_vp_unit"
                        )
                    with br3:
                        vs_br_out_unit = st.selectbox(
                            "Output unit",
                            options=["m/s", "km/s", "ft/s"],
                            key="ep_vs_br_out_unit"
                        )

                    if vp_col_br:
                        vp_preview = cur_df[vp_col_br].dropna()
                        pc1, pc2, pc3 = st.columns(3)
                        pc1.metric("Vp min",  f"{vp_preview.min():.3f}")
                        pc2.metric("Vp max",  f"{vp_preview.max():.3f}")
                        pc3.metric("Vp mean", f"{vp_preview.mean():.3f}")

                    st.markdown("---")

                    if vp_col_br is None:
                        st.warning("Select a Vp log to proceed.")
                    else:
                        ab3, _ = st.columns([2, 6])
                        with ab3:
                            apply_br = st.button(
                                "▶ Estimate & Save",
                                key="apply_vs_brocher",
                                type="primary",
                                use_container_width=True
                            )

                        if apply_br:
                            vs_df     = st.session_state["df"].copy()
                            vp_series = vs_df[vp_col_br].copy().replace(0, np.nan)

                            # Normalize Vp to km/s
                            if br_vp_unit == "m/s":
                                vp_kms = vp_series / 1000.0
                            elif br_vp_unit == "ft/s":
                                vp_kms = vp_series * 0.0003048
                            else:
                                vp_kms = vp_series

                            vs_kms = (
                                0.7858
                                - 1.2344 * vp_kms
                                + 0.7949 * vp_kms**2
                                - 0.1238 * vp_kms**3
                                + 0.0064 * vp_kms**4
                            ).clip(lower=0)

                            vs_ms = vs_kms * 1000.0
                            if vs_br_out_unit == "km/s":
                                vs_out = vs_kms.round(4)
                            elif vs_br_out_unit == "ft/s":
                                vs_out = (vs_ms / 0.3048).round(4)
                            else:
                                vs_out = vs_ms.round(4)

                            # vs_df["VS_EST"]              = vs_out
                            # st.session_state["df"]        = vs_df
                            # st.session_state["vs_source"] = "Brocher"

                            if "VS_EST" in vs_df.columns:
                                vs_df.drop(columns=["VS_EST"], inplace=True)
                            vs_df["VS_EST"]              = vs_out
                            st.session_state["df"]        = vs_df
                            st.session_state["vs_source"] = "Brocher"


                            vs_vals = vs_df["VS_EST"].dropna()
                            st.session_state["vs_success"] = {
                                "col"    : "VS_EST",
                                "source" : "Brocher Empirical",
                                "unit"   : vs_br_out_unit,
                                "min"    : float(vs_vals.min()),
                                "max"    : float(vs_vals.max()),
                                "mean"   : float(vs_vals.mean()),
                            }
                            st.rerun()

                # ─────────────────────────────────────────────
                #  METHOD 4 — ML FROM CONVENTIONAL LOGS
                # ─────────────────────────────────────────────
                elif method_vs == "ML from Conventional Logs":

                    with st.container(border=True):
                        st.markdown("**ML-Based Vs Prediction**")
                        st.caption(
                            "Predicts DTS for the current well using a model trained on a "
                            "separate well from the same field that has measured DTS. "
                            "Only conventional logs (GR, RHOB, NPHI, resistivity, etc.) "
                            "are used as features — DTS is the prediction target."
                        )

                    st.markdown("")

                    # ── Training well upload ──────────────────
                    st.markdown(
                        '<div class="petro-section-header">Training Well</div>',
                        unsafe_allow_html=True
                    )

                    train_las_file_vs = st.file_uploader(
                        "Upload a LAS file from a nearby well with measured DTS",
                        type=["las", "LAS"],
                        key="ep_vs_ml_train_las"
                    )

                    train_df_vs = None
                    if train_las_file_vs:
                        try:
                            import io
                            import lasio

                            file_content_vs = train_las_file_vs.read()
                            train_las_vs    = lasio.read(io.StringIO(
                                file_content_vs.decode("utf-8", errors="replace")
                            ))
                            train_df_vs     = train_las_vs.df().reset_index()
                            train_df_vs.columns = [c.upper() for c in train_df_vs.columns]

                            DTS_ALIASES_ML = ["DTS", "DTSM", "DTSH", "DTSV", "DT2", "DTST"]
                            dts_present    = any(c in train_df_vs.columns for c in DTS_ALIASES_ML)

                            tc1, tc2, tc3 = st.columns(3)
                            tc1.metric("Rows",        len(train_df_vs))
                            tc2.metric("Curves",      len(train_df_vs.columns))
                            tc3.metric("DTS present", "Yes" if dts_present else "No")

                        except Exception as e:
                            st.error(f"Failed to read training LAS file: {e}")
                            train_df_vs = None

                    if train_df_vs is not None:

                        # ── Target & depth column ─────────────
                        st.markdown(
                            '<div class="petro-section-header">Feature & Target Configuration</div>',
                            unsafe_allow_html=True
                        )

                        train_all_cols_vs = list(train_df_vs.columns)
                        dts_target_auto   = detect_curve(train_all_cols_vs, DTS_ALIASES_ML)

                        fc1, fc2 = st.columns(2)
                        with fc1:
                            vs_ml_target = st.selectbox(
                                "Target log (DTS — training well only)",
                                options=train_all_cols_vs,
                                index=(
                                    train_all_cols_vs.index(dts_target_auto)
                                    if dts_target_auto in train_all_cols_vs else 0
                                ),
                                key="ep_vs_ml_target"
                            )
                        with fc2:
                            vs_ml_depth_col_train = st.selectbox(
                                "Depth column (training well)",
                                options=train_all_cols_vs,
                                index=0,
                                key="ep_vs_ml_depth_train"
                            )

                        if train_df_vs[vs_ml_target].notna().sum() == 0:
                            st.error(
                                f"{vs_ml_target} has no valid values in the training well. "
                                "Select a different target log or upload a different LAS file."
                            )
                            st.stop()

                        # ── Feature pairing table ─────────────
                        VS_ALWAYS_EXCLUDE = {
                            vs_ml_target, vs_ml_depth_col_train,
                            "DTS", "DTSM", "DTSH", "DTSV", "DT2", "DTST",
                            "DT", "DTC", "DTCO", "AC",
                            "SPHI", "PR", "VPVS", "VP", "VS",
                            "VP_EST", "VS_EST", "AI", "SI", "DTCO_ML", "DTSO_ML"
                        }

                        candidate_train_features_vs = [
                            c for c in train_df_vs.columns if c not in VS_ALWAYS_EXCLUDE
                        ]
                        current_well_cols_vs  = list(st.session_state["df"].columns)
                        shared_by_name_vs     = [c for c in candidate_train_features_vs if c in current_well_cols_vs]
                        missing_in_current_vs = [c for c in candidate_train_features_vs if c not in current_well_cols_vs]

                        st.markdown("##### Feature Log Pairing")
                        st.caption(
                            "Select one log from the **training well** and pair it with the "
                            "corresponding log in the **current well**. "
                            "Use this to handle mnemonic mismatches (e.g. HCGR ↔ EGR)."
                        )

                        hc1, hc2, hc3 = st.columns([5, 5, 1])
                        hc1.caption("**Training well log**")
                        hc2.caption("**Current well log (paired)**")

                        if "ep_vs_ml_feature_pairs" not in st.session_state:
                            st.session_state["ep_vs_ml_feature_pairs"] = [
                                {"train": c, "current": c} for c in shared_by_name_vs
                            ]

                        vs_pairs  = st.session_state["ep_vs_ml_feature_pairs"]
                        to_remove = []

                        for i, pair in enumerate(vs_pairs):
                            pc1, pc2, pc3 = st.columns([5, 5, 1])
                            with pc1:
                                new_train = st.selectbox(
                                    f"VS Training log #{i+1}",
                                    options=candidate_train_features_vs,
                                    index=(
                                        candidate_train_features_vs.index(pair["train"])
                                        if pair["train"] in candidate_train_features_vs else 0
                                    ),
                                    key=f"ep_vs_ml_pair_train_{i}",
                                    label_visibility="collapsed"
                                )
                            with pc2:
                                new_current = st.selectbox(
                                    f"VS Current well log #{i+1}",
                                    options=current_well_cols_vs,
                                    index=(
                                        current_well_cols_vs.index(pair["current"])
                                        if pair["current"] in current_well_cols_vs else 0
                                    ),
                                    key=f"ep_vs_ml_pair_current_{i}",
                                    label_visibility="collapsed"
                                )
                            with pc3:
                                if st.button("✕", key=f"ep_vs_ml_pair_remove_{i}", help="Remove this pair"):
                                    to_remove.append(i)
                            vs_pairs[i] = {"train": new_train, "current": new_current}

                        for idx in sorted(to_remove, reverse=True):
                            vs_pairs.pop(idx)

                        if st.button("＋ Add feature pair", key="ep_vs_ml_add_pair"):
                            vs_pairs.append({
                                "train":   candidate_train_features_vs[0] if candidate_train_features_vs else "",
                                "current": current_well_cols_vs[0]        if current_well_cols_vs        else "",
                            })
                            st.rerun()

                        if missing_in_current_vs:
                            with st.expander(
                                f"Logs in training well not auto-matched ({len(missing_in_current_vs)})",
                                expanded=False
                            ):
                                st.caption(", ".join(missing_in_current_vs))
                                st.caption(
                                    "Add them manually above and pair with the correct mnemonic "
                                    "from the current well."
                                )

                        vs_ml_pairs          = [p for p in vs_pairs if p["train"] and p["current"]]
                        vs_train_feat_list   = [p["train"]   for p in vs_ml_pairs]
                        vs_current_feat_list = [p["current"] for p in vs_ml_pairs]
                        vs_dup_train         = len(vs_train_feat_list)   != len(set(vs_train_feat_list))
                        vs_dup_current       = len(vs_current_feat_list) != len(set(vs_current_feat_list))

                        # ── Options ───────────────────────────
                        res_aliases_vs  = ["AT10","AT20","AT30","AT60","AT90","RT","LLD","ILD","RDEP"]
                        res_in_feats_vs = [c for c in vs_train_feat_list if c in res_aliases_vs]

                        oc1, oc2 = st.columns(2)
                        with oc1:
                            use_log_res_vs = st.checkbox(
                                f"Log-transform resistivity"
                                f"{' (' + ', '.join(res_in_feats_vs) + ')' if res_in_feats_vs else ' (none detected)'}",
                                value=bool(res_in_feats_vs),
                                key="ep_vs_ml_log_res"
                            )
                        with oc2:
                            include_depth_vs = st.checkbox(
                                "Include depth as a feature",
                                value=False,
                                key="ep_vs_ml_include_depth"
                            )

                        st.markdown("---")

                        # ── Model configuration ───────────────
                        st.markdown(
                            '<div class="petro-section-header">Model Configuration</div>',
                            unsafe_allow_html=True
                        )

                        try:
                            from xgboost import XGBRegressor
                            HAS_XGB = True
                        except ImportError:
                            HAS_XGB = False

                        MODEL_OPTIONS_VS = ["Random Forest", "Gradient Boosting"]
                        if HAS_XGB:
                            MODEL_OPTIONS_VS.append("XGBoost")

                        mc1, mc2 = st.columns(2)
                        with mc1:
                            selected_models_vs = st.multiselect(
                                "Models to train",
                                options=MODEL_OPTIONS_VS,
                                default=MODEL_OPTIONS_VS,
                                key="ep_vs_ml_models"
                            )
                        with mc2:
                            n_estimators_vs = st.number_input(
                                "Estimators per model",
                                min_value=50, max_value=1000,
                                value=300, step=50,
                                key="ep_vs_ml_n_estimators"
                            )

                        mp1, mp2, mp3 = st.columns(3)
                        with mp1:
                            test_size_vs = st.slider(
                                "Test split",
                                min_value=0.1, max_value=0.4,
                                value=0.2, step=0.05,
                                key="ep_vs_ml_test_size"
                            )
                        with mp2:
                            cv_folds_vs = st.number_input(
                                "CV folds",
                                min_value=3, max_value=10,
                                value=5, step=1,
                                key="ep_vs_ml_cv_folds"
                            )
                        with mp3:
                            vs_ml_out = st.selectbox(
                                "Vs output unit",
                                options=["m/s", "km/s", "ft/s"],
                                key="ep_vs_ml_vp_out"
                            )

                        vs_ml_dt_unit = st.radio(
                            "DTS unit (training well target)",
                            options=["µs/ft", "µs/m"],
                            horizontal=True,
                            key="ep_vs_ml_dt_unit"
                        )

                        st.markdown("---")

                        # ── Validation & Train button ─────────
                        if not vs_ml_pairs:
                            st.warning("Add at least one feature pair to continue.")
                        elif not selected_models_vs:
                            st.warning("Select at least one model.")
                        elif vs_dup_train:
                            st.error("Duplicate training-well logs detected. Each training log should appear only once.")
                        elif vs_dup_current:
                            st.error("Duplicate current-well logs detected. Each current-well log should appear only once.")
                        else:
                            mbtn, _ = st.columns([2, 6])
                            with mbtn:
                                apply_vs_ml = st.button(
                                    "Train & Predict",
                                    key="apply_vs_ml",
                                    type="primary",
                                    use_container_width=True
                                )

                            if apply_vs_ml:
                                from sklearn.ensemble import (
                                    RandomForestRegressor, GradientBoostingRegressor
                                )
                                from sklearn.model_selection import (
                                    train_test_split, KFold, cross_val_score
                                )
                                from sklearn.preprocessing import RobustScaler
                                from sklearn.metrics import (
                                    r2_score, mean_squared_error, mean_absolute_error
                                )
                                from sklearn.pipeline import Pipeline
                                from sklearn.impute import SimpleImputer

                                with st.spinner("Training models…"):

                                    work_df_vs           = train_df_vs.copy()
                                    final_train_cols_vs  = list(vs_train_feat_list)
                                    final_current_cols_vs = list(vs_current_feat_list)

                                    # Log-transform resistivity
                                    if use_log_res_vs:
                                        new_t, new_c = [], []
                                        for tc, cc in zip(final_train_cols_vs, final_current_cols_vs):
                                            if tc in res_aliases_vs:
                                                log_tc = f"LOG_{tc}"
                                                log_cc = f"LOG_{cc}"
                                                work_df_vs[log_tc] = np.log10(
                                                    work_df_vs[tc].replace(0, np.nan)
                                                )
                                                new_t.append(log_tc)
                                                new_c.append(log_cc)
                                            else:
                                                new_t.append(tc)
                                                new_c.append(cc)
                                        final_train_cols_vs   = new_t
                                        final_current_cols_vs = new_c

                                    # Include depth
                                    if include_depth_vs and vs_ml_depth_col_train in work_df_vs.columns:
                                        final_train_cols_vs.append(vs_ml_depth_col_train)
                                        cur_depth_candidates = [
                                            c for c in current_well_cols_vs
                                            if c in ["DEPTH", "DEPT", "MD", "TVDSS", "TVD"]
                                        ]
                                        final_current_cols_vs.append(
                                            cur_depth_candidates[0]
                                            if cur_depth_candidates
                                            else vs_ml_depth_col_train
                                        )

                                    df_train_vs   = work_df_vs[work_df_vs[vs_ml_target].notna()].copy()
                                    valid_train_vs   = [c for c in final_train_cols_vs   if c in df_train_vs.columns]
                                    valid_current_vs = [c for c in final_current_cols_vs if c in st.session_state["df"].columns]
                                    n_valid_vs       = min(len(valid_train_vs), len(valid_current_vs))
                                    valid_train_vs   = valid_train_vs[:n_valid_vs]
                                    valid_current_vs = valid_current_vs[:n_valid_vs]

                                    X_vs = df_train_vs[valid_train_vs].values
                                    y_vs = df_train_vs[vs_ml_target].values

                                    X_train_vs, X_test_vs, y_train_vs, y_test_vs = train_test_split(
                                        X_vs, y_vs, test_size=test_size_vs, random_state=42
                                    )

                                    model_defs_vs = {}
                                    if "Random Forest" in selected_models_vs:
                                        model_defs_vs["Random Forest"] = Pipeline([
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("scaler",  RobustScaler()),
                                            ("model",   RandomForestRegressor(
                                                n_estimators=n_estimators_vs,
                                                max_depth=12,
                                                min_samples_leaf=3,
                                                n_jobs=-1,
                                                random_state=42
                                            ))
                                        ])
                                    if "Gradient Boosting" in selected_models_vs:
                                        model_defs_vs["Gradient Boosting"] = Pipeline([
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("scaler",  RobustScaler()),
                                            ("model",   GradientBoostingRegressor(
                                                n_estimators=n_estimators_vs,
                                                learning_rate=0.05,
                                                max_depth=5,
                                                random_state=42
                                            ))
                                        ])
                                    if HAS_XGB and "XGBoost" in selected_models_vs:
                                        model_defs_vs["XGBoost"] = Pipeline([
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("scaler",  RobustScaler()),
                                            ("model",   XGBRegressor(
                                                n_estimators=n_estimators_vs,
                                                learning_rate=0.05,
                                                max_depth=6,
                                                random_state=42,
                                                verbosity=0
                                            ))
                                        ])

                                    results_vs = {}
                                    for name, pipe in model_defs_vs.items():
                                        pipe.fit(X_train_vs, y_train_vs)
                                        y_pred_vs = pipe.predict(X_test_vs)
                                        results_vs[name] = {
                                            "pipe"        : pipe,
                                            "r2"          : r2_score(y_test_vs, y_pred_vs),
                                            "rmse"        : float(np.sqrt(mean_squared_error(y_test_vs, y_pred_vs))),
                                            "mae"         : float(mean_absolute_error(y_test_vs, y_pred_vs)),
                                            "y_pred_test" : y_pred_vs,
                                            "y_test"      : y_test_vs,
                                        }

                                    best_name_vs = max(results_vs, key=lambda k: results_vs[k]["r2"])
                                    best_pipe_vs = results_vs[best_name_vs]["pipe"]

                                    kf_vs     = KFold(n_splits=cv_folds_vs, shuffle=True, random_state=42)
                                    cv_vs     = cross_val_score(best_pipe_vs, X_vs, y_vs, cv=kf_vs, scoring="r2")
                                    cv_mean_vs = float(cv_vs.mean())
                                    cv_std_vs  = float(cv_vs.std())

                                    pred_df_vs = st.session_state["df"].copy()

                                    if use_log_res_vs:
                                        for tc, cc in zip(valid_train_vs, valid_current_vs):
                                            if tc.startswith("LOG_"):
                                                orig = cc[4:] if cc.startswith("LOG_") else cc
                                                if orig in pred_df_vs.columns:
                                                    pred_df_vs[cc] = np.log10(
                                                        pred_df_vs[orig].replace(0, np.nan)
                                                    )

                                    X_pred_vs   = pred_df_vs[[c for c in valid_current_vs if c in pred_df_vs.columns]].values
                                    dts_pred    = best_pipe_vs.predict(X_pred_vs)

                                    dts_series = pd.Series(dts_pred, index=pred_df_vs.index).replace(0, np.nan)
                                    vs_ms = (
                                        3.280840e5 / dts_series
                                        if vs_ml_dt_unit == "µs/ft"
                                        else 1.000000e6 / dts_series
                                    )

                                    if vs_ml_out == "km/s":
                                        vs_out = (vs_ms / 1000.0).round(4)
                                    elif vs_ml_out == "ft/s":
                                        vs_out = (vs_ms / 0.3048).round(4)
                                    else:
                                        vs_out = vs_ms.round(4)

                                    # pred_df_vs["DTSO_ML"] = dts_series.round(4)
                                    # pred_df_vs["VS_EST"]  = vs_out

                                    for _c in ["DTSO_ML", "VS_EST"]:
                                        if _c in pred_df_vs.columns:
                                            pred_df_vs.drop(columns=[_c], inplace=True)
                                    pred_df_vs["DTSO_ML"] = dts_series.round(4)
                                    pred_df_vs["VS_EST"]  = vs_out

                                    st.session_state["df"]        = pred_df_vs
                                    st.session_state["vs_source"] = "ML"

                                    vs_vals_ml = pred_df_vs["VS_EST"].dropna()
                                    best_obj_vs = best_pipe_vs.named_steps["model"]
                                    feat_imp_vs = (
                                        best_obj_vs.feature_importances_
                                        if hasattr(best_obj_vs, "feature_importances_")
                                        else None
                                    )

                                    feat_labels_vs = []
                                    for tc, cc in zip(valid_train_vs, valid_current_vs):
                                        feat_labels_vs.append(tc if tc == cc else f"{tc} (→{cc})")

                                    st.session_state["vs_success"] = {
                                        "col"    : "VS_EST",
                                        "source" : f"ML — {best_name_vs}",
                                        "unit"   : vs_ml_out,
                                        "min"    : float(vs_vals_ml.min()),
                                        "max"    : float(vs_vals_ml.max()),
                                        "mean"   : float(vs_vals_ml.mean()),
                                    }
                                    st.session_state["vs_ml_diagnostics"] = {
                                        "results" : {
                                            k: {
                                                "r2"          : v["r2"],
                                                "rmse"        : v["rmse"],
                                                "mae"         : v["mae"],
                                                "y_pred_test" : v["y_pred_test"].tolist(),
                                                "y_test"      : v["y_test"].tolist(),
                                            }
                                            for k, v in results_vs.items()
                                        },
                                        "best_name"        : best_name_vs,
                                        "cv_mean"          : cv_mean_vs,
                                        "cv_std"           : cv_std_vs,
                                        "cv_folds"         : cv_folds_vs,
                                        "feature_labels"   : feat_labels_vs,
                                        "feat_importances" : feat_imp_vs.tolist()
                                                            if feat_imp_vs is not None else None,
                                        "dts_unit"         : vs_ml_dt_unit,
                                    }
                                st.rerun()

                    # ── ML Diagnostics ────────────────────────
                    if st.session_state.get("vs_ml_diagnostics"):
                        diag_vs = st.session_state["vs_ml_diagnostics"]

                        st.markdown(
                            '<div class="petro-section-header">Model Diagnostics</div>',
                            unsafe_allow_html=True
                        )

                        metrics_rows_vs = [
                            {
                                "Model" : name,
                                "R²"    : f"{res['r2']:.4f}",
                                "RMSE"  : f"{res['rmse']:.3f}",
                                "MAE"   : f"{res['mae']:.3f}",
                            }
                            for name, res in diag_vs["results"].items()
                        ]
                        st.dataframe(
                            pd.DataFrame(metrics_rows_vs).set_index("Model"),
                            use_container_width=True
                        )
                        st.caption(
                            f"Best model: **{diag_vs['best_name']}**   |   "
                            f"{diag_vs['cv_folds']}-Fold CV R²: "
                            f"{diag_vs['cv_mean']:.4f} ± {diag_vs['cv_std']:.4f}"
                        )

                        best_res_vs   = diag_vs["results"][diag_vs["best_name"]]
                        yt_vs = np.array(best_res_vs["y_test"])
                        yp_vs = np.array(best_res_vs["y_pred_test"])
                        lims_vs = [min(yt_vs.min(), yp_vs.min()), max(yt_vs.max(), yp_vs.max())]

                        fig_vs_ml = go.Figure()
                        fig_vs_ml.add_trace(go.Scatter(
                            x=yt_vs, y=yp_vs,
                            mode="markers",
                            marker=dict(size=4, color="#01696f", opacity=0.5),
                            name="Test points"
                        ))
                        fig_vs_ml.add_trace(go.Scatter(
                            x=lims_vs, y=lims_vs,
                            mode="lines",
                            line=dict(color="#cc3333", width=1.5, dash="dash"),
                            name="1:1 line"
                        ))
                        fig_vs_ml.update_layout(
                            height=380,
                            xaxis_title=f"Actual DTS ({diag_vs['dts_unit']})",
                            yaxis_title=f"Predicted DTS ({diag_vs['dts_unit']})",
                            title=dict(
                                text=f"{diag_vs['best_name']} — Actual vs Predicted",
                                font=dict(size=12)
                            ),
                            plot_bgcolor="white", paper_bgcolor="white",
                            margin=dict(l=50, r=20, t=50, b=40),
                            showlegend=True,
                            legend=dict(font=dict(size=10)),
                        )
                        fig_vs_ml.update_xaxes(showgrid=True, gridcolor="#eeeeee")
                        fig_vs_ml.update_yaxes(showgrid=True, gridcolor="#eeeeee")
                        st.plotly_chart(fig_vs_ml, use_container_width=True)

                        if diag_vs.get("feat_importances") is not None:
                            fi_vs = pd.DataFrame({
                                "Feature"   : diag_vs["feature_labels"],
                                "Importance": diag_vs["feat_importances"],
                            }).sort_values("Importance", ascending=True)

                            fig_fi_vs = go.Figure(go.Bar(
                                x=fi_vs["Importance"],
                                y=fi_vs["Feature"],
                                orientation="h",
                                marker=dict(color="#01696f"),
                            ))
                            fig_fi_vs.update_layout(
                                height=max(300, len(fi_vs) * 26),
                                title=dict(text="Feature Importance", font=dict(size=12)),
                                xaxis_title="Importance",
                                plot_bgcolor="white", paper_bgcolor="white",
                                margin=dict(l=140, r=20, t=50, b=40),
                            )
                            fig_fi_vs.update_xaxes(showgrid=True, gridcolor="#eeeeee")
                            st.plotly_chart(fig_fi_vs, use_container_width=True)

            # ── Success banner (persists after rerun) ─────────────
            if st.session_state.get("vs_success"):
                s = st.session_state["vs_success"]
                st.success(
                    f"✅ **Vs saved as `{s['col']}`** via {s['source']}  |  "
                    f"Unit: {s['unit']}  |  "
                    f"Min: {s['min']:.1f}  Max: {s['max']:.1f}  Mean: {s['mean']:.1f}"
                )

        # ════════════════════════════════════════════════════
        #  EXPANDER 3 — ELASTIC MODULI LOG TRACKS
        # ════════════════════════════════════════════════════
        with st.expander("Elastic Moduli", expanded=False):

            from plotly.subplots import make_subplots

            # ── Initialize session state ──────────────────────
            for _key in ["moduli_success", "moduli_warnings", "moduli_prev_inputs"]:
                if _key not in st.session_state:
                    st.session_state[_key] = None

            # ── Deduplicate columns defensively ──────────────
            if st.session_state["df"].columns.duplicated().any():
                st.session_state["df"] = (
                    st.session_state["df"]
                    .loc[:, ~st.session_state["df"].columns.duplicated()]
                    .copy()
                )

            cur_df   = st.session_state["df"]
            all_cols = [c for c in cur_df.columns if c != depth_col]

            # ── Auto-detect required logs ─────────────────────
            vp_auto  = detect_curve(all_cols, ["VP", "VP_EST", "VEL", "VPCO"])
            vs_auto  = detect_curve(all_cols, ["VS", "VS_EST", "VSH"])
            rho_auto = detect_curve(all_cols, ["RHOB", "RHOZ", "DEN", "DENS", "DENSITY"])

            # ── Formula reference card ────────────────────────
            with st.container(border=True):
                st.markdown("**Formulas**")
                st.caption(
                    "Computed from Vp, Vs (m/s) and ρ (g/cc). "
                    "Moduli output in GPa / MPa / Pa. AI & SI in g/cc · m/s."
                )
                fc1, fc2 = st.columns(2)
                with fc1:
                    st.latex(r"\mu = \rho \, V_S^2 \quad \text{(Shear modulus)}")
                    st.latex(r"\lambda = \rho \, V_P^2 - 2\mu \quad \text{(Lamé's } \lambda\text{)}")
                    st.latex(r"K = \lambda + \tfrac{2}{3}\mu \quad \text{(Bulk modulus)}")
                    st.latex(r"E = \frac{9K\mu}{3K + \mu} \quad \text{(Young's modulus)}")
                with fc2:
                    st.latex(r"\nu = \frac{\lambda}{2(\lambda + \mu)} \quad \text{(Poisson's ratio)}")
                    st.latex(r"AI = \rho \cdot V_P \quad \text{(Acoustic impedance)}")
                    st.latex(r"SI = \rho \cdot V_S \quad \text{(Shear impedance)}")
                    st.latex(r"\lambda\rho = \lambda \cdot \rho, \quad \mu\rho = \mu \cdot \rho \quad \text{(LMR)}")

            st.markdown("")

            # ── Log selectors ─────────────────────────────────
            st.markdown(
                '<div class="petro-section-header">Input Log Selection</div>',
                unsafe_allow_html=True
            )

            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                vp_col_mod = st.selectbox(
                    "Vp log",
                    options=[None] + all_cols,
                    index=(all_cols.index(vp_auto) + 1 if vp_auto in all_cols else 0),
                    format_func=lambda v: "— select Vp —" if v is None else v,
                    key="ep_mod_vp"
                )
            with lc2:
                vs_col_mod = st.selectbox(
                    "Vs log",
                    options=[None] + all_cols,
                    index=(all_cols.index(vs_auto) + 1 if vs_auto in all_cols else 0),
                    format_func=lambda v: "— select Vs —" if v is None else v,
                    key="ep_mod_vs"
                )
            with lc3:
                rho_col_mod = st.selectbox(
                    "Density log (g/cc)",
                    options=[None] + all_cols,
                    index=(all_cols.index(rho_auto) + 1 if rho_auto in all_cols else 0),
                    format_func=lambda v: "— select RHOB —" if v is None else v,
                    key="ep_mod_rho"
                )

            uc1, uc2 = st.columns(2)
            with uc1:
                vp_unit_mod = st.selectbox(
                    "Vp / Vs input unit",
                    options=["m/s", "km/s", "ft/s"],
                    key="ep_mod_vp_unit"
                )
            with uc2:
                moduli_unit_out = st.selectbox(
                    "Moduli output unit",
                    options=["GPa", "MPa", "Pa"],
                    key="ep_mod_out_unit"
                )

            # ── Reset state when inputs change ───────────────
            _current_inputs = (vp_col_mod, vs_col_mod, rho_col_mod, vp_unit_mod, moduli_unit_out)
            if _current_inputs != st.session_state["moduli_prev_inputs"]:
                st.session_state["moduli_success"]     = None
                st.session_state["moduli_warnings"]    = None
                st.session_state["moduli_prev_inputs"] = _current_inputs

            # ── Outputs to compute ────────────────────────────
            st.markdown(
                '<div class="petro-section-header">Select Outputs to Compute</div>',
                unsafe_allow_html=True
            )

            oc1, oc2, oc3, oc4, oc5 = st.columns(5)
            calc_K      = oc1.checkbox("K  — Bulk modulus",       value=True, key="ep_mod_K")
            calc_G      = oc1.checkbox("μ  — Shear modulus",      value=True, key="ep_mod_G")
            calc_E      = oc2.checkbox("E  — Young's modulus",    value=True, key="ep_mod_E")
            calc_lam    = oc2.checkbox("λ  — Lamé's lambda",      value=True, key="ep_mod_lam")
            calc_nu     = oc3.checkbox("ν  — Poisson's ratio",    value=True, key="ep_mod_nu")
            calc_VR     = oc3.checkbox("Vp/Vs ratio",             value=True, key="ep_mod_VR")
            calc_AI     = oc4.checkbox("AI — Acoustic impedance", value=True, key="ep_mod_AI")
            calc_SI     = oc4.checkbox("SI — Shear impedance",    value=True, key="ep_mod_SI")
            calc_lamrho = oc5.checkbox("λρ — Lambda-Rho  (LMR)",  value=True, key="ep_mod_lamrho")
            calc_murho  = oc5.checkbox("μρ — Mu-Rho  (LMR)",      value=True, key="ep_mod_murho")

            st.markdown("---")

            # ── Preview metrics ───────────────────────────────
            if vp_col_mod and vs_col_mod and rho_col_mod:
                pv1, pv2, pv3 = st.columns(3)
                pv1.metric("Vp mean",   f"{cur_df[vp_col_mod].dropna().mean():.1f}")
                pv2.metric("Vs mean",   f"{cur_df[vs_col_mod].dropna().mean():.1f}")
                pv3.metric("RHOB mean", f"{cur_df[rho_col_mod].dropna().mean():.3f}")

            # ── Validation & compute ──────────────────────────
            missing_inputs = [
                name for name, col in
                [("Vp", vp_col_mod), ("Vs", vs_col_mod), ("Density", rho_col_mod)]
                if col is None
            ]

            if missing_inputs:
                st.warning(f"Select the following logs to proceed: {', '.join(missing_inputs)}")
            else:
                mb1, _ = st.columns([2, 6])
                with mb1:
                    apply_moduli = st.button(
                        "▶ Compute & Save",
                        key="apply_moduli",
                        type="primary",
                        use_container_width=True
                    )

                if apply_moduli:

                    # ── Column name map (safe ASCII names for df column labels) ──
                    COL_MAP = {
                        "K"    : "K",
                        "μ"    : "MU",
                        "E"    : "E",
                        "λ"    : "LAM",
                        "ν"    : "NU",
                        "VPVS" : "VPVS",
                        "AI"   : "AI",
                        "SI"   : "SI",
                        "λρ"   : "LAM_RHO",
                        "μρ"   : "MU_RHO",
                    }

                    mod_df  = st.session_state["df"].copy()

                    # ── Drop existing computed columns before re-saving ──
                    cols_to_drop = [c for c in COL_MAP.values() if c in mod_df.columns]
                    if cols_to_drop:
                        mod_df.drop(columns=cols_to_drop, inplace=True)

                    # ── Final deduplication guard ─────────────────────
                    if mod_df.columns.duplicated().any():
                        mod_df = mod_df.loc[:, ~mod_df.columns.duplicated()].copy()

                    vp_raw  = mod_df[vp_col_mod].copy().replace(0, np.nan)
                    vs_raw  = mod_df[vs_col_mod].copy().replace(0, np.nan)
                    rho_raw = mod_df[rho_col_mod].copy().replace(0, np.nan)

                    # Normalize velocities → m/s
                    if vp_unit_mod == "km/s":
                        vp_ms, vs_ms = vp_raw * 1000.0, vs_raw * 1000.0
                    elif vp_unit_mod == "ft/s":
                        vp_ms, vs_ms = vp_raw * 0.3048,  vs_raw * 0.3048
                    else:
                        vp_ms, vs_ms = vp_raw, vs_raw

                    # Density g/cc → kg/m³ for SI calculations
                    rho_si = rho_raw * 1000.0

                    # ── Core physics — SI units ───────────────────────
                    mu_pa   = rho_si * vs_ms**2
                    lam_pa  = rho_si * vp_ms**2 - 2.0 * mu_pa
                    K_pa    = lam_pa + (2.0 / 3.0) * mu_pa

                    denom_E  = (3.0 * K_pa + mu_pa).replace(0, np.nan)
                    E_pa     = (9.0 * K_pa * mu_pa) / denom_E

                    denom_nu = (2.0 * (lam_pa + mu_pa)).replace(0, np.nan)
                    nu_dim   = lam_pa / denom_nu

                    AI_val   = rho_raw * vp_ms
                    SI_val   = rho_raw * vs_ms
                    lamrho   = lam_pa  * rho_raw
                    murho    = mu_pa   * rho_raw
                    vpvs     = vp_ms   / vs_ms.replace(0, np.nan)

                    uf = {"GPa": 1e-9, "MPa": 1e-6, "Pa": 1.0}[moduli_unit_out]

                    # ── Safe column saver — no duplicates ────────────
                    computed_cols   = []   # display names (with unicode) for UI
                    computed_colmap = {}   # display name → df column name

                    def save_col(display_name, df_col_name, series):
                        mod_df[df_col_name] = series.round(4)
                        computed_cols.append(display_name)
                        computed_colmap[display_name] = df_col_name

                    if calc_K:      save_col("K",   "K",       K_pa   * uf)
                    if calc_G:      save_col("μ",   "MU",      mu_pa  * uf)
                    if calc_E:      save_col("E",   "E",       E_pa   * uf)
                    if calc_lam:    save_col("λ",   "LAM",     lam_pa * uf)
                    if calc_nu:     save_col("ν",   "NU",      nu_dim)
                    if calc_AI:     save_col("AI",  "AI",      AI_val)
                    if calc_SI:     save_col("SI",  "SI",      SI_val)
                    if calc_lamrho: save_col("λρ",  "LAM_RHO", lamrho)
                    if calc_murho:  save_col("μρ",  "MU_RHO",  murho)
                    if calc_VR:     save_col("Vp/Vs","VPVS",   vpvs)

                    # ── Physical sanity checks ────────────────────────
                    warnings_list = []
                    if "NU" in mod_df.columns:
                        nu_v = mod_df["NU"].dropna()
                        if (nu_v < -1).any() or (nu_v > 0.5).any():
                            warnings_list.append(
                                "**ν (Poisson's ratio)** has values outside [-1, 0.5] — "
                                "verify Vp and Vs inputs."
                            )
                    if "K" in mod_df.columns and (mod_df["K"].dropna() < 0).any():
                        warnings_list.append(
                            "**K (Bulk modulus)** has negative values — "
                            "check velocity and density magnitudes."
                        )
                    if "VPVS" in mod_df.columns:
                        vr_v = mod_df["VPVS"].dropna()
                        if (vr_v < 1.0).any():
                            warnings_list.append(
                                "**Vp/Vs ratio** below 1.0 detected — "
                                "physically impossible for most rocks."
                            )

                    st.session_state["df"]             = mod_df
                    st.session_state["moduli_success"] = {
                        "cols"      : computed_cols,
                        "colmap"    : computed_colmap,
                        "vp_unit"   : vp_unit_mod,
                        "mod_unit"  : moduli_unit_out,
                    }
                    st.session_state["moduli_warnings"] = warnings_list
                    st.rerun()

            # ══════════════════════════════════════════════════
            #  POST-COMPUTE — RESULTS & TRACKS
            # ══════════════════════════════════════════════════
            if st.session_state.get("moduli_success"):
                s      = st.session_state["moduli_success"]
                cur_df = st.session_state["df"]
                colmap = s.get("colmap", {c: c for c in s["cols"]})

                # ── Success banner ────────────────────────────
                st.success(
                    f"✅ Saved: **{', '.join(s['cols'])}**  |  "
                    f"Moduli in **{s['mod_unit']}**  |  Velocities in **{s['vp_unit']}**"
                )

                # ── Physical sanity warnings ──────────────────
                for w in (st.session_state.get("moduli_warnings") or []):
                    st.warning(w)

                # ── Stats QC table ────────────────────────────
                stat_rows = []
                for display_name in s["cols"]:
                    df_col = colmap.get(display_name, display_name)
                    if df_col in cur_df.columns:
                        d = cur_df[df_col].dropna()
                        stat_rows.append({
                            "Parameter" : display_name,
                            "Min"       : f"{d.min():.4f}",
                            "Max"       : f"{d.max():.4f}",
                            "Mean"      : f"{d.mean():.4f}",
                            "Valid pts" : f"{len(d):,}",
                        })
                if stat_rows:
                    st.dataframe(
                        pd.DataFrame(stat_rows).set_index("Parameter"),
                        use_container_width=True
                    )

                # ── Log tracks ───────────────────────────────
                TRACK_GROUPS = [
                    {
                        "title"  : f"Moduli ({s['mod_unit']})",
                        "keys"   : ["K", "μ", "E"],
                        "df_cols": ["K", "MU", "E"],
                        "colors" : ["#0a8b35", "#cc3333", "#323236"],
                    },
                    {
                        "title"  : f"λ ({s['mod_unit']})",
                        "keys"   : ["λ"],
                        "df_cols": ["LAM"],
                        "colors" : ["#5a4fcf"],
                    },
                    {
                        "title"  : "ν  &  Vp/Vs",
                        "keys"   : ["ν", "Vp/Vs"],
                        "df_cols": ["NU", "VPVS"],
                        "colors" : ["#01696f", "#cc3333"],
                    },
                    {
                        "title"  : "Impedances (g/cc·m/s)",
                        "keys"   : ["AI", "SI"],
                        "df_cols": ["AI", "SI"],
                        "colors" : ["#01696f", "#cc3333"],
                    },
                    {
                        "title"  : "LMR",
                        "keys"   : ["λρ", "μρ"],
                        "df_cols": ["LAM_RHO", "MU_RHO"],
                        "colors" : ["#5a4fcf", "#e00000"],
                    },
                ]

                active_tracks = [
                    grp for grp in TRACK_GROUPS
                    if any(c in cur_df.columns for c in grp["df_cols"])
                ]
                n_tracks = len(active_tracks)

                if n_tracks > 0:
                    st.markdown(
                        '<div class="petro-section-header">Log Tracks</div>',
                        unsafe_allow_html=True
                    )

                    depth_vals = cur_df[depth_col].values

                    fig = make_subplots(
                        rows=1,
                        cols=n_tracks,
                        shared_yaxes=True,
                        horizontal_spacing=0.02,
                        subplot_titles=[grp["title"] for grp in active_tracks],
                    )

                    for t_idx, grp in enumerate(active_tracks):
                        col_num = t_idx + 1
                        present = [
                            (key, df_col)
                            for key, df_col in zip(grp["keys"], grp["df_cols"])
                            if df_col in cur_df.columns
                        ]

                        for ci, (key, df_col) in enumerate(present):
                            fig.add_trace(
                                go.Scatter(
                                    x=cur_df[df_col].values,
                                    y=depth_vals,
                                    mode="lines",
                                    name=key,
                                    line=dict(
                                        color=grp["colors"][ci % len(grp["colors"])],
                                        width=1.2,
                                    ),
                                    legendgroup=key,
                                    showlegend=True,
                                ),
                                row=1, col=col_num,
                            )

                        xaxis_key = f"xaxis{'' if col_num == 1 else col_num}"
                        fig.update_layout(**{
                            xaxis_key: dict(
                                side="top",
                                showgrid=True,
                                gridcolor="#eeeeee",
                                tickfont=dict(size=8),
                                zeroline=True,
                                zerolinecolor="#bbbbbb",
                                zerolinewidth=1,
                                title=dict(text="", font=dict(size=1)),
                            )
                        })

                    fig.update_layout(
                        yaxis=dict(
                            autorange="reversed",
                            title=dict(text=depth_col, font=dict(size=10)),
                            showgrid=True,
                            gridcolor="#eeeeee",
                            tickfont=dict(size=8),
                        ),
                        height=580,
                        margin=dict(l=60, r=20, t=60, b=20),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        legend=dict(
                            orientation="v",
                            x=1.01, y=1.0,
                            font=dict(size=9),
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor="#dddddd",
                            borderwidth=1,
                        ),
                    )

                    for annotation in fig.layout.annotations:
                        annotation.font.size = 10
                        annotation.y         = 1.04

                    st.plotly_chart(fig, use_container_width=True)

                # ════════════════════════════════════════════════════
        #  EXPANDER 4 — ELASTIC CROSSPLOTS
        # ════════════════════════════════════════════════════
        with st.expander("Elastic Crossplots", expanded=False):

            # ── Deduplicate FIRST — before any column access ──
            _raw_df = st.session_state["df"]
            if _raw_df.columns.duplicated().any():
                # FIX 1: keep="last" preserves the most recently computed column
                _raw_df = _raw_df.loc[:, ~_raw_df.columns.duplicated(keep="last")].copy()
                st.session_state["df"] = _raw_df

            cur_df   = _raw_df
            all_cols = [c for c in cur_df.columns if c != depth_col]

            # ── Auto-detect logs ──────────────────────────────
            vp_auto_xp  = detect_curve(all_cols, ["VP", "VP_EST", "VEL", "VPCO"])
            vs_auto_xp  = detect_curve(all_cols, ["VS", "VS_EST"])
            rho_auto_xp = detect_curve(all_cols, ["RHOB", "RHOZ", "DEN", "DENS", "DENSITY"])
            ai_auto_xp  = detect_curve(all_cols, ["AI"])
            gr_auto_xp  = detect_curve(all_cols, ["GR", "CGR", "GRS", "HCGR", "GR_EDIT"])
            phi_auto_xp = detect_curve(all_cols, ["PHIE", "PHIT", "PHI", "NPHI", "POROSITY"])
            lam_auto_xp = detect_curve(all_cols, ["LAM_RHO", "LAMRHO"])
            mu_auto_xp  = detect_curve(all_cols, ["MU_RHO",  "MURHO"])

            # ── Reference line coefficients ───────────────────
            CAST_REF = {
                "Mudrock (Shale/Silt)": [-1.172,   0.8621],
                "Sandstone"           : [-0.8559,  0.8042],
                "Limestone"           : [-1.03049, 1.01677, -0.05508],
                "Dolomite"            : [-0.07775, 0.5832],
                "Shale"               : [-0.8674,  0.7696],
            }
            BROCHER_COEFFS = [0.7858, -1.2344, 0.7949, -0.1238, 0.0064]

            # ── Zone polygons (Vp vs Vs space, km/s) ─────────
            ZONE_POLYGONS = {
                "Gas Sand"  : dict(vp=[2.0,3.2,3.2,2.0,2.0], vs=[1.3,2.1,1.5,0.9,1.3],
                                fill="rgba(30,144,255,0.08)",  border="#1e90ff"),
                "Brine Sand": dict(vp=[2.8,4.2,4.2,2.8,2.8], vs=[1.4,2.5,2.0,1.1,1.4],
                                fill="rgba(0,180,120,0.08)",   border="#00b478"),
                "Shale"     : dict(vp=[2.0,4.0,4.0,2.0,2.0], vs=[0.5,2.0,1.5,0.3,0.5],
                                fill="rgba(160,110,50,0.08)",  border="#a06e32"),
                "Carbonate" : dict(vp=[4.5,6.5,6.5,4.5,4.5], vs=[2.3,3.7,3.2,1.9,2.3],
                                fill="rgba(180,0,180,0.08)",   border="#b400b4"),
            }

            # ══════════════════════════════════════════════════
            #  HELPER FUNCTIONS
            # ══════════════════════════════════════════════════

            def _col(df, name):
                """Safe 1D numpy extractor — handles duplicate-column DataFrames."""
                c = df[name]
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[:, 0]
                return c.to_numpy(dtype=float).ravel()

            def _to_kms(arr, unit):
                a = np.asarray(arr, dtype=float).ravel()
                if unit == "m/s":  return a / 1000.0
                if unit == "ft/s": return a * 0.0003048
                return a

            def _from_kms(arr, unit):
                a = np.asarray(arr, dtype=float).ravel()
                if unit == "m/s":  return a * 1000.0
                if unit == "ft/s": return a / 0.0003048
                return a

            def _castagna(vp_kms, litho):
                a = CAST_REF[litho]
                return np.clip(
                    sum(a[i] * vp_kms**i for i in range(len(a))), 0, None
                )

            def _brocher(vp_kms):
                return np.clip(
                    sum(BROCHER_COEFFS[i] * vp_kms**i
                        for i in range(len(BROCHER_COEFFS))), 0, None
                )

            def _make_scatter(x, y, c_vals, c_label, msize, cscale="RdYlGn_r"):
                x_a = np.asarray(x, dtype=float).ravel()
                y_a = np.asarray(y, dtype=float).ravel()
                if c_vals is not None:
                    c_a = np.asarray(c_vals, dtype=float).ravel()
                    return go.Scatter(
                        x=x_a, y=y_a, mode="markers",
                        marker=dict(
                            size=msize, color=c_a, colorscale=cscale,
                            colorbar=dict(
                                title=c_label, thickness=12, len=0.55,
                                tickfont=dict(size=8), titlefont=dict(size=9),
                            ),
                            opacity=0.75, showscale=True, line=dict(width=0),
                        ),
                        name="Data",
                    )
                return go.Scatter(
                    x=x_a, y=y_a, mode="markers",
                    marker=dict(
                        size=msize, color="#01696f",
                        opacity=0.65, line=dict(width=0),
                    ),
                    name="Data",
                )

            def _base_layout(title, xlab, ylab, height=500):
                return dict(
                    height=height,
                    title=dict(text=title, font=dict(size=12)),
                    xaxis_title=xlab,
                    yaxis_title=ylab,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(l=65, r=20, t=55, b=55),
                    legend=dict(
                        font=dict(size=10),
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="#dddddd",
                        borderwidth=1,
                    ),
                )

            def _add_grid(fig):
                fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
                fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)

            def _add_zones(fig, unit, toggles):
                for zname, zd in ZONE_POLYGONS.items():
                    if not toggles.get(zname):
                        continue
                    fig.add_trace(go.Scatter(
                        x=_from_kms(np.array(zd["vp"]), unit),
                        y=_from_kms(np.array(zd["vs"]), unit),
                        fill="toself",
                        fillcolor=zd["fill"],
                        line=dict(color=zd["border"], width=1.2, dash="dot"),
                        mode="lines",
                        name=zname,
                        hoverinfo="name",
                    ))

            def _hline(fig, y, label, color):
                fig.add_hline(
                    y=y,
                    line=dict(color=color, width=1.2, dash="dot"),
                    annotation_text=label,
                    annotation_position="right",
                    annotation_font_size=9,
                    annotation_font_color=color,
                )

            def _sel(label, auto, key, placeholder="— none —"):
                return st.selectbox(
                    label,
                    options=[None] + all_cols,
                    index=(all_cols.index(auto) + 1 if auto and auto in all_cols else 0),
                    format_func=lambda v: placeholder if v is None else v,
                    key=key,
                )

            # ══════════════════════════════════════════════════
            #  LAYOUT — Controls (left) | Plot (right)
            # ══════════════════════════════════════════════════
            ctrl_col, plot_col = st.columns([3, 7])

            with ctrl_col:

                st.markdown('<div class="petro-section-header">Log Selection</div>',
                            unsafe_allow_html=True)

                vp_col_xp     = _sel("Vp log",             vp_auto_xp,  "ep_xp_vp",     "— select Vp —")
                vs_col_xp     = _sel("Vs log",             vs_auto_xp,  "ep_xp_vs",     "— select Vs —")
                rho_col_xp    = _sel("Density log (g/cc)", rho_auto_xp, "ep_xp_rho",    "— select RHOB —")
                phi_col_xp    = _sel("Porosity log",       phi_auto_xp, "ep_xp_phi")
                ai_col_xp     = _sel("AI log",             ai_auto_xp,  "ep_xp_ai")
                lamrho_col_xp = _sel("λρ log (LAM_RHO)",  lam_auto_xp, "ep_xp_lamrho")
                murho_col_xp  = _sel("μρ log (MU_RHO)",   mu_auto_xp,  "ep_xp_murho")

                st.markdown("---")

                st.markdown('<div class="petro-section-header">Display Options</div>',
                            unsafe_allow_html=True)

                xp_vp_unit = st.selectbox(
                    "Velocity unit",
                    options=["m/s", "km/s", "ft/s"],
                    key="ep_xp_vp_unit",
                )

                color_mode = st.selectbox(
                    "Color by",
                    options=["Single color", "Log value (continuous)",
                            "GR lithology flag", "Depth gradient"],
                    key="ep_xp_color_mode",
                )

                color_col_xp = None
                gr_sand_cut  = 75.0
                gr_shale_cut = 120.0

                if color_mode == "Log value (continuous)":
                    color_col_xp = _sel("Color log", gr_auto_xp, "ep_xp_color_log",
                                        "— select log —")
                elif color_mode == "GR lithology flag":
                    color_col_xp = _sel("GR log", gr_auto_xp, "ep_xp_gr_log",
                                        "— select GR —")
                    cc1, cc2 = st.columns(2)
                    gr_sand_cut  = cc1.number_input("Sand  (GR ≤)",  value=75.0,
                                                    step=5.0, key="ep_xp_gr_sand")
                    gr_shale_cut = cc2.number_input("Shale (GR ≥)", value=120.0,
                                                    step=5.0, key="ep_xp_gr_shale")

                xp_marker_size = st.slider(
                    "Marker size", min_value=2, max_value=10,
                    value=4, step=1, key="ep_xp_marker_size",
                )

                st.markdown("---")

                st.markdown('<div class="petro-section-header">Depth Filter</div>',
                            unsafe_allow_html=True)

                _depth_vals = cur_df[depth_col].dropna()
                d_min = float(_depth_vals.min())
                d_max = float(_depth_vals.max())

                if "xplot_depth_range" not in st.session_state:
                    st.session_state["xplot_depth_range"] = (d_min, d_max)
                else:
                    cached = st.session_state["xplot_depth_range"]
                    if cached[0] < d_min or cached[1] > d_max:
                        st.session_state["xplot_depth_range"] = (d_min, d_max)

                depth_range = st.slider(
                    "Depth range",
                    min_value=d_min,
                    max_value=d_max,
                    value=st.session_state["xplot_depth_range"],
                    step=max(0.1, round((d_max - d_min) / 500, 1)),
                    format="%.1f",
                    key="ep_xp_depth_range",
                )
                st.session_state["xplot_depth_range"] = depth_range

                st.markdown("---")

                st.markdown('<div class="petro-section-header">Reference Lines</div>',
                            unsafe_allow_html=True)

                cast_line_litho = st.selectbox(
                    "Castagna line",
                    options=["None"] + list(CAST_REF.keys()),
                    index=1,
                    key="ep_xp_cast_litho",
                )
                show_brocher = st.checkbox("Brocher (2005)", value=False, key="ep_xp_brocher")
                show_gardner = st.checkbox("Gardner trend",  value=False, key="ep_xp_gardner")

                st.markdown("---")

                st.markdown('<div class="petro-section-header">Zone Annotations</div>',
                            unsafe_allow_html=True)

                zone_toggles = {
                    z: st.checkbox(z, value=False, key=f"ep_xp_zone_{z}")
                    for z in ZONE_POLYGONS
                }

            # ── Depth-filtered dataframe — always reset_index ──
            fdf = cur_df[
                cur_df[depth_col].between(depth_range[0], depth_range[1])
            ].copy().reset_index(drop=True)

            # ── Color array builder ───────────────────────────
            def _build_color(df, col, mode, gr_sand, gr_shale):
                if mode == "Log value (continuous)" and col and col in df.columns:
                    return _col(df, col), col, "RdYlGn_r"
                if mode == "GR lithology flag" and col and col in df.columns:
                    gr   = _col(df, col)
                    cats = np.where(gr <= gr_sand, 0.0,
                        np.where(gr >= gr_shale, 1.0, 0.5))
                    return cats, "Lithology", [[0,"#f4d44d"],[0.5,"#a0a0a0"],[1,"#6b4c2a"]]
                if mode == "Depth gradient":
                    return _col(df, depth_col), "Depth", "Viridis"
                return None, None, None

            _c_all, _c_label, _c_scale = _build_color(
                fdf, color_col_xp, color_mode, gr_sand_cut, gr_shale_cut
            )

            # FIX 2: Use positional iloc-safe slicing for color array
            def _cvals(pos_idx):
                """Slice color array by positional index.
                pos_idx comes from dropna'd subset of reset-indexed fdf — always safe."""
                if _c_all is None:
                    return None
                idx = np.asarray(pos_idx)
                # Guard: clamp indices to valid range
                idx = idx[idx < len(_c_all)]
                return _c_all[idx]

            # ══════════════════════════════════════════════════
            #  PLOT COLUMN — Six crossplot tabs
            # ══════════════════════════════════════════════════
            with plot_col:
                tabs = st.tabs([
                    "Vp vs Vs",
                    "Vp/Vs vs Vp",
                    "AI vs Vp/Vs",
                    "Vp vs RHOB",
                    "Vp vs Porosity",
                    "λρ vs μρ",
                ])

                # ── TAB 1 — Vp vs Vs ─────────────────────────
                with tabs[0]:
                    if vp_col_xp is None or vs_col_xp is None:
                        st.warning("Select Vp and Vs logs in the left panel.")
                    else:
                        _v1 = fdf[[depth_col, vp_col_xp, vs_col_xp]].dropna(
                            subset=[vp_col_xp, vs_col_xp]
                        ).reset_index(drop=True)          # FIX 2 — reset after dropna
                        _i1  = _v1.index.to_numpy()
                        vp_1 = _col(_v1, vp_col_xp)
                        vs_1 = _col(_v1, vs_col_xp)

                        fig1 = go.Figure()
                        fig1.add_trace(_make_scatter(
                            vp_1, vs_1, _cvals(_i1), _c_label,
                            xp_marker_size, _c_scale or "RdYlGn_r",
                        ))

                        vp_kms1 = _to_kms(vp_1, xp_vp_unit)

                        if cast_line_litho != "None":
                            vp_r = np.linspace(vp_kms1.min(), vp_kms1.max(), 300)
                            fig1.add_trace(go.Scatter(
                                x=_from_kms(vp_r, xp_vp_unit),
                                y=_from_kms(_castagna(vp_r, cast_line_litho), xp_vp_unit),
                                mode="lines",
                                line=dict(color="#cc3333", width=1.8, dash="dash"),
                                name=f"Castagna — {cast_line_litho}",
                            ))

                        if show_brocher:
                            vp_r = np.linspace(
                                max(1.5, vp_kms1.min()),
                                min(8.5, vp_kms1.max()), 300,
                            )
                            fig1.add_trace(go.Scatter(
                                x=_from_kms(vp_r, xp_vp_unit),
                                y=_from_kms(_brocher(vp_r), xp_vp_unit),
                                mode="lines",
                                line=dict(color="#e07b00", width=1.8, dash="dot"),
                                name="Brocher (2005)",
                            ))

                        _add_zones(fig1, xp_vp_unit, zone_toggles)
                        fig1.update_layout(**_base_layout(
                            "Vp vs Vs",
                            f"Vp ({xp_vp_unit})",
                            f"Vs ({xp_vp_unit})",
                        ))
                        _add_grid(fig1)
                        st.plotly_chart(fig1, use_container_width=True)

                # ── TAB 2 — Vp/Vs vs Vp ──────────────────────
                with tabs[1]:
                    if vp_col_xp is None or vs_col_xp is None:
                        st.warning("Select Vp and Vs logs in the left panel.")
                    else:
                        _v2 = fdf[[depth_col, vp_col_xp, vs_col_xp]].dropna(
                            subset=[vp_col_xp, vs_col_xp]
                        ).reset_index(drop=True)          # FIX 2
                        _i2   = _v2.index.to_numpy()
                        vp_2  = _col(_v2, vp_col_xp)
                        vs_2  = _col(_v2, vs_col_xp)
                        vs_2  = np.where(vs_2 == 0, np.nan, vs_2)
                        vpvs2 = np.where(np.isfinite(vs_2), vp_2 / vs_2, np.nan)

                        fig2 = go.Figure()
                        fig2.add_trace(_make_scatter(
                            vp_2, vpvs2, _cvals(_i2), _c_label,
                            xp_marker_size, _c_scale or "RdYlGn_r",
                        ))

                        vp_kms2 = _to_kms(vp_2, xp_vp_unit)
                        if cast_line_litho != "None":
                            vp_r  = np.linspace(vp_kms2.min(), vp_kms2.max(), 300)
                            vs_c  = _castagna(vp_r, cast_line_litho)
                            fig2.add_trace(go.Scatter(
                                x=_from_kms(vp_r, xp_vp_unit),
                                y=np.where(vs_c > 0, vp_r / vs_c, np.nan),
                                mode="lines",
                                line=dict(color="#cc3333", width=1.8, dash="dash"),
                                name=f"Castagna — {cast_line_litho}",
                            ))

                        _hline(fig2, 1.5, "Vp/Vs = 1.5  (gas)",   "#1e90ff")
                        _hline(fig2, 1.9, "Vp/Vs = 1.9  (brine)", "#00b478")

                        fig2.update_layout(**_base_layout(
                            "Vp/Vs Ratio vs Vp",
                            f"Vp ({xp_vp_unit})", "Vp/Vs",
                        ))
                        _add_grid(fig2)
                        st.plotly_chart(fig2, use_container_width=True)

                # ── TAB 3 — AI vs Vp/Vs ──────────────────────
                with tabs[2]:
                    if None in (ai_col_xp, vp_col_xp, vs_col_xp):
                        st.info(
                            "Requires Vp, Vs, and AI logs. "
                            "Compute AI in the **Elastic Moduli** expander first."
                        )
                    else:
                        _v3 = fdf[[depth_col, vp_col_xp, vs_col_xp, ai_col_xp]].dropna(
                            subset=[vp_col_xp, vs_col_xp, ai_col_xp]
                        ).reset_index(drop=True)          # FIX 2
                        _i3   = _v3.index.to_numpy()
                        vp_3  = _col(_v3, vp_col_xp)
                        vs_3  = _col(_v3, vs_col_xp)
                        ai_3  = _col(_v3, ai_col_xp)
                        vs_3  = np.where(vs_3 == 0, np.nan, vs_3)
                        vpvs3 = np.where(np.isfinite(vs_3), vp_3 / vs_3, np.nan)

                        fig3 = go.Figure()
                        fig3.add_trace(_make_scatter(
                            ai_3, vpvs3, _cvals(_i3), _c_label,
                            xp_marker_size, _c_scale or "RdYlGn_r",
                        ))

                        _hline(fig3, 1.5, "Vp/Vs = 1.5  (gas)",   "#1e90ff")
                        _hline(fig3, 1.9, "Vp/Vs = 1.9  (brine)", "#00b478")

                        fig3.update_layout(**_base_layout(
                            "AI vs Vp/Vs",
                            "AI  (g/cc · m/s)", "Vp/Vs",
                        ))
                        _add_grid(fig3)
                        st.plotly_chart(fig3, use_container_width=True)

                # ── TAB 4 — Vp vs RHOB ───────────────────────
                with tabs[3]:
                    if vp_col_xp is None or rho_col_xp is None:
                        st.warning("Select Vp and Density logs in the left panel.")
                    else:
                        _v4 = fdf[[depth_col, vp_col_xp, rho_col_xp]].dropna(
                            subset=[vp_col_xp, rho_col_xp]
                        ).reset_index(drop=True)          # FIX 2
                        _i4  = _v4.index.to_numpy()
                        vp_4 = _col(_v4, vp_col_xp)
                        rh_4 = _col(_v4, rho_col_xp)

                        fig4 = go.Figure()
                        fig4.add_trace(_make_scatter(
                            rh_4, vp_4, _cvals(_i4), _c_label,
                            xp_marker_size, _c_scale or "RdYlGn_r",
                        ))

                        if show_gardner:
                            vp_ms4   = _to_kms(vp_4, xp_vp_unit) * 1000.0
                            vp_range = np.linspace(vp_ms4.min(), vp_ms4.max(), 300)
                            rho_gard = 0.31 * vp_range**0.25
                            fig4.add_trace(go.Scatter(
                                x=rho_gard,
                                y=_from_kms(vp_range / 1000.0, xp_vp_unit),
                                mode="lines",
                                line=dict(color="#cc3333", width=1.8, dash="dash"),
                                name="Gardner  (a=0.31, b=0.25)",
                            ))

                        fig4.update_layout(**_base_layout(
                            "Vp vs RHOB  (Gardner validation)",
                            "RHOB  (g/cc)", f"Vp ({xp_vp_unit})",
                        ))
                        _add_grid(fig4)
                        st.plotly_chart(fig4, use_container_width=True)

                # ── TAB 5 — Vp vs Porosity ────────────────────
                with tabs[4]:
                    if vp_col_xp is None or phi_col_xp is None:
                        st.info("Select a Porosity log in the left panel.")
                    else:
                        _v5 = fdf[[depth_col, vp_col_xp, phi_col_xp]].dropna(
                            subset=[vp_col_xp, phi_col_xp]
                        ).reset_index(drop=True)          # FIX 2
                        _i5   = _v5.index.to_numpy()
                        vp_5  = _col(_v5, vp_col_xp)
                        phi_5 = _col(_v5, phi_col_xp)

                        fig5 = go.Figure()
                        fig5.add_trace(_make_scatter(
                            phi_5, vp_5, _cvals(_i5), _c_label,
                            xp_marker_size, _c_scale or "RdYlGn_r",
                        ))

                        wc1, wc2 = st.columns(2)
                        vf_w = wc1.number_input(
                            "Wyllie Vf — fluid (m/s)",
                            value=1500, step=100, key="ep_xp_wyllie_vf",
                        )
                        vm_w = wc2.number_input(
                            "Wyllie Vm — matrix (m/s)",
                            value=5500, step=100, key="ep_xp_wyllie_vm",
                        )

                        phi_r = np.linspace(
                            max(0.001, float(phi_5.min())),
                            min(0.45,  float(phi_5.max())),
                            200,
                        )
                        vf_safe = max(float(vf_w), 1.0)
                        vm_safe = max(float(vm_w), 1.0)
                        vp_wy   = 1.0 / (phi_r / vf_safe + (1.0 - phi_r) / vm_safe)

                        fig5.add_trace(go.Scatter(
                            x=phi_r,
                            y=_from_kms(vp_wy / 1000.0, xp_vp_unit),
                            mode="lines",
                            line=dict(color="#cc3333", width=1.8, dash="dash"),
                            name=f"Wyllie  (Vf={vf_w}, Vm={vm_w} m/s)",
                        ))

                        fig5.update_layout(**_base_layout(
                            "Vp vs Porosity  (Wyllie time-average)",
                            "Porosity (frac)", f"Vp ({xp_vp_unit})",
                        ))
                        _add_grid(fig5)
                        st.plotly_chart(fig5, use_container_width=True)

                # ── TAB 6 — λρ vs μρ (LMR) ───────────────────
                with tabs[5]:
                    if lamrho_col_xp is None or murho_col_xp is None:
                        st.info(
                            "Compute **LAM_RHO** and **MU_RHO** in the "
                            "**Elastic Moduli** expander first."
                        )
                    else:
                        _v6 = fdf[[depth_col, lamrho_col_xp, murho_col_xp]].dropna(
                            subset=[lamrho_col_xp, murho_col_xp]
                        ).reset_index(drop=True)          # FIX 2
                        _i6  = _v6.index.to_numpy()
                        lr_6 = _col(_v6, lamrho_col_xp)
                        mr_6 = _col(_v6, murho_col_xp)

                        fig6 = go.Figure()
                        fig6.add_trace(_make_scatter(
                            mr_6, lr_6, _cvals(_i6), _c_label,
                            xp_marker_size, _c_scale or "RdYlGn_r",
                        ))

                        _xr = np.linspace(float(mr_6.min()), float(mr_6.max()), 100)
                        fig6.add_trace(go.Scatter(
                            x=_xr, y=_xr,
                            mode="lines",
                            line=dict(color="#cc3333", width=1.5, dash="dash"),
                            name="λρ = μρ  (gas indicator)",
                        ))

                        _hline(fig6, 0, "λρ = 0  (clean gas sand)", "#1e90ff")

                        fig6.update_layout(**_base_layout(
                            "LMR Crossplot  (λρ vs μρ)",
                            "μρ", "λρ",
                        ))
                        _add_grid(fig6)
                        st.plotly_chart(fig6, use_container_width=True)

    

    # ════════════════════════════════════════════════════════
    #  TAB 7 — Rock Physics
    # ════════════════════════════════════════════════════════
    with tab7:

        # ════════════════════════════════════════════════════════════════
        # SECTION 1 — CRITICAL POROSITY MODEL (Nur et al., 1991, 1995)
        # ════════════════════════════════════════════════════════════════
        with st.expander("Critical Porosity Model", expanded=False):

            st.markdown('<div class="rp-section-header">Theory</div>', unsafe_allow_html=True)

            st.markdown("""
            The **Critical Porosity Model** proposed by Nur et al. (1991, 1995) is based on the
            concept of **critical porosity (ϕc)** — the porosity above which mineral grains lose
            contact and the rock loses its rigidity, transforming into a suspension where the
            **fluid phase** is the load-bearing component.

            - **ϕ > ϕc  (Suspension domain):** The rock falls apart. The fluid bears the load.
              Effective moduli are approximated by the **Reuss (iso-stress) average**.
              Since fluid shear modulus = 0, the effective shear modulus is also 0.
            - **ϕ < ϕc  (Load-bearing domain):** Mineral grains are in contact and bear the load.
              Moduli drop from mineral values at ϕ = 0 to suspension values at ϕ = ϕc.

            Grain sorting and angularity at deposition define ϕc. As porosity decreases through
            compaction and diagenesis, elastic stiffness rises and the rock moves upward along
            the trajectory.
            """)

            st.markdown("---")

            st.markdown('<div class="rp-section-header">Governing Equations</div>', unsafe_allow_html=True)

            with st.container(border=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Dry-frame bulk modulus**")
                    st.latex(r"K_{dry} = K_0 \left(1 - \frac{\phi}{\phi_c}\right)")
                with c2:
                    st.markdown("**Dry-frame shear modulus**")
                    st.latex(r"\mu_{dry} = \mu_0 \left(1 - \frac{\phi}{\phi_c}\right)")
                st.caption(
                    "K₀ = Mineral bulk modulus (GPa)  |  "
                    "µ₀ = Mineral shear modulus (GPa)  |  "
                    "ϕ = Porosity  |  ϕc = Critical porosity"
                )

            st.markdown("---")

            with st.expander("Reference table for Mineral Moduli", expanded=False):
                st.dataframe(pd.DataFrame([
                    {"Mineral": "Quartz (Sandstone)",  "K₀ (GPa)": 38.0, "µ₀ (GPa)": 44.0, "ρ (g/cc)": 2.65},
                    {"Mineral": "Calcite (Limestone)", "K₀ (GPa)": 76.8, "µ₀ (GPa)": 32.0, "ρ (g/cc)": 2.71},
                    {"Mineral": "Dolomite",            "K₀ (GPa)": 94.9, "µ₀ (GPa)": 45.0, "ρ (g/cc)": 2.87},
                    {"Mineral": "Clay (Shale)",        "K₀ (GPa)": 21.0, "µ₀ (GPa)": 7.0,  "ρ (g/cc)": 2.58},
                    {"Mineral": "Feldspar",            "K₀ (GPa)": 37.5, "µ₀ (GPa)": 15.0, "ρ (g/cc)": 2.62},
                ]), use_container_width=True, hide_index=True)

            st.markdown('<div class="rp-section-header">Input Parameters</div>', unsafe_allow_html=True)

            CPM_PRESETS = {
                "Sandstone": {"K0": 38.0, "G0": 44.0, "phic": 0.40},
                "Shale":     {"K0": 21.0, "G0": 7.0,  "phic": 0.55},
                "Limestone": {"K0": 76.8, "G0": 32.0, "phic": 0.40},
                "Dolomite":  {"K0": 94.9, "G0": 45.0, "phic": 0.40},
                "Chalk":     {"K0": 76.8, "G0": 32.0, "phic": 0.65},
            }

            cpm_selected = st.multiselect(
                "Select lithologies to compute & compare",
                options=list(CPM_PRESETS.keys()),
                default=["Sandstone", "Shale"],
                key="cpm_liths",
            )

            if not cpm_selected:
                st.info("Select at least one lithology above to proceed.")
            else:
                st.markdown("**Adjust parameters per lithology if needed:**")

                cpm_inputs = {}
                for lith in cpm_selected:
                    with st.expander(f"{lith}", expanded=False):
                        preset = CPM_PRESETS[lith]
                        lc1, lc2, lc3 = st.columns(3)
                        k0_val   = lc1.number_input("K₀ (GPa)",  value=preset["K0"],
                                      min_value=1.0, max_value=200.0,
                                      format="%.2f", key=f"cpm_K0_{lith}")
                        g0_val   = lc2.number_input("µ₀ (GPa)",  value=preset["G0"],
                                      min_value=1.0, max_value=200.0,
                                      format="%.2f", key=f"cpm_G0_{lith}")
                        phic_val = lc3.number_input("ϕc",        value=preset["phic"],
                                      min_value=0.01, max_value=0.80,
                                      format="%.3f", key=f"cpm_phic_{lith}")
                        cpm_inputs[lith] = {"K0": k0_val, "G0": g0_val, "phic": phic_val}

                st.markdown("---")

                btn_cpm, _ = st.columns([2, 6])
                with btn_cpm:
                    run_cpm = st.button("▶ Compute & Plot", key="run_cpm",
                                        type="primary", use_container_width=True)

                if run_cpm or st.session_state.get("cpm_results"):

                    if run_cpm:
                        cpm_computed = {}
                        for lith, params in cpm_inputs.items():
                            phi_arr = np.linspace(0.0, float(params["phic"]), 300)
                            K_dry   = params["K0"] * (1 - phi_arr / params["phic"])
                            G_dry   = params["G0"] * (1 - phi_arr / params["phic"])
                            cpm_computed[lith] = {
                                "phi": phi_arr, "K": K_dry, "G": G_dry,
                                "K0": params["K0"], "G0": params["G0"],
                                "phic": params["phic"],
                            }
                        st.session_state["cpm_results"] = cpm_computed

                    cpm_res = st.session_state["cpm_results"]

                    st.markdown('<div class="rp-section-header">Results Summary</div>',
                                unsafe_allow_html=True)
                    summary_rows = []
                    for lith, r in cpm_res.items():
                        summary_rows.append({
                            "Lithology":         lith,
                            "K₀ (GPa)":          f"{r['K0']:.2f}",
                            "µ₀ (GPa)":          f"{r['G0']:.2f}",
                            "ϕc":                f"{r['phic']:.3f}",
                            "K_dry @ ϕc (GPa)":  f"{r['K'][-1]:.4f}",
                            "µ_dry @ ϕc (GPa)":  f"{r['G'][-1]:.4f}",
                        })
                    st.dataframe(pd.DataFrame(summary_rows),
                                 use_container_width=True, hide_index=True)

                    st.markdown("---")

                    st.markdown('<div class="rp-section-header">Bulk & Shear Moduli vs Porosity</div>',
                                unsafe_allow_html=True)

                    lith_list = list(cpm_res.items())

                    for i in range(0, len(lith_list), 2):
                        cols = st.columns(2)
                        for j, (lith, r) in enumerate(lith_list[i:i+2]):
                            with cols[j]:
                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    x=r["phi"], y=r["K"],
                                    mode="lines",
                                    name="Bulk Modulus (GPa)",
                                    line=dict(color="#d62728", width=2),
                                ))
                                fig.add_trace(go.Scatter(
                                    x=r["phi"], y=r["G"],
                                    mode="lines",
                                    name="Shear Modulus (GPa)",
                                    line=dict(color="#1f77b4", width=2),
                                ))
                                fig.add_vline(
                                    x=float(r["phic"]),
                                    line=dict(color="gray", width=1.2, dash="dash"),
                                    annotation_text=f"ϕc = {r['phic']:.2f}",
                                    annotation_position="top right",
                                    annotation_font=dict(size=9, color="gray"),
                                )
                                fig.update_layout(
                                    title=dict(
                                        text=lith,
                                        font=dict(size=13), x=0.5, xanchor="center"),
                                    xaxis=dict(
                                        title="Porosity",
                                        showgrid=True, gridcolor="#eeeeee",
                                        showline=True, linecolor="#aaaaaa",
                                        zeroline=False, tickformat=".2f"),
                                    yaxis=dict(
                                        title="Elastic Properties (GPa)",
                                        showgrid=True, gridcolor="#eeeeee",
                                        showline=True, linecolor="#aaaaaa",
                                        zeroline=False),
                                    legend=dict(
                                        font=dict(size=9),
                                        bgcolor="rgba(255,255,255,0.85)",
                                        bordercolor="#dddddd", borderwidth=1,
                                        x=0.40, y=0.95),
                                    height=380,
                                    plot_bgcolor="white",
                                    paper_bgcolor="#fafafa",
                                    margin=dict(l=55, r=20, t=45, b=50),
                                )
                                st.plotly_chart(fig, use_container_width=True)
    
        # ════════════════════════════════════════════════════════════════
        # SECTION 2 — VRH AVERAGING
        # ════════════════════════════════════════════════════════════════
        with st.expander("Voigt–Reuss–Hill (VRH) Averaging", expanded=False):

            st.markdown('<div class="rp-section-header">Theory</div>', unsafe_allow_html=True)

            st.markdown("""
            To calculate the bulk modulus of the mineral matrix (K₀), the rock's mineral
            composition must be known. Once mineral abundances are determined, K₀ is derived
            by averaging the mineral constituents using the **Voigt-Reuss-Hill (VRH)** method.

            - **Voigt bound** — assumes uniform strain across all phases (upper bound, stiffest)
            - **Reuss bound** — assumes uniform stress across all phases (lower bound, softest)
            - **VRH average** — arithmetic mean of Voigt and Reuss bounds, used as the
              best practical estimate of the composite mineral modulus
            """)

            st.markdown("---")

            st.markdown('<div class="rp-section-header">Governing Equations</div>',
                        unsafe_allow_html=True)

            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Voigt bound (upper)**")
                    st.latex(r"K_V = F_1 K_1 + F_2 K_2")
                    st.latex(r"\mu_V = F_1 \mu_1 + F_2 \mu_2")
                with c2:
                    st.markdown("**Reuss bound (lower)**")
                    st.latex(r"K_R = \left(\frac{F_1}{K_1} + \frac{F_2}{K_2}\right)^{-1}")
                    st.latex(r"\mu_R = \left(\frac{F_1}{\mu_1} + \frac{F_2}{\mu_2}\right)^{-1}")
                with c3:
                    st.markdown("**VRH average**")
                    st.latex(r"K_0 = \frac{K_V + K_R}{2}")
                    st.latex(r"\mu_0 = \frac{\mu_V + \mu_R}{2}")
                st.caption(
                    "F₁, F₂ = volume fractions  |  "
                    "K₁, K₂ = bulk moduli (GPa)  |  "
                    "µ₁, µ₂ = shear moduli (GPa)"
                )

            st.markdown("---")

            with st.expander("Reference table for Mineral Moduli", expanded=False):
                st.dataframe(pd.DataFrame([
                    {"Mineral": "Quartz",    "K (GPa)": 38.0,  "µ (GPa)": 44.0,  "ρ (g/cc)": 2.65},
                    {"Mineral": "Clay",      "K (GPa)": 21.5,  "µ (GPa)": 7.0,   "ρ (g/cc)": 2.58},
                    {"Mineral": "Calcite",   "K (GPa)": 76.8,  "µ (GPa)": 32.0,  "ρ (g/cc)": 2.71},
                    {"Mineral": "Dolomite",  "K (GPa)": 94.9,  "µ (GPa)": 45.0,  "ρ (g/cc)": 2.87},
                    {"Mineral": "Feldspar",  "K (GPa)": 37.5,  "µ (GPa)": 15.0,  "ρ (g/cc)": 2.62},
                    {"Mineral": "Anhydrite", "K (GPa)": 56.3,  "µ (GPa)": 29.1,  "ρ (g/cc)": 2.98},
                    {"Mineral": "Pyrite",    "K (GPa)": 147.4, "µ (GPa)": 132.5, "ρ (g/cc)": 4.93},
                ]), use_container_width=True, hide_index=True)

            st.markdown('<div class="rp-section-header">Input Parameters</div>',
                        unsafe_allow_html=True)

            VRH_PRESETS = {
                "Quartz":    {"K": 38.0,  "mu": 44.0},
                "Clay":      {"K": 21.5,  "mu": 7.0},
                "Calcite":   {"K": 76.8,  "mu": 32.0},
                "Dolomite":  {"K": 94.9,  "mu": 45.0},
                "Feldspar":  {"K": 37.5,  "mu": 15.0},
                "Anhydrite": {"K": 56.3,  "mu": 29.1},
                "Pyrite":    {"K": 147.4, "mu": 132.5},
            }

            hc1, hc2, hc3, hc4 = st.columns([2, 2, 2, 2])
            hc1.caption("Mineral")
            hc2.caption("K (GPa)")
            hc3.caption("µ (GPa)")
            hc4.caption("Volume fraction")

            st.markdown("**Mineral 1**")
            v1c1, v1c2, v1c3, v1c4 = st.columns([2, 2, 2, 2])
            m1_choice = v1c1.selectbox("", options=list(VRH_PRESETS.keys()),
                                        index=0, key="vrh_m1",
                                        label_visibility="collapsed")
            m1_K  = v1c2.number_input("", value=VRH_PRESETS[m1_choice]["K"],
                                       format="%.2f", key="vrh_m1_K",
                                       label_visibility="collapsed")
            m1_mu = v1c3.number_input("", value=VRH_PRESETS[m1_choice]["mu"],
                                       format="%.2f", key="vrh_m1_mu",
                                       label_visibility="collapsed")
            m1_F  = v1c4.number_input("", value=0.80,
                                       min_value=0.0, max_value=1.0,
                                       format="%.3f", key="vrh_m1_F",
                                       label_visibility="collapsed")

            st.markdown("**Mineral 2**")
            v2c1, v2c2, v2c3, v2c4 = st.columns([2, 2, 2, 2])
            m2_choice = v2c1.selectbox("", options=list(VRH_PRESETS.keys()),
                                        index=1, key="vrh_m2",
                                        label_visibility="collapsed")
            m2_K  = v2c2.number_input("", value=VRH_PRESETS[m2_choice]["K"],
                                       format="%.2f", key="vrh_m2_K",
                                       label_visibility="collapsed")
            m2_mu = v2c3.number_input("", value=VRH_PRESETS[m2_choice]["mu"],
                                       format="%.2f", key="vrh_m2_mu",
                                       label_visibility="collapsed")
            m2_F  = v2c4.number_input("", value=0.20,
                                       min_value=0.0, max_value=1.0,
                                       format="%.3f", key="vrh_m2_F",
                                       label_visibility="collapsed")

            frac_sum = m1_F + m2_F
            if abs(frac_sum - 1.0) > 0.01:
                st.warning(f"Volume fractions sum to {frac_sum:.3f} — must sum to 1.000.")
            else:
                st.success(f"Volume fractions sum to {frac_sum:.3f} ✓")

            st.markdown("---")

            btn_vrh, _ = st.columns([2, 6])
            with btn_vrh:
                run_vrh = st.button("▶ Compute VRH", key="run_vrh",
                                    type="primary", use_container_width=True)

            # ── Always recompute to avoid stale session state key errors ──
            if run_vrh or st.session_state.get("vrh_K0"):

                Kv = m1_F * m1_K + m2_F * m2_K
                Kr = 1.0 / (m1_F / m1_K + m2_F / m2_K)
                K0 = (Kv + Kr) / 2.0

                Gv = m1_F * m1_mu + m2_F * m2_mu
                Gr = 1.0 / (m1_F / m1_mu + m2_F / m2_mu)
                G0 = (Gv + Gr) / 2.0

                f2_range = np.linspace(0.0, 1.0, 200)
                f1_range = 1.0 - f2_range

                Kv_arr   = f1_range * m1_K + f2_range * m2_K
                Kr_arr   = 1.0 / (f1_range / m1_K + f2_range / m2_K)
                Kvrh_arr = (Kv_arr + Kr_arr) / 2.0

                st.session_state["vrh_K0"] = K0
                st.session_state["vrh_G0"] = G0

                # ── Results ──────────────────────────────────────────────
                st.markdown('<div class="rp-section-header">Results</div>',
                            unsafe_allow_html=True)

                ra, rb, rc, rd = st.columns(4)
                ra.metric("K_Voigt (GPa)", f"{Kv:.4f}")
                rb.metric("K_Reuss (GPa)", f"{Kr:.4f}")
                rc.metric("K₀  VRH (GPa)", f"{K0:.4f}")
                rd.metric("µ₀  VRH (GPa)", f"{G0:.4f}")

                st.info(
                    f"K₀ = **{K0:.4f} GPa** and µ₀ = **{G0:.4f} GPa** "
                    "saved — used automatically in Hertz-Mindlin, Soft-Sand & Stiff-Sand."
                )

                # ── Plot ─────────────────────────────────────────────────
                st.markdown('<div class="rp-section-header">Composite Bulk Moduli vs Mineral Fraction</div>',
                            unsafe_allow_html=True)

                fig_vrh = go.Figure()

                fig_vrh.add_trace(go.Scatter(
                    x=f2_range, y=Kv_arr,
                    mode="lines", name="Voigt (upper bound)",
                    line=dict(color="#d62728", width=2),
                ))
                fig_vrh.add_trace(go.Scatter(
                    x=f2_range, y=Kr_arr,
                    mode="lines", name="Reuss (lower bound)",
                    line=dict(color="#1f77b4", width=2),
                ))
                fig_vrh.add_trace(go.Scatter(
                    x=f2_range, y=Kvrh_arr,
                    mode="lines", name="VRH average",
                    line=dict(color="#2ca02c", width=2, dash="dash"),
                ))
                fig_vrh.add_trace(go.Scatter(
                    x=[m2_F], y=[K0],
                    mode="markers",
                    name=f"Your mix  (F_{m2_choice} = {m2_F:.2f})",
                    marker=dict(color="#ff7f0e", size=10, symbol="circle"),
                ))

                fig_vrh.update_layout(
                    title=dict(
                        text=f"{m1_choice} / {m2_choice} Mixture",
                        font=dict(size=13), x=0.5, xanchor="center"),
                    xaxis=dict(
                        title=f"Volume fraction of {m2_choice}",
                        showgrid=True, gridcolor="#eeeeee",
                        showline=True, linecolor="#aaaaaa",
                        zeroline=False, tickformat=".2f"),
                    yaxis=dict(
                        title="Bulk Modulus (GPa)",
                        showgrid=True, gridcolor="#eeeeee",
                        showline=True, linecolor="#aaaaaa",
                        zeroline=False),
                    legend=dict(
                        font=dict(size=10),
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="#dddddd", borderwidth=1),
                    height=420,
                    plot_bgcolor="white",
                    paper_bgcolor="#fafafa",
                    margin=dict(l=65, r=20, t=55, b=55),
                )
                st.plotly_chart(fig_vrh, use_container_width=True)



        # ════════════════════════════════════════════════════════════════
        # SECTION 3 — HERTZ-MINDLIN CONTACT THEORY
        # ════════════════════════════════════════════════════════════════
        with st.expander("Hertz-Mindlin Contact Theory", expanded=False):

            st.markdown('<div class="rp-section-header">Theory</div>', unsafe_allow_html=True)

            st.markdown("""
            Hertz-Mindlin theory simulates the elastic modulus at high porosity (often critical
            porosity) as an elastic sphere pack driven by net confining pressure (Dvorkin et al., 1996).
            When porosity reduction is caused by mechanical compaction, this theory accurately predicts
            pressure dependence on elastic properties for any unconsolidated sediment (Avseth et al., 2001).

            The theory computes **K_HM** and **µ_HM** — the dry-frame bulk and shear moduli of the
            rock specifically at the critical porosity. These values serve as the high-porosity
            end-member for the Soft-Sand and Stiff-Sand models.
            """)

            st.markdown("---")

            # ── Equations ───────────────────────────────────────────────
            st.markdown('<div class="rp-section-header">Governing Equations</div>',
                        unsafe_allow_html=True)

            with st.container(border=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Dry-frame bulk modulus at ϕc**")
                    st.latex(
                        r"K_{HM} = \left[\frac{C^2(1-\phi_0)^2\mu_0^2}{18\pi^2(1-\nu)^2}P\right]^{1/3}"
                    )
                with c2:
                    st.markdown("**Dry-frame shear modulus at ϕc**")
                    st.latex(
                        r"\mu_{HM} = \frac{2+3f-\nu(1+3f)}{5(2-\nu)}"
                        r"\left[\frac{3C^2(1-\phi_0)^2\mu_0^2}{2\pi^2(1-\nu)^2}P\right]^{1/3}"
                    )
                st.caption(
                    "K_HM, µ_HM = dry-frame moduli at critical porosity  |  "
                    "P = confining pressure (MPa)  |  C = coordination number  |  "
                    "ν = Poisson's ratio  |  f = friction correction factor  |  "
                    "µ₀ = mineral shear modulus from VRH (GPa)"
                )

            st.markdown("---")

            # ── Parameter reference ──────────────────────────────────────
            with st.expander("Reference — Typical Parameter Values", expanded=False):
                st.dataframe(pd.DataFrame([
                    {"Parameter": "Coordination number  C", "Typical value": "6 – 9",
                     "Note": "8.4 for typical sandstone (Dvorkin et al., 1996)"},
                    {"Parameter": "Confining pressure  P",  "Typical value": "5 – 50 MPa",
                     "Note": "Effective pressure = overburden − pore pressure"},
                    {"Parameter": "Poisson's ratio  ν",     "Typical value": "0.10 – 0.30",
                     "Note": "Quartz = 0.06,  typical sand = 0.12 – 0.15"},
                    {"Parameter": "Friction factor  f",     "Typical value": "0 or 1",
                     "Note": "1 = ideal adhesion (dry pack),  0 = frictionless pack"},
                ]), use_container_width=True, hide_index=True)

            st.markdown("---")

            # ── Input Parameters ─────────────────────────────────────────
            st.markdown('<div class="rp-section-header">Input Parameters</div>',
                        unsafe_allow_html=True)

            hm1, hm2, hm3 = st.columns(3)
            hm_phic = hm1.number_input(
                "Critical porosity  ϕ₀",
                value=0.40, min_value=0.01, max_value=0.80,
                format="%.3f", key="hm_phic",
                help="Same ϕc as Critical Porosity Model — typically 0.36–0.40 for sandstone")
            hm_C = hm2.number_input(
                "Coordination number  C",
                value=8.4, min_value=1.0, max_value=20.0,
                format="%.1f", key="hm_C",
                help="Average grain contacts per grain — 8.4 for typical sandstone")
            hm_P = hm3.number_input(
                "Confining pressure  P (MPa)",
                value=10.0, min_value=0.1, max_value=200.0,
                format="%.1f", key="hm_P",
                help="Effective pressure = overburden pressure − pore pressure")

            hm4, hm5, _ = st.columns(3)
            hm_nu = hm4.number_input(
                "Poisson's ratio  ν",
                value=0.12, min_value=0.01, max_value=0.49,
                format="%.3f", key="hm_nu",
                help="Mineral Poisson's ratio — Quartz ≈ 0.06, typical sand ≈ 0.12")
            hm_f = hm5.number_input(
                "Friction factor  f",
                value=1.0, min_value=0.0, max_value=1.0,
                format="%.2f", key="hm_f",
                help="1 = ideal adhesion (dry pack),  0 = frictionless pack")

            # ── VRH source notice ────────────────────────────────────────
            K0_hm = st.session_state.get("vrh_K0", None)
            G0_hm = st.session_state.get("vrh_G0", None)

            if K0_hm is None or G0_hm is None:
                st.warning(
                    "K₀ and µ₀ not found — run **Section 2 (VRH)** first. "
                    "Defaulting to Quartz: K₀ = 38.0 GPa, µ₀ = 44.0 GPa."
                )
                K0_hm, G0_hm = 38.0, 44.0
            else:
                st.success(
                    f"Using K₀ = **{K0_hm:.4f} GPa** and µ₀ = **{G0_hm:.4f} GPa** "
                    "from Section 2 (VRH)."
                )

            st.markdown("---")

            # ── Compute button ────────────────────────────────────────────
            btn_hm, _ = st.columns([2, 6])
            with btn_hm:
                run_hm = st.button("▶ Compute Hertz-Mindlin", key="run_hm",
                                   type="primary", use_container_width=True)

            if run_hm or st.session_state.get("hm_K"):

                G0_Pa = G0_hm * 1e9
                P_Pa  = hm_P  * 1e6

                # ── Point computation ────────────────────────────────────
                term_K  = (hm_C**2 * (1 - hm_phic)**2 * G0_Pa**2 * P_Pa) / \
                          (18 * np.pi**2 * (1 - hm_nu)**2)
                K_HM    = (term_K ** (1/3)) * 1e-9

                term_mu = (3 * hm_C**2 * (1 - hm_phic)**2 * G0_Pa**2 * P_Pa) / \
                          (2 * np.pi**2 * (1 - hm_nu)**2)
                prefac  = (2 + 3*hm_f - hm_nu*(1 + 3*hm_f)) / (5*(2 - hm_nu))
                G_HM    = prefac * (term_mu ** (1/3)) * 1e-9

                st.session_state["hm_K"] = K_HM
                st.session_state["hm_G"] = G_HM

                # ── Results metrics ───────────────────────────────────────
                st.markdown('<div class="rp-section-header">Results</div>',
                            unsafe_allow_html=True)

                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("K_HM (GPa)",  f"{K_HM:.4f}",
                           help="Dry-frame bulk modulus at critical porosity")
                rc2.metric("µ_HM (GPa)",  f"{G_HM:.4f}",
                           help="Dry-frame shear modulus at critical porosity")
                rc3.metric("At ϕc",        f"{hm_phic:.3f}")
                rc4.metric("At P (MPa)",   f"{hm_P:.1f}")

                st.info(
                    f"K_HM = **{K_HM:.4f} GPa** and µ_HM = **{G_HM:.4f} GPa** "
                    "saved — used automatically in Soft-Sand and Stiff-Sand models."
                )

                st.markdown("---")

                # ── Pressure range for both plots ─────────────────────────
                P_range = np.linspace(1, 100, 200)
                P_Pa_arr = P_range * 1e6

                # ── Plot 1 arrays — K_HM & G_HM vs Pressure ──────────────
                K_HM_arr = ((hm_C**2 * (1-hm_phic)**2 * G0_Pa**2 * P_Pa_arr) /
                            (18 * np.pi**2 * (1-hm_nu)**2)) ** (1/3) * 1e-9

                G_HM_arr = prefac * ((3 * hm_C**2 * (1-hm_phic)**2 * G0_Pa**2 * P_Pa_arr) /
                            (2 * np.pi**2 * (1-hm_nu)**2)) ** (1/3) * 1e-9

                # ── Plot 2 arrays — G_HM vs Pressure for f values ─────────
                f_values  = [0.0, 0.25, 0.50, 0.75, 1.0]
                f_colors  = ["#1f77b4", "#17becf", "#2ca02c", "#ff7f0e", "#d62728"]
                f_labels  = ["f = 0.00 (frictionless)", "f = 0.25", "f = 0.50",
                             "f = 0.75", "f = 1.00 (ideal adhesion)"]

                base_term = ((3 * hm_C**2 * (1-hm_phic)**2 * G0_Pa**2 * P_Pa_arr) /
                             (2 * np.pi**2 * (1-hm_nu)**2)) ** (1/3) * 1e-9

                # ── Side-by-side plots ────────────────────────────────────
                st.markdown('<div class="rp-section-header">Diagnostic Plots</div>',
                            unsafe_allow_html=True)

                pcol1, pcol2 = st.columns(2)

                # ── Plot 1 ────────────────────────────────────────────────
                with pcol1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=P_range, y=K_HM_arr,
                        mode="lines", name="K_HM (GPa)",
                        line=dict(color="#d62728", width=2),
                    ))
                    fig1.add_trace(go.Scatter(
                        x=P_range, y=G_HM_arr,
                        mode="lines", name="µ_HM (GPa)",
                        line=dict(color="#1f77b4", width=2),
                    ))
                    fig1.add_vline(
                        x=hm_P,
                        line=dict(color="gray", width=1.2, dash="dash"),
                        annotation_text=f"P = {hm_P} MPa",
                        annotation_position="top right",
                        annotation_font=dict(size=9, color="gray"),
                    )
                    fig1.update_layout(
                        title=dict(
                            text="K_HM & µ_HM vs Confining Pressure",
                            font=dict(size=12), x=0.5, xanchor="center"),
                        xaxis=dict(
                            title="Confining pressure (MPa)",
                            showgrid=True, gridcolor="#eeeeee",
                            showline=True, linecolor="#aaaaaa",
                            zeroline=False),
                        yaxis=dict(
                            title="Modulus (GPa)",
                            showgrid=True, gridcolor="#eeeeee",
                            showline=True, linecolor="#aaaaaa",
                            zeroline=False),
                        legend=dict(
                            font=dict(size=9),
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor="#dddddd", borderwidth=1),
                        height=400,
                        plot_bgcolor="white",
                        paper_bgcolor="#fafafa",
                        margin=dict(l=60, r=20, t=50, b=50),
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                # ── Plot 2 ────────────────────────────────────────────────
                with pcol2:
                    fig2 = go.Figure()
                    for fv, fc, fl in zip(f_values, f_colors, f_labels):
                        pf = (2 + 3*fv - hm_nu*(1 + 3*fv)) / (5*(2 - hm_nu))
                        fig2.add_trace(go.Scatter(
                            x=P_range, y=pf * base_term,
                            mode="lines", name=fl,
                            line=dict(color=fc, width=2),
                        ))
                    fig2.add_vline(
                        x=hm_P,
                        line=dict(color="gray", width=1.2, dash="dash"),
                        annotation_text=f"P = {hm_P} MPa",
                        annotation_position="top right",
                        annotation_font=dict(size=9, color="gray"),
                    )
                    # Mark user's f value on plot
                    pf_user = (2 + 3*hm_f - hm_nu*(1 + 3*hm_f)) / (5*(2 - hm_nu))
                    G_at_P  = pf_user * (((3 * hm_C**2 * (1-hm_phic)**2 *
                                G0_Pa**2 * P_Pa) /
                                (2 * np.pi**2 * (1-hm_nu)**2)) ** (1/3)) * 1e-9
                    fig2.add_trace(go.Scatter(
                        x=[hm_P], y=[G_at_P],
                        mode="markers",
                        name=f"Your input (f = {hm_f:.2f})",
                        marker=dict(color="#ff7f0e", size=10, symbol="circle"),
                    ))
                    fig2.update_layout(
                        title=dict(
                            text="µ_HM vs Pressure — Friction Factor Sensitivity",
                            font=dict(size=12), x=0.5, xanchor="center"),
                        xaxis=dict(
                            title="Confining pressure (MPa)",
                            showgrid=True, gridcolor="#eeeeee",
                            showline=True, linecolor="#aaaaaa",
                            zeroline=False),
                        yaxis=dict(
                            title="µ_HM (GPa)",
                            showgrid=True, gridcolor="#eeeeee",
                            showline=True, linecolor="#aaaaaa",
                            zeroline=False),
                        legend=dict(
                            font=dict(size=9),
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor="#dddddd", borderwidth=1),
                        height=400,
                        plot_bgcolor="white",
                        paper_bgcolor="#fafafa",
                        margin=dict(l=60, r=20, t=50, b=50),
                    )
                    st.plotly_chart(fig2, use_container_width=True)


        # ════════════════════════════════════════════════════════════════
        # SECTION 4 — SOFT-SAND & STIFF-SAND MODELS (Combined)
        # ════════════════════════════════════════════════════════════════
        with st.expander("Soft-Sand & Stiff-Sand Models", expanded=False):

            # ── Theory (side by side) ────────────────────────────────────
            st.markdown('<div class="rp-section-header">Theory</div>', unsafe_allow_html=True)
            th1, th2 = st.columns(2)
            with th1:
                with st.container(border=True):
                    st.markdown("#### 🟦 Soft-Sand Model  *(HS Lower Bound)*")
                    st.markdown("""
                    Dvorkin & Nur (1996) — solid matter deposition **away from grain contacts**
                    causes porosity to reduce from the initial sand-pack value.

                    Uses the **modified lower HS bound** to interpolate between:
                    - ϕ = ϕc → K_HM, µ_HM  *(Hertz-Mindlin)*
                    - ϕ = 0 → K₀, µ₀  *(pure mineral)*

                    Represents **loose, shallow, unconsolidated** sands.
                    The interpolation uses **µ_HM** — the weakest possible shear term.
                    """)
            with th2:
                with st.container(border=True):
                    st.markdown("#### 🟥 Stiff-Sand Model  *(HS Upper Bound)*")
                    st.markdown("""
                    Cement deposition **at the grain contacts** causes porosity to reduce
                    from the initial sand-pack value.

                    Uses the **modified upper HS bound** to interpolate between the
                    **same two end-members** as the Soft-Sand model.

                    Represents **cemented, stiff, deeply buried** sands.
                    The interpolation uses **µ₀** — the stiffest possible shear term.
                    Since µ₀ > µ_HM, Stiff-Sand moduli are always **higher** at same ϕ.
                    """)

            st.markdown("---")

            # ── Equations (2 columns: Soft | Stiff) ─────────────────────
            st.markdown('<div class="rp-section-header">Governing Equations</div>',
                        unsafe_allow_html=True)
            with st.container(border=True):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown("**🟦 Soft-Sand  — Effective Bulk Modulus**")
                    st.latex(
                        r"K_{eff}^{soft} = \left[\frac{\phi/\phi_c}{K_{HM}+\frac{4}{3}\mu_{HM}}"
                        r"+ \frac{1-\phi/\phi_c}{K_0+\frac{4}{3}\mu_{HM}}\right]^{-1}"
                        r"- \frac{4}{3}\mu_{HM}"
                    )
                    st.markdown("**🟦 Soft-Sand  — Effective Shear Modulus**")
                    st.latex(
                        r"\mu_{eff}^{soft} = \left[\frac{\phi/\phi_c}{\mu_{HM}+z_{HM}}"
                        r"+ \frac{1-\phi/\phi_c}{\mu_0+z_{HM}}\right]^{-1} - z_{HM}"
                    )
                    st.latex(
                        r"z_{HM} = \frac{\mu_{HM}}{6}"
                        r"\left(\frac{9K_{HM}+8\mu_{HM}}{K_{HM}+2\mu_{HM}}\right)"
                    )
                with ec2:
                    st.markdown("**🟥 Stiff-Sand  — Effective Bulk Modulus**")
                    st.latex(
                        r"K_{eff}^{stiff} = \left[\frac{\phi/\phi_c}{K_{HM}+\frac{4}{3}\mu_0}"
                        r"+ \frac{1-\phi/\phi_c}{K_0+\frac{4}{3}\mu_0}\right]^{-1}"
                        r"- \frac{4}{3}\mu_0"
                    )
                    st.markdown("**🟥 Stiff-Sand  — Effective Shear Modulus**")
                    st.latex(
                        r"\mu_{eff}^{stiff} = \left[\frac{\phi/\phi_c}{\mu_{HM}+z_0}"
                        r"+ \frac{1-\phi/\phi_c}{\mu_0+z_0}\right]^{-1} - z_0"
                    )
                    st.latex(
                        r"z_0 = \frac{\mu_0}{6}"
                        r"\left(\frac{9K_0+8\mu_0}{K_0+2\mu_0}\right)"
                    )

                st.markdown("---")
                # with st.container(border=True):
                #     st.markdown("**Key difference — one term changes everything:**")
                #     kd1, kd2 = st.columns(2)
                #     with kd1:
                #         st.markdown("🟦 Soft-Sand uses **µ_HM** *(sand-pack shear)*")
                #         st.latex(r"K_{HM} + \tfrac{4}{3}\,\mu_{HM}")
                #     with kd2:
                #         st.markdown("🟥 Stiff-Sand uses **µ₀** *(mineral shear)*")
                #         st.latex(r"K_{HM} + \tfrac{4}{3}\,\mu_0")
                st.caption(
                    "K₀, µ₀ = mineral moduli from VRH  |  "
                    "K_HM, µ_HM = Hertz-Mindlin moduli at ϕc  |  "
                    "ϕc = critical porosity"
                )

            st.markdown("---")

            # ── Shared upstream values ───────────────────────────────────
            sh_K0   = st.session_state.get("vrh_K0",  None)
            sh_G0   = st.session_state.get("vrh_G0",  None)
            sh_KHM  = st.session_state.get("hm_K",    None)
            sh_GHM  = st.session_state.get("hm_G",    None)
            sh_phic = st.session_state.get("hm_phic", None)
            sh_C    = st.session_state.get("hm_C",    8.4)
            sh_P    = st.session_state.get("hm_P",    10.0)
            sh_nu   = st.session_state.get("hm_nu",   0.12)
            sh_f    = st.session_state.get("hm_f",    1.0)

            missing = []
            if sh_K0  is None or sh_G0  is None: missing.append("Section 2 (VRH)")
            if sh_KHM is None or sh_GHM is None: missing.append("Section 3 (Hertz-Mindlin)")

            if missing:
                st.warning(
                    f"Please run {' and '.join(missing)} first. "
                    "Defaulting to Quartz values."
                )
                sh_K0, sh_G0   = 38.0, 44.0
                sh_KHM, sh_GHM = 2.0,  2.0
                sh_phic        = 0.40
            else:
                with st.container(border=True):
                    st.markdown("**Parameters carried forward from previous sections:**")
                    cf1, cf2, cf3, cf4, cf5 = st.columns(5)
                    cf1.metric("K₀  (GPa)",  f"{sh_K0:.4f}",  help="Section 2 — VRH")
                    cf2.metric("µ₀  (GPa)",  f"{sh_G0:.4f}",  help="Section 2 — VRH")
                    cf3.metric("K_HM (GPa)", f"{sh_KHM:.4f}", help="Section 3 — Hertz-Mindlin")
                    cf4.metric("µ_HM (GPa)", f"{sh_GHM:.4f}", help="Section 3 — Hertz-Mindlin")
                    cf5.metric("ϕc",         f"{sh_phic:.3f}", help="Section 3 — Hertz-Mindlin")

            st.markdown("---")

            # ── Single compute button ────────────────────────────────────
            btn_col, _ = st.columns([2, 6])
            with btn_col:
                run_both = st.button("▶ Compute Both Models", key="run_both",
                                     type="primary", use_container_width=True)

            if run_both or st.session_state.get("both_done"):

                K0   = sh_K0
                G0   = sh_G0
                K_HM = sh_KHM
                G_HM = sh_GHM
                phic = float(sh_phic)
                nu   = float(sh_nu)
                f    = float(sh_f)

                phi_plot  = np.linspace(0.0, phic, 300)
                phi_table = np.arange(0.0, phic + 0.001, 0.05)
                phi_table = np.round(phi_table[phi_table <= phic + 0.001], 4)
                if round(phi_table[-1], 3) != round(phic, 3):
                    phi_table = np.append(phi_table, round(phic, 3))

                # ── HS helper terms ──────────────────────────────────────
                alpha_ss  = (G_HM / 6.0) * ((9*K_HM + 8*G_HM) / (K_HM + 2*G_HM))
                alpha_stf = (G0   / 6.0) * ((9*K0   + 8*G0  ) / (K0   + 2*G0  ))

                # ── Core model functions ─────────────────────────────────
                def soft_sand(phi, Khm, Ghm, al):
                    A    = phi / phic
                    B    = 1.0 - A
                    Keff = (A / (Khm + (4/3)*Ghm) + B / (K0 + (4/3)*Ghm)) ** (-1) - (4/3)*Ghm
                    Geff = (A / (Ghm + al)         + B / (G0 + al        )) ** (-1) - al
                    return Keff, Geff

                def stiff_sand(phi, Khm, Ghm, al):
                    A    = phi / phic
                    B    = 1.0 - A
                    Keff = (A / (Khm + (4/3)*G0) + B / (K0 + (4/3)*G0)) ** (-1) - (4/3)*G0
                    Geff = (A / (Ghm + al)        + B / (G0 + al       )) ** (-1) - al
                    return Keff, Geff

                # ── HM helper for sensitivity ────────────────────────────
                def hm_moduli(C_val, P_val):
                    G0_Pa = G0 * 1e9
                    P_Pa  = P_val * 1e6
                    Khm   = ((C_val**2 * (1-phic)**2 * G0_Pa**2 * P_Pa) /
                             (18 * np.pi**2 * (1-nu)**2)) ** (1/3) * 1e-9
                    pf    = (2 + 3*f - nu*(1 + 3*f)) / (5*(2 - nu))
                    Ghm   = pf * ((3 * C_val**2 * (1-phic)**2 * G0_Pa**2 * P_Pa) /
                             (2 * np.pi**2 * (1-nu)**2)) ** (1/3) * 1e-9
                    return Khm, Ghm

                # ── Compute main arrays ──────────────────────────────────
                ss_K,  ss_G  = soft_sand( phi_plot,  K_HM, G_HM, alpha_ss)
                stf_K, stf_G = stiff_sand(phi_plot,  K_HM, G_HM, alpha_stf)
                ss_Kt, ss_Gt   = soft_sand( phi_table, K_HM, G_HM, alpha_ss)
                stf_Kt, stf_Gt = stiff_sand(phi_table, K_HM, G_HM, alpha_stf)

                st.session_state.update({
                    "both_done": True,
                    "ss_done":   True,  "stf_done":  True,
                    "ss_phi":    phi_plot, "ss_Keff": ss_K,  "ss_Geff": ss_G,
                    "stf_phi":   phi_plot, "stf_Keff": stf_K, "stf_Geff": stf_G,
                })

                # ── Combined results table ────────────────────────────────
                st.markdown('<div class="rp-section-header">Results</div>',
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({
                    "Porosity ϕ":        np.round(phi_table,  4),
                    "K_soft (GPa)":      np.round(ss_Kt,      4),
                    "K_stiff (GPa)":     np.round(stf_Kt,     4),
                    "µ_soft (GPa)":      np.round(ss_Gt,      4),
                    "µ_stiff (GPa)":     np.round(stf_Gt,     4),
                }), use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown('<div class="rp-section-header">Diagnostic Plots</div>',
                            unsafe_allow_html=True)

                # ════════════════════════════════════════════════════════
                # ROW 1 — Overlay plots: Bulk | Shear
                # ════════════════════════════════════════════════════════
                r1c1, r1c2 = st.columns(2)

                def end_member_traces(phic, K_HM, K0):
                    t1 = go.Scatter(
                        x=[phic], y=[K_HM], mode="markers",
                        name=f"HM end-member (ϕc={phic:.3f})",
                        marker=dict(color="#555555", size=10, symbol="circle-open",
                                    line=dict(width=2)),
                        showlegend=True,
                    )
                    t2 = go.Scatter(
                        x=[0], y=[K0], mode="markers",
                        name="Mineral end-member (ϕ=0)",
                        marker=dict(color="#2ca02c", size=10, symbol="diamond"),
                        showlegend=True,
                    )
                    return t1, t2

                # ── Plot R1C1 — Bulk Modulus overlay ─────────────────────
                with r1c1:
                    fig_K = go.Figure()

                    # shaded band between bounds
                    fig_K.add_trace(go.Scatter(
                        x=np.concatenate([phi_plot, phi_plot[::-1]]),
                        y=np.concatenate([stf_K, ss_K[::-1]]),
                        fill="toself",
                        fillcolor="rgba(180,180,180,0.20)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="Real sands band",
                        showlegend=True,
                        hoverinfo="skip",
                    ))
                    fig_K.add_trace(go.Scatter(
                        x=phi_plot, y=ss_K,
                        mode="lines", name="K_eff  Soft-Sand (HS Lower)",
                        line=dict(color="#1f77b4", width=2.5, dash="dash"),
                    ))
                    fig_K.add_trace(go.Scatter(
                        x=phi_plot, y=stf_K,
                        mode="lines", name="K_eff  Stiff-Sand (HS Upper)",
                        line=dict(color="#d62728", width=2.5),
                    ))
                    t1, t2 = end_member_traces(phic, K_HM, K0)
                    fig_K.add_trace(t1); fig_K.add_trace(t2)

                    fig_K.update_layout(
                        title=dict(text="Bulk Modulus  K_eff — Soft vs Stiff",
                                   font=dict(size=12), x=0.5, xanchor="center"),
                        xaxis=dict(title="Porosity  ϕ", showgrid=True, gridcolor="#eeeeee",
                                   showline=True, linecolor="#aaaaaa",
                                   zeroline=False, tickformat=".2f"),
                        yaxis=dict(title="K_eff  (GPa)", showgrid=True, gridcolor="#eeeeee",
                                   showline=True, linecolor="#aaaaaa", zeroline=False),
                        legend=dict(font=dict(size=9), bgcolor="rgba(255,255,255,0.85)",
                                    bordercolor="#dddddd", borderwidth=1),
                        height=420, plot_bgcolor="white", paper_bgcolor="#fafafa",
                        margin=dict(l=60, r=20, t=50, b=50),
                    )
                    st.plotly_chart(fig_K, use_container_width=True)

                # ── Plot R1C2 — Shear Modulus overlay ────────────────────
                with r1c2:
                    fig_G = go.Figure()

                    fig_G.add_trace(go.Scatter(
                        x=np.concatenate([phi_plot, phi_plot[::-1]]),
                        y=np.concatenate([stf_G, ss_G[::-1]]),
                        fill="toself",
                        fillcolor="rgba(180,180,180,0.20)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="Real sands band",
                        showlegend=True,
                        hoverinfo="skip",
                    ))
                    fig_G.add_trace(go.Scatter(
                        x=phi_plot, y=ss_G,
                        mode="lines", name="µ_eff  Soft-Sand (HS Lower)",
                        line=dict(color="#1f77b4", width=2.5, dash="dash"),
                    ))
                    fig_G.add_trace(go.Scatter(
                        x=phi_plot, y=stf_G,
                        mode="lines", name="µ_eff  Stiff-Sand (HS Upper)",
                        line=dict(color="#d62728", width=2.5),
                    ))
                    t1g = go.Scatter(
                        x=[phic], y=[G_HM], mode="markers",
                        name=f"HM end-member (ϕc={phic:.3f})",
                        marker=dict(color="#555555", size=10, symbol="circle-open",
                                    line=dict(width=2)),
                    )
                    t2g = go.Scatter(
                        x=[0], y=[G0], mode="markers",
                        name="Mineral end-member (ϕ=0)",
                        marker=dict(color="#2ca02c", size=10, symbol="diamond"),
                    )
                    fig_G.add_trace(t1g); fig_G.add_trace(t2g)

                    fig_G.update_layout(
                        title=dict(text="Shear Modulus  µ_eff — Soft vs Stiff",
                                   font=dict(size=12), x=0.5, xanchor="center"),
                        xaxis=dict(title="Porosity  ϕ", showgrid=True, gridcolor="#eeeeee",
                                   showline=True, linecolor="#aaaaaa",
                                   zeroline=False, tickformat=".2f"),
                        yaxis=dict(title="µ_eff  (GPa)", showgrid=True, gridcolor="#eeeeee",
                                   showline=True, linecolor="#aaaaaa", zeroline=False),
                        legend=dict(font=dict(size=9), bgcolor="rgba(255,255,255,0.85)",
                                    bordercolor="#dddddd", borderwidth=1),
                        height=420, plot_bgcolor="white", paper_bgcolor="#fafafa",
                        margin=dict(l=60, r=20, t=50, b=50),
                    )
                    st.plotly_chart(fig_G, use_container_width=True)

                st.markdown("---")

                # ── Single moduli checkbox for rows 2 & 3 ────────────────
                st.markdown("**Select modulus for sensitivity plots (applies to both models):**")
                tog1, tog2, _ = st.columns([1, 1, 4])
                show_K = tog1.checkbox("K_eff (Bulk)",  value=True,  key="both_show_K")
                show_G = tog2.checkbox("µ_eff (Shear)", value=False, key="both_show_G")

                if not show_K and not show_G:
                    st.info("Select at least one modulus above to display the sensitivity plots.")
                else:
                    C_values = [4, 6, 8, 10, 12]
                    C_colors = ["#1f77b4", "#17becf", "#2ca02c", "#ff7f0e", "#d62728"]
                    P_values = [5, 10, 20, 40, 60]
                    P_colors = ["#1f77b4", "#17becf", "#2ca02c", "#ff7f0e", "#d62728"]
                    y_lbl    = ("K_eff (GPa)" if show_K and not show_G else
                                "µ_eff (GPa)" if show_G and not show_K else
                                "Modulus (GPa)")

                    def make_sensitivity_fig(title, traces):
                        fig = go.Figure(traces)
                        fig.update_layout(
                            title=dict(text=title, font=dict(size=12),
                                       x=0.5, xanchor="center"),
                            xaxis=dict(title="Porosity  ϕ", showgrid=True,
                                       gridcolor="#eeeeee", showline=True,
                                       linecolor="#aaaaaa", zeroline=False,
                                       tickformat=".2f"),
                            yaxis=dict(title=y_lbl, showgrid=True,
                                       gridcolor="#eeeeee", showline=True,
                                       linecolor="#aaaaaa", zeroline=False),
                            legend=dict(font=dict(size=9),
                                        bgcolor="rgba(255,255,255,0.85)",
                                        bordercolor="#dddddd", borderwidth=1),
                            height=400, plot_bgcolor="white", paper_bgcolor="#fafafa",
                            margin=dict(l=60, r=20, t=50, b=50),
                        )
                        return fig

                    def sensitivity_traces(model_fn, alpha_fn, C_or_P="C",
                                           fixed_val=None, vals=None, colors=None,
                                           label_prefix=""):
                        traces = []
                        for v, c in zip(vals, colors):
                            C_v = v        if C_or_P == "C" else float(sh_C)
                            P_v = float(sh_P) if C_or_P == "C" else v
                            Kh, Gh = hm_moduli(C_v, P_v)
                            al     = alpha_fn(Kh, Gh)
                            Ke, Ge = model_fn(phi_plot, Kh, Gh, al)
                            lbl    = f"C={v}" if C_or_P == "C" else f"P={v}MPa"
                            if show_K:
                                traces.append(go.Scatter(x=phi_plot, y=Ke, mode="lines",
                                    name=f"K  {lbl}", line=dict(color=c, width=2)))
                            if show_G:
                                traces.append(go.Scatter(x=phi_plot, y=Ge, mode="lines",
                                    name=f"µ  {lbl}", line=dict(color=c, width=2, dash="dot")))

                        # User marker
                        C_u = float(sh_C); P_u = float(sh_P)
                        Khu, Ghu = hm_moduli(C_u, P_u)
                        al_u     = alpha_fn(Khu, Ghu)
                        Ke_u, Ge_u = model_fn(np.array([phic/2]), Khu, Ghu, al_u)
                        lbl_u = f"Your C={sh_C:.1f}" if C_or_P == "C" else f"Your P={sh_P:.1f}MPa"
                        if show_K:
                            traces.append(go.Scatter(x=[phic/2], y=[Ke_u[0]], mode="markers",
                                name=f"{lbl_u}  K_eff",
                                marker=dict(color="#9467bd", size=10, symbol="star")))
                        if show_G:
                            traces.append(go.Scatter(x=[phic/2], y=[Ge_u[0]], mode="markers",
                                name=f"{lbl_u}  µ_eff",
                                marker=dict(color="#9467bd", size=10, symbol="star-open")))
                        return traces

                    def alpha_ss_fn(Kh, Gh):
                        return (Gh / 6.0) * ((9*Kh + 8*Gh) / (Kh + 2*Gh))

                    def alpha_stf_fn(Kh, Gh):
                        return (G0 / 6.0) * ((9*K0 + 8*G0) / (K0 + 2*G0))

                    # ════════════════════════════════════════════════════
                    # ROW 2 — Soft-Sand Sensitivity
                    # ════════════════════════════════════════════════════
                    st.markdown(
                        '<div class="rp-section-header">🟦 Soft-Sand — Sensitivity Analysis</div>',
                        unsafe_allow_html=True)
                    r2c1, r2c2 = st.columns(2)

                    with r2c1:
                        tr = sensitivity_traces(soft_sand, alpha_ss_fn, "C",
                                                vals=C_values, colors=C_colors)
                        st.plotly_chart(
                            make_sensitivity_fig(
                                "Soft-Sand — Coordination Number Sensitivity", tr),
                            use_container_width=True)

                    with r2c2:
                        tr = sensitivity_traces(soft_sand, alpha_ss_fn, "P",
                                                vals=P_values, colors=P_colors)
                        st.plotly_chart(
                            make_sensitivity_fig(
                                "Soft-Sand — Confining Pressure Sensitivity", tr),
                            use_container_width=True)

                    st.markdown("---")

                    # ════════════════════════════════════════════════════
                    # ROW 3 — Stiff-Sand Sensitivity
                    # ════════════════════════════════════════════════════
                    st.markdown(
                        '<div class="rp-section-header">🟥 Stiff-Sand — Sensitivity Analysis</div>',
                        unsafe_allow_html=True)
                    r3c1, r3c2 = st.columns(2)

                    with r3c1:
                        tr = sensitivity_traces(stiff_sand, alpha_stf_fn, "C",
                                                vals=C_values, colors=C_colors)
                        st.plotly_chart(
                            make_sensitivity_fig(
                                "Stiff-Sand — Coordination Number Sensitivity", tr),
                            use_container_width=True)

                    with r3c2:
                        tr = sensitivity_traces(stiff_sand, alpha_stf_fn, "P",
                                                vals=P_values, colors=P_colors)
                        st.plotly_chart(
                            make_sensitivity_fig(
                                "Stiff-Sand — Confining Pressure Sensitivity", tr),
                            use_container_width=True)


                # ════════════════════════════════════════════════════════════════
        # SECTION 6 — GASSMANN FLUID SUBSTITUTION (Path A — Forward Model)
        # ════════════════════════════════════════════════════════════════
        with st.expander("Gassmann Fluid Substitution — Forward Model", expanded=False):

            # ── Theory ──────────────────────────────────────────────────
            st.markdown('<div class="rp-section-header">Theory</div>', unsafe_allow_html=True)

            th1, th2 = st.columns(2)
            with th1:
                with st.container(border=True):
                    st.markdown("#### Forward Gassmann *(Dry → Saturated)*")
                    st.markdown("""
                    Gassmann's (1951) equation is the most widely used low-frequency fluid
                    substitution method in rock physics. It connects a rock's **saturated bulk
                    modulus** to its pore geometry, mineral composition, frame stiffness, and
                    the elastic properties of the pore-filling fluid.

                    Starting from the **dry-frame moduli** K\* and µ\* computed in the
                    Soft/Stiff Sand models, Gassmann predicts how K_sat changes when
                    pores are filled with any fluid — brine, oil, or gas.

                    - Valid at **seismic frequencies** (low-frequency limit)
                    - Assumes **homogeneous, isotropic**, monomineralic rock
                    - Requires **connected pore space** (drained conditions)
                    """)
            with th2:
                with st.container(border=True):
                    st.markdown("#### Why Shear Modulus is Unaffected")
                    st.markdown("""
                    A key result of Gassmann's theory is that **pore fluids do not resist shear
                    deformation** — they only resist compression. Therefore:

                    - µ_sat = µ\* always, regardless of fluid type
                    - This means **Vs is only weakly affected by fluid** (only through density)
                    - **Vp is strongly affected** by fluid — gas dramatically lowers Vp,
                      brine raises it

                    This contrast between Vp and Vs sensitivity to fluid is the
                    **physical basis of AVO analysis** and fluid discrimination in seismic.
                    """)

            st.markdown("---")

            # ── Equations ───────────────────────────────────────────────
            st.markdown('<div class="rp-section-header">Governing Equations</div>',
                        unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("**Saturated Bulk Modulus (Gassmann)**")
                st.latex(
                    r"K_{sat} = K^* + "
                    r"\frac{\left(1 - \dfrac{K^*}{K_0}\right)^2}"
                    r"{\dfrac{\phi}{K_{fl}} + \dfrac{1-\phi}{K_0} - \dfrac{K^*}{K_0^2}}"
                )
                st.markdown("---")
                st.markdown("**Shear Modulus (fluid independent)**")
                st.latex(r"\mu_{sat} = \mu^*")
                st.markdown("---")
                st.markdown("**Saturated Bulk Density**")
                st.latex(r"\rho_{sat} = \rho_0\,(1-\phi)\;+\;\rho_{fl}\cdot\phi")
                st.markdown("---")
                st.markdown("**P-wave and S-wave Velocities**")
                eq1, eq2 = st.columns(2)
                with eq1:
                    st.latex(
                        r"V_P = \sqrt{\frac{K_{sat} + \frac{4}{3}\,\mu_{sat}}{\rho_{sat}}}"
                    )
                with eq2:
                    st.latex(r"V_S = \sqrt{\frac{\mu_{sat}}{\rho_{sat}}}")
                st.markdown("---")
                st.caption(
                    "K\* = dry-frame bulk modulus  |  µ\* = dry-frame shear modulus  |  "
                    "K₀ = mineral bulk modulus from VRH  |  K_fl = fluid bulk modulus  |  "
                    "ϕ = porosity  |  ρ₀ = mineral grain density  |  ρ_fl = fluid density  |  "
                    "All moduli in GPa, densities in g/cc, velocities in m/s"
                )

            st.markdown("---")

            # ── Fluid Reference Table ────────────────────────────────────
            with st.expander("Reference Table for Fluid properties", expanded=False):
                st.dataframe(pd.DataFrame({
                    "Fluid Type":       ["Freshwater", "Brine (100k ppm)", "Brine (200k ppm)",
                                         "Dead Oil", "Live Oil (GOR~100)", "Light Oil",
                                         "Methane (gas)", "CO₂ (gas)", "Air"],
                    "K_fl (GPa)":       [2.25, 2.38, 2.50,
                                         0.90, 0.60, 1.10,
                                         0.04, 0.03, 0.00013],
                    "ρ_fl (g/cc)":      [1.00, 1.05, 1.09,
                                         0.85, 0.72, 0.78,
                                         0.15, 0.70, 0.001],
                    "Typical Vp (m/s)": [1480, 1550, 1615,
                                         1280, 1050, 1350,
                                         490,  450,  340],
                    "Notes": [
                        "Pure water at surface",
                        "Moderate salinity",
                        "High salinity brine",
                        "No dissolved gas",
                        "Moderate GOR",
                        "Low viscosity crude",
                        "Methane at reservoir conditions",
                        "Supercritical CO₂ ~approx",
                        "Surface conditions",
                    ]
                }), use_container_width=True, hide_index=True)
                st.caption(
                    "Values are approximate and vary with temperature, pressure, and salinity. "
                    "Use Batzle-Wang (1992) equations for precise in-situ fluid properties."
                )

            st.markdown("---")

            # ── Read upstream values ─────────────────────────────────────
            gk0       = st.session_state.get("vrh_K0",   None)
            gg0       = st.session_state.get("vrh_G0",   None)
            gKdry     = st.session_state.get("ss_Keff",  None)
            gGdry     = st.session_state.get("ss_Geff",  None)
            gKdry_stf = st.session_state.get("stf_Keff", None)
            gGdry_stf = st.session_state.get("stf_Geff", None)
            gphi      = st.session_state.get("ss_phi",   None)
            gphic     = st.session_state.get("hm_phic",  None)

            missing = []
            if gk0 is None or gg0 is None:     missing.append("Section 2 (VRH)")
            if gKdry is None or gGdry is None:  missing.append("Section 4&5 (Sand Models)")

            # ── FIX 2: replaced st.stop() with else-guard ────────────────
            if missing:
                st.warning(f"⚠️ Please run {' and '.join(missing)} first to use this section.")
            else:
                with st.container(border=True):
                    st.markdown("**Parameters carried forward from previous sections:**")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("K₀  (GPa)", f"{gk0:.4f}")
                    m2.metric("µ₀  (GPa)", f"{gg0:.4f}")
                    m3.metric("ϕc",        f"{gphic:.3f}")

                st.markdown("---")

                # ── Fluid Input Panel ────────────────────────────────────────
                st.markdown('<div class="rp-section-header">Fluid Properties</div>',
                            unsafe_allow_html=True)

                FLUID_PRESETS = {
                    "Brine":  {"Kfl": 2.50, "rhofl": 1.09},
                    "Oil":    {"Kfl": 0.90, "rhofl": 0.80},
                    "Gas":    {"Kfl": 0.04, "rhofl": 0.15},
                    "Custom": {"Kfl": 1.00, "rhofl": 1.00},
                }

                FLUID_COLORS_MAP = {
                    "Brine":  "#008bee",
                    "Oil":    "#00c53b",
                    "Gas":    "#ff2b2b",
                    "Custom": "#fff70e",
                }

                FLUID_DESC = {
                    "Brine":  "Saline formation water (~2.5 GPa)",
                    "Oil":    "Dead oil (~0.9 GPa)",
                    "Gas":    "Methane gas (~0.04 GPa)",
                    "Custom": "User-defined fluid",
                }

                with st.container(border=True):
                    st.markdown("**Select fluid(s)**")
                    fcols = st.columns(4)
                    fluid_selected = {}
                    for i, (fname, fprops) in enumerate(FLUID_PRESETS.items()):
                        with fcols[i]:
                            checked = st.checkbox(
                                fname, value=(fname == "Brine"),
                                key=f"gas_fwd_fluid_{fname}",  # ✅ FIX 1: unique key
                            )
                            if checked:
                                fluid_selected[fname] = fprops.copy()

                    if fluid_selected:
                        st.markdown("---")
                        st.markdown("**Edit fluid properties if needed:**")
                        edit_cols = st.columns(len(fluid_selected))
                        for i, (fname, fprops) in enumerate(fluid_selected.items()):
                            with edit_cols[i]:
                                with st.container(border=True):
                                    st.markdown(f"**{fname}**")
                                    fluid_selected[fname]["Kfl"] = st.number_input(
                                        "K_fl (GPa)", value=fprops["Kfl"],
                                        min_value=0.001, max_value=10.0,
                                        format="%.4f", key=f"gas_kfl_{fname}",
                                    )
                                    fluid_selected[fname]["rhofl"] = st.number_input(
                                        "ρ_fl (g/cc)", value=fprops["rhofl"],
                                        min_value=0.01, max_value=3.0,
                                        format="%.4f", key=f"gas_rhofl_{fname}",
                                    )
                    else:
                        st.info("Select at least one fluid above.")

                st.markdown("---")

                # ── Mineral density + model selector ─────────────────────────
                with st.container(border=True):
                    pc1, pc2, pc3, pc4, pc5 = st.columns([1.2, 1, 1, 1, 1.5])
                    with pc1:
                        rho0 = st.number_input(
                            "ρ_mineral (g/cc)",
                            value=2.65, min_value=1.0, max_value=5.0,
                            format="%.3f", key="gas_rho0",
                        )
                    with pc2:
                        use_soft  = st.checkbox("Use Soft-Sand",  value=True,
                                                key="gas_use_ss")
                    with pc3:
                        use_stiff = st.checkbox("Use Stiff-Sand", value=True,
                                                key="gas_use_stf",
                                                disabled=(gKdry_stf is None))
                    with pc4:
                        use_crit  = st.checkbox("Use Crit-Por",   value=False,
                                                key="gas_use_crit")
                    with pc5:
                        if gKdry_stf is None:
                            st.caption("⚠ Stiff-Sand unavailable — run Section 4&5 first.")
                        else:
                            st.caption("All selected models overlaid on plots.")

                st.markdown("---")

                # ── Compute button ────────────────────────────────────────────
                btn_g, _ = st.columns([2, 6])
                with btn_g:
                    run_gass = st.button("▶ Run Gassmann Forward Model", key="run_gass",
                                         type="primary", use_container_width=True)

                # ── FIX 3: clear stale state on new button click ──────────────
                if run_gass:
                    st.session_state["gass_done"] = False

                if run_gass or st.session_state.get("gass_done"):

                    # ── FIX 2: replaced st.stop() with else-guard ─────────────
                    if not fluid_selected:
                        st.warning("⚠️ Select at least one fluid to proceed.")
                    else:
                        K0  = float(gk0)
                        phi = np.array(gphi)

                        # ── Core functions ────────────────────────────────────────
                        def gassmann_forward(Kdry, K0, Kfl, phi):
                            num   = (1.0 - Kdry / K0) ** 2
                            denom = (phi / Kfl) + ((1.0 - phi) / K0) - (Kdry / K0**2)
                            return Kdry + num / denom

                        def critical_porosity_model(phi, K0, G0, phic):
                            factor = np.clip(1.0 - phi / phic, 0.0, 1.0)
                            return K0 * factor, G0 * factor

                        # ── Build model list ─────────────────────────────────────
                        model_pairs = []
                        if use_soft  and gKdry     is not None:
                            model_pairs.append(("Soft-Sand",  np.array(gKdry),     np.array(gGdry)))
                        if use_stiff and gKdry_stf is not None:
                            model_pairs.append(("Stiff-Sand", np.array(gKdry_stf), np.array(gGdry_stf)))
                        if use_crit:
                            Kcrit, Gcrit = critical_porosity_model(
                                phi, float(gk0), float(gg0), float(gphic))
                            model_pairs.append(("Crit-Por", Kcrit, Gcrit))

                        # ── FIX 2: replaced st.stop() with else-guard ─────────────
                        if not model_pairs:
                            st.warning("⚠️ Select at least one model.")
                        else:
                            results = {}
                            for mname, Kdry_arr, Gdry_arr in model_pairs:
                                for fname, fprops in fluid_selected.items():
                                    Kfl   = fprops["Kfl"]
                                    rhofl = fprops["rhofl"]

                                    Ksat      = gassmann_forward(Kdry_arr, K0, Kfl, phi)
                                    musat     = Gdry_arr.copy()
                                    rhosat    = rho0 * (1.0 - phi) + rhofl * phi
                                    rhosat_si = rhosat * 1000.0

                                    Vp_ms = np.sqrt((Ksat + (4.0/3.0)*musat) * 1e9 / rhosat_si)
                                    Vs_ms = np.sqrt(musat * 1e9 / rhosat_si)
                                    VpVs  = np.where(Vs_ms > 0, Vp_ms / Vs_ms, np.nan)
                                    Ip    = rhosat * Vp_ms

                                    results[(mname, fname)] = {
                                        "phi":    phi,
                                        "Ksat":   Ksat,
                                        "Vp":     Vp_ms,
                                        "Vs":     Vs_ms,
                                        "Ip":     Ip,
                                        "VpVs":   VpVs,
                                        "rhosat": rhosat,
                                        "color":  FLUID_COLORS_MAP.get(fname, "#999999"),
                                    }

                            st.session_state["gass_done"]    = True
                            st.session_state["gass_results"] = results
                            st.session_state["gass_phi"]     = phi

                            # ── Results table ─────────────────────────────────────────
                            st.markdown('<div class="rp-section-header">Results</div>',
                                        unsafe_allow_html=True)

                            phi_table = np.arange(0.0, float(gphic) + 0.001, 0.05)
                            phi_table = np.round(phi_table[phi_table <= float(gphic) + 0.001], 4)
                            if round(phi_table[-1], 3) != round(float(gphic), 3):
                                phi_table = np.append(phi_table, round(float(gphic), 3))

                            table_data = {"Porosity ϕ": np.round(phi_table, 4)}
                            for (mname, fname), res in results.items():
                                lbl = f"{mname[:4]}-{fname[:3]}"
                                table_data[f"Vp {lbl} (m/s)"] = np.round(
                                    np.interp(phi_table, res["phi"], res["Vp"]), 1)
                                table_data[f"Vs {lbl} (m/s)"] = np.round(
                                    np.interp(phi_table, res["phi"], res["Vs"]), 1)
                                table_data[f"Ip {lbl}"]        = np.round(
                                    np.interp(phi_table, res["phi"], res["Ip"]), 2)

                            st.dataframe(pd.DataFrame(table_data),
                                         use_container_width=True, hide_index=True)

                            st.markdown("---")
                            st.markdown('<div class="rp-section-header">Diagnostic Plots</div>',
                                        unsafe_allow_html=True)

                            LINESTYLES  = {
                                "Soft-Sand":  "solid",
                                "Stiff-Sand": "dash",
                                "Crit-Por":   "dot",
                            }
                            MODEL_COLORS = {
                                "Soft-Sand":  "#2ca02c",
                                "Stiff-Sand": "#000000",
                                "Crit-Por":   "#9467bd",
                            }

                            def base_layout(title, xtitle, ytitle, height=400):
                                return dict(
                                    title=dict(text=title, font=dict(size=12),
                                               x=0.5, xanchor="center"),
                                    xaxis=dict(title=xtitle, showgrid=True, gridcolor="#eeeeee",
                                               showline=True, linecolor="#aaaaaa",
                                               zeroline=False, tickformat=".2f"),
                                    yaxis=dict(title=ytitle, showgrid=True, gridcolor="#eeeeee",
                                               showline=True, linecolor="#aaaaaa", zeroline=False),
                                    legend=dict(font=dict(size=9), bgcolor="rgba(255,255,255,0.85)",
                                                bordercolor="#dddddd", borderwidth=1),
                                    height=height, plot_bgcolor="white", paper_bgcolor="#fafafa",
                                    margin=dict(l=65, r=20, t=50, b=50),
                                )

                            # ════════════════════════════════════════════════════════
                            # ROW 1 — Model comparison (fixed fluid, vary model)
                            # ════════════════════════════════════════════════════════
                            st.markdown("##### Model Comparison with fixed fluid")

                            row1_fluid_options = list(fluid_selected.keys())
                            r1f_col, _ = st.columns([2, 5])
                            with r1f_col:
                                row1_fluid = st.selectbox(
                                    "Select fluid",
                                    options=row1_fluid_options,
                                    key="gas_row1_fluid"
                                )

                            r1c1, r1c2 = st.columns(2)

                            with r1c1:
                                fig_r1_vp = go.Figure()
                                for mname, _, _ in model_pairs:
                                    res = results.get((mname, row1_fluid))
                                    if res:
                                        fig_r1_vp.add_trace(go.Scatter(
                                            x=res["phi"], y=res["Vp"], mode="lines",
                                            name=mname,
                                            line=dict(color=MODEL_COLORS.get(mname, "#555"),
                                                      width=2.5,
                                                      dash=LINESTYLES.get(mname, "solid")),
                                        ))
                                fig_r1_vp.update_layout(**base_layout(
                                    f"Vp vs Porosity  [{row1_fluid}]", "Porosity  ϕ", "Vp  (m/s)"))
                                st.plotly_chart(fig_r1_vp, use_container_width=True)

                            with r1c2:
                                fig_r1_vs = go.Figure()
                                for mname, _, _ in model_pairs:
                                    res = results.get((mname, row1_fluid))
                                    if res:
                                        fig_r1_vs.add_trace(go.Scatter(
                                            x=res["phi"], y=res["Vs"], mode="lines",
                                            name=mname,
                                            line=dict(color=MODEL_COLORS.get(mname, "#555"),
                                                      width=2.5,
                                                      dash=LINESTYLES.get(mname, "solid")),
                                        ))
                                fig_r1_vs.update_layout(**base_layout(
                                    f"Vs vs Porosity  [{row1_fluid}]", "Porosity  ϕ", "Vs  (m/s)"))
                                st.plotly_chart(fig_r1_vs, use_container_width=True)

                            # ════════════════════════════════════════════════════════
                            # ROW 2 — Fluid comparison (fixed model, vary fluid)
                            # ════════════════════════════════════════════════════════
                            if len(fluid_selected) > 1:
                                st.markdown("---")
                                st.markdown("##### Fluid Comparison with fixed model")

                                row2_model_options = [m for m, _, _ in model_pairs]
                                r2m_col, _ = st.columns([2, 5])
                                with r2m_col:
                                    row2_model = st.selectbox(
                                        "Select model",
                                        options=row2_model_options,
                                        key="gas_row2_model"
                                    )

                                r2c1, r2c2 = st.columns(2)

                                with r2c1:
                                    fig_r2_vp = go.Figure()
                                    for fname in fluid_selected:
                                        res = results.get((row2_model, fname))
                                        if res:
                                            fig_r2_vp.add_trace(go.Scatter(
                                                x=res["phi"], y=res["Vp"], mode="lines",
                                                name=fname,
                                                line=dict(
                                                    color=FLUID_COLORS_MAP.get(fname, "#999"),
                                                    width=2.5),
                                            ))
                                    fig_r2_vp.update_layout(**base_layout(
                                        f"Vp vs Porosity  [{row2_model}]",
                                        "Porosity  ϕ", "Vp  (m/s)"))
                                    st.plotly_chart(fig_r2_vp, use_container_width=True)

                                with r2c2:
                                    fig_r2_vs = go.Figure()
                                    for fname in fluid_selected:
                                        res = results.get((row2_model, fname))
                                        if res:
                                            fig_r2_vs.add_trace(go.Scatter(
                                                x=res["phi"], y=res["Vs"], mode="lines",
                                                name=fname,
                                                line=dict(
                                                    color=FLUID_COLORS_MAP.get(fname, "#999"),
                                                    width=2.5),
                                            ))
                                    fig_r2_vs.update_layout(**base_layout(
                                        f"Vs vs Porosity  [{row2_model}]",
                                        "Porosity  ϕ", "Vs  (m/s)"))
                                    st.plotly_chart(fig_r2_vs, use_container_width=True)

                            # ── Save to session state for Section 8 RPT ───────────────
                            st.session_state["rpt_model_curves"] = results
                            st.success("✅ Gassmann complete — results saved for RPT.")


        # ════════════════════════════════════════════════════════════════
        # SECTION 6B — GASSMANN LOG-BASED FLUID SUBSTITUTION (Path B)
        # ════════════════════════════════════════════════════════════════
        with st.expander("Gassmann - Log Based Fluid Substitution", expanded=False):

            st.markdown('<div class="rp-section-header">Theory</div>', unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("#### Log-Based Fluid Substitution")
                st.markdown("""
                Uses **real well logs** (Vp, Vs, RHOB, ϕ) instead of a theoretical dry frame.
                The rock's dry-frame modulus K\* is extracted sample-by-sample via **Inverse Gassmann**,
                then re-saturated with any fluid using **Forward Gassmann**.

                **Key advantages:**
                - No sand model required — frame is derived directly from logs
                - Applied at every depth sample for full log-resolution output
                - Intermediate logs (K_SAT, K_DRY) saved for QC
                - Ideal for comparing Brine → Oil → Gas saturation scenarios
                """)

            st.markdown("---")

            # ── Equations ────────────────────────────────────────────────
            st.markdown('<div class="rp-section-header">Governing Equations</div>',
                        unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("**Log-derived saturated moduli**")
                eq_a, eq_b = st.columns(2)
                with eq_a:
                    st.latex(r"\mu = \rho\,V_S^2")
                with eq_b:
                    st.latex(r"K_{sat} = \rho V_P^2 - \tfrac{4}{3}\,\mu")

                st.markdown("---")
                st.markdown("**Inverse Gassmann (extract dry frame)**")
                st.latex(
                    r"K^* = \frac{K_{sat}"
                    r"\!\left(\dfrac{\phi\,K_0}{K_{fl,old}} + 1 - \phi\right) - K_0}"
                    r"{\dfrac{\phi\,K_0}{K_{fl,old}} + \dfrac{K_{sat}}{K_0} - 1}"
                )

                st.markdown("---")
                st.markdown("**Forward Gassmann (new fluid)**")
                st.latex(
                    r"K_{sat,new} = K^* + "
                    r"\frac{\left(1 - \dfrac{K^*}{K_0}\right)^2}"
                    r"{\dfrac{\phi}{K_{fl,new}} + \dfrac{1-\phi}{K_0} - \dfrac{K^*}{K_0^2}}"
                )

                st.markdown("---")
                st.markdown("**Density update and new velocities**")
                eq_c, eq_d, eq_e = st.columns(3)
                with eq_c:
                    st.latex(r"\rho_{new} = \rho - \rho_{fl,old}\phi + \rho_{fl,new}\phi")
                with eq_d:
                    st.latex(
                        r"V_{P,new} = \sqrt{\frac{K_{sat,new} + \frac{4}{3}\mu}{\rho_{new}}}"
                    )
                with eq_e:
                    st.latex(r"V_{S,new} = \sqrt{\frac{\mu}{\rho_{new}}}")

                st.caption(
                    "All moduli in GPa  |  densities in g/cc → converted to kg/m³ internally  |  "
                    "velocities output in m/s"
                )

            st.markdown("---")

            # ── Fluid Reference Table ─────────────────────────────────────
            with st.expander("Reference Table for Fluid Properties", expanded=False):
                st.dataframe(pd.DataFrame({
                    "Fluid Type":       ["Freshwater", "Brine (100k ppm)", "Brine (200k ppm)",
                                         "Dead Oil", "Live Oil (GOR~100)", "Light Oil",
                                         "Methane (gas)", "CO₂ (gas)", "Air"],
                    "K_fl (GPa)":       [2.25, 2.38, 2.50,
                                         0.90, 0.60, 1.10,
                                         0.04, 0.03, 0.00013],
                    "ρ_fl (g/cc)":      [1.00, 1.05, 1.09,
                                         0.85, 0.72, 0.78,
                                         0.15, 0.70, 0.001],
                    "Typical Vp (m/s)": [1480, 1550, 1615,
                                         1280, 1050, 1350,
                                         490,  450,  340],
                    "Notes": [
                        "Pure water at surface",
                        "Moderate salinity",
                        "High salinity brine",
                        "No dissolved gas",
                        "Moderate GOR",
                        "Low viscosity crude",
                        "Methane at reservoir conditions",
                        "Supercritical CO₂ ~approx",
                        "Surface conditions",
                    ]
                }), use_container_width=True, hide_index=True)
                st.caption(
                    "Values are approximate and vary with temperature, pressure, and salinity. "
                    "Use Batzle-Wang (1992) equations for precise in-situ fluid properties."
                )

            st.markdown("---")

            # ── Read upstream values ──────────────────────────────────────
            gk0   = st.session_state.get("vrh_K0", None)
            df_rp = st.session_state.get("df",     None)

            if gk0 is None:
                st.warning("⚠️ Please run VRH Section first to get K₀.")
            elif df_rp is None or len(df_rp) == 0:
                st.warning("⚠️ No well log data found. Please upload a LAS/CSV file first.")
            else:
                # ── Dedup guard ───────────────────────────────────────────
                if df_rp.columns.duplicated().any():
                    df_rp = df_rp.loc[:, ~df_rp.columns.duplicated(keep="last")].copy()
                    st.session_state["df"] = df_rp

                allcols = list(df_rp.columns)

                with st.container(border=True):
                    st.markdown("**K₀ carried from VRH Section:**")
                    st.metric("K₀ (GPa)", f"{gk0:.4f}")

                st.markdown("---")

                # ── Log column selectors ──────────────────────────────────
                st.markdown('<div class="rp-section-header">Log Column Selection</div>',
                            unsafe_allow_html=True)

                def best_match(candidates, columns):
                    for c in candidates:
                        for col in columns:
                            if c.upper() in col.upper():
                                return col
                    return columns[0]

                with st.container(border=True):
                    lc1, lc2, lc3, lc4 = st.columns(4)
                    with lc1:
                        vp_col_b = st.selectbox(
                            "Vp log (m/s)", options=allcols,
                            index=allcols.index(best_match(["VP_EST", "VP", "VPCO"], allcols)),
                            key="pathb_vp_col"
                        )
                    with lc2:
                        vs_col_b = st.selectbox(
                            "Vs log (m/s)", options=allcols,
                            index=allcols.index(best_match(["VS_EST", "VS"], allcols)),
                            key="pathb_vs_col"
                        )
                    with lc3:
                        rho_col_b = st.selectbox(
                            "RHOB log (g/cc)", options=allcols,
                            index=allcols.index(best_match(["RHOB", "RHOZ", "DENS", "DEN"], allcols)),
                            key="pathb_rho_col"
                        )
                    with lc4:
                        phi_col_b = st.selectbox(
                            "Porosity log", options=allcols,
                            index=allcols.index(best_match(["PHIE", "PHIT", "NPHI", "PHI"], allcols)),
                            key="pathb_phi_col"
                        )

                    preview_cols = list(dict.fromkeys([vp_col_b, vs_col_b, rho_col_b, phi_col_b]))
                    st.markdown("**Selected log preview (first 5 valid rows):**")
                    st.dataframe(
                        df_rp[preview_cols].dropna().head(5).round(4),
                        use_container_width=True, hide_index=True
                    )

                st.markdown("---")

                # ── Fluid selectors ───────────────────────────────────────
                st.markdown('<div class="rp-section-header">Fluid Selection</div>',
                            unsafe_allow_html=True)

                FLUID_PRESETS_B = {
                    "Brine":  {"Kfl": 2.50, "rhofl": 1.09},
                    "Oil":    {"Kfl": 0.90, "rhofl": 0.80},
                    "Gas":    {"Kfl": 0.04, "rhofl": 0.15},
                    "Custom": {"Kfl": 1.00, "rhofl": 1.00},
                }

                with st.container(border=True):
                    fc1, fc2 = st.columns(2)

                    with fc1:
                        st.markdown("**Existing fluid in rock (original)**")
                        old_fluid = st.selectbox(
                            "Original fluid type",
                            options=list(FLUID_PRESETS_B.keys()),
                            index=0, key="pathb_old_fluid"
                        )
                        kfl_old = st.number_input(
                            "K_fl_old (GPa)",
                            value=FLUID_PRESETS_B[old_fluid]["Kfl"],
                            min_value=0.001, max_value=10.0,
                            format="%.4f", key="pathb_kfl_old"
                        )
                        rfl_old = st.number_input(
                            "ρ_fl_old (g/cc)",
                            value=FLUID_PRESETS_B[old_fluid]["rhofl"],
                            min_value=0.01, max_value=3.0,
                            format="%.4f", key="pathb_rfl_old"
                        )

                    with fc2:
                        st.markdown("**New fluid to substitute**")
                        new_fluid = st.selectbox(
                            "New fluid type",
                            options=list(FLUID_PRESETS_B.keys()),
                            index=2, key="pathb_new_fluid"
                        )
                        kfl_new = st.number_input(
                            "K_fl_new (GPa)",
                            value=FLUID_PRESETS_B[new_fluid]["Kfl"],
                            min_value=0.001, max_value=10.0,
                            format="%.4f", key="pathb_kfl_new"
                        )
                        rfl_new = st.number_input(
                            "ρ_fl_new (g/cc)",
                            value=FLUID_PRESETS_B[new_fluid]["rhofl"],
                            min_value=0.01, max_value=3.0,
                            format="%.4f", key="pathb_rfl_new"
                        )

                st.markdown("---")

                # ── Run button ────────────────────────────────────────────
                btn_b, _ = st.columns([2, 6])
                with btn_b:
                    run_pathb = st.button(
                        "▶ Run Log Fluid Substitution",
                        key="run_pathb",
                        type="primary",
                        use_container_width=True
                    )

                # ── FIX: persist results in session_state so they survive rerenders ──
                if run_pathb:
                    sub_df = df_rp[[vp_col_b, vs_col_b, rho_col_b, phi_col_b]].copy()
                    sub_df = sub_df.replace([np.inf, -np.inf], np.nan).dropna()

                    if len(sub_df) == 0:
                        st.error("No valid rows after removing NaN/Inf. Check log columns.")
                        st.session_state.pop("pathb_results", None)
                    else:
                        Vp  = sub_df[vp_col_b].values.astype(float)
                        Vs  = sub_df[vs_col_b].values.astype(float)
                        rho = sub_df[rho_col_b].values.astype(float)
                        phi = sub_df[phi_col_b].values.astype(float)

                        K0 = float(gk0)

                        rho_si = rho * 1000.0
                        mu_si  = rho_si * Vs**2
                        Ksat   = (rho_si * Vp**2 - (4.0 / 3.0) * mu_si) / 1e9
                        mu     = mu_si / 1e9

                        def gassmann_inverse(Ksat_gpa, K0_gpa, Kfl_gpa, phi_arr):
                            A   = phi_arr * K0_gpa / Kfl_gpa
                            num = Ksat_gpa * (A + 1.0 - phi_arr) - K0_gpa
                            den = A + Ksat_gpa / K0_gpa - 1.0
                            return np.where(np.abs(den) > 1e-9, num / den, np.nan)

                        def gassmann_forward_b(Kdry_gpa, K0_gpa, Kfl_gpa, phi_arr):
                            num   = (1.0 - Kdry_gpa / K0_gpa) ** 2
                            denom = (phi_arr / Kfl_gpa) + ((1.0 - phi_arr) / K0_gpa) - (Kdry_gpa / K0_gpa**2)
                            return np.where(np.abs(denom) > 1e-9,
                                            Kdry_gpa + num / denom, np.nan)

                        Kdry     = gassmann_inverse(Ksat, K0, kfl_old, phi)
                        Ksat_new = gassmann_forward_b(Kdry, K0, kfl_new, phi)

                        rho_new    = rho - rfl_old * phi + rfl_new * phi
                        rho_new_si = rho_new * 1000.0

                        Vp_new = np.sqrt(
                            np.clip((Ksat_new + (4.0 / 3.0) * mu) * 1e9 / rho_new_si, 0, None))
                        Vs_new = np.sqrt(
                            np.clip(mu * 1e9 / rho_new_si, 0, None))

                        # ── Save substituted logs to df ───────────────────
                        _work_df = st.session_state["df"]
                        for _c in ["VP_SUB", "VS_SUB", "KSAT_LOG", "KDRY_LOG"]:
                            if _c in _work_df.columns:
                                _work_df.drop(columns=[_c], inplace=True)

                        _work_df.loc[sub_df.index, "VP_SUB"]   = pd.Series(Vp_new,  index=sub_df.index)
                        _work_df.loc[sub_df.index, "VS_SUB"]   = pd.Series(Vs_new,  index=sub_df.index)
                        _work_df.loc[sub_df.index, "KSAT_LOG"] = pd.Series(Ksat,    index=sub_df.index)
                        _work_df.loc[sub_df.index, "KDRY_LOG"] = pd.Series(Kdry,    index=sub_df.index)
                        st.session_state["df"] = _work_df

                        # ── Cache results so they render after button click ─
                        depth_arr = st.session_state["df"].loc[sub_df.index, depth_col].values
                        st.session_state["pathb_results"] = {
                            "Vp":        Vp,
                            "Vs":        Vs,
                            "Vp_new":    Vp_new,
                            "Vs_new":    Vs_new,
                            "depth":     depth_arr,
                            "n_valid":   int((~(np.isnan(Vp_new) | np.isnan(Vs_new))).sum()),
                            "n_total":   len(sub_df),
                            "old_fluid": old_fluid,
                            "new_fluid": new_fluid,
                        }

                # ── Display results from session_state (survives rerender) ──
                if st.session_state.get("pathb_results"):
                    res       = st.session_state["pathb_results"]
                    Vp        = res["Vp"]
                    Vs        = res["Vs"]
                    Vp_new    = res["Vp_new"]
                    Vs_new    = res["Vs_new"]
                    depth_vals= res["depth"]
                    n_valid   = res["n_valid"]
                    n_total   = res["n_total"]
                    old_fluid = res["old_fluid"]
                    new_fluid = res["new_fluid"]

                    st.markdown('<div class="rp-section-header">Results</div>',
                                unsafe_allow_html=True)

                    rv1, rv2, rv3, rv4 = st.columns(4)
                    rv1.metric("Samples processed",   n_total)
                    rv2.metric("Valid output samples", n_valid)
                    rv3.metric("Vp_new range (m/s)",
                               f"{np.nanmin(Vp_new):.0f} – {np.nanmax(Vp_new):.0f}")
                    rv4.metric("Vs_new range (m/s)",
                               f"{np.nanmin(Vs_new):.0f} – {np.nanmax(Vs_new):.0f}")

                    if n_valid < n_total:
                        st.warning(
                            f"⚠️ {n_total - n_valid} samples returned NaN — likely due to "
                            "near-zero Gassmann denominator (extreme porosity or moduli values). "
                            "These depths are left as NaN in VP_SUB / VS_SUB."
                        )

                    st.markdown("---")

                    st.markdown('<div class="rp-section-header">Diagnostic Plots</div>',
                                unsafe_allow_html=True)
                    st.caption(
                        "Original log vs fluid-substituted result. "
                        "VP_SUB and VS_SUB are now saved to the dataframe."
                    )

                    dp1, dp2 = st.columns(2)

                    def track_layout(title, xtitle, height=600):
                        return dict(
                            title=dict(text=title, font=dict(size=12),
                                       x=0.5, xanchor="center"),
                            xaxis=dict(title=xtitle, showgrid=True,
                                       gridcolor="#eeeeee", zeroline=False),
                            yaxis=dict(title="Depth", autorange="reversed",
                                       showgrid=True, gridcolor="#eeeeee"),
                            legend=dict(font=dict(size=9),
                                        bgcolor="rgba(255,255,255,0.85)",
                                        bordercolor="#dddddd", borderwidth=1),
                            height=height, plot_bgcolor="white",
                            paper_bgcolor="#fafafa",
                            margin=dict(l=65, r=20, t=50, b=50),
                        )

                    with dp1:
                        fig_vp = go.Figure()
                        fig_vp.add_trace(go.Scatter(
                            x=Vp, y=depth_vals, mode="lines",
                            name=f"Vp original ({old_fluid})",
                            line=dict(color="#0095ff", width=1)
                        ))
                        fig_vp.add_trace(go.Scatter(
                            x=Vp_new, y=depth_vals, mode="lines",
                            name=f"Vp substituted ({new_fluid})",
                            line=dict(color="#ff0303", width=1)
                        ))
                        fig_vp.update_layout(
                            **track_layout("Vp — Original vs Substituted", "Vp  (m/s)"))
                        st.plotly_chart(fig_vp, use_container_width=True, key="pathb_fig_vp")

                    with dp2:
                        fig_vs = go.Figure()
                        fig_vs.add_trace(go.Scatter(
                            x=Vs, y=depth_vals, mode="lines",
                            name=f"Vs original ({old_fluid})",
                            line=dict(color="#0095ff", width=1)
                        ))
                        fig_vs.add_trace(go.Scatter(
                            x=Vs_new, y=depth_vals, mode="lines",
                            name=f"Vs substituted ({new_fluid})",
                            line=dict(color="#ff0303", width=1)
                        ))
                        fig_vs.update_layout(
                            **track_layout("Vs — Original vs Substituted", "Vs  (m/s)"))
                        st.plotly_chart(fig_vs, use_container_width=True, key="pathb_fig_vs")

                    st.success(
                        f"✅ Fluid substitution complete  |  "
                        f"{n_valid:,} valid samples  |  "
                        f"VP_SUB and VS_SUB saved to dataframe — "
                        f"available in Crossplot (Tab 4) and Elastic Properties (Tab 6)."
                    )
        
        
                # ════════════════════════════════════════════════════════════════
        # SECTION 7 — ROCK PHYSICS TEMPLATE (RPT)
        # ════════════════════════════════════════════════════════════════
        with st.expander("Rock Physics Template (RPT)", expanded=False):

            st.markdown('<div class="rp-section-header">Theory</div>', unsafe_allow_html=True)

            st.markdown("""
            The **Rock Physics Template (RPT)** is a crossplot of two elastic attributes
            — typically **Acoustic Impedance (Ip)** vs **Vp/Vs ratio** — where theoretical
            model curves form a background grid and real well-log data is overlaid as a scatter.

            The template allows direct visual comparison of:
            - **Where the data falls** relative to model predictions
            - **Which fluid** the pore space likely contains (brine shifts Ip high, gas shifts it low)
            - **Which rock model** best describes the formation (Soft-Sand vs Stiff-Sand)
            - **Saturation trends** — points colored by Sw show the fluid gradient

            The RPT is the primary diagnostic tool connecting rock physics models to
            seismic observations (Avseth et al., 2005).
            """)

            st.markdown("---")

            # ════════════════════════════════════════════════════════════
            # STEP 1 — PREREQUISITE CHECK
            # ════════════════════════════════════════════════════════════
            rpt_curves = st.session_state.get("rpt_model_curves", None)
            rpt_phi    = st.session_state.get("gass_phi",         None)
            rpt_phic   = st.session_state.get("hm_phic",          None)
            df_rpt     = st.session_state.get("df",               None)

            missing_prereqs = []
            if rpt_curves is None:
                missing_prereqs.append("**Section 6 — Gassmann Forward Model** (to generate RPT curves)")

            if missing_prereqs:
                st.warning(
                    "⚠️ The following section(s) must be run before the RPT can be displayed:\n\n"
                    + "\n".join(f"- {m}" for m in missing_prereqs)
                )
            else:
                # ════════════════════════════════════════════════════════
                # STEP 2 — AXIS CONFIGURATION
                # ════════════════════════════════════════════════════════
                st.markdown('<div class="rp-section-header">Axis Configuration</div>',
                            unsafe_allow_html=True)

                AXIS_OPTIONS = ["Ip  (kg/m²·s × 10³)", "Vp/Vs", "Vp  (m/s)", "Vs  (m/s)"]
                AXIS_KEY_MAP = {
                    "Ip  (kg/m²·s × 10³)": "Ip",
                    "Vp/Vs":               "VpVs",
                    "Vp  (m/s)":           "Vp",
                    "Vs  (m/s)":           "Vs",
                }

                ax1, ax2 = st.columns(2)
                with ax1:
                    x_axis_label = st.selectbox(
                        "X-axis", options=AXIS_OPTIONS,
                        index=0, key="rpt_xaxis"
                    )
                with ax2:
                    y_axis_label = st.selectbox(
                        "Y-axis", options=AXIS_OPTIONS,
                        index=1, key="rpt_yaxis"
                    )

                x_key = AXIS_KEY_MAP[x_axis_label]
                y_key = AXIS_KEY_MAP[y_axis_label]

                if x_key == y_key:
                    st.warning("X and Y axes are the same — select two different attributes.")
                else:
                    st.markdown("---")

                    # ════════════════════════════════════════════════════
                    # STEP 3 — MODEL CURVE SELECTION
                    # ════════════════════════════════════════════════════
                    st.markdown('<div class="rp-section-header">Model Curve Selection</div>',
                                unsafe_allow_html=True)

                    all_curve_keys = list(rpt_curves.keys())
                    # all_curve_keys are tuples like ("Soft-Sand", "Brine")

                    LINESTYLES_RPT = {
                        "Soft-Sand":  "solid",
                        "Stiff-Sand": "dash",
                        "Crit-Por":   "dot",
                    }
                    FLUID_COLORS_RPT = {
                        "Brine":  "#008bee",
                        "Oil":    "#00c53b",
                        "Gas":    "#ff2b2b",
                        "Custom": "#fff70e",
                    }

                    with st.container(border=True):
                        st.markdown("**Select model–fluid combinations to overlay on RPT:**")
                        curve_cols = st.columns(min(len(all_curve_keys), 4))
                        selected_curves = {}
                        for i, (mname, fname) in enumerate(all_curve_keys):
                            col_idx = i % len(curve_cols)
                            label   = f"{mname} — {fname}"
                            with curve_cols[col_idx]:
                                checked = st.checkbox(
                                    label,
                                    value=True,
                                    key=f"rpt_curve_{mname}_{fname}"
                                )
                            if checked:
                                selected_curves[(mname, fname)] = rpt_curves[(mname, fname)]

                    st.markdown("---")

                    # ════════════════════════════════════════════════════
                    # STEP 4 — POROSITY LABEL INTERVAL
                    # ════════════════════════════════════════════════════
                    phi_label_col, _ = st.columns([2, 5])
                    with phi_label_col:
                        phi_tick_step = st.number_input(
                            "Porosity label interval along curves",
                            value=0.05, min_value=0.01, max_value=0.20,
                            format="%.2f", key="rpt_phi_tick",
                            help="Porosity values annotated along each model curve"
                        )

                    st.markdown("---")

                    # ════════════════════════════════════════════════════
                    # STEP 5 — LOG OVERLAY CONFIGURATION
                    # ════════════════════════════════════════════════════
                    st.markdown('<div class="rp-section-header">Log Data Overlay</div>',
                                unsafe_allow_html=True)

                    has_df    = df_rpt is not None and len(df_rpt) > 0
                    df_cols   = list(df_rpt.columns) if has_df else []

                    # ── Overlay A — Original Logs ─────────────────────────
                    with st.container(border=True):
                        st.markdown("#### Overlay A — Original Well Logs")

                        use_overlay_a = st.checkbox(
                            "Enable original log overlay",
                            value=has_df, key="rpt_use_overlay_a",
                            disabled=not has_df
                        )

                        if not has_df:
                            st.caption("No well log data found — upload a LAS/CSV file to enable.")

                        if use_overlay_a and has_df:
                            oa1, oa2 = st.columns(2)

                            # ── X log selector ───────────────────────────
                            X_LOG_CANDIDATES = {
                                "Ip  (kg/m²·s × 10³)": ["AI", "IP"],
                                "Vp/Vs":               ["VPVS", "VP_VS"],
                                "Vp  (m/s)":           ["VP", "VP_EST", "VPCO"],
                                "Vs  (m/s)":           ["VS", "VS_EST"],
                            }
                            Y_LOG_CANDIDATES = X_LOG_CANDIDATES

                            def best_log_col(candidates, columns, fallback_idx=0):
                                for c in candidates:
                                    for col in columns:
                                        if c.upper() == col.upper():
                                            return col
                                for c in candidates:
                                    for col in columns:
                                        if c.upper() in col.upper():
                                            return col
                                return columns[fallback_idx] if columns else None

                            xcands_a = X_LOG_CANDIDATES.get(x_axis_label, [])
                            ycands_a = Y_LOG_CANDIDATES.get(y_axis_label, [])

                            with oa1:
                                xcol_a_default = best_log_col(xcands_a, df_cols)
                                xcol_a = st.selectbox(
                                    f"X log  [{x_axis_label}]",
                                    options=df_cols,
                                    index=df_cols.index(xcol_a_default) if xcol_a_default in df_cols else 0,
                                    key="rpt_xcol_a"
                                )
                            with oa2:
                                ycol_a_default = best_log_col(ycands_a, df_cols)
                                ycol_a = st.selectbox(
                                    f"Y log  [{y_axis_label}]",
                                    options=df_cols,
                                    index=df_cols.index(ycol_a_default) if ycol_a_default in df_cols else 0,
                                    key="rpt_ycol_a"
                                )

                            # ── Color property selector ──────────────────
                            SW_COLS  = [c for c in ["SWA", "SWS", "SW"] if c in df_cols]
                            PHI_COLS = [c for c in ["PHIE", "PHIT", "DPHI", "NPHI"] if c in df_cols]
                            VSH_COLS = [c for c in ["VSH"] if c in df_cols]
                            DEPTH_COLS = [c for c in df_cols if "DEPT" in c.upper() or "DEPTH" in c.upper() or "MD" in c.upper()]

                            color_options_a = (
                                (["Sw — " + c for c in SW_COLS])  +
                                (["Porosity — " + c for c in PHI_COLS]) +
                                (["VSH — " + c for c in VSH_COLS]) +
                                (["Depth — " + c for c in DEPTH_COLS])
                            )

                            oc1, oc2 = st.columns([2, 3])
                            with oc1:
                                if color_options_a:
                                    color_choice_a = st.selectbox(
                                        "Color scatter by",
                                        options=color_options_a,
                                        index=0, key="rpt_color_a"
                                    )
                                    # Parse selected column
                                    color_col_a = color_choice_a.split(" — ")[-1]

                                    COLORSCALE_MAP = {
                                        "Sw":       "RdBu_r",
                                        "Porosity": "viridis",
                                        "VSH":      "YlOrBr",
                                        "Depth":    "plasma",
                                    }
                                    color_prop_a = color_choice_a.split(" — ")[0]
                                    colorscale_a = COLORSCALE_MAP.get(color_prop_a, "viridis")
                                else:
                                    st.caption("No suitable color log found.")
                                    color_col_a  = None
                                    colorscale_a = "viridis"

                            with oc2:
                                marker_size_a = st.slider(
                                    "Marker size", min_value=2, max_value=12,
                                    value=5, key="rpt_markersize_a"
                                )
                                marker_opacity_a = st.slider(
                                    "Marker opacity", min_value=0.1, max_value=1.0,
                                    value=0.70, step=0.05, key="rpt_opacity_a"
                                )

                    # ── Overlay B — Fluid-Substituted Logs ───────────────
                    with st.container(border=True):
                        st.markdown("#### Overlay B — Fluid-Substituted Logs")

                        has_sub = has_df and "VP_SUB" in df_cols and "VS_SUB" in df_cols
                        use_overlay_b = st.checkbox(
                            "Enable substituted log overlay",
                            value=has_sub, key="rpt_use_overlay_b",
                            disabled=not has_sub
                        )

                        if not has_sub:
                            st.caption(
                                "VP_SUB / VS_SUB not found — run **Section 6B (Log-Based Fluid Substitution)** first."
                            )

                        if use_overlay_b and has_sub:
                            # Compute Ip_sub and VpVs_sub from VP_SUB, VS_SUB, RHOB
                            RHOB_COLS = [c for c in ["RHOB", "RHOZ", "DEN", "ZDEN"] if c in df_cols]

                            ob1, ob2 = st.columns(2)
                            with ob1:
                                if RHOB_COLS:
                                    rhob_col_b = st.selectbox(
                                        "RHOB log for Ip_sub  (g/cc)",
                                        options=RHOB_COLS,
                                        key="rpt_rhob_b"
                                    )
                                else:
                                    rhob_col_b = st.selectbox(
                                        "RHOB log for Ip_sub  (g/cc)",
                                        options=df_cols, key="rpt_rhob_b"
                                    )
                            with ob2:
                                marker_size_b = st.slider(
                                    "Marker size (sub)", min_value=2, max_value=12,
                                    value=5, key="rpt_markersize_b"
                                )

                    st.markdown("---")

                    # ════════════════════════════════════════════════════
                    # STEP 6 — PLOT RENDERING
                    # ════════════════════════════════════════════════════
                    btn_rpt, _ = st.columns([2, 6])
                    with btn_rpt:
                        run_rpt = st.button(
                            "▶ Generate RPT",
                            key="run_rpt",
                            type="primary",
                            use_container_width=True
                        )

                    if run_rpt:
                        st.session_state["rpt_done"] = True

                    if run_rpt or st.session_state.get("rpt_done"):

                        fig_rpt = go.Figure()

                        # ── Helper: extract axis values from a result dict ──
                        def get_axis_vals(res, key):
                            if key == "Ip":
                                return res["Ip"] / 1000.0   # convert to 10³ units
                            return res[key]

                        # ── Layer 1: Model Curves ─────────────────────────
                        phi_arr_full = np.array(rpt_phi)

                        for (mname, fname), res in selected_curves.items():
                            xvals = get_axis_vals(res, x_key)
                            yvals = get_axis_vals(res, y_key)
                            phi_c = np.array(res["phi"])

                            curve_color = FLUID_COLORS_RPT.get(fname, "#999999")
                            line_dash   = LINESTYLES_RPT.get(mname, "solid")
                            label       = f"{mname} — {fname}"

                            fig_rpt.add_trace(go.Scatter(
                                x=xvals, y=yvals,
                                mode="lines",
                                name=label,
                                line=dict(color=curve_color, width=2.5, dash=line_dash),
                                hovertemplate=(
                                    f"<b>{label}</b><br>"
                                    f"X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>"
                                    "ϕ: %{customdata:.3f}<extra></extra>"
                                ),
                                customdata=phi_c,
                            ))

                            # ── Porosity tick annotations along curve ─────
                            phi_ticks = np.arange(0.0, float(rpt_phic) + 0.001, phi_tick_step)
                            phi_ticks = phi_ticks[phi_ticks <= float(rpt_phic) + 0.001]

                            for ptick in phi_ticks:
                                idx    = np.argmin(np.abs(phi_c - ptick))
                                xt     = float(xvals[idx])
                                yt     = float(yvals[idx])
                                fig_rpt.add_annotation(
                                    x=xt, y=yt,
                                    text=f"ϕ={ptick:.2f}",
                                    showarrow=False,
                                    font=dict(size=8, color=curve_color),
                                    bgcolor="rgba(255,255,255,0.6)",
                                    borderpad=1,
                                    xanchor="left",
                                    yanchor="bottom",
                                )

                            # ── End-of-curve fluid label ──────────────────
                            fig_rpt.add_annotation(
                                x=float(xvals[-1]), y=float(yvals[-1]),
                                text=f"<b>{fname}</b>",
                                showarrow=False,
                                font=dict(size=9, color=curve_color),
                                bgcolor="rgba(255,255,255,0.75)",
                                bordercolor=curve_color,
                                borderwidth=1,
                                borderpad=2,
                                xanchor="left",
                            )

                        # ── Layer 2: Original Log Scatter (Overlay A) ─────
                        if use_overlay_a and has_df and color_col_a:
                            try:
                                sub_a = df_rpt[[xcol_a, ycol_a, color_col_a]].dropna()

                                xa_vals = sub_a[xcol_a].values
                                ya_vals = sub_a[ycol_a].values
                                ca_vals = sub_a[color_col_a].values

                                # Ip unit conversion if needed
                                if x_key == "Ip":
                                    xa_vals = xa_vals / 1000.0
                                if y_key == "Ip":
                                    ya_vals = ya_vals / 1000.0

                                fig_rpt.add_trace(go.Scatter(
                                    x=xa_vals,
                                    y=ya_vals,
                                    mode="markers",
                                    name=f"Logs — colored by {color_col_a}",
                                    marker=dict(
                                        color=ca_vals,
                                        colorscale=colorscale_a,
                                        size=marker_size_a,
                                        opacity=marker_opacity_a,
                                        colorbar=dict(
                                            title=dict(text=color_col_a, side="right"),
                                            thickness=14,
                                            len=0.6,
                                            x=1.02,
                                        ),
                                        showscale=True,
                                        line=dict(width=0),
                                    ),
                                    hovertemplate=(
                                        f"<b>Log Data</b><br>"
                                        f"X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>"
                                        f"{color_col_a}: %{{marker.color:.3f}}"
                                        "<extra></extra>"
                                    ),
                                ))
                            except Exception as e:
                                st.warning(f"Could not plot Overlay A: {e}")

                        # ── Layer 3: Substituted Log Scatter (Overlay B) ──
                        if use_overlay_b and has_sub:
                            try:
                                sub_b = df_rpt[["VP_SUB", "VS_SUB", rhob_col_b]].dropna()

                                vp_sub  = sub_b["VP_SUB"].values
                                vs_sub  = sub_b["VS_SUB"].values
                                rho_sub = sub_b[rhob_col_b].values

                                ip_sub   = (rho_sub * vp_sub) / 1000.0   # 10³ units
                                vpvs_sub = np.where(vs_sub > 0, vp_sub / vs_sub, np.nan)

                                x_log_map_b = {
                                    "Ip":   ip_sub,
                                    "VpVs": vpvs_sub,
                                    "Vp":   vp_sub,
                                    "Vs":   vs_sub,
                                }
                                xb_vals = x_log_map_b.get(x_key, ip_sub)
                                yb_vals = x_log_map_b.get(y_key, vpvs_sub)

                                fig_rpt.add_trace(go.Scatter(
                                    x=xb_vals,
                                    y=yb_vals,
                                    mode="markers",
                                    name="Substituted Logs (VP_SUB / VS_SUB)",
                                    marker=dict(
                                        color="#ff7f0e",
                                        size=marker_size_b,
                                        opacity=0.65,
                                        symbol="circle-open",
                                        line=dict(width=1.5, color="#ff7f0e"),
                                    ),
                                    hovertemplate=(
                                        "<b>Substituted Log</b><br>"
                                        "X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
                                    ),
                                ))
                            except Exception as e:
                                st.warning(f"Could not plot Overlay B: {e}")

                        # ── Layout ────────────────────────────────────────
                        fig_rpt.update_layout(
                            title=dict(
                                text=f"Rock Physics Template — {x_axis_label} vs {y_axis_label}",
                                font=dict(size=14),
                                x=0.5, xanchor="center"
                            ),
                            xaxis=dict(
                                title=x_axis_label,
                                showgrid=True, gridcolor="#eeeeee",
                                showline=True, linecolor="#aaaaaa",
                                zeroline=False,
                            ),
                            yaxis=dict(
                                title=y_axis_label,
                                showgrid=True, gridcolor="#eeeeee",
                                showline=True, linecolor="#aaaaaa",
                                zeroline=False,
                            ),
                            legend=dict(
                                font=dict(size=9),
                                bgcolor="rgba(255,255,255,0.88)",
                                bordercolor="#dddddd",
                                borderwidth=1,
                                x=0.01, y=0.99,
                                xanchor="left", yanchor="top",
                            ),
                            height=620,
                            plot_bgcolor="white",
                            paper_bgcolor="#fafafa",
                            margin=dict(l=70, r=90, t=60, b=60),
                        )

                        st.plotly_chart(fig_rpt, use_container_width=True, key="fig_rpt_main")

                        # ── Save figure state ─────────────────────────────
                        st.session_state["rpt_done"] = True
                        st.success(
                            "✅ RPT generated — model curves and log overlay plotted successfully."
                        )