import streamlit as st
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from pathlib import Path

# ---------------- Config ----------------
st.set_page_config(page_title="SOURCE_KEY Status", layout="wide")
st.title("SOURCE_KEY Status")

K = 2.5  # z-threshold

# ---------------- Data helpers ----------------
CWD = Path.cwd()

def read_csv(name: str) -> pd.DataFrame:
    p = CWD / name
    if not p.exists():
        raise FileNotFoundError(f"Missing: {name} (looked in {CWD})")
    return pd.read_csv(p, parse_dates=['DATE_TIME'])

@st.cache_data(show_spinner=False)
def load_generation() -> pd.DataFrame:
    g1 = read_csv("Plant_1_Generation_Data.csv")
    g2 = read_csv("Plant_2_Generation_Data.csv")
    gen = pd.concat([g1, g2], ignore_index=True, sort=False)

    need = {'DATE_TIME','PLANT_ID','SOURCE_KEY','DAILY_YIELD'}
    miss = need - set(gen.columns)
    if miss:
        raise ValueError(f"Generation CSVs missing columns: {miss}")

    gen['date'] = gen['DATE_TIME'].dt.date
    daily = (gen.groupby(['SOURCE_KEY','PLANT_ID','date'], as_index=False)['DAILY_YIELD']
               .sum()
               .rename(columns={'DAILY_YIELD':'yield'}))
    return daily  # SOURCE_KEY, PLANT_ID, date, yield

def _daily_sensor_from_weather(wx: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    wx = wx.copy()
    wx['date'] = wx['DATE_TIME'].dt.date
    for c in sensor_cols:
        if c in wx.columns:
            col = c
            break
    else:
        num = [c for c in wx.select_dtypes('number').columns if c != 'PLANT_ID']
        if not num:
            raise ValueError("No numeric sensor column in weather file.")
        col = num[0]
    return wx.groupby('date', as_index=False)[col].mean().rename(columns={col: 'sensor'})

@st.cache_data(show_spinner=False)
def load_weather_daily_by_file() -> dict:
    SCOLS = ['IRRADIANCE','IRRADIATION','GLOBAL_HORIZON_IRRADIANCE']
    w1 = _daily_sensor_from_weather(read_csv("Plant_1_Weather_Sensor_Data.csv"), SCOLS)
    w2 = _daily_sensor_from_weather(read_csv("Plant_2_Weather_Sensor_Data.csv"), SCOLS)
    return {1: w1, 2: w2}

@st.cache_data(show_spinner=False)
def plant_id_map() -> dict:
    g1 = read_csv("Plant_1_Generation_Data.csv")
    g2 = read_csv("Plant_2_Generation_Data.csv")
    pids1 = g1['PLANT_ID'].dropna().unique()
    pids2 = g2['PLANT_ID'].dropna().unique()
    if len(pids1) != 1 or len(pids2) != 1:
        raise ValueError(f"Expected exactly one PLANT_ID per generation file; got {pids1=} {pids2=}")
    return {int(pids1[0]): 1, int(pids2[0]): 2}  # REAL PLANT_ID -> which weather file (1 or 2)

@st.cache_data(show_spinner=False)
def build_training_frame() -> pd.DataFrame:
    gen_daily = load_generation()
    wx_daily_files = load_weather_daily_by_file()
    pid_to_file = plant_id_map()

    parts = []
    for pid, grp in gen_daily.groupby('PLANT_ID'):
        idx = pid_to_file.get(int(pid))
        if idx is None:
            continue
        wx = wx_daily_files[idx]
        merged = pd.merge(grp, wx, on='date', how='inner')  # join on date only
        parts.append(merged)

    if not parts:
        raise ValueError("No overlaps after date-only merge. Check that weather and generation dates overlap.")
    return pd.concat(parts, ignore_index=True, sort=False)  # SOURCE_KEY, PLANT_ID, date, yield, sensor

# ---------------- Scoring (MAD z, K=2.5) ----------------
@st.cache_data(show_spinner=False)
def score_all(k: float = K) -> pd.DataFrame:
    df = build_training_frame().copy()
    x = df['sensor'].to_numpy()
    y = df['yield'].to_numpy()
    if x.size < 2:
        raise ValueError(f"Need >=2 samples to fit isotonic regression; got {x.size}.")

    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    y_hat = iso.fit_transform(x, y)

    res = y - y_hat
    mad = np.median(np.abs(res - np.median(res)))
    sigma = 1.4826 * mad if mad > 0 else np.std(res)
    sigma = sigma if sigma > 0 else 1.0

    z = res / sigma
    under = z < -k
    over  = z >  k

    df['expected_y'] = y_hat
    df['z'] = z
    df['label'] = np.where(under, 'underperforming',
                    np.where(over,  'overperforming', 'normal'))
    return df

# ---------------- Overview helpers ----------------
def fmt_date(d):
    try:
        return pd.to_datetime(d).strftime("%B %d, %Y")
    except Exception:
        return str(d)

SOURCE_META_DEFAULT = {
    "location": "W53V+3FP, Santanpur, Charanka, Gujarat 385350, India",
    "model": "550W Monocrystalline",
    "manufacturer": "Waaree Energies",
    "installed": "",
    "last_inspection": "",
    "inspector": "",
    "notes": "",
}
SOURCE_META = {}  # per-key overrides if you have them

def get_meta(source_key: str) -> dict:
    base = SOURCE_META_DEFAULT.copy()
    base.update(SOURCE_META.get(source_key, {}))
    return {
        "location": base.get("location", "—") or "—",
        "model": base.get("model", "—") or "—",
        "manufacturer": base.get("manufacturer", "—") or "—",
        "installed": fmt_date(base.get("installed", "")),
        "last_inspection": fmt_date(base.get("last_inspection", "")),
        "inspector": base.get("inspector", "—") or "—",
        "notes": base.get("notes", "—") or "—",
    }

def compute_overview(source_key: str, k: float = K) -> dict:
    df = score_all(k)
    key = source_key.strip()
    sub = df[df['SOURCE_KEY'] == key].copy()
    if sub.empty:
        examples = ", ".join(map(str, df['SOURCE_KEY'].unique()[:5]))
        raise ValueError(f"SOURCE_KEY not found: {key!r}. Examples: {examples}")

    latest = sub['date'].max()
    row = sub.loc[sub['date'] == latest].iloc[0]
    label = str(row['label']).lower()
    z = float(row['z'])
    last_y = float(row['yield'])
    expected = float(row['expected_y'])

    flagged = sub[(sub['z'] < -k) | (sub['z'] > k)].sort_values('date')
    last_flagged_date = flagged['date'].max() if not flagged.empty else None

    sub_sorted = sub.sort_values('date')
    sub_sorted['under_bool'] = sub_sorted['yield'] < sub_sorted['expected_y']
    streak_days = int(sub_sorted['under_bool'].tail(3).sum())

    meta = get_meta(key)
    return {
        "source_key": key,
        "status": label,
        "z": z,
        "latest_date": latest,
        "last_yield": last_y,
        "expected_yield": expected,
        "streak_under_3": streak_days,
        "last_flagged_date": last_flagged_date,
        "meta": meta
    }

def status_badge(label: str) -> str:
    if label == "normal":
        return "✅ Working Normally"
    if label == "overperforming":
        return "⚠️ Overperforming"
    return "⚠️ Underperforming"

def normal_range_text(last_y, expected_y):
    diff = last_y - expected_y
    pct = (diff / expected_y * 100.0) if expected_y else 0.0
    side = "above" if diff > 0 else "below" if diff < 0 else "at"
    return f"{last_y:,.0f} (≈ {abs(pct):.0f}% {side} expected of {expected_y:,.0f})"

# ---------------- UI ----------------

# 1) Form for lookups (debounced)
with st.form("lookup", clear_on_submit=False):
    # keep last-entered key in the box between reruns
    default_val = st.session_state.get("last_source_key", "")
    sk = st.text_input("Enter SOURCE_KEY", value=default_val, placeholder="paste exact key")
    go = st.form_submit_button("Check status")

# 2) If a new lookup was requested, compute & store it
if go:
    try:
        ov = compute_overview(sk, k=K)
        st.session_state["last_source_key"] = sk.strip()
        st.session_state["last_overview"] = ov
        # a new lookup should clear prior scheduling state
        st.session_state.pop("scheduled", None)
    except Exception as e:
        st.session_state.pop("last_overview", None)  # clear stale result
        st.error(str(e))

# 3) Render the most recent overview (persists across reruns)
ov = st.session_state.get("last_overview")
if ov:
    meta = ov["meta"]
    st.divider()

    def fmt_date(d):
        try:
            return pd.to_datetime(d).strftime("%B %d, %Y")
        except Exception:
            return "—"

    last_flagged_line = (
        f"**Most Recent Flagged Day (|z|>{K}):** {fmt_date(ov['last_flagged_date'])}"
        if ov["last_flagged_date"] else
        f"**Most Recent Flagged Day (|z|>{K}):** —"
    )

    st.markdown(
        f"""
**Panel ID:** {ov['source_key']}  
**Location:** {meta['location']}  
**Model:** {meta['model']}  
**Manufacturer:** {meta['manufacturer']}  
**Installation:** {meta['installed']}

**Status:** {status_badge(ov['status'])}  
**Last Output:** {normal_range_text(ov['last_yield'], ov['expected_yield'])}  
{last_flagged_line}  
**Inspection Notes:** {meta['notes']}

**Suggestion:** {"This panel has been operating under expected for the past 3 days." if ov["streak_under_3"] >= 3 else "No action needed. Performance is within expected range recently."}
        """
    )

    # 4) CTA + confirmation (these cause reruns but we still have last_overview)
    if st.button("⭐ Schedule Check-In", key="schedule_btn"):
        st.session_state["scheduled"] = True

    if st.session_state.get("scheduled"):
        st.success(f"Check-in scheduled for {ov['source_key']}! Check email for confirmation.")
else:
    st.caption(f"Enter a SOURCE_KEY and press **Check status** (z-threshold = {K}).")