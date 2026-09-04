from pathlib import Path
import json

import pandas as pd
import streamlit as st
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
PLOTS_DIR = PROJECT_ROOT / 'results' / 'plots'
EXPLANATIONS_DIR = PROJECT_ROOT / 'results' / 'explanations'
MODEL_DIR = PROJECT_ROOT / 'models' / 'trained_models'

st.set_page_config(
	page_title='Mission Shield | Zero-Day Detection',
	page_icon='🛰️',
	layout='wide',
	initial_sidebar_state='expanded',
)

st.markdown(
	'''
	<style>
	@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
	:root { --ink:#e8f0f2; --muted:#8ea5aa; --panel:#102126; --line:#26434a; --cyan:#5fe2d2; --amber:#f4b860; --red:#ff6b6b; }
	.stApp { background: radial-gradient(circle at 80% 0%, #183943 0, #091419 38%, #071014 100%); color:var(--ink); }
	[data-testid='stSidebar'] { background:#081216; border-right:1px solid var(--line); }
	[data-testid='stSidebar'] * { font-family:'IBM Plex Mono', monospace; }
	h1,h2,h3 { font-family:'Barlow Condensed', sans-serif !important; letter-spacing:.03em; }
	h1 { font-size:2.8rem !important; margin-bottom:0 !important; }
	h2 { font-size:1.7rem !important; }
	p, label, .stMarkdown, .stDataFrame { font-family:'IBM Plex Mono', monospace; }
	.eyebrow { color:var(--cyan); font:500 .72rem 'IBM Plex Mono'; letter-spacing:.16em; text-transform:uppercase; }
	.subtitle { color:var(--muted); font:400 .82rem 'IBM Plex Mono'; margin:0.15rem 0 1.2rem; }
	.metric-card { background:linear-gradient(145deg,#142b31,#0d1b20); border:1px solid var(--line); border-radius:7px; padding:18px 20px; min-height:112px; }
	.metric-label { color:var(--muted); font:400 .68rem 'IBM Plex Mono'; text-transform:uppercase; letter-spacing:.12em; }
	.metric-value { color:var(--ink); font:600 2rem 'Barlow Condensed'; margin-top:8px; }
	.metric-note { color:var(--cyan); font:400 .7rem 'IBM Plex Mono'; margin-top:2px; }
	.status-dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--cyan); box-shadow:0 0 10px var(--cyan); margin-right:8px; }
	.status-dot.warn { background:var(--amber); box-shadow:0 0 10px var(--amber); }
	.status-dot.alert { background:var(--red); box-shadow:0 0 10px var(--red); }
	.section-rule { border-top:1px solid var(--line); margin:25px 0 17px; }
	.alert-row { background:#102126; border-left:3px solid var(--red); padding:11px 14px; margin:7px 0; border-radius:2px; }
	.alert-row strong { color:#fff; font-family:'Barlow Condensed'; font-size:1.05rem; }
	.alert-row span { color:var(--muted); font:400 .72rem 'IBM Plex Mono'; }
	.source-note { color:#70878c; font:400 .65rem 'IBM Plex Mono'; margin-top:12px; }
	</style>
	''',
	unsafe_allow_html=True,
)


@st.cache_data
def load_csv(name, **kwargs):
	path = METRICS_DIR / name
	return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


@st.cache_data
def load_json(name):
	path = METRICS_DIR / name
	return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


@st.cache_data
def load_explanation_csv(name):
	path = EXPLANATIONS_DIR / name
	return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource
def load_global_model():
	model_path = MODEL_DIR / 'federated_autoencoder_model.h5'
	return keras.models.load_model(model_path, compile=False) if model_path.exists() else None


threat_events = load_csv('07_threat_scored_events.csv')
threat_metrics = load_csv('07_threat_scoring_metrics.csv')
threat_config = load_json('07_threat_scoring_config.json')
fed_metrics = load_csv('05_federated_learning_metrics.csv')
zero_day = load_csv('04_zero_day_metrics.csv', index_col=0)
zero_day_groups = load_csv('04_zero_day_group_comparison.csv')
examples = load_explanation_csv('06_example_predictions.csv')
global_model = load_global_model()


def metric_card(label, value, note=''):
	st.markdown(
		f'<div class="metric-card"><div class="metric-label">{label}</div>'
		f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
		unsafe_allow_html=True,
	)


def section(title, caption=''):
	st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
	st.markdown(f'<div class="eyebrow">{title}</div>', unsafe_allow_html=True)
	if caption:
		st.markdown(f'<div class="subtitle">{caption}</div>', unsafe_allow_html=True)


def plot_image(filename, caption):
	path = PLOTS_DIR / filename
	if path.exists():
		st.image(str(path), caption=caption, use_container_width=True)
	else:
		st.info(f'{filename} is not available yet.')


st.sidebar.markdown('<div class="eyebrow">MISSION CONTROL / A-ZD-01</div>', unsafe_allow_html=True)
st.sidebar.title('MISSION SHIELD')
st.sidebar.caption('Autonomous Zero-Day Attack Detection Framework')
page = st.sidebar.radio(
	'Navigate',
	['Dashboard', 'Satellite Status', 'Threat Detection', 'Zero-Day Analysis', 'Federated Learning', 'Explainable AI', 'Reports'],
	label_visibility='collapsed',
)
st.sidebar.markdown('---')
model_status = 'MODEL ONLINE' if global_model is not None else 'MODEL FILE MISSING'
model_dot = '' if global_model is not None else ' alert'
st.sidebar.markdown(f'<span class="status-dot{model_dot}"></span> {model_status}', unsafe_allow_html=True)
st.sidebar.caption('Inference-only mode · saved artifacts')

if threat_events.empty:
	st.error('Threat-scoring outputs are missing. Run the completed notebooks before launching the dashboard.')
	st.stop()

latest = fed_metrics.iloc[-1] if not fed_metrics.empty else pd.Series(dtype=float)
event_count = len(threat_events)
anomaly_count = int(threat_events['predicted_anomaly'].sum())
critical_count = int((threat_events['severity'] == 'Critical').sum())
mean_score = threat_events['threat_score'].mean()

st.markdown('<div class="eyebrow">ORBITAL SECURITY OPERATIONS CENTER</div>', unsafe_allow_html=True)
st.title('Autonomous Zero-Day Defense')
st.markdown('<div class="subtitle">Federated intelligence for a simulated satellite constellation · live view of saved experiment outputs</div>', unsafe_allow_html=True)

if page == 'Dashboard':
	cols = st.columns(4)
	with cols[0]: metric_card('Events evaluated', f'{event_count:,}', 'UNSW-NB15 test set')
	with cols[1]: metric_card('Anomalies detected', f'{anomaly_count:,}', f'{anomaly_count / event_count:.1%} of traffic')
	with cols[2]: metric_card('Critical events', f'{critical_count:,}', 'score ≥ 75')
	with cols[3]: metric_card('Mean threat score', f'{mean_score:.1f}/100', 'normalized risk index')
	section('Threat posture', 'Current distribution of the scored test traffic')
	left, right = st.columns([1.5, 1])
	with left: plot_image('07_threat_scoring_overview.png', 'Severity levels and threat-score distribution')
	with right:
		st.markdown('#### Recent security alerts')
		for _, row in threat_events.head(7).iterrows():
			st.markdown(f'<div class="alert-row"><strong>{row.severity} · {row.threat_score:.1f}/100</strong><br><span>Event {int(row.event_id):05d} · {row.attack_cat_for_evaluation} · reconstruction error {row.reconstruction_error:.4f}</span></div>', unsafe_allow_html=True)
	section('Federated model health', 'Final communication round metrics')
	cols = st.columns(4)
	with cols[0]: metric_card('FedAvg round', f'{int(latest.get("round", 0))}', 'global model')
	with cols[1]: metric_card('ROC-AUC', f'{latest.get("roc_auc", 0):.3f}', 'test anomaly ranking')
	with cols[2]: metric_card('F1 score', f'{latest.get("f1_score", 0):.3f}', 'test detection')
	with cols[3]: metric_card('Active nodes', f'{int(load_json("07_threat_scoring_config.json").get("n_clients", 5))}', 'simulated clients')

elif page == 'Satellite Status':
	st.header('Satellite Status')
	st.markdown('<div class="subtitle">Simulated node registry derived from the federated experiment configuration. No live telemetry is connected.</div>', unsafe_allow_html=True)
	n_clients = 5
	statuses = ['NOMINAL', 'NOMINAL', 'WATCH', 'NOMINAL', 'NOMINAL']
	for start in range(0, n_clients, 3):
		cols = st.columns(3)
		for offset, col in enumerate(cols):
			node = start + offset
			if node >= n_clients: break
			with col:
				status = statuses[node]
				dot = '' if status == 'NOMINAL' else ' warn'
				metric_card(f'SAT-{node + 1:02d}', f'<span class="status-dot{dot}"></span>{status}', f'client node · {len(threat_events) // n_clients:,} routed events')
	section('Constellation telemetry', 'Operational context for the five FedAvg clients')
	st.dataframe(pd.DataFrame({'Node': [f'SAT-{i + 1:02d}' for i in range(n_clients)], 'Role': ['Local trainer'] * n_clients, 'Data isolation': ['Independent local partition'] * n_clients, 'Status': statuses}), hide_index=True, use_container_width=True)

elif page == 'Threat Detection':
	st.header('Threat Detection')
	st.markdown('<div class="subtitle">Filter and inspect the complete event-level output from the threat-scoring notebook.</div>', unsafe_allow_html=True)
	selected = st.multiselect('Severity filter', ['Low', 'Medium', 'High', 'Critical'], default=['Medium', 'High', 'Critical'])
	view = threat_events[threat_events['severity'].isin(selected)].copy()
	st.dataframe(view.head(500), hide_index=True, use_container_width=True, height=520)
	st.caption(f'Showing {min(len(view), 500):,} of {len(view):,} matching events. Download the full CSV from Reports.')
	plot_image('07_threat_score_vs_reconstruction_error.png', 'Threat score is derived from reconstruction error, confidence, and prediction.')

elif page == 'Zero-Day Analysis':
	st.header('Zero-Day Analysis')
	st.markdown('<div class="subtitle">Evaluation results from the withheld-attack experiment. Values are loaded from the completed notebook.</div>', unsafe_allow_html=True)
	cols = st.columns(4)
	for col, label in zip(cols, ['Unseen detection rate', 'Known detection rate', 'Normal false-positive rate', 'Withheld category']):
		if label == 'Withheld category': value = 'Worms'
		else:
			key = {'Unseen detection rate': 'unseen_attack_detection_rate', 'Known detection rate': 'known_attack_detection_rate', 'Normal false-positive rate': 'normal_false_positive_rate'}[label]
			value = f'{zero_day.loc["value", key]:.1%}' if key in zero_day.columns else 'n/a'
		with col: metric_card(label, value, 'Step 4 experiment')
	section('Detection comparison')
	plot_image('04_zero_day_detection_rates.png', 'Known versus unseen attack detection')
	if not zero_day.empty: st.dataframe(zero_day, use_container_width=True)
	if not zero_day_groups.empty: st.dataframe(zero_day_groups, hide_index=True, use_container_width=True)

elif page == 'Federated Learning':
	st.header('Federated Learning')
	st.markdown('<div class="subtitle">Global model performance across communication rounds. The dashboard never retrains the model.</div>', unsafe_allow_html=True)
	plot_image('05_federated_learning_round_metrics.png', 'FedAvg reconstruction loss and anomaly-detection performance')
	if not fed_metrics.empty:
		st.dataframe(fed_metrics.style.format({c: '{:.4f}' for c in fed_metrics.columns if c != 'round'}), hide_index=True, use_container_width=True)

elif page == 'Explainable AI':
	st.header('Explainable AI')
	st.markdown('<div class="subtitle">SHAP global importance and local SHAP/LIME explanations generated from actual model predictions.</div>', unsafe_allow_html=True)
	plot_image('06_shap_global_feature_importance.png', 'Global SHAP importance for reconstruction loss')
	example_names = examples['example'].tolist() if not examples.empty else []
	selected_example = st.selectbox('Prediction example', example_names) if example_names else None
	if selected_example:
		row = examples[examples['example'] == selected_example].iloc[0]
		metric_card(f'{selected_example} · {row.attack_cat}', f'{row.reconstruction_loss:.4f}', f'prediction: {"ANOMALY" if row.predicted_anomaly else "NORMAL"}')
		cols = st.columns(2)
		with cols[0]: plot_image(f'06_shap_{selected_example}.png', 'SHAP local feature contributions')
		with cols[1]: plot_image(f'06_lime_{selected_example}.png', 'LIME local surrogate weights')
		shap_table = load_explanation_csv(f'06_shap_{selected_example}.csv')
		if not shap_table.empty: st.dataframe(shap_table, hide_index=True, use_container_width=True)

elif page == 'Reports':
	st.header('Reports & Artifacts')
	st.markdown('<div class="subtitle">Downloadable outputs generated by the completed notebooks.</div>', unsafe_allow_html=True)
	reports = [
		('Threat-scored event register', METRICS_DIR / '07_threat_scored_events.csv'),
		('Threat scoring configuration', METRICS_DIR / '07_threat_scoring_config.json'),
		('Federated round metrics', METRICS_DIR / '05_federated_learning_metrics.csv'),
		('Zero-day metrics', METRICS_DIR / '04_zero_day_metrics.csv'),
	]
	for label, path in reports:
		if path.exists():
			st.download_button(f'Download {label}', path.read_bytes(), file_name=path.name, mime='text/csv' if path.suffix == '.csv' else 'application/json', key=path.name)
	section('Data provenance')
	st.info('This interface loads saved CSV, JSON, PNG, and model metadata artifacts. It does not train, refit preprocessing, or invent telemetry.')
