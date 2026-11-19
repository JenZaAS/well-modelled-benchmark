"""Local Timestral gap validation and new-well inference dashboard."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from benchmark.run import load_dataset
from wmb.curves import CORE
from workstreams.red_foundation_models.contender import Timestral
from workstreams.red_foundation_models.dashboard_data import load_local, predict_tracks, validate


def plot_tracks(frame: pd.DataFrame, target: str) -> None:
    """Plot aligned log tracks with increasing measured depth downward."""
    fig = go.Figure()
    colors = {'Observed': '#35627a', 'Hidden truth': '#222222', 'P50': '#d65616',
              'P10': '#ad7951', 'P90': '#ad7951', 'Interpolation': '#668e47'}
    for label, color in colors.items():
        if label in frame and frame[label].notna().any():
            fig.add_scatter(x=frame[label], y=frame.Depth, name=label, mode='lines+markers',
                            marker={'size': 2}, connectgaps=False,
                            line={'color': color, 'dash': 'dot' if label in ('P10', 'P90') else 'solid'})
    fig.update_layout(height=650, xaxis_title=target, yaxis_title='Measured depth (m)',
                      yaxis_autorange='reversed', legend_orientation='h')
    st.plotly_chart(fig, width='stretch')


def benchmark_view(dataset: str, target: str) -> None:
    """Show validation inventory, scores, and selectable per-well tracks."""
    st.caption('Seed 0 · 20% validation split by well · zero demonstrations · no facies or weight updates')
    key = (dataset, target)
    if st.button('Run validation', type='primary'):
        with st.spinner('Scoring held-out gaps locally…'):
            train, test = load_dataset(dataset, 0)
            result = validate(Timestral(shots=0), train, test, target)
            st.session_state.validation = (key, result)
    stored = st.session_state.get('validation')
    if not stored or stored[0] != key:
        return
    result = stored[1]
    st.subheader('Dataset splits')
    st.dataframe(pd.DataFrame(result.inventory), hide_index=True)
    masked = sum(int(frame['Hidden truth'].notna().sum()) for frame in result.tracks.values())
    st.subheader('Validation gaps')
    st.write(f'{len(result.tracks)} wells with scorable gaps · {masked:,} deliberately masked samples')
    st.dataframe(pd.DataFrame(result.scores).T)
    st.caption('n = paired predictions. Coverage is a fraction; nominal one-sigma coverage is 0.683.')
    name = st.selectbox('Validation well', list(result.tracks))
    plot_tracks(result.tracks[name], target)


def inference_view(dataset: str, target: str) -> None:
    """Run inference on a local well, retaining observed target context."""
    st.caption('Use a local LAS or FORCE-format CSV. Files are read in place; nothing is uploaded.')
    path = st.text_input('Local file path')
    if st.button('Read wells'):
        wells = load_local(Path(path).expanduser())
        if not wells:
            raise ValueError('The file has no wells.')
        st.session_state.source = (path, wells)
        st.session_state.pop('inference', None)
    source = st.session_state.get('source')
    if not source or source[0] != path:
        return
    index = st.selectbox('Input well', range(len(source[1])), format_func=lambda i: source[1][i].name)
    well = source[1][index]
    key = (dataset, path, well.name, target)
    if st.button('Predict missing samples', type='primary'):
        with st.spinner('Running local inference…'):
            train, _ = load_dataset(dataset, 0)
            model = Timestral(shots=1)
            model.fit([w for w in train if w.name != well.name], target)
            st.session_state.inference = (key, predict_tracks(model, well, target))
    stored = st.session_state.get('inference')
    if stored and stored[0] == key:
        frame = stored[1]
        missing = int(frame.Observed.isna().sum())
        st.write(f'{frame.P50.notna().sum():,} of {missing:,} missing samples predicted. '
                 'No accuracy score: missing values have no ground truth.')
        plot_tracks(frame, target)


def main() -> None:
    """Render the local dashboard without loading a checkpoint on startup."""
    st.set_page_config(page_title='Timestral | Well logs', layout='wide')
    st.title('Timestral · Well logs')
    page = st.sidebar.radio('Workspace', ['Benchmark', 'New well'])
    dataset = st.sidebar.selectbox('Reference dataset', ['force2020', 'synthetic'])
    target = st.sidebar.selectbox('Target log', CORE, index=list(CORE).index('RHOB'))
    try:
        if page == 'Benchmark':
            benchmark_view(dataset, target)
        else:
            inference_view(dataset, target)
    except (OSError, ValueError, ImportError, KeyError, RuntimeError) as exc:
        st.error(f'Could not complete the local operation: {exc}')


if __name__ == '__main__':
    main()