import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so pages can `from app import core`

from app import core  # noqa: E402


@st.cache_resource
def _load():
    return core.load_artifacts()


def get_artifacts() -> dict:
    try:
        return _load()
    except core.ArtifactVersionError as e:
        st.error(str(e))
        st.stop()


def setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} · Team 8 Bike Demand", page_icon="🚲", layout="wide")
    st.title(title)
