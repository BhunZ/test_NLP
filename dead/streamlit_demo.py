from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from backend.rag_service import RagService


def _split_by_citations(
    text: str,
) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    pat = re.compile(r"\[(\d{1,3})\]")
    last = 0
    for m in pat.finditer(text):
        if m.start() > last:
            parts.append({"type": "text", "value": text[last : m.start()]})
        parts.append({"type": "cite", "n": int(m.group(1))})
        last = m.end()
    if last < len(text):
        parts.append({"type": "text", "value": text[last:]})
    return parts


def _render_answer_with_links(text: str) -> None:
    # Turn [n] into in-page links to #source-n
    parts = _split_by_citations(text)
    out = []
    for p in parts:
        if p["type"] == "text":
            out.append(p["value"])
        else:
            n = p["n"]
            out.append(f'<a href="#source-{n}" style="text-decoration:none;">'
                       f'<span style="display:inline-block;padding:0 6px;'
                       f'border:1px solid rgba(139,92,246,.35);'
                       f'border-radius:6px;background:rgba(139,92,246,.14);'
                       f'font-size:12px;font-weight:700;vertical-align:2px;'
                       f'color:#c4b5fd;">[{n}]</span></a>')
    st.markdown("".join(out), unsafe_allow_html=True)


def _youtube_iframe(video_id: str, start_seconds: int) -> str:
    start_seconds = max(0, int(start_seconds))
    return f"""
<div style="position:relative;padding-top:56.25%;">
  <iframe
    src="https://www.youtube.com/embed/{video_id}?start={start_seconds}&autoplay=0"
    style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen
    title="YouTube player"
  ></iframe>
</div>
"""


@st.cache_resource(show_spinner=False)
def _svc() -> RagService:
    # Loads indexes lazily on first ask. Cache ensures one instance per Streamlit process.
    return RagService()


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict]


st.set_page_config(
    page_title="Stanford NLP Tutor",
    page_icon="📚",
    layout="centered",
)

_init_state()

st.title("📚 Stanford NLP Tutor")
st.caption("Hoi ve NLP bang tieng Viet, tra loi co trich dan timestamp YouTube.")

with st.sidebar:
    st.subheader("Cau hinh")
    provider = st.selectbox("LLM provider", ["groq", "mistral"], index=0)
    top_k = st.slider("Top-K sources", min_value=1, max_value=12, value=6, step=1)
    rerank = st.checkbox("Rerank (slower, slightly better)", value=False)
    enable_rewrite = st.checkbox("Enable rewrite (requires Ollama)", value=False)
    course_filter = st.text_input("Course filter (exact match)", value="")

    st.divider()
    st.caption("Backend: in-process via `backend/rag_service.py` (no FastAPI needed).")


EXAMPLES = [
    ("Co che attention", "Co che attention hoat dong ra sao?"),
    ("BERT vs GPT", "BERT vs GPT khac nhau o dau?"),
    ("Positional encoding", "Tai sao can positional encoding?"),
    ("RLHF la gi", "RLHF la gi?"),
]

if not st.session_state.messages:
    st.markdown("### Goi y")
    cols = st.columns(2)
    for i, (title, q) in enumerate(EXAMPLES):
        with cols[i % 2]:
            if st.button(f"{title}", use_container_width=True):
                st.session_state._prefill = q


def _render_sources(sources: List[Dict[str, Any]]) -> None:
    st.markdown("### Nguon")
    for i, s in enumerate(sources, 1):
        # Anchor so [n] can link-scroll here.
        st.markdown(f'<div id="source-{i}"></div>', unsafe_allow_html=True)

        with st.container(border=True):
            c1, c2 = st.columns([1, 3], vertical_alignment="top")
            with c1:
                st.image(s.get("thumbnail_url"), use_container_width=True)
                st.caption(f"[{i}] {s.get('timestamp_display')}")
            with c2:
                title = s.get("video_title") or s.get("video_id") or "Unknown"
                st.markdown(f"**{title}**")
                meta = []
                if s.get("course"):
                    meta.append(str(s["course"]))
                if s.get("score") is not None:
                    meta.append(f"score={s['score']:.3f}")
                if s.get("retrieved_by"):
                    meta.append("by=" + ",".join(s["retrieved_by"]))
                if meta:
                    st.caption(" • ".join(meta))

                snippet = (s.get("text_en") or "").strip().replace("\n", " ")
                if snippet:
                    st.write(f"“{snippet[:220]}...”")

                btn_cols = st.columns([1, 1, 4])
                with btn_cols[0]:
                    if st.button("Mo video", key=f"open-{s.get('chunk_id')}-{i}"):
                        st.session_state._open_source = s
                with btn_cols[1]:
                    st.link_button("YouTube", s.get("youtube_url") or "#")


# Render chat history
for m in st.session_state.messages:
    role = m["role"]
    with st.chat_message(role):
        if role == "assistant":
            _render_answer_with_links(m["text"])
            if m.get("meta"):
                st.caption(
                    f"⏱ {m['meta'].get('total_latency_ms', 0)}ms • "
                    f"sources={len(m.get('sources') or [])}"
                )
            if m.get("sources"):
                _render_sources(m["sources"])
        else:
            st.write(m["text"])


prefill = st.session_state.pop("_prefill", "")
prompt = st.chat_input("Hoi gi do ve NLP...", key="chat_input")
if prefill and not prompt:
    # Streamlit doesn't support programmatic set of chat_input; we just show it.
    st.info(f"Da chon: {prefill}")


if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Dang tim nguon va viet cau tra loi..."):
            res = _svc().ask(
                query=prompt,
                top_k=int(top_k),
                rerank=bool(rerank),
                llm_provider=str(provider),
                enable_rewrite=bool(enable_rewrite),
                course_filter=(course_filter.strip() or None),
            )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": res["answer"]["text_vi"],
                "sources": res["sources"],
                "meta": res["meta"],
            }
        )

        _render_answer_with_links(res["answer"]["text_vi"])
        st.caption(f"⏱ {res['meta'].get('total_latency_ms', 0)}ms • sources={len(res['sources'])}")
        _render_sources(res["sources"])


# "Modal" video panel (simple dialog-like section)
open_source: Optional[Dict[str, Any]] = st.session_state.pop("_open_source", None)
if open_source:
    st.divider()
    st.subheader("Player")
    st.write(f"{open_source.get('video_title') or open_source.get('video_id')} ({open_source.get('timestamp_display')})")
    components.html(
        _youtube_iframe(str(open_source.get("video_id")), int(open_source.get("start_seconds") or 0)),
        height=420,
    )
    st.link_button("Open on YouTube", open_source.get("youtube_url") or "#")

