import os
import streamlit as st

# ✅ 배포/로컬 공통: secrets 우선, 없으면 OS env 사용
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "DEEPSEEK_API_KEY" in st.secrets:
    os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
os.environ["DEEPSEEK_BASE_URL"] = st.secrets.get(
    "DEEPSEEK_BASE_URL",
    os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

import rag_core
from rag_core import Chunk
from rag_core import load_vectorstore, rag_qa
from rag_core import answer_stream


# =========================
# Streamlit page config
# =========================
st.set_page_config(
    page_title="백남준 챗봇",
    page_icon="🧠",
    layout="centered",
)

# =========================
# Minimal ChatGPT-like CSS
# =========================
st.markdown("""
<style>
/* Chat container width */
.block-container { max-width: 860px; }

/* Make chat a bit tighter */
.stChatMessage { padding: 8px 0; }

/* Code blocks */
pre { border-radius: 12px !important; }

/* Sidebar titles */
.sidebar-title { font-weight: 700; font-size: 14px; margin-bottom: 8px; }

/* Smaller captions */
.small-caption { font-size: 12px; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)


# =========================
# Load VectorStore once
# =========================
@st.cache_resource(show_spinner=True)
def get_vs(vs_dir: str):
    return load_vectorstore(vs_dir)


# =========================
# Catalog / sources
# =========================
# ⚠️ 여기는 네 chunks metadata에 들어있는 source 이름과 맞아야 함.
# 보통 너가 생성할 때 넣었던 SOURCES["source"] 값들.
# =========================
# Catalog / sources + (A안) 파일별 한줄 설명
# =========================
ALL_SOURCES = [
    "백남준 연대기_백남준문화예술재단.xlsx",
    "백남준 퍼포먼스 목록.pdf",
    "백남준 참여전시목록_그룹전.pdf",
    "백남준 참여전시목록_개인전.pdf",
    "백남준 참고문헌목록_일반.pdf",
    "백남준 참고문헌목록_인터뷰.pdf",
    "백남준 참고문헌목록_기사.pdf",
    "백남준 작품목록_영화.pdf",
    "백남준 작품목록_단채널비디오.pdf",
    "백남준 작품목록_다큐멘터리 비디오.pdf",
    "The_Worlds_of_Nam_June_Paik_2000.pdf",
    "백남준 해외기사_NER,POS.xlsx",
    "백남준-아카이브전-전시-서문.pdf",
    "김금미_백남준기념관-강연-원고.pdf",
    "8.The-Mysteries-of-Encounters-between-Nam_June_Paik-John_Cage-and-Joseph_Beuys.pdf",
    "[백남준 작품 13선] 소장처 정리 (2).xlsx",
    "백남준 해외소장 현황 업데이트_2026ver.xlsx",
    "백남준 말에서 크리스토 (OCR PDF)",

]

# ✅ (A안 핵심) 파일별 "무슨 내용인지" 한 줄 설명
# - 라우터(DeepSeek)가 파일명을 보고 추측하는 대신, 설명까지 보고 더 정확히 sources를 고릅니다.
SOURCE_DESCRIPTIONS = {
    "백남준 연대기_백남준문화예술재단.xlsx": "백남준 생애/활동 연표(연도별 주요 사건, 전시/활동 맥락)",
    "백남준 퍼포먼스 목록.pdf": "퍼포먼스/행위 관련 목록(작품/행사명, 시기, 장소 등)",
    "백남준 참여전시목록_그룹전.pdf": "그룹전 참여 전시 목록(전시명, 기간, 장소 등)",
    "백남준 참여전시목록_개인전.pdf": "개인전 전시 목록(전시명, 기간, 장소 등)",
    "백남준 참고문헌목록_일반.pdf": "참고문헌 목록(일반 도서/자료 서지정보)",
    "백남준 참고문헌목록_인터뷰.pdf": "참고문헌 중 인터뷰 자료 목록(인터뷰이/매체/연도 등)",
    "백남준 참고문헌목록_기사.pdf": "참고문헌 중 기사/보도 자료 목록(신문/잡지/날짜 등)",
    "백남준 작품목록_영화.pdf": "작품 목록(영화 관련) – 작품명/제작연도/형식 등",
    "백남준 작품목록_단채널비디오.pdf": "작품 목록(단채널 비디오) – 작품명/연도/형식 등",
    "백남준 작품목록_다큐멘터리 비디오.pdf": "작품 목록(다큐멘터리 비디오) – 작품명/연도 등",
    "The_Worlds_of_Nam_June_Paik_2000.pdf": "전시/도록 성격의 자료(작품/에세이/전시 맥락 포함 가능)",
    "백남준 해외기사_NER,POS.xlsx": "해외 기사 텍스트/분석 결과(인물/지명/키워드, 문장 단위 정보 가능)",
    "백남준-아카이브전-전시-서문.pdf": "아카이브 전시 서문/기획 글(전시 의도/해석/맥락)",
    "김금미_백남준기념관-강연-원고.pdf": "강연 원고(해설/비평/맥락 설명 중심)",
    "8.The-Mysteries-of-Encounters-between-Nam_June_Paik-John_Cage-and-Joseph_Beuys.pdf": "논문/에세이(백남준-케이지-보이스 관계/해석 중심)",
    "[백남준 작품 13선] 소장처 정리 (2).xlsx": "백남준 작품 13선 작품별(시트별) 버전 현황/소장처/관련 정보 정리",
    "백남준 해외소장 현황 업데이트_2026ver.xlsx": "백남준 해외 소장 현황 업데이트(2026ver) - 소장처/작품 정보(첫 시트만 사용)",
    "백남준 말에서 크리스토 (OCR PDF)": "백남준 일생 일화들 소개, 생전 백남준의 말 수록 (페이지 기반 텍스트, 인용/발언/에세이 수록)"

}

DEFAULT_SOURCES = [
    "백남준 연대기_백남준문화예술재단.xlsx",
    "백남준 해외기사_NER,POS.xlsx",
    "The_Worlds_of_Nam_June_Paik_2000.pdf",
    "백남준 말에서 크리스토 (OCR PDF)", 
]

# ✅ 라우터에게 "파일명 + 한줄설명" 카탈로그를 보여줌
# 형식 예:
# - 파일명 :: 설명
CATALOG_TEXT = "\n".join([
    f"- {s} :: {SOURCE_DESCRIPTIONS.get(s, '설명 없음')}"
    for s in ALL_SOURCES
])


# Fixed settings (슬라이더 제거 + 값 고정)
# =========================
vs_dir = os.getenv("PAIK_VS_DIR", "data/paik_vs")

TOP_N_EVIDENCE = 8        # 근거 개수 고정
SCORE_THRESHOLD = 0.22    # 근거 충분성 임계값 고정


# =========================
# Load VS (cached)
# =========================
try:
    vs = get_vs(vs_dir)
except Exception as e:
    st.error(f"VectorStore 로드 실패: {e}")
    st.stop()


# =========================
# Session State: messages
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요. 백남준 챗봇입니다.\n\n백남준의 작업과 생각에 대해 무엇이든 물어보세요."}
    ]

# =========================
# Render chat history
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# =========================
# Chat input
# =========================
user_q = st.chat_input("백남준에 대해 질문해보세요…")

if user_q:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # ✅ 최근 5턴(=user+assistant 10개 메시지)만 history로 전달
    recent_history = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")]
    recent_history = recent_history[-10:]  # 5턴 = 최대 10개 메시지

    # generate answer (streaming)
    with st.chat_message("assistant"):
        # 1) retrieval 먼저 (이건 스트리밍 아님)
        with st.spinner("검색 중…"):
            out = rag_qa(
                vs=vs,
                question=user_q,
                catalog_text=CATALOG_TEXT,
                all_sources=ALL_SOURCES,
                default_sources=DEFAULT_SOURCES,
                history=recent_history,          # ✅ (현재는 선택, 그래도 넘겨둠)
                use_rerank= True,
                top_n_evidence=TOP_N_EVIDENCE,
                score_threshold=SCORE_THRESHOLD,
            )

        # 2) 이제 LLM 답변을 streaming으로 생성
        placeholder = st.empty()
        full_text = ""

        for token in rag_core.answer_stream(
            question=user_q,
            evidence=out["evidence_chunks"],
            mode=out["mode"],
            history=recent_history,             # ✅ 멀티턴 핵심: 여기로 전달
        ):
            full_text += token
            placeholder.markdown(full_text + "▌")

        # 3) 스트리밍 완료 후: 답변에는 출처를 붙이지 않음(상세보기에서만 노출)
        final_text = full_text.strip()
        placeholder.markdown(final_text)


        # expandable debug info
        with st.expander("🔎 검색/라우팅 상세 보기"):
            st.write("mode:", out.get("mode"))
            st.write("top_score:", out.get("top_score"))
            st.json(out["route"])
            st.write("evidence:")
            for i, ev in enumerate(out["evidence"], 1):
                st.markdown(f"**[{i}] {ev['cite']}**")
                st.write(ev["text"])
            st.divider()
            st.write("출처:")
            if out.get("mode") == "grounded" and out.get("cite_lines"):
                for line in out["cite_lines"]:
                    st.markdown(f"- {line}")
            else:
                st.markdown("- (현재 RAG 데이터에서 직접 근거를 찾지 못했습니다)")


    # store assistant message
    st.session_state.messages.append({"role": "assistant", "content": final_text})

