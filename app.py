import streamlit as st
from rag_graph import app, programmes, programme_map

st.set_page_config(
    page_title="TCET College Assistant",
    page_icon="🎓",
    layout="centered",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    .main .block-container { padding-top: 2rem; }
    div[data-testid="stSidebar"] { background-color: #0f1116; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar: programme selection ----------
with st.sidebar:
    st.title("🎓 TCET Assistant")
    st.caption("Ask about academics, fees, or placements.")

    selected_label = st.selectbox("Select your programme", programmes, index=0)
    selected_number = selected_label.split(".")[0].strip()
    student_programme = programme_map.get(selected_number, "B.E. Computer Engineering")

    st.success(f"Programme: **{student_programme}**")

    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.title("TCET College Assistant")
st.caption(f"Chatting as a **{student_programme}** student")

# ---------- Chat state ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "human"/"ai", "content": str}

# Render existing history
for msg in st.session_state.chat_history:
    role = "user" if msg["role"] == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# ---------- Chat input ----------
user_query = st.chat_input("Ask something about attendance, fees, placements...")

if user_query:
    st.session_state.chat_history.append({"role": "human", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Rebuild the LangGraph "messages" list from chat history so the
            # graph sees the full conversation, exactly like the CLI version did.
            langgraph_messages = [
                (m["role"], m["content"]) for m in st.session_state.chat_history
            ]

            result = app.invoke({
                "programme": student_programme,
                "messages": langgraph_messages,
            })

            ai_reply = result["messages"][-1].content
            st.markdown(ai_reply)

    st.session_state.chat_history.append({"role": "ai", "content": ai_reply})