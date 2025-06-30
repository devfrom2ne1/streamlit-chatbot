import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response


st.set_page_config(page_title="튜닝마법사 챗봇", page_icon="🧙🏻")


st.title("🧙🏻 튜닝마법사 챗봇")
st.caption("튜닝마법사 챗봇은 Oracle SQL 튜닝에 대한 질문에 답변합니다.")


load_dotenv()


if 'message_list' not in st.session_state:
    st.session_state.message_list = []


for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input("튜닝마법사에게 튜닝할 SQL을 알려주세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})