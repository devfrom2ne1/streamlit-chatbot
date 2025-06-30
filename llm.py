from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples

# 세션 기록 저장용 store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Pinecone 기반 벡터 검색기
def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'oracle-tuning-index'  # 너가 생성한 Pinecone index 이름으로 변경
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

# LLM 객체 생성
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm

# 과거 대화문맥 반영 retriever
def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

# RAG 체인 구성
def get_rag_chain():
    llm = get_llm()
    
    # 예시 대화 입력
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 Oracle SQL 성능 튜닝 전문가이며, 사용자의 SQL 쿼리 성능 문제를 분석하고 개선 방법을 제공합니다. "
        "아래 조건을 반드시 따라 답변하세요:\n\n"
        
        "1. 질문이 주어지면 성능 저하 원인을 분석해 설명합니다. (인덱스 미사용, 암묵적 형변환 등)\n"
        "2. Oracle 공식문서 기준으로 관련 내용을 설명하고, 필요시 문서 경로를 포함하세요.\n"
        "3. 개선된 쿼리(After)를 작성해줍니다.\n"
        "4. 추가로 튜닝 포인트나 인덱스 전략, 힌트 등을 제안합니다.\n\n"
        
        "응답은 다음 형식을 따릅니다:\n"
        "① 문제 원인 설명\n"
        "② Oracle 문서 인용 (있을 경우)\n"
        "③ 튜닝 후 쿼리 예시\n"
        "④ 실무 팁 (선택)\n\n"

        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain

# 최종 응답 생성 함수
def get_ai_response(user_message):
    rag_chain = get_rag_chain()
    ai_response = rag_chain.stream(
        {
            "input": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )
    return ai_response
