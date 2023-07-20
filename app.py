import streamlit as st
import openai
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import sys, re

# 페이지 레이아웃을 wide로 지정
st.set_page_config(layout="wide")

# CSS 적용 (버튼에 전체 적용)
st.markdown("""
    <style>
    .stButton>button {
        color: white;
        background-color: #FF6347;
        float:right;
    }
    </style>
    """, unsafe_allow_html=True)

#st.title("한화생명 ChatGPT CPSP 검사기") 을 써도 되지만, Css 사용을 위해 markdown을 주입 함
st.markdown("<h1 style='text-align: center; color: black;'>한화생명 ChatGPT CPSP 검사기</h1>", unsafe_allow_html=True)

st.text("\n")
st.text("\n")
st.text("\n")
# Create two columns
col1, col2 = st.columns(2)

# Use the columns like normal st calls
col1.subheader("[ 데이터 입력 ]")
col1.markdown("##### 1. CPSP 검사를 위한 한화생명 룰셋을 업로드하세요. (샘플 : java)")

# 1) get_excel_chunks 메소드
# 설명 : to_string으로 변환된 엑셀 데이터를 chunks 로 쪼갬
def get_excel_chunks(data_st):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2048,        # 언어모델이 한번에 처리할 수 있는 토큰 수 : 보통 1024, 2048 등으로 처리. 무작정 크게하면 메모리 오버플로우나, 성능 저하 초래 가능성 있음
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(data_st)
    return chunks

# 2) get_vectorstore 메소드
# 설명 : excel chunks를 벡터화해서 FAISS라는 검색 라이브러리에 저장함 (쉽게 말해, 쪼갠 chunks 들을 vector store 에 저장한다고 보면 됌)
def get_vectorstore(excel_text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["api_key"])
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=excel_text_chunks, embedding=embeddings)
    return vectorstore

# 3) get_conversation_chain 메소드 (핵심 로직, gpt-4 사용)
# 설명 : 사용자와 gpt 챗봇 간의 ConversationalRetrievalChain 을 생성함
#       이 대화체인을 통해, 사용자의 질문에 대한 적절한 응답을 생성하기 위해 언어모델, 메모리, 벡터 검색기를 사용함
# 참고 : Retriever는 사용자의 질문이나 주제에 대해 미리 학습된 데이터에서 가장 관련이 있는 정보를 찾아주는 역할을 함. 이걸 쓰기 위해서 벡터db에 저장해야 함
def get_conversation_chain(vectorstore):
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = ChatOpenAI(model="gpt-4",openai_api_key=st.secrets["api_key"],streaming=True, callbacks=callbacks)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),       
        memory=memory,
    )
    return conversation_chain

# 4) run 메소드 (실제 최초 동작부)
def run():
    uploaded_file = col1.file_uploader("Choose a Excel file", type="xlsx")
    if uploaded_file is not None:
            # 엑셀 파일을 pandas dataframe으로 변환
            data = pd.read_excel(uploaded_file)
            
            # 데이터 출력
            with col1.expander("적용된 룰셋을 상세히 보기"):
                
                # expander에서 검색어 입력 받기
                keyword = st.text_input("검색어를 입력하세요")

                if keyword:  # 검색어가 있는 경우
                    # 데이터프레임에서 검색어를 포함하는 행 필터링
                    # 아래는 전체 룰셋에서 검색할 때 쓰는 거고, 이번에는 RULE_NAME 에만 적용
                    # search_result = data[data.apply(lambda row: keyword.lower() in row.astype(str).values, axis=1)]
                    search_result = data[data['RULE_NAME'].astype(str).str.lower().str.contains(keyword.lower())] # RULE_NAME 에만 적용
                    st.write(search_result)
                else:  # 검색어가 없는 경우
                    st.write(data)        

                # 엑셀데이터를 to_string으로 바꿔서, chunks로 쪼개기 위해 get_excel_chunks 메소드 호출
                data_st = data.to_string()
                excel_text_chunks = get_excel_chunks(data_st)

                # chunks 단위로 쪼갠 엑셀데이터를 벡터스토어에 저장
                vectorstore = get_vectorstore(excel_text_chunks)

                # 저장한 벡터스토어를 기반으로 대화체인(conversation chain)을 생성함
                st.session_state.conversation = get_conversation_chain(vectorstore)
    else:
        col1.write("파일을 업로드 해주세요.")

if __name__ == '__main__':
    run()

col1.text("\n")
col1.text("\n")
col1.text("\n")
col1.markdown("##### 2. CPSP 검사를 위한 소스코드를 입력하세요.")
col2.subheader("[ 결과 출력 ]")

# 검사를 위한 소스코드 입력 부분 (사용자 소스코드 입력 영역)
user_input = col1.text_area("Please enter your text here")

# 5) handle_userinput 메소드
# 설명 : 사용자가 입력한 소스코드를 라인단위로 읽어 string 연산 한 결과를 가져다가, 위에서 구현한 대화체인에게 질의함
#       질의한 결과를 st.session_state.displayed_chat_history에 append 함
def handle_userinput(check_datas):
    response = st.session_state.conversation({'question': check_datas})
    st.session_state.chat_history = response['chat_history']
    
    # process 버튼을 누를 때마다 결과출력 부분을 비우기 위함
    st.session_state.displayed_chat_history = []

    # prefix로 지정한 질의는 chat_history에서 짝수 인덱스에 위치하므로, 홀수 인덱스(gpt 응답결과만)만 보여주기 위해 i%2 != 0을 씀
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 != 0:
            st.session_state.displayed_chat_history.append(message.content)


if col1.button("검사시작", key="button"):
    with st.spinner("검사 중입니다..."):
        # 사용자가 입력한 input value를 check_data에 세팅
        check_data = user_input
        check_datas = ''
        line_all = ''
        lines = check_data.splitlines()     # 문자열을 라인별로 나누어 리스트로 반환
        for line in lines:
            line_all = line_all + line      # 개선필요) 라인단위로 string 연산
            print('line: '+line_all)

        # 별도 질의를 사용자가 입력하지 않기 위함
        check_datas = line_all+ '\n 다음 코드가 엑셀 업로드 한 규칙 중에 위배되는 항목이 있는지 찾아주고, 내용과 올바른 코드를 알려줘'

        # 사용자가 입력한 코드를 라인단위로 읽어, string 연산을 한 결과를 파라미터로 넘김
        handle_userinput(check_datas)
        st.session_state.previous_question = line_all

        clearer = re.compile('<.*?>')

        # st.session_state의 displayed_chat_history에 질의한 결과가 담겨 있으면, clear 한 다음 col2.code에 java lang으로 출력 함
        if 'displayed_chat_history' in st.session_state:
            for message in st.session_state.displayed_chat_history:
                rmT = re.sub(clearer, '', message)
                col2.code(rmT, language='java')

        # 이전 질의에 대한 flush
        if 'previous_question' not in st.session_state:
            st.session_state.previous_question = ""
