import json
import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from typing import List
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.retrievers.multi_query import MultiQueryRetriever


class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        # print("lines: ", len(lines))
        # print("lines: ", lines)
        return list(filter(None, lines))  # Remove empty lines


class SZUChat():
    def __init__(self, file_path, promptTemplate, embed_model_name, chat_model, parent_split=True):
        self.file_path = file_path
        self.prompt = ChatPromptTemplate.from_messages(promptTemplate)
        self.embed_model_name = embed_model_name
        self.chat_model = chat_model
        self.parent_split = parent_split

        self.text = None
        self.split_text = []
        self.parent_splitter = None
        self.child_splitter = None
        self.llm_chain = None
        self.retriever = None
        self.mutil_retriever = None

    def mutilQueryRetriver(self, search_k=1):
        output_parser = LineListOutputParser()
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""您是一个AI语言模型助手。您的任务是生成用户提出的问题的三个不同版本，以便从向量数据库中检索相关文档。通过对用户问题的多个视角进行生成，您的目标是帮助用户克服基于距离的相似性搜索的一些局限性。请将这些替代问题用换行符分隔开。\n原始问题：{question}""",
        )
        llm = ChatOllama(model=self.chat_model, temperature=0, main_gpu=0)
        llm_chain = QUERY_PROMPT | llm | output_parser

        self.textDBEmbedding(search_k=1)
        self.mutil_retriever = MultiQueryRetriever(retriever=self.retriever, llm_chain=llm_chain, include_original=False)

    def mutilQueryQuestion(self, question):
        unique_docs = self.mutil_retriever.invoke(question)
        print("问题扩展数量:", len(unique_docs))
        print("扩展问题: ", unique_docs)
        return unique_docs

    def loadDataFromTxt(self):
        loader = TextLoader(self.file_path, encoding='utf-8')
        text = loader.load()
        return text


    def splitText(self):
        text = self.loadDataFromTxt()
        parent_splitter = CharacterTextSplitter(
            separator="\t\n\t",
            chunk_size=0,
            chunk_overlap=0,
            length_function=len,
        )
        self.text = text
        self.parent_splitter = parent_splitter

        if self.parent_split:
            child_splitter = RecursiveCharacterTextSplitter(
                separators=["\n", "。", "；", "!"],
                chunk_size=20,
                chunk_overlap=0,
                length_function=len,
            )
            self.child_splitter = child_splitter


    def textDBEmbedding(self, search_k=1):
        embed_model_kwargs = {'device': 'cuda'}
        embed_encode_kwargs = {'batch_size': 64, 'normalize_embeddings': True}

        embed_model = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            model_kwargs=embed_model_kwargs,
            encode_kwargs=embed_encode_kwargs
        )

        if not self.parent_split:
            self.split_text = self.parent_splitter.split_documents(self.text)
            db_faiss = FAISS.from_documents(documents=self.split_text, embedding=embed_model)
            retriever = db_faiss.as_retriever(search_kwargs={'k': search_k})
        else:
            vectorstore = FAISS.from_texts(["initialize faiss"], embedding=embed_model)
            store = InMemoryStore()
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
                search_kwargs={'k': search_k}
            )
            retriever.add_documents(self.text)

        self.retriever = retriever

    def retrieveDBText(self, question):
        results = self.retriever.invoke(question)
        return results


    def loadLlmChain(self):
        chat_model = ChatOllama(model=self.chat_model, main_gpu=0)
        llm_chain = (
                {"content": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt
                | chat_model
                | StrOutputParser()
        )
        self.llm_chain = llm_chain

    def chatOnline(self, question, history=None):
        _content = ""
        content = self.retrieveDBText(question)
        for ct in content:
            _content += ct.page_content + '\n\n'
            print(ct.page_content)

        return self.llm_chain.invoke({"content": _content, "question": question})

    def mutilQueryChatOnline(self, question, history=None):
        _content = ""
        content = self.mutilQueryQuestion(question)
        for ct in content:
            _content += ct.page_content + '\n\n'
        print("检索文档：", _content)

        return self.llm_chain.invoke({"content": _content, "question": question})


if __name__ == "__main__":
    file_path = "data/rag/szu_data.txt"
    promptTemplate = [
      ("system", "你是深圳大学的信息AI助手，熟悉学校概况、院系机构以及计算机与软件学院的师资队伍信息，你需要根据下面提供的知识库信息用中文直接回答问题。知识库中与问题无关的信息，你需要忽略。知识库中若没有相关信息，你只能说'很抱歉，我不知道。'。不管信息是否存在，你都不能提及知识库。\n 知识库内容：\n {content} \n"),
      ("human", "你好！"),
      ("ai", "你好！"),
      ("human", "{question}"),
    ]
    embed_model_name = 'embedding_model/bce-embedding-base_v1'
    chat_model = "qwen2:7b"

    szuChat = SZUChat(file_path=file_path, promptTemplate=promptTemplate, embed_model_name=embed_model_name, chat_model=chat_model, parent_split=True)
    szuChat.splitText()

    szuChat.mutilQueryRetriver()
    szuChat.loadLlmChain()
    chatUI = gr.ChatInterface(fn=szuChat.mutilQueryChatOnline, title="深圳大学信息助手", description="")
    chatUI.launch()

