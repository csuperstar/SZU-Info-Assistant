from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

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

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import jieba


def cut_words(text):
    """
        利用jieba库进行文本分词
    """
    return jieba.lcut(text)

class SZUAI():
    def __init__(self, data_file, embed_model_name, sft_finetune_model):
        self.file_path = data_file
        self.embed_model_name = embed_model_name
        self.sft_finetune_model = sft_finetune_model
        self.chat_model = None
        self.ensemble_retriever = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_finetune_model, trust_remote_code=True)
        self.loadRetriever()
        self.loadLLM()
    
    def loadTextDataFromTxt(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            f.close()
        
        text = text.split('\t\n\t')
        return text
        
    def loadDocDataFromTxt(self):
        loader = TextLoader(self.file_path, encoding='utf-8')
        text = loader.load()
        return text
    
    def loadTextSplitter(self):
        parent_splitter = CharacterTextSplitter(
            separator="\t\n\t",
            chunk_size=0,
            chunk_overlap=0,
            length_function=len,
        )

        child_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", "。", "；", "!"],
            chunk_size=20,
            chunk_overlap=0,
            length_function=len,
        )
        return parent_splitter, child_splitter
    
    def loadRetriever(self, search_k=1):
        text = self.loadTextDataFromTxt()
        parent_splitter, child_splitter = self.loadTextSplitter()
        
        # 关键词检索
        bm25_retriever = BM25Retriever.from_texts(text, preprocess_func=cut_words, k=search_k)
        
        # 向量检索
        embed_model_kwargs = {'device': 'cuda'}
        embed_encode_kwargs = {'batch_size': 64, 'normalize_embeddings': True}

        embed_model = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            model_kwargs=embed_model_kwargs,
            encode_kwargs=embed_encode_kwargs
        )

        
        vectorstore = FAISS.from_texts(["initialize faiss"], embedding=embed_model)
        store = InMemoryStore()
        cp_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={'k': search_k}
        )
        doc_text = self.loadDocDataFromTxt()
        cp_retriever.add_documents(doc_text)

        self.ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, cp_retriever], weights=[0.5, 0.5])
    
    def loadLLM(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.sft_finetune_model,
            device_map="auto",
            top_k = 1,
            top_p = 0,
            temperature = 0,
            trust_remote_code=True
        ).eval()
        
        self.chat_model = model
    
    def loadPrompt(self, question):
        docs_text = self.ensemble_retriever.invoke(question)
        
        concatenated_string = ''
        for i, block in enumerate(docs_text, start=1):
            concatenated_string += f"{i}. {block.page_content}\n\n"
        
        prompt = "作为深圳大学的信息助手，你的回答必须基于以下知识库中的内容，且只能回答与深圳大学相关的问题。如果问题不在你的知识范围内或超出范围，请回答：“抱歉，我暂时回答不了这个问题，请提供更详细的信息。”\n 知识库：\n {} \n 问题：{} \n 回答：".format(concatenated_string, question)
        return prompt
    
    def chatOnline(self, question, history=None):
        prompt = self.loadPrompt(question)
        print("prompt: {}\n".format(prompt))
        response, history = self.chat_model.chat(self.tokenizer, prompt, history=None)
        print("response: {}\n\n".format(response))
        return response

    

if __name__ == "__main__":
    data_file = "data/rag/szu_data.txt"
    embed_model_name = "embedding_model/bce-embedding-base_v1"
    sft_finetune_model = "finetune_model/qwen/Qwen-1_8B-Chat-Int4"
    
    
    
    szu = SZUAI(data_file, embed_model_name, sft_finetune_model)
    chatUI = gr.ChatInterface(fn=szu.chatOnline, title="深圳大学信息助手", description="")
    chatUI.launch(share=True)