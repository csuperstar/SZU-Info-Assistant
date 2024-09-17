import json
import random
import re
from tqdm import tqdm
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


def read_and_split_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 按 '\t\n\t' 切分内容
    split_content = content.split('\t\n\t')
    
    return split_content

# 基于加权概率选取块的数量
def select_random_blocks(blocks):
    # 权重定义：1块的权重最大，依次减小
    weights = [0.6, 0.2, 0.15, 0.05]  # 1块、2块、3块、4块的概率权重
    num_blocks = random.choices([1, 2, 3, 4], weights)[0]  # 加权选择数量
    selected_blocks = random.sample(blocks, num_blocks)  # 随机选取指定数量的块
    return selected_blocks

# 拼接文本块并标号
def concatenate_blocks(blocks):
    concatenated_string = ''
    
    # 遍历每个块并按顺序编号
    for i, block in enumerate(blocks, start=1):
        concatenated_string += f"{i}. {block}\n\n"  # 加编号和块内容，每块后加换行
    
    return concatenated_string


def replace_data_values(new_id, knowledge_base, new_user_value, new_assistant_value):
    data_prompt = {
        "id": "",
        "conversations": [
            {
                "from": "user",
                "value": "" 
            },
            {
                "from": "assistant",
                "value": "" 
            }
        ]
    }
    
    # 替换id
    data_prompt["id"] = "identity_" + str(new_id)
    # 替换user和assistant的value
    data_prompt["conversations"][0]["value"] = "作为深圳大学的信息助手，你的回答必须基于以下知识库中的内容，且只能回答与深圳大学相关的问题。如果问题不在你的知识范围内或超出范围，请回答：“抱歉，我暂时回答不了这个问题，请提供更详细的信息。”\n 知识库：\n {} \n 问题：{} \n 回答：".format(knowledge_base, new_user_value)
    data_prompt["conversations"][1]["value"] = new_assistant_value
    
    return data_prompt
    

def loadLlmChain(chat_model, prompt):
    chat_model = ChatOllama(model=chat_model, main_gpu=1, temperature=0)
    llm_chain = prompt | chat_model | StrOutputParser()
    
    return llm_chain


if __name__ == "__main__":
    file_path = "data/rag/szu_data.txt"
    split_data = read_and_split_txt(file_path)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["content"],
        template="""请根据以下提供的深圳大学相关数据块，生成 4 个问题并给出中文答案，使用json格式输出('question': '', 'answer:': '')：
1. 与信息库相关的 2 个问题：这些问题必须基于文档块中的内容，并提供基于文档内容的准确答案。
2. 与信息库无关的 2 个问题：这些问题是文档中没有提到的，请回答：“抱歉，我暂时回答不了这个问题，请提供更详细的信息。”
    
信息内容：
{content}
""")
    llm_chain = loadLlmChain(chat_model="qwen2:72b", prompt=QUERY_PROMPT)
    
    all_id = 0
    valid_epoch = 1000
    training_data = []
    output_file = 'data/finetune/szu_finetune_data.json'
    while valid_epoch > 0:
        selected_blocks = select_random_blocks(split_data)
        result_string = concatenate_blocks(selected_blocks)
        
        response = llm_chain.invoke(result_string)
        
        try:
            cleaned_string = response.replace('```json', '').replace('```', '')
            qa_data = json.loads(cleaned_string)
            for i, qa in enumerate(qa_data, start=1):
                data_value = replace_data_values(all_id, result_string, qa.get('question'), qa.get('answer'))
                all_id += 1
                training_data.append(data_value)
            
            valid_epoch -= 1
            print("valid_epoch: \n", valid_epoch)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=4)
                f.close()
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}\n")

