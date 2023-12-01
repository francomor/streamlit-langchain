import asyncio
import os
import pickle

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


def get_text_chunks(text, chunk_size, chunk_overlap):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_llm_chain(
    api_key,
    model,
    prompt_system_role,
    prompt_user,
    temperature,
    presence_penalty,
    frequency_penalty,
    max_tokens,
):
    system_message_prompt = SystemMessage(content=prompt_system_role)
    human_message_prompt = HumanMessagePromptTemplate.from_template(prompt_user)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        },
        verbose=True,
    )
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    return chain


def read_dict_from_disk(file_name):
    file_name = f"./save_files/{file_name}.pkl"
    if os.path.isfile(file_name):
        with open(file_name, "rb") as fp:
            d_values = pickle.load(fp)
    else:
        d_values = {}
    return d_values


def save_dict_on_disk(file_name, d_values):
    file_name = f"./save_files/{file_name}.pkl"
    with open(file_name, "wb") as fp:
        pickle.dump(d_values, fp, protocol=pickle.HIGHEST_PROTOCOL)


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
