from datetime import datetime
from typing import Tuple

import streamlit as st

from langchain.callbacks import get_openai_callback

from .shared import get_llm_chain, get_text_chunks, get_vectorstore, save_dict_on_disk

NUMBER_OF_LINE_BREAKS_EMBEDDING = 3


async def run_auto_cm(
    save_file_name,
    text_data,
    chunk_size,
    chunk_overlap,
    system_role_prompt,
    user_prompt,
    api_key,
    model_type,
    temperature,
    presence_penalty,
    frequency_penalty,
    max_tokens,
    embedding_query,
    number_of_docs
):
    vector_store = split_text_data_in_chunks(
        text_data,
        api_key,
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
    )
    summary_data = query_embedding(
        vector_store,
        embedding_query,
        number_of_docs,
    )
    summary, total_tokens, total_cost = await call_gpt(
        summary_data,
        api_key,
        model_type,
        system_role_prompt,
        user_prompt,
        temperature,
        presence_penalty,
        frequency_penalty,
        max_tokens,
    )
    save_dict_on_disk(
        save_file_name,
        {
            "gpt_model_index": 0 if model_type == "gpt-4" else 1,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "text_data": text_data,
            "data_chunk_size": chunk_size,
            "data_chunk_overlap": chunk_overlap,
            "embedding_query": embedding_query,
            "number_of_docs": number_of_docs,
            "system_role_prompt": system_role_prompt,
            "user_prompt": user_prompt,
        },
    )
    return (
        summary,
        total_tokens,
        total_cost,
    )


def split_text_data_in_chunks(
    text_data,
    open_ai_api_key: str,
    chunk_size=2500,
    chunk_overlap=200,
):
    st_write_time("Splitting the data in chunks...")
    text_chunks = get_text_chunks(text_data, chunk_size, chunk_overlap)
    if not text_chunks:
        raise Exception("Can't split the data in chunks.")
    return get_vectorstore(text_chunks, open_ai_api_key)


def query_embedding(
    vectorstore,
    embedding_query,
    number_of_docs_per_section=2,
):
    st_write_time("Querying the embedding...")
    docs = vectorstore.similarity_search(embedding_query, k=number_of_docs_per_section)
    if not docs:
        raise Exception("Can't find any section in the embeddings.")
    join_exp = "\n" * NUMBER_OF_LINE_BREAKS_EMBEDDING
    summary_data = join_exp.join([doc.page_content for doc in docs])
    return summary_data


async def call_gpt(
    summary_data,
    api_key,
    model_type,
    system_role_prompt,
    user_prompt,
    temperature,
    presence_penalty,
    frequency_penalty,
    max_tokens,
) -> Tuple[str, int, float]:
    st_write_time("Calling GPT...")
    llm_chain = get_llm_chain(
        api_key,
        model_type,
        system_role_prompt,
        user_prompt,
        temperature,
        presence_penalty,
        frequency_penalty,
        max_tokens,
    )
    with get_openai_callback() as cb:
        result = await llm_chain.agenerate([{"summary_data": summary_data}])
        generation = result.generations[0][0].text
        total_tokens = cb.total_tokens
        total_cost = cb.total_cost
    st_write_time("Calling GPT complete.")
    return generation, total_tokens, total_cost


def st_write_time(message: str):
    st.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
