import os
from datetime import datetime

import streamlit as st

from app.langchain import run_auto_cm
from app.prompts import system_prompt, user_prompt
from app.shared import get_or_create_eventloop, read_dict_from_disk


def summarize(save_file_name: str, page_title: str):
    if not st.session_state.get("authentication_status", None):
        st.error("Login required")
    else:
        st.set_page_config(page_title=page_title)
        st.markdown(f"# {page_title}")
        st.sidebar.header(page_title)

        last_values = read_dict_from_disk(save_file_name)
        if not last_values:
            last_values = {
                "gpt_model_index": 1,
                "max_tokens": 500,
                "temperature": 0.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "text_data": "",
                "data_chunk_size": 2500,
                "data_chunk_overlap": 200,
                "embedding_query": "",
                "number_of_docs": 2,
                "system_role_prompt": system_prompt,
                "user_prompt": user_prompt,
            }

        input_api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            max_chars=55,
            help="Enter your OpenAI API Key",
        )
        col_gpt_1, col_gpt_2 = st.columns(2)
        with col_gpt_1:
            input_model_type = st.selectbox(
                "Model",
                ["gpt-4", "gpt-3.5-turbo"],
                index=last_values["gpt_model_index"],
                help="Select the model to use.",
            )
        with col_gpt_2:
            input_max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=4000,
                value=last_values["max_tokens"],
                help="Defaults to 500.",
            )
        col_gpt_slicer_1, col_gpt_slicer_2, col_gpt_slicer_3 = st.columns(3)
        with col_gpt_slicer_1:
            input_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=last_values["temperature"],
                step=0.1,
                help="Defaults to 1. Between 0 and 2. Higher values like 0.8 will make the output more random, while lower values "
                "like 0.2 will make it more focused and deterministic.",
            )
        with col_gpt_slicer_2:
            input_presence_penalty = st.slider(
                "Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=last_values["presence_penalty"],
                step=0.1,
                help="Defaults to 0. Number between -2.0 and 2.0. Positive values increase the model's likelihood to talk about "
                "new topics.",
            )
        with col_gpt_slicer_3:
            input_frequency_penalty = st.slider(
                "Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=last_values["frequency_penalty"],
                step=0.1,
                help="Defaults to 0. Number between -2.0 and 2.0. Number between -2.0 and 2.0. Positive values decrease the model's"
                "likelihood to repeat the same line verbatim.",
            )

        input_system_role_prompt = st.text_area(
            "System Role",
            value=last_values["system_role_prompt"],
            height=200,
            max_chars=5000,
        )
        input_user_prompt = st.text_area(
            "User Prompt",
            value=last_values["user_prompt"],
            height=10,
            max_chars=5000
        )

        input_text_data = st.text_area(
            "Text to summarize",
            value=last_values["text_data"],
            height=200,
            max_chars=20000,
        )

        col_data_1, col_data_2 = st.columns(2)
        with col_data_1:
            data_chunk_size = st.number_input(
                "Embedding query chunk size",
                min_value=1,
                max_value=10000,
                value=last_values["data_chunk_size"],
            )
        with col_data_2:
            data_chunk_overlap = st.number_input(
                "Embedding query chunk overlap",
                min_value=1,
                max_value=2000,
                value=last_values["data_chunk_overlap"],
            )

        col_embedding_1, col_embedding_2 = st.columns(2)
        with col_embedding_1:
            input_embedding_query = st.text_input(
                "Query To Retrieve Embedding",
                value=last_values["embedding_query"],
                max_chars=1000,
                help="Enter the query for the embedding to retrieve.",
            )
        with col_embedding_2:
            number_of_docs = st.number_input(
                "Number of text chunks to retrieve",
                min_value=1,
                max_value=10,
                value=last_values["number_of_docs"],
            )

        if st.button("Run"):
            try:
                with st.status("Processing text data...", expanded=True) as status:
                    start_time = datetime.now()
                    loop = get_or_create_eventloop()
                    (
                        result,
                        total_tokens,
                        total_cost,
                    ) = loop.run_until_complete(
                        run_auto_cm(
                            save_file_name,
                            input_text_data,
                            data_chunk_size,
                            data_chunk_overlap,
                            input_system_role_prompt,
                            input_user_prompt,
                            input_api_key,
                            input_model_type,
                            input_temperature,
                            input_presence_penalty,
                            input_frequency_penalty,
                            input_max_tokens,
                            input_embedding_query,
                            number_of_docs
                        )
                    )
                    status.update(
                        label=f"Text data processed! Elapsed time: {(datetime.now() - start_time).seconds} seconds",
                        state="complete",
                        expanded=False,
                    )

                with st.container():
                    st.header("Output")
                    with st.expander(
                        f"result ({total_tokens} total tokens consumed). Cost: ${total_cost:.5f}"
                    ):
                        st.text(result)
            except Exception as e:
                st.write(e)
