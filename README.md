# semantic-search
Small streamlit app showing semantic search with OpenAI.

1. You can add files with text using the upload function. For each file an embedding will retrieved using the embedding-ada-002 model.
2. Write a question! Another embedding will be calculated for this bit of text.
3. The list of documents is sorted based on the similarity of document embeddings vs question embedding.
4. A completion request is sent to the davinci model: Answer the question given the document. 

The setup is heavily influenced by the [openai-cookbook](https://github.com/openai/openai-cookbook) and [openai-in-a-day](https://github.com/csiebler/openai-in-a-day). But then in streamlit :).

## How to run
1. Create the environment (conda create -f streamlit-app/st-app-env.yml)
2. Put your OpenAI key and organization in .streamlit/secrets.toml (or use another mechanism)
3. In the streamlit-app folder: streamlit run semantic_search.py
