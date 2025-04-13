import re
from bs4 import BeautifulSoup
import unicodedata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import pipeline
import html
from bs4 import BeautifulSoup
import json
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import textwrap
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import pipeline
from sec_edgar_downloader import Downloader
import requests
import os
import time
import faiss
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util
import torch
from sentence_transformers import SentenceTransformer, util
import json
import torch
from sentence_transformers import CrossEncoder
from collections import defaultdict

from operator import itemgetter
import re


###INITIALIZED
########################################################################

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vector_db = FAISS.load_local(
#     "vector_db",
#     embeddings=embeddings,
#     allow_dangerous_deserialization=True
# )
#
# with open('section_summaries.json', 'r', encoding='utf-8') as f:
#     section_summaries = json.load(f)
#
# query = "Apple's strategy in global smartphone markets"
# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
#
# summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY", device=-1)


########################################################################


###FUNCTIONS
########################################################################
# def find_items(query, section_summaries, removed = []):

#     # Load a sentence transformer model
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Prepare data
#     items = list(section_summaries.items())
#     item_keys = [item[0] for item in items]
#     summaries = [item[1] for item in items]

#     # Compute embeddings
#     summary_embeddings = model.encode(summaries, convert_to_tensor=True)
#     query_embedding = model.encode(query, convert_to_tensor=True)

#     # Compute cosine similarities
#     cosine_scores = util.cos_sim(query_embedding, summary_embeddings)[0]

#     # Get top 3 results
#     top_k = 3
#     top_results = torch.topk(cosine_scores, k=top_k)

#     top_matches = []
#     for score, idx in zip(top_results.values, top_results.indices):
#         top_matches.append(item_keys[idx])

#     return top_matches


def find_items(query, section_summaries, removed=[]):
    # Load a sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Prepare data
    items = list(section_summaries.items())

    # Filter out removed items
    filtered_items = [item for item in items if item[0] not in removed]

    item_keys = [item[0] for item in filtered_items]
    summaries = [item[1] for item in filtered_items]

    # Compute embeddings
    summary_embeddings = model.encode(summaries, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(query_embedding, summary_embeddings)[0]

    # Get top 3 results
    top_k = 3
    top_results = torch.topk(cosine_scores, k=top_k)

    top_matches = []
    for score, idx in zip(top_results.values, top_results.indices):
        top_matches.append(item_keys[idx])

    return top_matches


def rerank_documents(query, retrieved_docs, cross_encoder):
    # Prepare the inputs for the cross-encoder
    cross_encoder_inputs = [[query, doc.page_content] for doc in retrieved_docs]

    # Compute relevance scores
    relevance_scores = cross_encoder.predict(cross_encoder_inputs)

    # Attach scores to documents
    pairs_list = []
    for idx, doc in enumerate(retrieved_docs):
        pairs_list.append((doc, relevance_scores[idx]))

    # Sort documents by relevance score in descending order
    sorted_docs = sorted(pairs_list, key=lambda x: x[1], reverse=True)

    # Final output
    reranked_docs = [doc for doc, _ in sorted_docs]

    return reranked_docs


def filtered_retrieval(query, vector_db, cross_encoder, filter, fetch_k=100, k=5):
    retriever = vector_db.as_retriever(
        search_kwargs={
            "filter": filter,
            "fetch_k": fetch_k,
            "k": k
        }
    )

    filtered_docs = retriever.invoke(query)
    print(filtered_docs)
    reranked = rerank_documents(query, filtered_docs, cross_encoder)

    return reranked


# Load summarizer
def summarize_text(text, max_len=100, min_len=40):
    text = text[:4000]  # Prevent exceeding token limit
    return summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']


def recursive_summarize(text, max_chunk_chars=3500):
    """Break long text into chunks and summarize, then summarize summaries."""
    if len(text) <= max_chunk_chars:
        return summarize_text(text)

    # Split into smaller chunks
    chunks = textwrap.wrap(text, max_chunk_chars)
    chunk_summaries = [summarize_text(chunk) for chunk in chunks]

    # Combine and summarize again
    final_input = " ".join(chunk_summaries)
    return summarize_text(final_input)


def summarize_10k_chunks(all_chunks: list, item=[]):
    section_summaries = {}

    # Group chunks by section
    sections = {}
    for doc in all_chunks:
        section = doc.metadata["section"]
        if section not in sections:
            sections[section] = []
        sections[section].append(doc)

    for section, docs in sections.items():
        chunk_summaries = []

        for doc in docs:
            i = doc.metadata["chunk_index"]
            if i % 3 == 1:
                chunk_summary = summarize_text(doc.page_content)
                doc.metadata["chunk_summary"] = chunk_summary
                chunk_summaries.append(chunk_summary)
                print(chunk_summary)

        combined_summary_text = " ".join(chunk_summaries)
        section_summary = recursive_summarize(combined_summary_text)

        if "Maybe you might find this helpful." in section_summary:
            for doc in docs:
                if doc.metadata['chunk_index'] == 0:
                    section_summary = doc.page_content[:30]
                    break

        section_summaries[section] = section_summary
        print(section_summary)
        print("__________________________________________")

    return section_summaries


########################################################################


###SCRIPT
########################################################################

# # similarity search between section_summaries and query, output item 
# top_matches = find_items(query, section_summaries)
# print(top_matches)

# retrieve topk chunks that is relevant to query in each item
def retrieve_topk_chunks(top_matches):
    all_relevant_chunks = []
    for item in top_matches:
        filter = {"company": "apple",
                  "year": "2017",
                  "section": item}
        try:
            retrieved_docs = filtered_retrieval(query, vector_db, cross_encoder, filter)
            all_relevant_chunks.append(retrieved_docs)
        except:
            print(f"There's insufficient context being retrieved from the item section {item}")

    return all_relevant_chunks


# all_relevant_chunks = retrieve_topk_chunks(top_matches)

def flatten_chunks(all_relevant_chunks):
    # Flatten all chunks from all items
    flattened_chunks = []
    for chunk_list in all_relevant_chunks:
        for chunk in chunk_list:
            flattened_chunks.append(chunk)  # Assumes each chunk is a dict with at least 'content' key
    return flattened_chunks


# flattened_chunks = flattened_chunks(all_relevant_chunks)

def rerank2(flattened_chunks):
    # Create inputs for the cross encoder (query, chunk)
    rerank_inputs = [(query, doc.page_content) for doc in flattened_chunks]

    # Get similarity scores from cross_encoder
    scores = cross_encoder.predict(rerank_inputs)

    # Attach scores to the documents
    scored_chunks = []
    for doc, score in zip(flattened_chunks, scores):
        scored_chunks.append({
            "document": doc,
            "score": float(score)
        })

    # Sort by score descending
    reranked_chunks = sorted(scored_chunks, key=itemgetter("score"), reverse=True)
    return reranked_chunks


def generate_retrieve_context(reranked_chunks):
    retrieved_context = [
        f"{i}. {item['document'].page_content.strip()}"
        for i, item in enumerate(reranked_chunks[:3], 1)
    ]

    return retrieved_context


# reranked_chunks = rerank2(flattened_chunks)
# retrieved_context = generate_retrieve_context(reranked_chunks)
# print(retrieved_context)


def retrieve_process(retrieved_context, query):
    prompt = f"""
        You are given a user query and a set of retrieved documents. Determine whether the retrieved documents are relevant to answering the query.

        Query: "{query}"

        Retrieved Context:
        {retrieved_context}

        Instructions:
        - If the documents are not relevant, respond exactly: "No"
        - If the documents are relevant, respond exactly in the following format:
        Yes, from line X. Respond

        Where X is the line number in the Retrieved Context that contains the relevant information. And Respond is the answer generated utilizing the retrieved context to answer the query.
        
        Do not place any words at the start of the respond other than "Yes" or "No"
        
        Example:
        Yes, from line 2
    """

    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"

    # Remove the following line when push to github
    api_key = os.environ["HUGGINGFACE_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}"}

    data = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=data)

    text = response.json()[0]["generated_text"]

    return (text[text.index(prompt) + len(prompt) + 1:])


# llm_output = retrieve_process(retrieved_context, query)


def update_section_summaries(llm_output, reranked_chunks, section_summaries):
    llm_output = llm_output.strip()

    if not llm_output.lower().startswith("yes"):
        return None  # Skip if not relevant

    # Extract the line number from llm_output
    match = re.search(r"line\s+(\d+)", llm_output.lower())
    if not match:
        return None  # Line number not found

    line_number = int(match.group(1)) - 1  # Convert to 0-based index

    # Safety check
    if line_number < 0 or line_number >= len(reranked_chunks):
        return None

    # Get the relevant document and summarize
    relevant_doc = [reranked_chunks[line_number]["document"]]
    new_summarisation = summarize_10k_chunks(relevant_doc)

    # Update the section_summaries dictionary
    for key, value in new_summarisation.items():
        if key in section_summaries:
            section_summaries[key] += " " + value
        else:
            section_summaries[key] = value

    with open('section_summaries.json', 'w', encoding='utf-8') as f:
        json.dump(section_summaries, f, ensure_ascii=False, indent=4)

    return section_summaries


# updated_section_summaries = update_section_summaries(llm_output, reranked_chunks, section_summaries)


def pipeline(query):
    top_matches = find_items(query, section_summaries)
    print(top_matches)

    all_relevant_chunks = retrieve_topk_chunks(top_matches)
    print(all_relevant_chunks)

    flattened_chunks = flatten_chunks(all_relevant_chunks)

    reranked_chunks = rerank2(flattened_chunks)
    retrieved_context = generate_retrieve_context(reranked_chunks)
    print(retrieved_context)

    llm_output = retrieve_process(retrieved_context, query)
    print(llm_output)

    llm_output = llm_output.strip()

    return llm_output, top_matches

    # if not llm_output.lower().startswith("yes"):
    #     # no, then need to remove from the list, and scan through another 3 items
    #     return None  # Skip if not relevant
    # else:
    #     update_section_summaries(llm_output, reranked_chunks, section_summaries)


def exhaustive_query(query, section_summaries, removed):
    if len(removed) == section_summaries:
        return "No context found from our Database"
    else:
        output, top_matches = pipeline(query)
        if output.startswith("No"):
            removed.extend(top_matches)
            return exhaustive_query(query, section_summaries, removed)
        else:
            return output




if __name__ == "__main__":
    # Initial Setup for the running the query
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(
        "vector_db",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    with open('section_summaries.json', 'r', encoding='utf-8') as f:
        section_summaries = json.load(f)

    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY", device=-1)

    query = "What's the revenue for Apple in 2017"
    print(query)
    removed = []
    exhaustive_query(query, section_summaries, removed)
