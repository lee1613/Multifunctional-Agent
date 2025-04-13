import re
from sentence_transformers import SentenceTransformer, util
import statistics
import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import pandas as pd
from bs4 import BeautifulSoup
import os
from sentence_transformers import CrossEncoder

# Load the cross-encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# URL of the earnings call transcript
url = 'https://www.fool.com/earnings/call-transcripts/2024/10/31/apple-aapl-q4-2024-earnings-call-transcript/'

# Load OPENAPI Key
os.environ["OPENAI_API_KEY"] = ""

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")


def fetch_and_save_transcript(url):
    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the article-body div
    article_body = soup.find('div', class_='article-body')
    if not article_body:
        raise ValueError("Could not find the article body.")

    # Remove all h2 tags from the article body
    for h2 in article_body.find_all('h2'):
        h2.decompose()

    # Extract all paragraph text
    paragraphs = article_body.find_all('p')
    transcript = '\n'.join([para.get_text() for para in paragraphs])

    return transcript

def chunk_transcript(text):

    

    # Pattern to detect speakers including Operator
    speaker_pattern = re.compile(r'^(?:[A-Z][a-z]+(?: [A-Z][a-z]+)* -- .+|Operator)$', re.MULTILINE)
    chunks = []

    matches = list(speaker_pattern.finditer(text))
    pending_question = None
    question_speaker = None
    chunk_id = 1

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        speaker_line = match.group(0).strip()
        speaker_text = text[start:end].strip().replace('\n', ' ')

        # Skip empty text
        if not speaker_text:
            continue

        # Skip Operator chunks
        if speaker_line == "Operator":
            continue

        # Store analyst question temporarily
        if "Analyst" in speaker_line:
            pending_question = speaker_text
            question_speaker = speaker_line
            continue

        # Build final chunk with or without question
        chunk = {
            "chunk_id": f"chunk_{chunk_id:03}",
            "speaker": speaker_line,
            "text": speaker_text,
            "question": pending_question if pending_question else ""
        }

        chunks.append(chunk)
        chunk_id += 1
        pending_question = None  # Clear after attaching to next answer
    
    # convert documents
    documents = [
        Document(
            page_content=entry["text"],
            metadata={
                "chunk_id": entry["chunk_id"],
                "speaker": entry["speaker"]
            }
        )
        for entry in chunks if entry["text"].strip()  
    ]

    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Build FAISS index
    faiss_index = FAISS.from_documents(documents, embedding_model)

    return faiss_index

def retrieve_relevant_docs(query, vector_store, k=5):
    # Retrieve top-k similar document chunks for the query
    retrieved_docs = vector_store.similarity_search(query, k=k)
    return retrieved_docs

def retrieve_and_rerank(query, vector_store, cross_encoder, initial_k=10, final_k=5):
    # Step 1: Initial retrieval using bi-encoder
    initial_retrieved_docs = retrieve_relevant_docs(query, vector_store, k=initial_k)

    # Step 2: Re-rank the initially retrieved documents using cross-encoder
    reranked_docs = rerank_documents(query, initial_retrieved_docs, cross_encoder)

    # Step 3: Select the top-k documents after re-ranking
    top_k_docs = reranked_docs[:final_k]

    return top_k_docs

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

def build_prompt(retrieved_docs, query, prompt_engineering):
    context_chunks = []

    for i, doc in enumerate(retrieved_docs, start=1):
        # Use only speaker metadata now
        speaker = doc.metadata.get('speaker', 'Unknown')
        source_info = f"\n[Source: {speaker}]"

        # Add question if it exists
        question = doc.metadata.get("question", "")
        if question:
            chunk_text = f"Q: {question}\nA: {doc.page_content}"
        else:
            chunk_text = doc.page_content

        chunk_text = chunk_text.replace("\n", " ")
        context_chunks.append(f"{i}. {chunk_text}{source_info}")

    # Join all context chunks
    context = "\n\n".join(context_chunks)

    # Prompt template
    prompt = (
        "You are a financial analyst reviewing an earnings call transcript. Your task is to extract and summarize key factual claims and sentiment-related insights expressed by management that can later be verified against the company's 10-K filing."
        " The 10-K contains detailed data in sections such as the Management Discussion and Analysis (MD&A), Financial Statements, Business Overview, and Risk Factors.\n\n"

        "Instructions:\n\n"
        "Based on the retrieved context from earnings call transcripts below:\n\n"
        "#### Context/Retrieved Statements:\n{context}\n\n"
        "Identify and extract factual claims regarding the company’s performance, outlook, and any sentiment indications (for example, optimism about growth, caution regarding risks, or confidence in future performance).\n\n"
        "For each claim, provide the following details in a structured format:\n\n"
        "Claim Text: A concise statement of the fact or sentiment.\n\n"
        "[Source: speaker]\n\n"
        "Metric/Detail (if applicable): Any quantifiable data (e.g., “revenue growth of 15%”, “EBITDA margin improvement”).\n\n"
        "Relevant Reporting Period: Indicate the fiscal year or quarter mentioned (e.g., “FY2023”).\n\n"
        "Target 10-K Section: Suggest which section of the 10-K is most appropriate to verify this claim (for example, “MD&A”, “Financial Statements”, “Risk Factors”, “Business Overview”).\n\n"
        "#### Query: {query}\n\n"
    ).format(context=context, query=query, prompt_engineering=prompt_engineering)

    return prompt

def generate_insight(prompt, model="gpt-4-turbo", temperature=0.1, max_tokens=512):
    """
    Generate analysis using the given prompt via the OpenAI ChatCompletion API.
    """
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are interested in analyzing a company's sentiment level."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def pipeline(query):
    prompt_engineering = ""
    transcript = fetch_and_save_transcript(url)
    faiss_index = chunk_transcript(transcript)
    retrieved_docs = retrieve_and_rerank(query, faiss_index, cross_encoder, initial_k=10, final_k=5)
    prompt = build_prompt(retrieved_docs, query, prompt_engineering)
    insight = generate_insight(prompt)
    print(insight)
    return insight

if __name__ == "__main__":
    query = "How did iPhone sales perform across regions, and what contributed to the growth in 2024 Q4?"
    print(query)
    pipeline(query)
