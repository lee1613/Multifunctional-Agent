import re
from sentence_transformers import SentenceTransformer, util
import statistics
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import librosa
import numpy as np
import os
from pydub import AudioSegment
import nltk
from nltk.tokenize import word_tokenize

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
import json


# Load the cross-encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# URL of the earnings call transcript
url = 'https://www.fool.com/earnings/call-transcripts/2024/10/31/apple-aapl-q4-2024-earnings-call-transcript/'

# Load OPENAPI Key
os.environ["OPENAI_API_KEY"] = ""

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
json_path = "audio_chunks/apple_q4_2024_AUDIO_chunks.json"


# scraping the motley fool earnings call transcript 
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


# Function to extract the relevant audio features from the audio file
def extract_audio_features(audio_path, sr=16000):

    # Load the audio segment using librosa.
    y, sr = librosa.load(audio_path, sr=sr)

    # Compute 13 MFCCs and take the mean across time for each coefficient.
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_avg = np.mean(mfccs, axis=1).tolist()

    # Use librosa.pyin to estimate pitch (f0). fmin and fmax defined for typical speech range.
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    # Compute mean pitch over voiced regions (ignore unvoiced).
    if f0 is not None and np.any(voiced_flag):
        pitch_mean = float(np.mean(f0[voiced_flag]))
    else:
        pitch_mean = None

    # Calculate Root Mean Square (RMS) energy and compute the mean value.
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))

    return {
        'mfccs_avg': mfccs_avg,
        'pitch_mean': pitch_mean,
        'rms_mean': rms_mean
    }

# Splitting audio features into chunks
def split_audio_by_chunks(audio_file_path, chunks, output_dir="audio_chunks"):

    # Load audio
    full_audio = AudioSegment.from_file(audio_file_path, format="mp3")
    total_audio_ms = len(full_audio)

    # Total word count across all chunks
    def count_words(text):
        return len(word_tokenize(text))
    
    total_words = sum(count_words(chunk['text']) for chunk in chunks)
    
    current_start_ms = 0
    os.makedirs(output_dir, exist_ok=True)

    for chunk in chunks:
        wc = count_words(chunk['text'])
        chunk_duration_ms = int((wc / total_words) * total_audio_ms)
        chunk['start_time'] = current_start_ms
        chunk['end_time'] = current_start_ms + chunk_duration_ms

        audio_segment = full_audio[chunk['start_time']:chunk['end_time']]
        out_filename = f"{chunk['chunk_id']}.mp3"
        out_path = os.path.join(output_dir, out_filename)
        audio_segment.export(out_path, format="mp3")
        chunk['audio_file'] = out_path

        current_start_ms += chunk_duration_ms

    return chunks

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

    return chunks


def process_audio_features(chunks):
    
    pitch_list = [chunk['audio_features']['pitch_mean'] for chunk in chunks if 'audio_features' in chunk]
    rms_list   = [chunk['audio_features']['rms_mean'] for chunk in chunks if 'audio_features' in chunk]

    pitch_global_mean = np.mean(pitch_list)
    pitch_global_std  = np.std(pitch_list)
    rms_global_mean   = np.mean(rms_list)
    rms_global_std    = np.std(rms_list)

    
    for chunk in chunks:
        if 'audio_features' in chunk:
            features = chunk['audio_features']

            # Standardize pitch and RMS (z-score)
            pitch = features.get('pitch_mean', 0)
            rms   = features.get('rms_mean', 0)

            pitch_z = (pitch - pitch_global_mean) / (pitch_global_std + 1e-8)
            rms_z   = (rms - rms_global_mean) / (rms_global_std + 1e-8)

            features['pitch_z'] = pitch_z
            features['rms_z'] = rms_z

            # Composite intensity score
            features['composite_intensity'] = abs(pitch_z) + abs(rms_z)

            # MFCCs analysis
            mfccs = np.array(features.get('mfccs_avg', []))
            if mfccs.size > 0:
                features['mfccs_mean_abs'] = float(np.mean(np.abs(mfccs)))
                features['mfccs_variance'] = float(np.var(mfccs))
            else:
                features['mfccs_mean_abs'] = None
                features['mfccs_variance'] = None

    return chunks

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

def create_audio_summary(audio_features):

    pitch_z = audio_features.get('pitch_z', 'N/A')
    rms_z = audio_features.get('rms_z', 'N/A')
    composite_intensity = audio_features.get('composite_intensity', 'N/A')

    summary = (f"\nAudio Analysis Summary: The normalized pitch is {pitch_z:.4f} and "
               f"the normalized RMS energy is {rms_z:.4f}, leading to a composite intensity of "
               f"{composite_intensity:.4f}.")
    return summary

def build_prompt(retrieved_docs, query, prompt_engineering):
    context_chunks = []

    for i, doc in enumerate(retrieved_docs, start=1):
        # Use only speaker metadata now
        speaker = doc.metadata.get('speaker', 'Unknown')
        source_info = f"\n[Source: {speaker}]"

        # Use question if it exists
        question = doc.metadata.get("question", "")
        if question:
            chunk_text = f"Q: {question}\nA: {doc.page_content}"
        else:
            chunk_text = doc.page_content

        # Incorporate audio features (if available) from metadata
        audio_feat = doc.metadata.get("audio_features", None)
        if audio_feat:
            audio_summary = create_audio_summary(audio_feat)
        else:
            audio_summary = ""


        chunk_text = chunk_text.replace("\n", " ")


        context_chunks.append(f"{i}. {chunk_text}{source_info}{audio_summary}")

    # Join all context chunks
    context = "\n\n".join(context_chunks)

    # Build the final multimodal prompt (including instructions and query)
    prompt = (
        "You are a financial analyst reviewing an earnings call transcript and its associated audio features. "
        "Your task is to extract and summarize key factual claims and sentiment-related insights expressed by management "
        "that can later be verified against the company's 10-K filing. The 10-K contains detailed data in sections such as "
        "the Management Discussion and Analysis (MD&A), Financial Statements, Business Overview, and Risk Factors.\n\n"
        "Take into account the question, speaker, earnings call transcript text and audio features (like pitch, RMS energy, "
        "and composite intensity) in your elaboration to provide a comprehensive analysis.\n\n"

        "Instructions:\n\n"
        "Based on the retrieved context from earnings call transcripts (with their corresponding audio features) below:\n\n"
        "#### Context/Retrieved Statements:\n{context}\n\n"
        "Identify and extract factual claims regarding the company’s performance, outlook, and any sentiment indications "
        "(for example, optimism about growth, caution regarding risks, or confidence in future performance).\n\n"
        "For each claim, provide the following details in a structured format:\n\n"
        "Claim Text: A concise statement of the fact or sentiment.\n\n"
        "[Source: speaker]\n\n"
        "Metric/Detail (if applicable): Any quantifiable data (e.g., 'revenue growth of 15%', 'EBITDA margin improvement').\n\n"
        "Relevant Reporting Period: Indicate the fiscal year or quarter mentioned (e.g., 'FY2023').\n\n"
        "Target 10-K Section: Suggest which section of the 10-K is most appropriate to verify this claim (for example, 'MD&A', 'Financial Statements', 'Risk Factors', 'Business Overview').\n\n"
        "#### Query: {query}\n\n"
    ).format(context=context, query=query, prompt_engineering=prompt_engineering)

    return prompt

def generate_insight(prompt, model="gpt-4-turbo", temperature=0.1, max_tokens=1080):
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
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunks = process_audio_features(chunks)

    documents = [
        Document(
            page_content=entry["text"],
            metadata={
                "chunk_id": entry["chunk_id"],
                "speaker": entry["speaker"],
                "question": entry["question"],
                "audio_features": entry["audio_features"]
            }
        )
        for entry in chunks if entry["text"].strip()
    ]

    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    faiss_index = FAISS.from_documents(documents, embedding_model)

    
    retrieved_docs = retrieve_and_rerank(query, faiss_index, cross_encoder, initial_k=10, final_k=5)
    prompt = build_prompt(retrieved_docs, query, prompt_engineering)
    insight = generate_insight(prompt)

    print("============= transcript results =============")
    print(insight)
    return insight


if __name__ == "__main__":
    query = "Analyze Apples's sentiment level."
    company = 'Apple' 
    year = '2024'
    pipeline(query)
