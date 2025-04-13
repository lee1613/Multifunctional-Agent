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




txt_folder = 'database/apple_10k_txts'
html_folder = 'database/apple_10k_html'


#######################################################################################################################################

def download_data():
    CIK = "0000320193"  # Apple
    EMAIL = "your_email@example.com"  # Use your real email
    SAVE_FOLDER = txt_folder
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    HEADERS = {
        "User-Agent": f"MyApp/1.0 ({EMAIL})"
    }

    def get_10k_filing_info(cik):
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(url, headers=HEADERS)
        data = res.json()
        
        info = []
        for i, form in enumerate(data["filings"]["recent"]["form"]):
            if form == "10-K":
                acc_num = data["filings"]["recent"]["accessionNumber"][i]  # e.g., 0000320193-23-000106
                acc_num_clean = acc_num.replace("-", "")  # e.g., 000032019323000106
                info.append((acc_num_clean, acc_num))  # (folder, file)
        return info

    def download_txt_file(folder_acc, file_acc):
        url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{folder_acc}/{file_acc}.txt"
        res = requests.get(url, headers=HEADERS)
        if res.status_code == 200:
            file_path = os.path.join(SAVE_FOLDER, f"{file_acc}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(res.text)
            print(f"Downloaded: {file_path}")
        else:
            print(f"Failed to download: {file_acc} (status code: {res.status_code})")

    # === MAIN ===
    filings = get_10k_filing_info(CIK)

    for folder_acc, file_acc in filings:
        try:
            download_txt_file(folder_acc, file_acc)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error downloading {file_acc}: {e}")

def convert_txt_file_to_html():
    # Create output folder if it doesn't exist
    os.makedirs(html_folder, exist_ok=True)

    # Process all .txt files
    for filename in os.listdir(txt_folder):
        if filename.endswith('.txt'):
            txt_path = os.path.join(txt_folder, filename)
            html_filename = filename.replace('.txt', '.html')
            html_path = os.path.join(html_folder, html_filename)

            # Read .txt (HTML content)
            with open(txt_path, 'r', encoding='utf-8') as file:
                html_content = file.read()

            # Write as .html
            with open(html_path, 'w', encoding='utf-8') as html_file:
                html_file.write(html_content)

            print(f"Converted: {filename} → {html_filename}")

    print("All conversions complete.")

def normalize_text(s):
    return unicodedata.normalize("NFKC", s).replace('\xa0', ' ').strip()

def extract_clean_text(file_path):
    # Step 1: Load and clean up the HTML
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Step 2: Convert to plain text
    text = soup.get_text(separator='\n')

    # Step 3: Skip everything until the real content begins (after TOC)
    # We'll skip everything until the second "Item " occurrence
    item_matches = list(re.finditer(r"(Item\s+[0-9A-Za-z]+(?:[A-Z])?)\.", text, flags=re.IGNORECASE))
    # item_pattern = r"(Item\s+[0-9A-Za-z]+(?:[A-Z])?)\."

    # Use only the real content (skip the TOC)
    content_start = item_matches[1].start() if len(item_matches) > 1 else 0
    main_text = text[content_start:]

    # Step 4: Extract items and their contents
    pattern = re.compile(r"(Item\s+[0-9A-Za-z]+(?:[A-Z])?)\.", re.IGNORECASE)
    matches = list(pattern.finditer(main_text))

    extracted_clean = {}

    for i in range(len(matches)):
        item_title = normalize_text(matches[i].group(1))
        if item_title.lower() == "item 16":
            continue  # Skip Item 16
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(main_text)
        content = main_text[start:end].strip()
        
        # Clean content: normalize spaces and remove excessive line breaks
        cleaned_content = re.sub(r'\s+', ' ', content)
        
        extracted_clean[item_title] = cleaned_content

    return extracted_clean

# Load summarizer
summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY", device=-1)

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

def chunk_10k_sections(parsed_sections: dict, company: str, year: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    all_chunks = []

    for section_title, section_text in parsed_sections.items():
        normalized_title = section_title.upper().replace(".", "").replace("ITEM", "Item").strip()
        section_docs = splitter.create_documents([section_text])

        for i, doc in enumerate(section_docs):
            doc.metadata = {
                "company": company,
                "year": year,
                "section": normalized_title,
                "chunk_index": i
            }
            all_chunks.append(doc)

    return all_chunks

def summarize_10k_chunks(all_chunks: list, item = []):
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

def add_documents(vector_db, chunks):
  vector_db.add_documents(chunks)
  return vector_db

def add_10_K(file_path, vector_db, company, year, section_summaries):

    extracted_clean = extract_clean_text(file_path)
    chunks = chunk_10k_sections(extracted_clean, company=company, year=year)

    # Collect chunks that belong to new sections (not yet summarized)
    summarise_this_chunks = [doc for doc in chunks if doc.metadata['section'] not in section_summaries]

    # If there are new sections, summarize and update the dict
    if summarise_this_chunks:
        new_summaries = summarize_10k_chunks(summarise_this_chunks)
        section_summaries.update({k: v for k, v in new_summaries.items() if k not in section_summaries})

        with open('section_summaries.json', 'w', encoding='utf-8') as f:
          json.dump(section_summaries, f, ensure_ascii=False, indent=4)

    # Add all chunks to the vector DB
    vector_db = add_documents(vector_db, chunks)

    return vector_db, section_summaries

#######################################################################################################################################

# download_data()
# convert_txt_file_to_html()

for i, filename in enumerate(os.listdir(html_folder)):
    if filename == '.DS_Store':
        continue

    file_path = os.path.join(html_folder, filename)
    
    if os.path.isfile(file_path):
        full_path = os.path.join(html_folder, filename)  # This is now the full path
        print(f"Working On: {full_path}")

        company = full_path.split('/')[1].split('_')[0]
        year = '20' + full_path.split('/')[-1].split('-')[1]

        if (year == '2014') or (year == '2015') or (year == '2016'):
            continue 

        if i == 0:
            ## initialise chunks and vector database and section_summaries
            print(f"This is a first file: {full_path}")
            
            extracted_clean = extract_clean_text(full_path)
            chunks = chunk_10k_sections(extracted_clean, company=company, year=year)
            
            # test
            # section_summaries = summarize_10k_chunks(chunks)

            # with open('section_summaries.json', 'w', encoding='utf-8') as f:
            #     json.dump(section_summaries, f, ensure_ascii=False, indent=4)

            # no need to run the summarise_10k
            with open('section_summaries.json', 'r', encoding='utf-8') as f:
                section_summaries = json.load(f)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_db = FAISS.from_documents(chunks, embeddings)
            vector_db.save_local("vector_db")

        else:
            vector_db = FAISS.load_local(
                "vector_db",
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )

            add_10_K(full_path, vector_db, company, year, section_summaries)
            print(vector_db.index.ntotal)
    
            vector_db.save_local("vector_db")
