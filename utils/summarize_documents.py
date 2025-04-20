# Run the initialization whenever the ChatBot is started
import textwrap

from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter


summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY", device=-1)

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
                # print(chunk_summary)

        combined_summary_text = " ".join(chunk_summaries)
        section_summary = recursive_summarize(combined_summary_text)

        if "Maybe you might find this helpful." in section_summary:
            for doc in docs:
                if doc.metadata['chunk_index'] == 0:
                    section_summary = doc.page_content[:30]
                    break

        section_summaries[section] = section_summary
        # print(section_summary)
        # print("__________________________________________")

    return section_summaries