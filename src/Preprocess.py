from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import pipeline

summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY", device=-1)
