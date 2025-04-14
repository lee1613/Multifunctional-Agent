from retrieve_transcript import pipeline as transcript_pipeline
from retrieve_10k import pipeline as tenk_pipeline
import os 
import openai

os.environ["OPENAI_API_KEY"] = ""


def multimodal_agent(query):
    print(f"Running multimodal RAG agents on query: {query}")

    # Step 1: Run both pipelines
    transcript_output = transcript_pipeline(query)
    tenk_output = tenk_pipeline(query)

    # Step 2: Feed both into the cross-checker agent
    final_summary = cross_check(transcript_output, tenk_output)

    print("\n📊 Final Investment Insight:\n")
    print(final_summary)
    return final_summary



def cross_check(transcript_output, tenk_output):
    prompt = f"""
        You are a financial cross-checking agent.

        Given the following two analyses:

        📄 Transcript Analysis:
        {transcript_output}

        📄 10-K Analysis:
        {tenk_output}

        Please:
        1. Check if the transcript claims are supported by the 10-K.
        2. Highlight any optimistic commentary not grounded in financials.
        3. Identify new information or guidance present in the transcript but not in the 10-K.
        4. Summarize the investment outlook based on both sources.

        Respond clearly in a concise format.
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a critical-thinking financial assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800,
    )

    return response.choices[0].message.content




if __name__ == "__main__":
    query = "Did Apple experience growth in emerging markets in Q4 2024?"
    multimodal_agent(query)




