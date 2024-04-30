import requests
import pandas as pd

API_URL = "https://qvo5pcuqml2jk85x.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": "Bearer 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",
	"Content-Type": "application/json" 
}

df = pd.read_excel("llama_results.xlsx")

sentiment_results = []
for idx, row in df.iterrows():
    city = row['city']
    country = row['country']
    place = f"{city}, {country}"
    text = row['review_text']
    typee = row['type']

    
    # prompt = f"""
    #         Input:
    #         ```
    #         Review Text: {text}
    #         ```

    #         This review is for: Riyadh, Saudi Arabia, Hospital

    #         Task:
    #         1. Overall Sentiment Analysis and Scoring:
    #         - Analyze the overall sentiment of the review.
    #         - Classify the sentiment into one of the following categories: Very Negative, Negative, Neutral, Positive, Very Positive.
    #         - Assign a score to the overall sentiment ranging from -100 (most negative) to 100 (most positive), where 0 represents a neutral sentiment.

    #         2. Aspect-Based Sentiment Analysis and Scoring:
    #         - Identify if key aspects are mentioned in the review (e.g. cleanliness, staff behavior, waiting time, cost, treatment effectiveness, teaching staff, or any specific aspect they have reviewer has about).
    #         - For each aspect identified, analyze the sentiment and classify it into one of the categories as mentioned above.
    #         - Assign a score to each aspect's sentiment ranging from -100 to 100, where 0 is neutral.

    #         Output Should be JSON Like Below:
    #         Provide json like:
    #         {{
    #         "overall": {{
    #             "Overall Sentiment": "Classified Sentiment",
    #             "Overall Score": "Numerical Score"
    #         }},
    #         "aspects": {{
    #             "aspect1": {{
    #             "Sentiment": "Classified Sentiment",
    #             "Score": "Numerical Score"
    #             }},
    #             "aspect2": {{
    #             "Sentiment": "Classified Sentiment",
    #             "Score": "Numerical Score"
    #             }},
    #             // Additional aspects can be added similarly
    #         }}
    #         }}

    #         Instructions:
    #         - Pay attention to modifiers and context that may alter the sentiment intensity (e.g., "not good" vs. "good").
    #         - Consider the overall tone and specific words used to describe experiences or opinions about the {place}.
    #         - **Return only the JSON output.**
    #         - Do not return any additional text or code snippets.
    #          """
    
    prompt = f"""
    You are a Robot that only return's JSON.
    You reply in JSON format:
    {{
    "overall": {{
    "Overall Sentiment": "Assign one of them: Very Negative, Negative, Neutral, Positive, Very Positive",
    "Overall Score": "Assign a score sentiment ranging from -100 to 100, Neutral Should be 0.."
    }},
    "aspects": {{
    "aspect1": {{
    "Sentiment": "Assign one of them: Very Negative, Negative, Neutral, Positive, Very Positive",
    "Score": "Assign a score to each aspect's sentiment ranging from -100 to 100, Neutral Should be 0."
    }},
    "aspect2": {{
    "Sentiment": "Assign one of them: Very Negative, Negative, Neutral, Positive, Very Positive",
    "Score": "Assign a score to each aspect's sentiment ranging from -100 to 100, Neutral Should be 0."
    }},
    // Additional aspects can be added similarly if it has.
    }}
    }}

    Now Here is My Review Text: {text}
        
    Task:
    - Analyze the following review for sentiment and return only the JSON output without any additional explanations or code snippets
    - Consider the overall tone and specific words used to describe experiences or opinions about the {typee}.     
    """
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    result = query({
        "inputs": prompt,
        "parameters": {
            "top_k": 10,
            "top_p": 0.9,
            "temperature": 0.3,
            "max_new_tokens": 100,
            "return_full_text": False,
        },
    })

    
    parts = result[0]['generated_text'].split('\n\n\n\n\n')
    cleaned_text = parts[0]
    
    sentiment_results.append(cleaned_text)

    df.loc[idx, 'llama_result'] = result
    
    print(cleaned_text,)

df.to_excel("llama_results.xlsx", index=False)


             
