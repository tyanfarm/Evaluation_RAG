import requests
import json
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

def call_dify_api(question, api_key, api_url):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}    
    data = {
        "inputs": {"query": question},
        "response_mode": "blocking",
        "user": "abc-123"
    }
    response = requests.post(api_url, headers=headers, json=data)

    return response.json().get("answer", "")

def evaluate_responses(ground_truths, api_key, api_url):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    scores = []

    for item in ground_truths:
        question = item["question"]
        expected_answer = item["answer"]
        generated_answer = call_dify_api(question, api_key, api_url)

        # ROUGE Score
        rouge_scores = scorer.score(expected_answer, generated_answer)
        rouge1 = rouge_scores['rouge1'].fmeasure
        rougeL = rouge_scores['rougeL'].fmeasure

        # Consine Similarity
        embedding1 = model.encode(expected_answer, convert_to_tensor=True)
        embedding2 = model.encode(generated_answer, convert_to_tensor=True)
        consine_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

        scores.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "rouge1": rouge1,
            "rougeL": rougeL,
            "cosine_similarity": consine_similarity
        })

    return scores

if __name__ == "__main__":
    API_KEY = "app-m87ZoSsAabTadDO8oSz4SvQL"
    API_URL = "http://localhost/v1/completion-messages"

    # Data for test
    ground_truths = [
        {"question": "which team is Messi playing for now?", "answer": "Lionel Messi is playing for Inter Miami FC"},
        {"question": "Summary of match developments between Messi's team & opponent team", "answer": "Inter Miami beat Mexican superpower Club America 3-2 on penalty kicks after the teams played to a 2-2 deadlock in regulation."}
    ]

    results = evaluate_responses(ground_truths, API_KEY, API_URL)

    # Output
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False) 

    print("Evaluation completed! Results saved in evaluation_results.json")   