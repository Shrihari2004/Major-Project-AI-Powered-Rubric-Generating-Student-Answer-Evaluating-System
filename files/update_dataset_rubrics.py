import json
import urllib.request
import concurrent.futures
import time
import sys

api_key = "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"
url = "https://integrate.api.nvidia.com/v1/chat/completions"

def generate_descriptive_rubric(question, rubric_list):
    short_criteria = "\n".join([f"{i+1}. {r['criterion']} [{r['max_marks']} mark(s)]" for i, r in enumerate(rubric_list)])
    
    prompt = f"""You are an expert educator creating grading rubrics.
I have a question and a preliminary grading rubric containing short criterion names.
Your task is to expand these short criterion names into highly descriptive, full sentences that explain exactly what the student must do to achieve the marks.

Question: {question}
Preliminary Rubric:
{short_criteria}

Please provide the finalized rubric.
FORMAT REQUIREMENT:
You must format your output EXACTLY as a numbered list where each point ends with the marks in brackets, like "[X mark(s)]".
Example:
1. The solution correctly identifies the main concept and applies it with no errors. [5 mark(s)]
2. The time complexity is clearly explained and optimized. [3 mark(s)]

Do not include any intro or outro text. Only output the numbered list.
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 512
    }
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content'].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429: # Rate limit
                time.sleep(5)
            else:
                time.sleep(2)
        except Exception as e:
            time.sleep(2)
            if attempt == 2:
                return short_criteria # Fallback on final error

def process_item(item):
    question = item.get("question", "")
    rubric = item.get("rubric", [])
    if question and rubric:
        descriptive_rubric = generate_descriptive_rubric(question, rubric)
        item["api_generated_rubric"] = descriptive_rubric
    return item

def main():
    print("Loading dataset...")
    try:
        with open("ml_ready_master_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: ml_ready_master_dataset.json not found.")
        sys.exit(1)
        
    print(f"Processing {len(data)} items concurrently...")
    
    updated_data = []
    # Using 10 workers to avoid rate limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_item, item): item for item in data}
        
        count = 0
        for future in concurrent.futures.as_completed(futures):
            count += 1
            updated_data.append(future.result())
            if count % 50 == 0:
                print(f"Processed {count}/{len(data)} items...")

    print("Saving to ml_ready_master_dataset_enhanced.json...")
    with open("ml_ready_master_dataset_enhanced.json", "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2)
        
    print("Dataset enhancement complete!")

if __name__ == "__main__":
    main()
