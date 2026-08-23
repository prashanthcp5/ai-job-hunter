import pandas as pd
from jobspy import scrape_jobs
from google import genai
import os
import json
import re

# --- CONFIGURATION ---
API_KEY = os.getenv("GEMINI_API_KEY")
SEARCH_TERM = "Supply Chain"
LOCATION = "Boston, MA"
BATCH_SIZE = 10  # Evaluate 10 jobs at a time in a single AI call
# ---------------------

print("🚀 Starting the HIGH-PERFORMANCE AI supply chain job analyzer...")

try:
    ai_client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"❌ Failed to initialize AI Client: {e}")
    exit()

MY_PROFILE = """
- Education: Master of Science (MS) in Supply Chain Management from Boston University (Graduated May 2026).
- Certifications: Certified Lean Six Sigma Green Belt.
- Core Interests & Projects: Procurement, inventory optimization, RFID tracking technology implementation, data-driven methodologies (Decision Matrices).
- Target Roles: Entry-to-mid level Supply Chain Analyst, Logistics Planner, Inventory Analyst, Procurement Specialist.
"""

# SUPPLY CHAIN LOCAL KEYWORD FILTER
KEYWORDS = ["supply chain", "procurement", "inventory", "logistics", "forecasting", "sap", "six sigma", "analyst", "planning"]

try:
    print(f"🔍 Scraping jobs for '{SEARCH_TERM}' in {LOCATION}...")
    jobs = scrape_jobs(
        site_name=["linkedin", "indeed", "google"],
        search_term=SEARCH_TERM,
        location=LOCATION,
        results_wanted=30, 
        hours_old=72,
        country_only="USA"
    )

    if jobs.empty:
        print("❌ No matching roles found in the last 72 hours.")
        exit()

    initial_count = len(jobs)
    
    # 1. OPTIMIZATION: Drop duplicate listings across multiple job boards
    jobs = jobs.drop_duplicates(subset=["title", "company"])
    print(f"📋 Cleaned duplicates. Reduced from {initial_count} down to {len(jobs)} unique jobs.")

    # 2. OPTIMIZATION: Pre-screen locally with keyword scoring to remove noise
    def calculate_local_score(text):
        if not text: return 0
        text_lower = str(text).lower()
        return sum(1 for kw in KEYWORDS if kw in text_lower)

    jobs["keyword_score"] = jobs["description"].apply(calculate_local_score)
    
    # Only keep jobs that mention at least 1 core supply chain keyword
    jobs = jobs[jobs["keyword_score"] > 0].copy()
    print(f"🎯 Local keyword filtering kept {len(jobs)} relevant jobs.")

    # Reset index so we can map batch responses back perfectly
    jobs = jobs.reset_index(drop=True)
    
    # Create empty columns for AI results
    jobs['match_score'] = 0
    jobs['ai_analysis'] = "Not evaluated."

    # 3. OPTIMIZATION: Batching rows and executing via itertuples
    batches = [jobs[i:i + BATCH_SIZE] for i in range(0, len(jobs), BATCH_SIZE)]
    print(f"📦 Grouped rows into {len(batches)} efficient AI batches.")

    for batch_idx, batch in enumerate(batches):
        print(f"🤖 Processing AI Batch {batch_idx + 1}/{len(batches)}...")
        
        # Build a compact jobs summary string for this batch
        batch_jobs_payload = []
        for row in batch.itertuples():
            # Truncate description to 2500 chars to minimize token overhead
            truncated_desc = str(row.description)[:2500].replace('\n', ' ')
            batch_jobs_payload.append({
                "job_index": row.Index,
                "title": row.title,
                "company": row.company,
                "description": truncated_desc
            })

        # Inject strict JSON schema formatting into the prompt instructions
        prompt = f"""
        You are an elite corporate Recruiter. Evaluate this batch of job opportunities against the Candidate Profile.
        
        [Candidate Profile]
        {MY_PROFILE}
        
        [Jobs Batch Data]
        {json.dumps(batch_jobs_payload, indent=2)}
        
        Task: Analyze each job. Return a JSON array matching this exact format structure:
        [
          {{
            "job_index": 0,
            "score": 9,
            "reason": "Clear explanation sentence."
          }}
        ]
        Ensure your response contains ONLY the valid JSON array payload. Do not wrap it in markdown code blocks like ```json.
        """

        try:
            # Enforce structured JSON configuration mode natively
            response = ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            
            # Direct structural parsing completely safely
            analysis_results = json.loads(response.text.strip())
            
            for item in analysis_results:
                idx = item.get("job_index")
                jobs.at[idx, 'match_score'] = int(item.get("score", 0))
                jobs.at[idx, 'ai_analysis'] = item.get("reason", "No evaluation summary returned.")
                
        except Exception as batch_error:
            print(f"⚠️ Warning: Failed processing data within batch {batch_idx + 1}: {batch_error}")

    # Final presentation sorting logic
    jobs = jobs.sort_values(by="match_score", ascending=False)
    
    # Added 'date_posted' back to file data presentation columns
    clean_jobs = jobs[["match_score", "date_posted", "title", "company", "job_url", "location", "ai_analysis"]]
    
    output_file = "ai_ranked_supply_chain_jobs.csv"
    clean_jobs.to_csv(output_file, index=False)
    
    print(f"\n🎉 Execution complete! Curated file generated: '{output_file}'")
    print("📈 Open your spreadsheet app to see highly targeted leads cleanly ranked by real fit!")

except Exception as e:
    print(f"⚠️ Script stopped unexpectedly due to an error: {e}")