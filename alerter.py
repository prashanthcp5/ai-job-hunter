import os
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import pandas as pd
from jobspy import scrape_jobs
from google import genai
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
SEARCH_TERM = "Supply Chain"
LOCATION = "Boston, MA"
MIN_ALERT_SCORE = 7  # Only email opportunities meeting or exceeding this threshold
SENDER_EMAIL = os.getenv("ALERT_EMAIL_SENDER")
SENDER_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD")
RECIPIENT_EMAIL = os.getenv("ALERT_EMAIL_RECIPIENT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Complete, unabridged candidate profile matching interface.py
MY_PROFILE = """
PREETHI SRINIVASAN 
Boston, MA | (617) 893-8446 | preesrini99@gmail.com | https://www.linkedin.com/in/preethi-srinivasan-8513221aa/ 

SUMMARY
Supply Chain and Procurement professional with an MS in Supply Chain Management, a Lean Six Sigma Green Belt, and two years of corporate experience as a Senior Procurement Analyst. Expert in workflow optimization, strategic sourcing, and inventory management. Skilled in leveraging data analytics to drive operational efficiency.

EDUCATION
- Master of Science in Supply Chain Management 
  Boston University, Boston, Massachusetts, USA 
  Certifications: Lean Six Sigma Green Belt | Timeline: Jan 2025 - May 2026 
  Relevant Coursework: Global Supply Chains, Enterprise Risk Management, Quantitative and Qualitative Decision Making, Six Sigma Quality Methods
- Bachelor of Engineering in Electronics and Instrumentation 
  Anna University, Chennai, Tamil Nadu, India | Timeline: Aug 2017 - April 2021

EXPERIENCE
- Capstone Consultant
  American Surgical Company, Beverly, MA, USA | Jan 2026 - April 2026
  * Modernizing operational efficiency for a medical device manufacturer by designing scalable inventory systems, evaluating RFID technology and ERP integration capabilities
  * Optimized restocking levels for 200-300 SKUs.
  * Developed streamlined sampling workflows to ensure 100% quality assurance compliance.

- Graduate Assistant (Research & Teaching) 
  Boston University, Boston, MA, USA | Sep 2025 - May 2026
  * Research: Engineered a decision-support model and intuitive data visualizations that used a "willing to pay" framework to assist hospital administrators in their evaluation of ancillary service options, with the resulting methodology presented at the Northeast Decision Sciences Institute (NEDSI) 2026 Conference.
  * Teaching: Facilitated the graduate-level 'Global Supply Chains' course for 35+ students, delivering targeted feedback to enhance mastery of logistics and strategic supply chain management.

- Senior Procurement Analyst
  Flextronics Technologies Pvt. Ltd, Chennai, India | Oct 2021 - Oct 2023 
  * Implemented a strategic vendor management system using performance metrics to maintain high on-time delivery and operational continuity. 
  * Utilized data analysis to support strategic negotiations, resulting in sustained delivery performance and supplier reliability. 
  * Reduced material lead times and streamlined procurement workflows by conducting comprehensive root-cause analyses to resolve service and delivery bottlenecks. 
  * Managed all procurement duties until the shipment reached production, collaborating with internal teams and external vendors to resolve complex issues and ensure data accuracy. 

SKILLS
- Procurement & Sourcing: Strategic Sourcing Analysis, Purchasing, Vendor Management, Supplier Relationship Management 
- Supply Chain & Operations: Inventory Optimization, Forecasting, Workflow Optimization, Project Management 
- Systems & Technology: ERP Integration, RFID System Evaluation, Item Serialization, MS Excel (Advanced: Pivot Tables, VLOOKUPS, Macros, etc.) 
- Certification: Lean Six Sigma Green Belt

TARGET ROLES
Supply Chain Analyst, Procurement Analyst, Sourcing Specialist, Inventory Analyst, Procurement Specialist.
"""

# --- DATABASE DEDUPLICATION ---
def init_db():
    conn = sqlite3.connect("seen_jobs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sent_jobs (
            job_url TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            score INTEGER,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def is_job_seen(conn, job_url):
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM sent_jobs WHERE job_url = ?", (job_url,))
    return cursor.fetchone() is not None

def mark_job_sent(conn, job_url, title, company, score):
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO sent_jobs (job_url, title, company, score) VALUES (?, ?, ?, ?)",
                   (job_url, title, company, score))
    conn.commit()

# --- EMAIL DISPATCHER ---
def send_email_digest(matches):
    if not matches:
        print("No new high-scoring matches to dispatch.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🎯 {len(matches)} New High-Fit Supply Chain Roles Found"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL

    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.5;">
        <h2 style="color: #0b5394;">🎯 New Supply Chain Job Matches (Score &ge; {MIN_ALERT_SCORE}/10)</h2>
        <p>Here are the latest roles matching your background extracted during the recent run:</p>
        <hr style="border: 0; border-top: 1px solid #ccc;" />
    """

    for job in matches:
        html_content += f"""
        <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #fafafa;">
            <h3 style="margin-top: 0; color: #111;">{job['title']} — <span style="color: #27ae60;">{job['score']}/10</span></h3>
            <p style="margin: 5px 0;"><strong>Company:</strong> {job['company']} | <strong>Location:</strong> {job['location']} | <strong>Source:</strong> {job['site']}</p>
            <p style="margin: 5px 0;"><strong>Top Match Reasons:</strong> {job['strengths']}</p>
            <p style="margin: 5px 0; color: #c0392b;"><strong>Identified Gaps:</strong> {job['gaps']}</p>
            <div style="margin-top: 10px;">
                <a href="{job['job_url']}" style="background-color: #0066cc; color: #fff; padding: 8px 14px; text-decoration: none; border-radius: 4px; display: inline-block;">Apply Now &rarr;</a>
            </div>
        </div>
        """

    html_content += "</body></html>"
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
    print(f"Successfully emailed {len(matches)} opportunities to {RECIPIENT_EMAIL}")

# --- MAIN EXECUTION PIPELINE ---
def run_pipeline():
    conn = init_db()
    print("Fetching active listings...")
    
    raw_jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "google"],
        search_term=SEARCH_TERM,
        google_search_term=f"{SEARCH_TERM} in {LOCATION}",
        location=LOCATION,
        results_wanted=40,
        hours_old=24,
        country_only="USA",
        linkedin_fetch_description=True
    )

    if raw_jobs.empty:
        print("No jobs found in this cycle.")
        return

    unseen_jobs = []
    for row in raw_jobs.itertuples():
        if not is_job_seen(conn, row.job_url):
            unseen_jobs.append(row)

    if not unseen_jobs:
        print("All scraped roles have already been evaluated.")
        return

    unseen_df = pd.DataFrame(unseen_jobs)
    unseen_df["description"] = unseen_df["description"].fillna("").astype(str)
    
    # Pre-filter out clearance, citizenship blockers, and empty listings
    eval_pool = unseen_df[
        (unseen_df["description"].str.strip().str.len() > 50) &
        (~unseen_df["description"].str.contains("security clearance|us citizen|u.s. citizen|green card required", case=False))
    ].head(10)

    if eval_pool.empty or not GEMINI_API_KEY:
        print("No eligible unblocked descriptions ready for evaluation.")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    payload = [{
        "job_index": int(row.Index),
        "title": str(row.title),
        "company": str(row.company),
        "description": str(row.description)[:1200]
    } for row in eval_pool.itertuples()]

    prompt = f"""
    You are an executive talent acquisition advisor matching a candidate against job requisitions.
    
    Candidate Profile:
    {MY_PROFILE}
    
    SENIORITY & EXPERIENCE FIT CRITERIA:
    - Candidate has 2 years corporate procurement experience + MS SCM degree (Target: 0-3 years experience roles).
    - If the job explicitly requires 5+ years or executive leadership, penalize score to 1-3.
    - If the role matches entry/mid analyst level (0-3 yrs) with supply chain/procurement/inventory skills, score 7-10.
    
    Jobs Batch JSON:
    {json.dumps(payload, indent=2)}
    
    Return a raw JSON array of objects with:
    - 'job_index': int
    - 'score': int (1-10)
    - 'strengths': short comma-separated list of 3-4 top matching competencies
    - 'gaps': concise explanation of actual experience/seniority gaps
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={'response_mime_type': 'application/json'}
    )
    
    results = json.loads(response.text.strip())
    matched_to_send = []

    for res in results:
        j_idx = res.get("job_index")
        score = int(res.get("score", 0))
        target_row = eval_pool.loc[j_idx]
        
        mark_job_sent(conn, target_row["job_url"], target_row["title"], target_row["company"], score)

        if score >= MIN_ALERT_SCORE:
            matched_to_send.append({
                "title": target_row["title"],
                "company": target_row["company"],
                "location": target_row["location"],
                "site": str(target_row["site"]).replace("_jobs", "").capitalize(),
                "job_url": target_row["job_url"],
                "score": score,
                "strengths": res.get("strengths", ""),
                "gaps": res.get("gaps", "")
            })

    send_email_digest(matched_to_send)

if __name__ == "__main__":
    run_pipeline()