import streamlit as st
import pandas as pd
import re
from jobspy import scrape_jobs
from google import genai
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

# 1. PAGE SETUP & STYLING
st.set_page_config(page_title="Supply Chain Job Finder", layout="wide", page_icon="🔍")

st.title("🔍 Supply Chain Job Intelligence & Fit Analyzer")
st.write("Evaluate live market requisitions or analyze custom job descriptions directly against your background.")

# 2. KEYWORD MANAGEMENT WITH STRATEGIC WEIGHTING
KEYWORDS_WEIGHTS = {
    "supply chain": 5,
    "procurement": 5,
    "buyer": 5,
    "purchasing": 5,
    "inventory": 4,
    "six sigma": 4,
    "green belt": 4,
    "rfid": 4,
    "forecasting": 3,
    "logistics": 3,
    "power bi": 3,
    "sql": 3,
    "sap": 3,
    "planning": 2,
    "analyst": 1
}

# 3. LINKEDIN CONNECTIONS CACHING (REFERRAL ENGINE)
@st.cache_data(ttl=86400)
def load_linkedin_connections():
    try:
        if os.path.exists("connections.csv"):
            df = pd.read_csv("connections.csv")
            if 'Company' in df.columns:
                return set(df['Company'].dropna().str.lower().str.strip().unique())
        return set()
    except Exception:
        return set()

# 4. LOCAL HEURISTICS & PRE-FILTERING (ZERO API TOKENS)
def extract_ats_keywords(job_description):
    if not job_description:
        return []
    desc_lower = str(job_description).lower()
    return list(set([kw for kw, weight in KEYWORDS_WEIGHTS.items() if kw in desc_lower and weight >= 3]))

def evaluate_local_heuristics(title, description, resume_text):
    title_lower = str(title).lower()
    desc_lower = str(description).lower()
    combined_text = f"{title_lower} {desc_lower}"
    
    flags = []
    hard_blocker = False
    
    # 1. Clearance & Citizenship Flags
    if "security clearance" in combined_text or "clearance required" in combined_text or "secret clearance" in combined_text:
        flags.append("🔴 [Clearance Required]")
        hard_blocker = True
    if "us citizen" in combined_text or "u.s. citizen" in combined_text or "green card required" in combined_text or "must be a u.s. citizen" in combined_text:
        flags.append("🔴 [US Citizen Flag]")
        hard_blocker = True
        
    # 2. Executive / Senior Title Check
    senior_titles = ["senior manager", "sr. manager", "director", "vp", "vice president", "head of", "principal"]
    is_senior_title = any(st_word in title_lower for st_word in senior_titles)
    
    # 3. Years of Experience Check via Regex
    yoe_matches = re.findall(r'(\d+)\+?\s*(?:to|-)\s*(\d+)?\s*years?', desc_lower)
    max_yoe_found = 0
    for match in yoe_matches:
        try:
            val = int(match[0])
            if val > max_yoe_found:
                max_yoe_found = val
        except ValueError:
            pass
            
    is_senior_yoe = max_yoe_found >= 5
    
    # 4. Keyword Match Ratio
    job_keywords = extract_ats_keywords(description)
    resume_lower = str(resume_text).lower()
    found_kw = [kw for kw in job_keywords if kw in resume_lower]
    missing_kw = [kw for kw in job_keywords if kw not in resume_lower]
    
    if hard_blocker:
        local_score = 1
    elif is_senior_title or is_senior_yoe:
        local_score = 2
    elif job_keywords:
        local_score = int((len(found_kw) / len(job_keywords)) * 10)
        local_score = max(3, min(9, local_score))
    else:
        local_score = 5
        
    return local_score, flags, found_kw, missing_kw, hard_blocker, is_senior_title or is_senior_yoe

# 5. DATA INGESTION PIPELINE (CACHED)
@st.cache_data(ttl=3600)
def fetch_and_clean_jobs(search_term, location, num_results, proxies_list=None):
    try:
        raw_jobs = scrape_jobs(
            site_name=["linkedin", "indeed", "google"],
            search_term=search_term,
            google_search_term=f"{search_term} in {location}",
            location=location,
            results_wanted=num_results, 
            hours_old=24,
            country_only="USA",
            linkedin_fetch_description=True,
            proxies=proxies_list if proxies_list else None
        )
        if raw_jobs.empty:
            return pd.DataFrame()
            
        clean_df = raw_jobs.drop_duplicates(subset=["title", "company"]).copy()
        clean_df["description"] = clean_df["description"].fillna("").astype(str)
        return clean_df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Search Error: {e}")
        return pd.DataFrame()

# 6. SIDEBAR SETTINGS
st.sidebar.header("Pipeline Settings")
search_term = st.sidebar.text_input("Job Title", value="Supply Chain Analyst")
location = st.sidebar.text_input("Location", value="Boston, MA")
num_results = st.sidebar.slider("Jobs to Scrape (Stage 1)", 20, 60, 40)
top_ai_count = st.sidebar.slider("Top Jobs for Deep AI Evaluation (Stage 2)", 5, 15, 10)

with st.sidebar.expander("⚙️ Advanced Settings"):
    api_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    model_choice = st.selectbox("Reasoning Engine", ["gemini-2.5-flash", "gemini-2.5-flash-lite"])
    proxies_input = st.text_area("Rotating Proxies (Optional)", value="", placeholder="http://user:pass@host:port")

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

linkedin_connections = load_linkedin_connections()

# 7. ON-DEMAND DIALOG CO-PILOTS
@st.dialog("✨ Tailor Resume Bullet (Verified Grounding)")
def open_bullet_generator(job_title, company, job_description, missing_keywords):
    st.write(f"**Target Role:** {job_title} at **{company}**")
    st.write("---")
    
    if missing_keywords:
        st.warning(f"**Identified Keywords / Skill Gaps:** {', '.join(missing_keywords)}")
    else:
        st.info("Your background aligns well with the keywords detected for this role.")
        
    st.write("#### 💬 Experience Verification")
    st.write("Did you complete projects or tasks related to these gaps during coursework, Capstone, or past roles?")
    
    user_notes = st.text_area(
        "Your raw context / notes:",
        placeholder="e.g., 'Used regression forecasting in BU supply chain classes' or 'Created supplier scorecards at Flextronics' (Leave blank if none)",
        height=80
    )
    
    if st.button("Generate Tailored Bullet", use_container_width=True):
        if not api_key:
            st.error("Please provide an API Key under Advanced Settings.")
            return
            
        with st.spinner("Drafting authentic, evidence-backed bullet..."):
            try:
                client = genai.Client(api_key=api_key)
                prompt = f"""
                You are a senior supply chain recruiter tailoring a single resume bullet point for a candidate.
                
                Candidate Profile:
                {MY_PROFILE}
                
                Target Job: {job_title} at {company}
                Job Excerpt: {str(job_description)[:1500]}
                Job Gaps: {', '.join(missing_keywords)}
                Candidate Notes: "{user_notes.strip() if user_notes.strip() else 'No direct experience with gaps. Do NOT fabricate.'}"
                
                RULES:
                1. If candidate provided notes, translate that exact experience into a high-impact bullet using the target job's terminology.
                2. If notes are empty, generate a transferable skills bullet using ONLY verified profile accomplishments (SKU optimization, vendor management, Green Belt).
                3. DO NOT hallucinate tools, metrics, or responsibilities.
                4. Length: Exactly 1 bullet starting with a strong past-tense action verb.
                
                Return raw text only.
                """
                
                response = client.models.generate_content(model=model_choice, contents=prompt)
                st.write("---")
                st.success("✅ **Tailored Bullet:**")
                st.code(response.text.strip(), language="text")
                st.caption("Grounded exclusively in your actual background and explicit notes.")
            except Exception as e:
                st.error(f"Generation failed: {e}")

@st.dialog("📝 Application Portal Assistant")
def open_answer_assistant(job_title, company, job_description):
    st.write(f"**Target Role:** {job_title} at **{company}**")
    app_question = st.text_area("Paste the application question here:", placeholder="e.g., Why are you the right fit for this role?", height=100)
    
    if st.button("Generate Answer", use_container_width=True):
        if not app_question.strip():
            st.warning("Please enter a question first.")
            return
        if not api_key:
            st.error("Please provide an API Key under Advanced Settings.")
            return
            
        with st.spinner("Drafting humanized response..."):
            try:
                client = genai.Client(api_key=api_key)
                prompt = f"""
                Draft a short answer response for a job application web form.
                
                Candidate Profile: {MY_PROFILE}
                Target Job: {job_title} at {company}
                Context: {str(job_description)[:1500]}
                Application Question: "{app_question}"
                
                RULES:
                - Exactly 2 to 3 concise sentences.
                - Authentic, direct, professional tone.
                - ZERO AI buzzwords (no 'thrilled', 'delve', 'dynamic', 'passionate', 'testament').
                - Connect candidate's MS in SCM, Green Belt, or Flextronics procurement background to the position's requirements.
                
                Return raw text only.
                """
                response = client.models.generate_content(model=model_choice, contents=prompt)
                st.write("---")
                st.success("📝 **Copy-Paste Ready Answer:**")
                st.text_area("Result:", value=response.text.strip(), height=120, label_visibility="collapsed")
            except Exception as err:
                st.error(f"Error: {err}")

# 8. TAB NAVIGATION
tab_search, tab_custom = st.tabs(["🔍 Live Job Search Pipeline", "📄 Custom Job / JD Fit Analyzer"])

# ============================================================================
# TAB 1: LIVE SCRAPER & 2-STAGE PIPELINE
# ============================================================================
with tab_search:
    if st.sidebar.button("🚀 Find & Analyze Jobs", use_container_width=True, type="primary"):
        with st.spinner("Stage 1: Scraping job boards & applying local heuristic filters..."):
            processed_proxies = [p.strip() for p in proxies_input.split(",") if p.strip()] if proxies_input else None
            jobs_df = fetch_and_clean_jobs(search_term, location, num_results, processed_proxies)
            
            if jobs_df.empty:
                st.sidebar.warning("No jobs found matching your criteria.")
            else:
                jobs_df['match_score'] = 5
                jobs_df['ai_evaluated'] = False
                jobs_df['strengths'] = ""
                jobs_df['gaps'] = ""
                jobs_df['gap_severity'] = 1
                jobs_df['resume_bullet'] = ""
                jobs_df['flags'] = ""
                jobs_df['has_referral'] = False
                jobs_df['found_keywords'] = ""
                jobs_df['missing_keywords'] = ""
                
                for idx, row in jobs_df.iterrows():
                    local_score, flags, found_kw, missing_kw, hard_block, senior_block = evaluate_local_heuristics(
                        row.get('title', ''), row.get('description', ''), MY_PROFILE
                    )
                    jobs_df.at[idx, 'match_score'] = local_score
                    jobs_df.at[idx, 'flags'] = " ".join(flags)
                    jobs_df.at[idx, 'found_keywords'] = ", ".join(found_kw) if found_kw else "None"
                    jobs_df.at[idx, 'missing_keywords'] = ", ".join(missing_kw) if missing_kw else "None"
                    jobs_df.at[idx, 'strengths'] = f"Matching skills: {', '.join(found_kw[:4])}" if found_kw else "Title aligns with focus."
                    
                    company_clean = str(row.get('company', '')).lower().strip()
                    if any(company_clean in conn for conn in linkedin_connections) or any(conn in company_clean for conn in linkedin_connections):
                        jobs_df.at[idx, 'has_referral'] = True

                eligible_for_ai = jobs_df[
                    (jobs_df['description'].str.strip().str.len() > 50) & 
                    (~jobs_df['flags'].str.contains("Clearance|Citizen", case=False, na=False)) &
                    (jobs_df['match_score'] >= 3)
                ].sort_values(by='match_score', ascending=False).head(top_ai_count)
                
                if not eligible_for_ai.empty and api_key:
                    st.info(f"Stage 2: Running deep qualitative AI analysis on top {len(eligible_for_ai)} target opportunities...")
                    try:
                        ai_client = genai.Client(api_key=api_key)
                        batch_payload = []
                        for row in eligible_for_ai.itertuples():
                            truncated_desc = str(row.description)[:1200].replace('\n', ' ')
                            batch_payload.append({
                                "job_index": int(row.Index),
                                "title": str(row.title),
                                "company": str(row.company),
                                "description": truncated_desc
                            })

                        prompt = f"""
                        You are an expert executive talent acquisition advisor matching a candidate against job requisitions.
                        Candidate Profile:
                        {MY_PROFILE}
                        
                        SENIORITY & EXPERIENCE FIT CRITERIA:
                        - Candidate has 2 years of corporate procurement experience + MS SCM degree (Target: 0-3 years experience roles).
                        - If the job explicitly requires 5+ years or executive leadership, penalize score to 1-3.
                        - If the role matches entry/mid analyst level (0-3 yrs) with supply chain/procurement/inventory skills, score 7-10.
                        
                        Jobs Batch JSON:
                        {json.dumps(batch_payload, indent=2)}
                        
                        Return a raw JSON array of objects with:
                        - 'job_index': int
                        - 'score': int (1-10)
                        - 'strengths': short comma-separated list of 3-4 top matching competencies
                        - 'gaps': concise explanation of actual experience/seniority gaps
                        - 'gap_severity': int from 1 (minor) to 5 (critical blocker)
                        - 'resume_bullet': one suggested resume bullet angle
                        """
                        
                        response = ai_client.models.generate_content(
                            model=model_choice,
                            contents=prompt,
                            config={'response_mime_type': 'application/json'}
                        )
                        
                        results = json.loads(response.text.strip())
                        for item in results:
                            raw_idx = item.get("job_index")
                            if raw_idx is not None:
                                j_idx = int(raw_idx)
                                if j_idx in jobs_df.index:
                                    jobs_df.at[j_idx, 'match_score'] = int(item.get("score", 5))
                                    jobs_df.at[j_idx, 'strengths'] = str(item.get("strengths", ""))
                                    jobs_df.at[j_idx, 'gaps'] = str(item.get("gaps", ""))
                                    jobs_df.at[j_idx, 'gap_severity'] = int(item.get("gap_severity", 1))
                                    jobs_df.at[j_idx, 'resume_bullet'] = str(item.get("resume_bullet", ""))
                                    jobs_df.at[j_idx, 'ai_evaluated'] = True
                    except Exception as ai_err:
                        st.warning(f"AI Deep Analysis skipped: {ai_err}. Local rankings displayed.")
                        
                st.session_state.jobs_df = jobs_df.sort_values(by="match_score", ascending=False).reset_index(drop=True)
                st.success("Search complete!")

    if 'jobs_df' in st.session_state:
        master_df = st.session_state.jobs_df
        
        for col, default_val in [
            ('match_score', 5), ('ai_evaluated', False), ('has_referral', False), 
            ('flags', ''), ('strengths', ''), ('gaps', ''), ('gap_severity', 1),
            ('resume_bullet', ''), ('missing_keywords', 'None'), ('found_keywords', 'None')
        ]:
            if col not in master_df.columns:
                master_df[col] = default_val

        st.write("### 🎛️ Dashboard Filters")
        filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
        
        with filter_col1:
            min_score = st.slider("Minimum Strategic Fit Score", 1, 10, 5)
        with filter_col2:
            platform_options = ["All Platforms"] + sorted(list(master_df['site'].str.replace("_jobs", "").str.capitalize().unique())) if 'site' in master_df.columns else ["All Platforms"]
            selected_source = st.selectbox("Job Platform", options=platform_options)
        with filter_col3:
            st.write("")
            st.write("")
            show_referrals_only = st.checkbox("Show Referrals Only 🤝", value=False)
            
        filtered_df = master_df[master_df["match_score"] >= min_score].copy()
        if selected_source != "All Platforms" and 'site' in master_df.columns:
            filtered_df = filtered_df[filtered_df["site"].str.replace("_jobs", "").str.capitalize() == selected_source]
        if show_referrals_only:
            filtered_df = filtered_df[filtered_df["has_referral"] == True]
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jobs Scraped (Stage 1)", len(master_df))
        m2.metric("AI Evaluated (Stage 2)", int((master_df['ai_evaluated'] == True).sum()))
        m3.metric("Referrals Available", int((master_df['has_referral'] == True).sum()))
        m4.metric("Filtered Opportunities", len(filtered_df))
        
        st.write("### 💼 Evaluated Opportunities")
        if filtered_df.empty:
            st.info("No jobs match your current filter preferences. Try lowering the Minimum Strategic Fit Score.")
        else:
            for idx, row in filtered_df.iterrows():
                with st.container(border=True):
                    info_col, action_col = st.columns([4, 2])
                    
                    min_sal, max_sal = row.get('min_amount'), row.get('max_amount')
                    interval = row.get('interval', 'yearly')
                    if pd.notna(min_sal) or pd.notna(max_sal):
                        if min_sal == max_sal: salary_str = f"${min_sal:,.0f}"
                        elif pd.notna(min_sal) and pd.notna(max_sal): salary_str = f"${min_sal:,.0f} - ${max_sal:,.0f}"
                        elif pd.notna(min_sal): salary_str = f"${min_sal:,.0f}+"
                        else: salary_str = f"Up to ${max_sal:,.0f}"
                        if pd.notna(interval) and interval: salary_str += f" ({interval})"
                    else:
                        salary_str = "Not listed"
                        
                    flags_text = f" {row.get('flags', '')}" if row.get('flags') else ""
                    if row.get('has_referral'):
                        flags_text += " 🤝 [Referral Available]"
                        
                    score_val = int(row.get('match_score', 5))
                    star_rating = "⭐" * max(1, min(5, round(score_val / 2)))
                    ai_badge = " [🤖 Deep Evaluated]" if row.get('ai_evaluated') else " [⚡ Local Ranked]"
                    
                    with info_col:
                        st.subheader(f"{row.get('title', 'Position')}{flags_text} — {score_val}/10 {star_rating}")
                        site_str = str(row.get('site', 'web')).replace('_jobs', '').capitalize()
                        st.write(f"🏢 **{row.get('company', 'Unknown')}** | 📍 {row.get('location', 'N/A')} | 📂 Source: {site_str} | 📅 Posted: {row.get('date_posted', 'Recent')} | 💰 Salary: {salary_str}{ai_badge}")
                        
                        if row.get('strengths'):
                            st.write(f"**Top Strengths / Match:** {row.get('strengths')}")
                            
                    with action_col:
                        st.link_button("🔗 Apply Now", row.get('job_url', '#'), use_container_width=True)
                        
                        raw_missing = str(row.get('missing_keywords', ''))
                        missing_list = [k.strip() for k in raw_missing.split(",") if k.strip() and k.strip() != "None"]
                        
                        if st.button("✨ Tailor Resume Bullet", key=f"tailor_{idx}", use_container_width=True):
                            open_bullet_generator(row.get('title', ''), row.get('company', ''), row.get('description', ''), missing_list)
                        if st.button("📝 Draft Portal Answer", key=f"ans_{idx}", use_container_width=True):
                            open_answer_assistant(row.get('title', ''), row.get('company', ''), row.get('description', ''))
                    
                    with st.expander("📋 View Fit Analysis & Full Job Description"):
                        det1, det2 = st.columns(2)
                        severity_num = int(row.get('gap_severity', 1))
                        
                        if severity_num >= 4:
                            sev_label = f"⚠️ Red Flag Level: {severity_num}/5 (Seniority/Experience Gap)"
                        elif severity_num >= 3:
                            sev_label = f"🟡 Red Flag Level: {severity_num}/5 (Moderate Gap)"
                        else:
                            sev_label = f"✅ Red Flag Level: {severity_num}/5 (Minor - Great Fit)"
                            
                        with det1:
                            if row.get('gaps'):
                                st.error(f"**Gaps & Experience Evaluation:**\n\n*{sev_label}*\n\n{row.get('gaps')}")
                            else:
                                st.info(f"**Missing Keywords Identified Locally:**\n{row.get('missing_keywords', 'None')}")
                        with det2:
                            if row.get('resume_bullet'):
                                st.info(f"**AI Strategy Pivot Point:**\n*\"{row.get('resume_bullet')}\"*")
                            else:
                                st.write("Use the **Tailor Resume Bullet** button above to generate a grounded bullet.")
                                
                        st.write("---")
                        st.write("**Full Source Job Description:**")
                        desc_text = str(row.get('description', '')).strip()
                        if desc_text in ["", "nan"]:
                            st.warning("Full description restricted by job board security. Click 'Apply Now' to view details on the company site.")
                        else:
                            st.write(desc_text)
    else:
        st.info("Click '🚀 Find & Analyze Jobs' in the sidebar to scrape and evaluate positions.")

# ============================================================================
# TAB 2: CUSTOM JOB / JD FIT ANALYZER (DIRECT PASTE)
# ============================================================================
with tab_custom:
    st.write("### 📄 Analyze Any Job Description Directly")
    st.write("Found a role on a company site or in an email? Paste the details below for an instant fit score, clearance check, and tailored resume bullet.")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        custom_title = st.text_input("Job Title", value="Procurement Specialist / Analyst", placeholder="e.g. Supply Chain Analyst")
    with col_c2:
        custom_company = st.text_input("Company Name", value="Sensirion", placeholder="e.g. Boston Scientific")
        
    custom_jd = st.text_area(
        "Job Description Text:",
        placeholder="Paste the full job requirements, qualifications, and role responsibilities here...",
        height=250
    )
    
    if st.button("📊 Evaluate Custom Job Fit", use_container_width=True, type="primary"):
        if not custom_jd.strip():
            st.warning("Please paste a job description first.")
        else:
            with st.spinner("Running local heuristics and AI evaluation..."):
                # 1. Local Heuristics Check
                local_score, flags, found_kw, missing_kw, hard_block, senior_block = evaluate_local_heuristics(
                    custom_title, custom_jd, MY_PROFILE
                )
                
                # Check Referral
                company_clean = custom_company.lower().strip()
                has_referral = any(company_clean in conn for conn in linkedin_connections) or any(conn in company_clean for conn in linkedin_connections)
                
                ai_score = local_score
                strengths = f"Matching skills: {', '.join(found_kw[:4])}" if found_kw else "Alignment with core domain."
                gaps = "Local heuristic evaluation only (API key not provided)."
                gap_severity = 1
                resume_bullet = "Review qualifications on the target company portal."
                
                # 2. Deep Gemini Fit Analysis
                if api_key:
                    try:
                        ai_client = genai.Client(api_key=api_key)
                        prompt = f"""
                        You are an expert executive talent acquisition advisor matching a candidate against a job requisition.
                        
                        Candidate Profile:
                        {MY_PROFILE}
                        
                        Target Job: {custom_title} at {custom_company}
                        Job Description Text:
                        {custom_jd[:2500]}
                        
                        SENIORITY & EXPERIENCE FIT CRITERIA:
                        - Candidate has 2 years of corporate procurement analytics experience + MS SCM degree (Target: 0-3 years experience roles).
                        - If the job explicitly requires 5+ years or executive leadership, penalize score to 1-3.
                        - If the role matches entry/mid analyst level (0-3 yrs) with supply chain/procurement/inventory skills, score 7-10.
                        
                        Return a raw JSON object (no markdown):
                        {{
                            "score": int (1-10),
                            "strengths": "short comma-separated list of 3-4 top matching competencies",
                            "gaps": "concise explanation of actual experience/seniority gaps",
                            "gap_severity": int (1 to 5),
                            "resume_bullet": "one suggested resume bullet angle"
                        }}
                        """
                        response = ai_client.models.generate_content(
                            model=model_choice,
                            contents=prompt,
                            config={'response_mime_type': 'application/json'}
                        )
                        res_json = json.loads(response.text.strip())
                        ai_score = int(res_json.get("score", local_score))
                        strengths = str(res_json.get("strengths", strengths))
                        gaps = str(res_json.get("gaps", gaps))
                        gap_severity = int(res_json.get("gap_severity", 1))
                        resume_bullet = str(res_json.get("resume_bullet", resume_bullet))
                    except Exception as err:
                        st.warning(f"AI evaluation failed: {err}. Displaying local heuristic metrics.")
                        
                # Display Results Card
                st.write("---")
                st.write("### 🎯 Analysis Results")
                
                flags_text = f" {' '.join(flags)}" if flags else ""
                if has_referral:
                    flags_text += " 🤝 [Referral Available]"
                    
                star_rating = "⭐" * max(1, min(5, round(ai_score / 2)))
                
                with st.container(border=True):
                    c_info, c_act = st.columns([4, 2])
                    
                    with c_info:
                        st.subheader(f"{custom_title}{flags_text} — {ai_score}/10 {star_rating}")
                        st.write(f"🏢 **{custom_company}** | 📊 Analysis: **{'AI Deep Evaluated' if api_key else 'Local Heuristic'}**")
                        st.write(f"**Top Strengths / Match:** {strengths}")
                        if found_kw:
                            st.write(f"**Matched Keywords:** {' • '.join([f'✓ {k}' for k in found_kw])}")
                            
                    with c_act:
                        if st.button("✨ Tailor Resume Bullet", key="custom_tailor_btn", use_container_width=True):
                            open_bullet_generator(custom_title, custom_company, custom_jd, missing_kw)
                        if st.button("📝 Draft Portal Answer", key="custom_ans_btn", use_container_width=True):
                            open_answer_assistant(custom_title, custom_company, custom_jd)
                            
                    st.write("---")
                    det_col1, det_col2 = st.columns(2)
                    
                    if gap_severity >= 4:
                        sev_label = f"⚠️ Red Flag Level: {gap_severity}/5 (Seniority/Experience Gap)"
                    elif gap_severity >= 3:
                        sev_label = f"🟡 Red Flag Level: {gap_severity}/5 (Moderate Gap)"
                    else:
                        sev_label = f"✅ Red Flag Level: {gap_severity}/5 (Minor - Great Fit)"
                        
                    with det_col1:
                        st.error(f"**Gaps & Experience Evaluation:**\n\n*{sev_label}*\n\n{gaps}")
                    with det_col2:
                        st.info(f"**Suggested Strategy Pivot Point:**\n*\"{resume_bullet}\"*")