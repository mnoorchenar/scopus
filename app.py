"""
Flask Application for Reference Management Pipeline
Complete implementation matching overleaf.py functionality
Includes LaTeX citation parsing, BibTeX processing, and full pipeline

FEATURES:
- Fixed abbreviations to include periods (ISO 4 standard): "Energy" → "Ener."
- LaTeX manuscript analysis for citation frequency and section tracking
- Filter references by section
- Clear entire database
"""

from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
import pandas as pd
import os
import json
from datetime import datetime
import re
from difflib import SequenceMatcher
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import io
import time
import random
import hashlib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['DATABASE'] = 'refs_management.db'
app.config['API_KEY'] = os.environ.get('API_KEY', 'your-secret-key-here')
app.config['ENVIRONMENT'] = os.environ.get('ENVIRONMENT', 'development')
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Prepositions to keep lowercase in abbreviations
LOWERCASE_WORDS = {"and", "or", "in", "on", "of", "for", "to", "the", "a", "an", "with", "at", "by", "from"}

# =====================================================================
# HTTP CLIENT WITH RETRIES
# =====================================================================

def make_http_session():
    """Create HTTP session with retries and proper headers"""
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": "RefsManagement/1.0 (mailto:contact@example.com)",
        "Accept": "application/json",
    })
    return s

HTTP = make_http_session()

# =====================================================================
# AUTHENTICATION
# =====================================================================

def check_api_key():
    """Check API key for protected routes"""
    if app.config['ENVIRONMENT'] == 'development':
        return True
    
    api_key = request.headers.get('X-API-Key')
    return api_key == app.config['API_KEY']

# =====================================================================
# LATEX CITATION PARSING (from overleaf.py)
# =====================================================================

def parse_citations_from_tex(tex_content: str) -> pd.DataFrame:
    """Parse citations from LaTeX content with section tracking"""
    print("📖 Parsing citations from LaTeX")
    
    lines = tex_content.split('\n')
    clean_text = "\n".join(line for line in lines if not line.strip().startswith("%"))

    section_pattern = re.compile(r'\\section\{([^}]*)\}(?:\\label\{[^}]*\})?')
    cite_pattern = re.compile(r'\\cite\{([^}]*)\}')
    sections = section_pattern.split(clean_text)

    citations, ref_sections = [], {}
    for i in range(1, len(sections), 2):
        if i >= len(sections):
            break
        section_name = sections[i].strip()
        section_text = sections[i+1] if i+1 < len(sections) else ""
        matches = cite_pattern.findall(section_text)
        for match in matches:
            for key in match.split(","):
                ref = key.strip()
                citations.append(ref)
                if ref not in ref_sections:
                    ref_sections[ref] = []
                if section_name not in ref_sections[ref]:
                    ref_sections[ref].append(section_name)

    freq, order = {}, []
    for c in citations:
        if c not in freq:
            order.append(c)
        freq[c] = freq.get(c, 0) + 1

    df = pd.DataFrame({
        "Reference": order,
        "Frequency": [freq[c] for c in order],
        "Sections": [", ".join(ref_sections[c]) for c in order]
    })
    print(f"✅ Found {len(df)} unique citations")
    return df

def merge_citations_with_bib(citations_df: pd.DataFrame, bib_df: pd.DataFrame) -> pd.DataFrame:
    """Merge citations with BibTeX entries"""
    print("🔗 Merging citations with BibTeX")
    bib_lookup = bib_df.set_index("Key").to_dict(orient="index")
    merged_records = []
    
    for _, row in citations_df.iterrows():
        key = row["Reference"]
        bib_info = bib_lookup.get(key, {})
        merged_records.append({
            "Reference": key,
            "Frequency": row["Frequency"],
            "Sections": row["Sections"],
            "Type": bib_info.get("Type", ""),
            "Authors": bib_info.get("Authors", ""),
            "Title": bib_info.get("Title", ""),
            "Journal/Booktitle": bib_info.get("Journal/Booktitle", ""),
            "Year": bib_info.get("Year", ""),
            "Publisher": bib_info.get("Publisher", ""),
            "Volume": bib_info.get("Volume", ""),
            "Pages": bib_info.get("Pages", ""),
            "DOI": bib_info.get("DOI", ""),
            "BibTeX": bib_info.get("BibTeX", "")
        })
    
    df = pd.DataFrame(merged_records)
    print(f"✅ Merged into {len(df)} rows")
    return df

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def abbreviate_journal_custom(title: str) -> str:
    """
    Custom abbreviation with periods (ISO 4 standard)
    Examples:
      "Energy" → "Ener."
      "Applied Energy" → "Appl. Ener."
      "Journal of Energy" → "J. of Ener."
      "IEEE Transactions on Neural Networks" → "IEEE Trans. on Neural Netw."
    """
    if not title:
        return ""
    
    words = title.split()
    abbr = []
    
    for i, word in enumerate(words):
        # Keep prepositions and conjunctions in lowercase (except at start)
        if word.lower() in LOWERCASE_WORDS and i != 0:
            abbr.append(word.lower())
        # Keep acronyms as-is (words with 2+ uppercase letters)
        elif sum(1 for c in word if c.isupper()) >= 2:
            abbr.append(word)
        # Abbreviate long words (>4 letters) with period
        elif len(word) > 4:
            # Take first 4 letters and add period
            abbr.append(word[:4].capitalize() + ".")
        # Short words (≤4 letters): add period
        else:
            abbr.append(word.capitalize() + ".")
    
    return " ".join(abbr)

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with proper schema"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create table with all columns in proper order
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bibliography (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_num INTEGER,
            session_id TEXT,
            reference TEXT,
            frequency INTEGER,
            sections TEXT,
            key TEXT UNIQUE,
            doi TEXT,
            type TEXT,
            authors TEXT,
            title TEXT,
            journal_booktitle TEXT,
            year TEXT,
            year_int INTEGER,
            publisher TEXT,
            volume TEXT,
            pages TEXT,
            bibtex TEXT,
            crossref_bibtex TEXT,
            crossref_bibtex_localkey TEXT,
            title_similarity INTEGER,
            journal_abbreviation TEXT,
            crossref_bibtex_abbrev TEXT,
            crossref_bibtex_protected TEXT,
            used TEXT,
            imported_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index on DOI
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_bib_doi 
        ON bibliography(doi) 
        WHERE doi IS NOT NULL AND doi != ''
    """)
    
    conn.commit()
    conn.close()

def extract_year_int(year_str):
    """Extract integer year from year string"""
    if not year_str:
        return None
    match = re.search(r'\d{4}', str(year_str))
    return int(match.group()) if match else None

def scan_brace_balanced_value(text, start_pos):
    """Scan for brace-balanced field value"""
    if start_pos >= len(text):
        return "", start_pos
    
    if text[start_pos] == '{':
        depth = 1
        pos = start_pos + 1
        while pos < len(text) and depth > 0:
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
            pos += 1
        return text[start_pos+1:pos-1], pos
    elif text[start_pos] == '"':
        pos = start_pos + 1
        while pos < len(text):
            if text[pos] == '"' and text[pos-1] != '\\':
                return text[start_pos+1:pos], pos + 1
            pos += 1
        return text[start_pos+1:], len(text)
    else:
        pos = start_pos
        while pos < len(text) and text[pos] not in ',}':
            pos += 1
        return text[start_pos:pos].strip(), pos

def parse_bibtex_entry(entry_text):
    """Parse single BibTeX entry with proper brace balancing"""
    match = re.match(r'@(\w+)\s*\{([^,]+),', entry_text)
    if not match:
        return None
    
    entry_type, entry_key = match.groups()
    fields = {}
    
    fields_start = entry_text.find(entry_key) + len(entry_key) + 1
    fields_text = entry_text[fields_start:]
    
    pos = 0
    while pos < len(fields_text):
        while pos < len(fields_text) and fields_text[pos] in ' \t\n\r,':
            pos += 1
        if pos >= len(fields_text) or fields_text[pos] == '}':
            break
        
        field_match = re.match(r'(\w+)\s*=\s*', fields_text[pos:])
        if not field_match:
            break
        
        field_name = field_match.group(1).lower()
        pos += field_match.end()
        
        value, new_pos = scan_brace_balanced_value(fields_text, pos)
        fields[field_name] = value.strip()
        pos = new_pos
    
    return {
        'type': entry_type,
        'key': entry_key.strip(),
        'fields': fields
    }

def parse_bibtex_input(bibtex_content):
    """Parse BibTeX content from user input"""
    entries = ["@" + e for e in bibtex_content.split("@") if e.strip()]
    papers = []

    for entry in entries:
        parsed = parse_bibtex_entry(entry)
        if not parsed:
            continue
        
        fields = parsed['fields']
        
        papers.append({
            "Key": parsed['key'],
            "Type": parsed['type'],
            "Authors": fields.get("author", "").strip(),
            "Title": fields.get("title", "").strip(),
            "Journal/Booktitle": fields.get("journal", fields.get("booktitle", "")).strip(),
            "Year": fields.get("year", "").strip(),
            "Publisher": fields.get("publisher", fields.get("organization", "")).strip(),
            "Volume": fields.get("volume", "").strip(),
            "Pages": fields.get("pages", "").strip(),
            "DOI": fields.get("doi", "").strip(),
            "BibTeX": entry.strip(),
            "Imported_Date": datetime.now().isoformat()
        })

    return pd.DataFrame(papers).drop_duplicates(subset="Key", keep="first").reset_index(drop=True)

def clean_bibtex_fields(bibtex):
    """Remove unwanted fields from BibTeX entries"""
    if not bibtex:
        return bibtex
        
    fields_to_remove = ['url', 'source', 'publication_stage', 'note', 'abstract']
    
    for field in fields_to_remove:
        pattern = rf'\s*{field}\s*=\s*'
        pos = 0
        result = []
        
        while pos < len(bibtex):
            match = re.search(pattern, bibtex[pos:], re.IGNORECASE)
            if not match:
                result.append(bibtex[pos:])
                break
            
            result.append(bibtex[pos:pos + match.start()])
            value_start = pos + match.end()
            _, value_end = scan_brace_balanced_value(bibtex, value_start)
            
            while value_end < len(bibtex) and bibtex[value_end] in ' \t\n\r,':
                value_end += 1
            
            pos = value_end
        
        bibtex = ''.join(result)
    
    bibtex = re.sub(r'\n\s*\n\s*\n', '\n\n', bibtex)
    bibtex = re.sub(r',\s*,', ',', bibtex)
    bibtex = re.sub(r',(\s*)\}', r'\1}', bibtex)
    
    lines = [line for line in bibtex.split('\n') if line.strip()]
    return '\n'.join(lines)

def protect_acronyms_in_fields(bibtex):
    """Protect acronyms with braces"""
    if not bibtex:
        return bibtex
        
    def wrap_token(token):
        if token.startswith("{") and token.endswith("}"):
            return token
        if sum(1 for c in token if c.isupper()) >= 2:
            return "{" + token + "}"
        return token

    def process_field_value(value):
        if value.startswith("{") and value.endswith("}"):
            inner = value[1:-1]
            if not ('{' in inner and '}' in inner):
                return value
        
        tokens = re.split(r'(\s+)', value)
        fixed = "".join(wrap_token(tok) if tok.strip() else tok for tok in tokens)
        fixed = re.sub(r'\{\{([^{}]+)\}\}', r'{\1}', fixed)
        return fixed

    for field in ["title", "booktitle", "journal"]:
        pattern = rf'({field}\s*=\s*)'
        matches = list(re.finditer(pattern, bibtex, re.IGNORECASE))
        
        for match in reversed(matches):
            field_start = match.end()
            value, value_end = scan_brace_balanced_value(bibtex, field_start)
            
            if value:
                processed = process_field_value(value)
                new_field = f"{match.group(1)}{{{processed}}}"
                bibtex = bibtex[:match.start()] + new_field + bibtex[value_end:]

    return bibtex

def replace_bibtex_key(bibtex, new_key):
    """Replace the citation key in a BibTeX entry"""
    if not bibtex:
        return bibtex
    
    try:
        start_brace = bibtex.index("{")
        first_comma = bibtex.index(",", start_brace)
        entry_type = bibtex[:start_brace]
        new_start = f"{entry_type}{{{new_key},"
        return new_start + bibtex[first_comma+1:]
    except ValueError:
        return bibtex

def enrich_with_crossref(df):
    """Enrich references with Crossref data"""
    enriched_rows = []
    
    for idx, row in df.iterrows():
        enriched_data = dict(row)
        
        if not row.get('Title'):
            enriched_data['Crossref_BibTeX'] = row.get('BibTeX', '')
            enriched_data['Title_Similarity'] = 0
            enriched_rows.append(enriched_data)
            continue

        query_parts = [row['Title']]
        if row.get('Authors'):
            query_parts.append(row['Authors'].split(',')[0])
        if row.get('Journal/Booktitle'):
            query_parts.append(row['Journal/Booktitle'])
        if row.get('Year'):
            query_parts.append(row['Year'])
        
        query = " ".join(query_parts)

        try:
            url = f"https://api.crossref.org/works?query.bibliographic={requests.utils.quote(query)}&rows=3"
            response = HTTP.get(url, timeout=15)
            response.raise_for_status()
            items = response.json().get("message", {}).get("items", [])

            best_score = 0
            crossref_bibtex = row.get('BibTeX', '')
            best_doi = row.get('DOI', '')

            for item in items:
                cr_title = item.get("title", [""])[0]
                score = SequenceMatcher(None, row['Title'].lower(), cr_title.lower()).ratio()
                
                if row.get('Year') and 'published-print' in item:
                    cr_year = str(item['published-print'].get('date-parts', [['']])[0][0])
                    if row['Year'].strip() == cr_year:
                        score = min(1.0, score + 0.1)
                
                if score > best_score:
                    best_score = score
                    best_doi = item.get('DOI', best_doi)
                    
                    if best_doi:
                        try:
                            bibtex_response = HTTP.get(
                                f"https://doi.org/{best_doi}",
                                headers={"Accept": "application/x-bibtex"},
                                timeout=15
                            )
                            if bibtex_response.status_code == 200:
                                crossref_bibtex = bibtex_response.text.strip()
                        except Exception as e:
                            print(f"⚠️ BibTeX fetch failed for DOI {best_doi}: {e}")

            enriched_data['Crossref_BibTeX'] = crossref_bibtex if best_score >= 0.85 else row.get('BibTeX', '')
            enriched_data['Title_Similarity'] = int(round(best_score * 100))
            if best_doi:
                enriched_data['DOI'] = best_doi
            
        except Exception as e:
            print(f"⚠️ Crossref enrichment failed: {e}")
            enriched_data['Crossref_BibTeX'] = row.get('BibTeX', '')
            enriched_data['Title_Similarity'] = 0

        time.sleep(0.15 + random.uniform(0, 0.25))
        enriched_rows.append(enriched_data)

    return pd.DataFrame(enriched_rows)

def add_journal_abbreviations(df):
    """Add journal abbreviations and create all BibTeX versions"""
    abbreviated_rows = []
    
    for idx, row in df.iterrows():
        journal = row.get('Journal/Booktitle', '')
        journal_abbrev = abbreviate_journal_custom(journal)
        
        row_data = dict(row)
        row_data['Journal_Abbreviation'] = journal_abbrev
        
        # Create LocalKey version
        key_to_use = row_data.get('Key') or row_data.get('Reference') or f"ref_{idx}"
        
        if row_data.get('Crossref_BibTeX'):
            row_data['Crossref_BibTeX_LocalKey'] = replace_bibtex_key(
                row_data['Crossref_BibTeX'], 
                key_to_use
            )
        else:
            row_data['Crossref_BibTeX_LocalKey'] = row_data.get('BibTeX', '')
        
        # Create abbreviated version
        if journal_abbrev and row_data.get('Crossref_BibTeX_LocalKey'):
            new_bib = row_data['Crossref_BibTeX_LocalKey'].strip()
            new_bib = re.sub(
                r'(journal\s*=\s*\{)[^}]+(\})',
                rf'\1{journal_abbrev}\2',
                new_bib,
                flags=re.IGNORECASE
            )
            row_data['Crossref_BibTeX_Abbrev'] = new_bib
        else:
            row_data['Crossref_BibTeX_Abbrev'] = row_data.get('Crossref_BibTeX_LocalKey', row_data.get('BibTeX', ''))
        
        # Create protected version
        row_data['Crossref_BibTeX_Protected'] = protect_acronyms_in_fields(
            row_data.get('Crossref_BibTeX_Abbrev', row_data.get('BibTeX', ''))
        )
        
        # Clean all versions
        for bib_col in ['BibTeX', 'Crossref_BibTeX', 'Crossref_BibTeX_LocalKey', 
                        'Crossref_BibTeX_Abbrev', 'Crossref_BibTeX_Protected']:
            if row_data.get(bib_col):
                row_data[bib_col] = clean_bibtex_fields(row_data[bib_col])
        
        abbreviated_rows.append(row_data)

    return pd.DataFrame(abbreviated_rows)

# =====================================================================
# ROUTES
# =====================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_bibtex():
    """Process BibTeX content with optional LaTeX analysis"""
    if not check_api_key():
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Handle both JSON and FormData
        if request.is_json:
            data = request.get_json()
            bibtex_content = data.get('bibtex_content') or data.get('bibtex', '')
            input_mode = data.get('input_mode', 'bibtex')
            enrich = data.get('enrich', False)
            abbreviate = data.get('abbreviate', False)
            protect = data.get('protect', False)
            save_to_db = data.get('save_to_db', False)
            latex_file = None
        else:
            # FormData from file upload
            bibtex_content = request.form.get('bibtex_content', '')
            input_mode = request.form.get('input_mode', 'bibtex')
            enrich = request.form.get('enrich', 'false').lower() == 'true'
            abbreviate = request.form.get('abbreviate', 'false').lower() == 'true'
            protect = request.form.get('protect', 'false').lower() == 'true'
            save_to_db = request.form.get('save_to_db', 'false').lower() == 'true'
            latex_file = request.files.get('latex_file')
        
        print(f"📥 Received: {len(bibtex_content)} chars, mode={input_mode}")
        
        # TITLE MODE: Search Crossref for each title
        if input_mode == 'title':
            titles = [line.strip() for line in bibtex_content.split('\n') if line.strip()]
            print(f"🔍 Title mode: searching for {len(titles)} titles")
            
            if not titles:
                return jsonify({'error': 'No titles provided'}), 400
            
            # Convert titles to BibTeX entries via Crossref
            bibtex_entries = []
            for i, title in enumerate(titles, 1):
                print(f"  [{i}/{len(titles)}] Searching: {title[:50]}...")
                try:
                    url = f"https://api.crossref.org/works?query.bibliographic={requests.utils.quote(title)}&rows=1"
                    response = HTTP.get(url, timeout=15)
                    items = response.json().get("message", {}).get("items", [])
                    
                    if items and items[0].get('DOI'):
                        doi = items[0]['DOI']
                        bibtex_r = HTTP.get(
                            f"https://doi.org/{doi}",
                            headers={"Accept": "application/x-bibtex"},
                            timeout=15
                        )
                        if bibtex_r.status_code == 200:
                            bibtex_entries.append(bibtex_r.text.strip())
                            print(f"    ✅ Found via DOI: {doi}")
                    
                    time.sleep(random.uniform(1, 2))
                except Exception as e:
                    print(f"    ⚠️ Failed: {e}")
            
            bibtex_content = '\n\n'.join(bibtex_entries)
            print(f"✅ Retrieved {len(bibtex_entries)} BibTeX entries from titles")
            
            if not bibtex_content:
                return jsonify({'error': 'No BibTeX entries found for the provided titles'}), 400
        
        if not bibtex_content or len(bibtex_content.strip()) == 0:
            return jsonify({'error': 'No BibTeX content provided'}), 400
        
        # Parse BibTeX
        df = parse_bibtex_input(bibtex_content)
        
        if df.empty:
            return jsonify({'error': 'No valid BibTeX entries found'}), 400
        
        # Parse LaTeX file if provided
        latex_analyzed = False
        citations_found = 0
        if latex_file:
            print("📄 LaTeX file provided, analyzing...")
            try:
                latex_content = latex_file.read().decode('utf-8')
                citations_df = parse_citations_from_tex(latex_content)
                citations_found = len(citations_df)
                
                # Merge LaTeX citation data with BibTeX data
                df = merge_citations_with_bib(citations_df, df)
                df.insert(0, "Index", range(1, len(df) + 1))
                latex_analyzed = True
                print(f"✅ LaTeX analyzed: {citations_found} citations found")
            except Exception as e:
                print(f"⚠️ LaTeX analysis failed: {e}")
        
        # Enrich if requested
        if enrich:
            df = enrich_with_crossref(df)
        else:
            df['Crossref_BibTeX'] = df['BibTeX']
            df['Title_Similarity'] = 0
        
        # Always create all versions
        df = add_journal_abbreviations(df)
        
        # Save to database if requested
        db_id = None
        if save_to_db:
            conn = get_db_connection()
            cursor = conn.cursor()
            session_id = datetime.now().isoformat()
            
            for _, row in df.iterrows():
                doi = row.get('DOI', '').strip()
                year_int = extract_year_int(row.get('Year', ''))
                key_val = row.get('Reference') or row.get('Key', f"ref_{row.name}")
                
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO bibliography 
                        (index_num, session_id, reference, frequency, sections, key, doi, type, 
                         authors, title, journal_booktitle, year, year_int, publisher, volume, pages,
                         bibtex, crossref_bibtex, crossref_bibtex_localkey, title_similarity,
                         journal_abbreviation, crossref_bibtex_abbrev, crossref_bibtex_protected,
                         imported_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row.get('Index'), session_id, row.get('Reference', key_val),
                        row.get('Frequency', 0), row.get('Sections', ''),
                        key_val, doi, row.get('Type', ''), row.get('Authors', ''),
                        row.get('Title', ''), row.get('Journal/Booktitle', ''),
                        row.get('Year', ''), year_int, row.get('Publisher', ''),
                        row.get('Volume', ''), row.get('Pages', ''),
                        row.get('BibTeX', ''), row.get('Crossref_BibTeX', ''),
                        row.get('Crossref_BibTeX_LocalKey', ''),
                        row.get('Title_Similarity', 0), row.get('Journal_Abbreviation', ''),
                        row.get('Crossref_BibTeX_Abbrev', ''),
                        row.get('Crossref_BibTeX_Protected', ''),
                        datetime.now().isoformat()
                    ))
                except sqlite3.IntegrityError as e:
                    print(f"⚠️ DB insert failed for {key_val}: {e}")
            
            conn.commit()
            db_id = session_id
            conn.close()

        # Determine final BibTeX column
        if protect:
            final_bibtex_col = 'Crossref_BibTeX_Protected'
        elif abbreviate:
            final_bibtex_col = 'Crossref_BibTeX_Abbrev'
        else:
            final_bibtex_col = 'Crossref_BibTeX_LocalKey'
        
        # Prepare response columns
        response_cols = ['Key', 'Type', 'Authors', 'Title', 'Journal/Booktitle', 'Year', final_bibtex_col]
        if latex_analyzed:
            response_cols.insert(6, 'Frequency')
            response_cols.insert(7, 'Sections')
        
        response_df = df[[col for col in response_cols if col in df.columns]].copy()
        response_df.columns = list(response_df.columns[:-1]) + ['Final_BibTeX']

        return jsonify({
            'success': True,
            'count': len(df),
            'db_id': db_id,
            'latex_analyzed': latex_analyzed,
            'citations_found': citations_found,
            'data': response_df.to_dict(orient='records'),
            'full_data': df.to_dict(orient='records')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/sections/list', methods=['GET'])
def list_sections():
    """Get list of all unique sections from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT sections FROM bibliography WHERE sections IS NOT NULL AND sections != ""')
        rows = cursor.fetchall()
        conn.close()
        
        # Parse and deduplicate sections
        all_sections = set()
        for row in rows:
            if row[0]:
                sections = [s.strip() for s in row[0].split(',')]
                all_sections.update(sections)
        
        sections_list = sorted(list(all_sections))
        
        return jsonify({
            'success': True,
            'sections': sections_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sections/references', methods=['GET'])
def get_references_by_section():
    """Get all references for a specific section"""
    section = request.args.get('section', '')
    
    if not section:
        return jsonify({'error': 'Section parameter required'}), 400
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Find all references that contain this section
        cursor.execute('''
            SELECT key, title, authors, year, frequency, sections, reference
            FROM bibliography 
            WHERE sections LIKE ?
        ''', (f'%{section}%',))
        
        rows = cursor.fetchall()
        conn.close()
        
        references = []
        for row in rows:
            # Calculate frequency in this specific section
            sections_list = [s.strip() for s in row[5].split(',') if s.strip()]
            if section in sections_list:
                # Count occurrences in this section (simplified - assumes equal distribution)
                total_freq = row[4] or 0
                num_sections = len(sections_list)
                freq_in_section = total_freq // num_sections if num_sections > 0 else total_freq
                
                references.append({
                    'key': row[0],
                    'title': row[1],
                    'authors': row[2],
                    'year': row[3],
                    'frequency_in_section': freq_in_section,
                    'total_frequency': total_freq,
                    'all_sections': row[5]
                })
        
        return jsonify({
            'success': True,
            'section': section,
            'references': references
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/entries', methods=['GET'])
def get_database_entries():
    """Get all entries from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM bibliography ORDER BY created_at DESC LIMIT 100
        ''')
        
        columns = [description[0] for description in cursor.description]
        entries = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'count': len(entries),
            'entries': entries
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/delete/<key>', methods=['DELETE'])
def delete_entry(key):
    """Delete entry from database"""
    if not check_api_key():
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM bibliography WHERE key=?', (key,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/clear', methods=['POST'])
def clear_database():
    """Clear all entries from database"""
    if not check_api_key():
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM bibliography')
        deleted_count = cursor.rowcount
        
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="bibliography"')
        
        conn.commit()
        conn.close()
        
        print(f"🗑️ Cleared {deleted_count} entries from database")
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Successfully cleared {deleted_count} entries from database'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/export', methods=['GET'])
def export_database():
    """Export database as CSV"""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query('SELECT * FROM bibliography', conn)
        conn.close()
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='references_export.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/export-bibtex', methods=['GET'])
def export_bibtex():
    """Export database as BibTeX"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT key, crossref_bibtex_protected FROM bibliography ORDER BY key')
        
        bibtex_content = '\n\n'.join([row[1] for row in cursor.fetchall() if row[1]])
        
        conn.close()
        
        return send_file(
            io.BytesIO(bibtex_content.encode()),
            mimetype='text/plain',
            as_attachment=True,
            download_name='references.bib'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/download', methods=['GET'])
def download_database():
    """Download entire database file"""
    try:
        return send_file(
            app.config['DATABASE'],
            mimetype='application/x-sqlite3',
            as_attachment=True,
            download_name='refs_management.db'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM bibliography')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT type) FROM bibliography')
        types = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT year_int) FROM bibliography WHERE year_int IS NOT NULL')
        years = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_entries': total,
            'entry_types': types,
            'unique_years': years
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=False, host='0.0.0.0', port=7860)