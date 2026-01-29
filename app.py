"""
Flask Application for Reference Management Pipeline
Users can paste BibTeX content, view results, and save to database
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

app = Flask(__name__)
app.config['DATABASE'] = 'refs_management.db'
app.config['API_KEY'] = os.environ.get('API_KEY', 'your-secret-key-here')  # Change in production
app.config['ENVIRONMENT'] = os.environ.get('ENVIRONMENT', 'development')  # 'development' or 'production'

# Prepositions to keep lowercase in abbreviations
LOWERCASE_WORDS = {"and", "or", "in", "on", "of", "for", "to", "the", "a", "an"}

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
        "User-Agent": "RefsManagement/1.0 (mailto:contact@example.com)",  # Update with your email
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
# UTILITY FUNCTIONS
# =====================================================================

def load_journal_abbreviations(ltwa_file="ltwa.txt"):
    """Load journal abbreviations from ltwa.txt file"""
    abbreviations = {}
    if os.path.exists(ltwa_file):
        try:
            with open(ltwa_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        full_name = parts[0].strip()
                        abbrev = parts[1].strip()
                        abbreviations[full_name.lower()] = abbrev
        except Exception as e:
            print(f"⚠️ Error loading LTWA: {e}")
    return abbreviations

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with proper schema"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bibliography (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            key TEXT,
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
            title_similarity INTEGER,
            journal_abbreviation TEXT,
            crossref_bibtex_abbrev TEXT,
            crossref_bibtex_protected TEXT,
            imported_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_bib_doi 
        ON bibliography(doi) 
        WHERE doi IS NOT NULL AND doi != ''
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_bib_key 
        ON bibliography(key) 
        WHERE key IS NOT NULL AND key != ''
    """)
    conn.commit()
    conn.close()

def extract_year_int(year_str):
    """Extract integer year from year string"""
    if not year_str:
        return None
    match = re.search(r'\d{4}', str(year_str))
    return int(match.group()) if match else None

def generate_stable_key(title, year, authors):
    """Generate stable key from title, year, and authors"""
    normalized = f"{title.lower().strip()}_{year}_{authors.split(',')[0] if authors else ''}"
    return hashlib.md5(normalized.encode()).hexdigest()[:16]

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
        # Unquoted value - read until comma or closing brace
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
    
    # Find start of fields
    fields_start = entry_text.find(entry_key) + len(entry_key) + 1
    fields_text = entry_text[fields_start:]
    
    # Parse fields
    pos = 0
    while pos < len(fields_text):
        # Skip whitespace and commas
        while pos < len(fields_text) and fields_text[pos] in ' \t\n\r,':
            pos += 1
        if pos >= len(fields_text) or fields_text[pos] == '}':
            break
        
        # Read field name
        field_match = re.match(r'(\w+)\s*=\s*', fields_text[pos:])
        if not field_match:
            break
        
        field_name = field_match.group(1).lower()
        pos += field_match.end()
        
        # Read field value
        value, new_pos = scan_brace_balanced_value(fields_text, pos)
        fields[field_name] = value.strip()
        pos = new_pos
    
    return {
        'type': entry_type,
        'key': entry_key.strip(),
        'fields': fields
    }

def parse_bibtex_input(bibtex_content):
    """Parse BibTeX content from user input with improved parsing"""
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
    """Remove unwanted fields from BibTeX entries with proper brace handling"""
    fields_to_remove = ['url', 'source', 'publication_stage', 'note', 'abstract']
    
    for field in fields_to_remove:
        # More robust pattern that handles nested braces
        pattern = rf'\s*{field}\s*=\s*'
        pos = 0
        result = []
        
        while pos < len(bibtex):
            match = re.search(pattern, bibtex[pos:], re.IGNORECASE)
            if not match:
                result.append(bibtex[pos:])
                break
            
            # Add text before match
            result.append(bibtex[pos:pos + match.start()])
            
            # Skip the field value
            value_start = pos + match.end()
            _, value_end = scan_brace_balanced_value(bibtex, value_start)
            
            # Skip trailing comma if present
            while value_end < len(bibtex) and bibtex[value_end] in ' \t\n\r,':
                value_end += 1
            
            pos = value_end
        
        bibtex = ''.join(result)
    
    # Clean up formatting
    bibtex = re.sub(r'\n\s*\n\s*\n', '\n\n', bibtex)
    bibtex = re.sub(r',\s*,', ',', bibtex)
    bibtex = re.sub(r',(\s*)\}', r'\1}', bibtex)
    
    lines = [line for line in bibtex.split('\n') if line.strip()]
    return '\n'.join(lines)

def protect_acronyms_in_fields(bibtex):
    """Protect acronyms with braces - improved to handle nested braces"""
    def wrap_token(token):
        if token.startswith("{") and token.endswith("}"):
            return token
        if sum(1 for c in token if c.isupper()) >= 2:
            return "{" + token + "}"
        return token

    def process_field_value(value):
        # Don't process if already fully braced
        if value.startswith("{") and value.endswith("}"):
            inner = value[1:-1]
            # Check if it's a simple brace wrap
            if not ('{' in inner and '}' in inner):
                return value
        
        tokens = re.split(r'(\s+)', value)
        fixed = "".join(wrap_token(tok) if tok.strip() else tok for tok in tokens)
        # Remove double bracing
        fixed = re.sub(r'\{\{([^{}]+)\}\}', r'{\1}', fixed)
        return fixed

    for field in ["title", "booktitle", "journal"]:
        # Find field and extract its value properly
        pattern = rf'({field}\s*=\s*)'
        matches = list(re.finditer(pattern, bibtex, re.IGNORECASE))
        
        for match in reversed(matches):  # Process in reverse to maintain positions
            field_start = match.end()
            value, value_end = scan_brace_balanced_value(bibtex, field_start)
            
            if value:
                processed = process_field_value(value)
                # Determine original delimiter
                if field_start < len(bibtex) and bibtex[field_start] == '{':
                    new_field = f"{match.group(1)}{{{processed}}}"
                else:
                    new_field = f"{match.group(1)}{{{processed}}}"
                
                bibtex = bibtex[:match.start()] + new_field + bibtex[value_end:]

    return bibtex

def enrich_with_crossref(df):
    """Enrich references with Crossref data using HTTP session"""
    enriched_rows = []
    
    for idx, row in df.iterrows():
        enriched_data = dict(row)
        
        if not row['Title']:
            enriched_data['Crossref_BibTeX'] = row['BibTeX']
            enriched_data['Title_Similarity'] = 0
            enriched_rows.append(enriched_data)
            continue

        # Build query
        query_parts = [row['Title']]
        if row['Authors']:
            query_parts.append(row['Authors'].split(',')[0])
        if row['Journal/Booktitle']:
            query_parts.append(row['Journal/Booktitle'])
        if row['Year']:
            query_parts.append(row['Year'])
        
        query = " ".join(query_parts)

        try:
            url = f"https://api.crossref.org/works?query.bibliographic={requests.utils.quote(query)}&rows=3"
            response = HTTP.get(url, timeout=15)
            response.raise_for_status()
            items = response.json().get("message", {}).get("items", [])

            best_score = 0
            crossref_bibtex = row['BibTeX']
            best_doi = row.get('DOI', '')

            for item in items:
                cr_title = item.get("title", [""])[0]
                score = SequenceMatcher(None, row['Title'].lower(), cr_title.lower()).ratio()
                
                # Enhanced matching: also check year and first author
                year_match = False
                if row['Year'] and 'published-print' in item:
                    cr_year = str(item['published-print'].get('date-parts', [['']])[0][0])
                    year_match = row['Year'].strip() == cr_year
                
                # Adjust score based on year match
                if year_match:
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

            # Use threshold of 0.85 instead of 0.95 for better coverage
            enriched_data['Crossref_BibTeX'] = crossref_bibtex if best_score >= 0.85 else row['BibTeX']
            enriched_data['Title_Similarity'] = int(round(best_score * 100))
            if best_doi:
                enriched_data['DOI'] = best_doi
            
        except Exception as e:
            print(f"⚠️ Crossref enrichment failed: {e}")
            enriched_data['Crossref_BibTeX'] = row['BibTeX']
            enriched_data['Title_Similarity'] = 0

        time.sleep(0.15 + random.uniform(0, 0.25))
        enriched_rows.append(enriched_data)

    return pd.DataFrame(enriched_rows)

def get_bibtex_from_titles(titles_list):
    """Get BibTeX from Crossref using only paper titles"""
    papers = []
    
    for idx, title in enumerate(titles_list):
        title = title.strip()
        if not title:
            continue
            
        try:
            url = f"https://api.crossref.org/works?query.title={requests.utils.quote(title)}&rows=1"
            response = HTTP.get(url, timeout=15)
            response.raise_for_status()
            items = response.json().get("message", {}).get("items", [])
            
            if items:
                item = items[0]
                cr_title = item.get("title", [""])[0]
                score = SequenceMatcher(None, title.lower(), cr_title.lower()).ratio()
                
                if score >= 0.7:
                    bibtex_text = ""
                    doi = item.get("DOI", "")
                    
                    if doi:
                        try:
                            bibtex_response = HTTP.get(
                                f"https://doi.org/{doi}",
                                headers={"Accept": "application/x-bibtex"},
                                timeout=15
                            )
                            if bibtex_response.status_code == 200:
                                bibtex_text = bibtex_response.text.strip()
                        except Exception as e:
                            print(f"⚠️ BibTeX fetch failed: {e}")
                    
                    if bibtex_text:
                        parsed = parse_bibtex_entry(bibtex_text)
                        if parsed:
                            fields = parsed['fields']
                            papers.append({
                                "Key": parsed['key'],
                                "Type": parsed['type'],
                                "Authors": fields.get("author", "").strip(),
                                "Title": fields.get("title", cr_title).strip(),
                                "Journal/Booktitle": fields.get("journal", fields.get("booktitle", "")).strip(),
                                "Year": fields.get("year", "").strip(),
                                "Publisher": fields.get("publisher", "").strip(),
                                "Volume": fields.get("volume", "").strip(),
                                "Pages": fields.get("pages", "").strip(),
                                "DOI": doi,
                                "BibTeX": bibtex_text,
                                "Crossref_BibTeX": bibtex_text,
                                "Title_Similarity": int(round(score * 100)),
                                "Imported_Date": datetime.now().isoformat()
                            })
                        else:
                            # Fallback to metadata
                            papers.append(create_paper_from_crossref_metadata(item, idx, score))
                    else:
                        papers.append(create_paper_from_crossref_metadata(item, idx, score))
                else:
                    papers.append(create_not_found_entry(title, idx, score))
            else:
                papers.append(create_not_found_entry(title, idx, 0))
                
        except Exception as e:
            papers.append(create_error_entry(title, idx, str(e)))
        
        time.sleep(0.2 + random.uniform(0, 0.3))
    
    return pd.DataFrame(papers)

def create_paper_from_crossref_metadata(item, idx, score):
    """Create paper entry from Crossref metadata"""
    authors = []
    if "author" in item:
        authors = [f"{a.get('family', '')}, {a.get('given', '')}" for a in item.get("author", [])]
    
    year = ""
    if "published-print" in item:
        year = str(item["published-print"].get("date-parts", [[""]])[0][0])
    elif "published-online" in item:
        year = str(item["published-online"].get("date-parts", [[""]])[0][0])
    
    cr_title = item.get("title", [""])[0]
    
    return {
        "Key": f"crossref_{idx+1}",
        "Type": item.get("type", "article"),
        "Authors": " and ".join(authors),
        "Title": cr_title,
        "Journal/Booktitle": item.get("container-title", [""])[0] if item.get("container-title") else "",
        "Year": year,
        "Publisher": item.get("publisher", ""),
        "Volume": item.get("volume", ""),
        "Pages": item.get("page", ""),
        "DOI": item.get("DOI", ""),
        "BibTeX": "",
        "Crossref_BibTeX": "",
        "Title_Similarity": int(round(score * 100)),
        "Imported_Date": datetime.now().isoformat()
    }

def create_not_found_entry(title, idx, score):
    """Create entry for title not found"""
    suffix = " (NOT FOUND - low match)" if score > 0 else " (NOT FOUND)"
    return {
        "Key": f"not_found_{idx+1}",
        "Type": "article",
        "Authors": "",
        "Title": title + suffix,
        "Journal/Booktitle": "",
        "Year": "",
        "Publisher": "",
        "Volume": "",
        "Pages": "",
        "DOI": "",
        "BibTeX": "",
        "Crossref_BibTeX": "",
        "Title_Similarity": int(round(score * 100)) if score > 0 else 0,
        "Imported_Date": datetime.now().isoformat()
    }

def create_error_entry(title, idx, error_msg):
    """Create entry for error case"""
    return {
        "Key": f"error_{idx+1}",
        "Type": "article",
        "Authors": "",
        "Title": title + f" (ERROR: {error_msg})",
        "Journal/Booktitle": "",
        "Year": "",
        "Publisher": "",
        "Volume": "",
        "Pages": "",
        "DOI": "",
        "BibTeX": "",
        "Crossref_BibTeX": "",
        "Title_Similarity": 0,
        "Imported_Date": datetime.now().isoformat()
    }

def add_journal_abbreviations(df):
    """Add journal abbreviations to dataframe"""
    abbreviations = load_journal_abbreviations()
    
    abbreviated_rows = []
    for idx, row in df.iterrows():
        journal = row.get('Journal/Booktitle', '')
        journal_abbrev = abbreviations.get(journal.lower(), '')
        
        row_data = dict(row)
        row_data['Journal_Abbreviation'] = journal_abbrev
        
        if journal_abbrev and row_data.get('Crossref_BibTeX'):
            new_bib = row_data['Crossref_BibTeX'].strip()
            new_bib = re.sub(
                r'(journal\s*=\s*\{)[^}]+(\})',
                rf'\1{journal_abbrev}\2',
                new_bib,
                flags=re.IGNORECASE
            )
            row_data['Crossref_BibTeX_Abbrev'] = new_bib
        else:
            row_data['Crossref_BibTeX_Abbrev'] = row_data.get('Crossref_BibTeX', row['BibTeX'])
        
        row_data['Crossref_BibTeX_Protected'] = protect_acronyms_in_fields(
            row_data.get('Crossref_BibTeX_Abbrev', row['BibTeX'])
        )
        
        # Clean unwanted fields from all BibTeX versions
        row_data['BibTeX'] = clean_bibtex_fields(row_data['BibTeX'])
        if row_data.get('Crossref_BibTeX'):
            row_data['Crossref_BibTeX'] = clean_bibtex_fields(row_data['Crossref_BibTeX'])
        if row_data.get('Crossref_BibTeX_Abbrev'):
            row_data['Crossref_BibTeX_Abbrev'] = clean_bibtex_fields(row_data['Crossref_BibTeX_Abbrev'])
        if row_data.get('Crossref_BibTeX_Protected'):
            row_data['Crossref_BibTeX_Protected'] = clean_bibtex_fields(row_data['Crossref_BibTeX_Protected'])
        
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
def process_references():
    """Process BibTeX or Title input"""
    try:
        data = request.json
        input_content = data.get('bibtex_content', '')
        input_mode = data.get('input_mode', 'bibtex')
        enrich = data.get('enrich', False)
        abbreviate = data.get('abbreviate', False)
        protect = data.get('protect', False)
        save_to_db = data.get('save_to_db', False)
        
        if not input_content.strip():
            return jsonify({'error': 'No content provided'}), 400

        if input_mode == 'title':
            titles = [line.strip() for line in input_content.strip().split('\n') if line.strip()]
            if not titles:
                return jsonify({'error': 'No titles provided'}), 400
            
            df = get_bibtex_from_titles(titles)
            if df.empty:
                return jsonify({'error': 'No results found from Crossref'}), 400
            
            if 'Crossref_BibTeX' not in df.columns:
                df['Crossref_BibTeX'] = df['BibTeX']
            if 'Title_Similarity' not in df.columns:
                df['Title_Similarity'] = 0
        else:
            df = parse_bibtex_input(input_content)
            if df.empty:
                return jsonify({'error': 'No valid BibTeX entries found'}), 400

            if enrich:
                df = enrich_with_crossref(df)
            else:
                df['Crossref_BibTeX'] = df['BibTeX']
                df['Title_Similarity'] = 0

        if abbreviate:
            df = add_journal_abbreviations(df)
        else:
            df['Journal_Abbreviation'] = ''
            df['Crossref_BibTeX_Abbrev'] = df['Crossref_BibTeX']
            df['Crossref_BibTeX_Protected'] = df['Crossref_BibTeX']

        if protect:
            df['Crossref_BibTeX_Protected'] = df['Crossref_BibTeX_Abbrev'].apply(protect_acronyms_in_fields)

        db_id = None
        if save_to_db:
            conn = get_db_connection()
            cursor = conn.cursor()
            session_id = datetime.now().isoformat()
            
            for _, row in df.iterrows():
                doi = row.get('DOI', '').strip()
                year_int = extract_year_int(row.get('Year', ''))
                
                try:
                    if doi:
                        # Use DOI as primary key
                        cursor.execute('''
                            INSERT INTO bibliography 
                            (session_id, key, doi, type, authors, title, journal_booktitle, year, year_int,
                             publisher, volume, pages, bibtex, crossref_bibtex, 
                             title_similarity, journal_abbreviation, crossref_bibtex_abbrev, 
                             crossref_bibtex_protected, imported_date)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(doi) DO UPDATE SET
                                session_id=excluded.session_id,
                                key=excluded.key,
                                type=excluded.type,
                                authors=excluded.authors,
                                title=excluded.title,
                                journal_booktitle=excluded.journal_booktitle,
                                year=excluded.year,
                                year_int=excluded.year_int,
                                publisher=excluded.publisher,
                                volume=excluded.volume,
                                pages=excluded.pages,
                                bibtex=excluded.bibtex,
                                crossref_bibtex=excluded.crossref_bibtex,
                                title_similarity=excluded.title_similarity,
                                journal_abbreviation=excluded.journal_abbreviation,
                                crossref_bibtex_abbrev=excluded.crossref_bibtex_abbrev,
                                crossref_bibtex_protected=excluded.crossref_bibtex_protected,
                                imported_date=excluded.imported_date
                        ''', (
                            session_id, row['Key'], doi, row['Type'], row['Authors'],
                            row['Title'], row['Journal/Booktitle'], row['Year'], year_int,
                            row['Publisher'], row.get('Volume', ''), row.get('Pages', ''),
                            row['BibTeX'], row.get('Crossref_BibTeX', row['BibTeX']),
                            row.get('Title_Similarity', 0), row.get('Journal_Abbreviation', ''),
                            row.get('Crossref_BibTeX_Abbrev', row['BibTeX']),
                            row.get('Crossref_BibTeX_Protected', row['BibTeX']),
                            row.get('Imported_Date', datetime.now().isoformat())
                        ))
                    else:
                        # Fall back to key-based insert
                        cursor.execute('''
                            INSERT OR REPLACE INTO bibliography 
                            (session_id, key, doi, type, authors, title, journal_booktitle, year, year_int,
                             publisher, volume, pages, bibtex, crossref_bibtex, 
                             title_similarity, journal_abbreviation, crossref_bibtex_abbrev, 
                             crossref_bibtex_protected, imported_date)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            session_id, row['Key'], doi, row['Type'], row['Authors'],
                            row['Title'], row['Journal/Booktitle'], row['Year'], year_int,
                            row['Publisher'], row.get('Volume', ''), row.get('Pages', ''),
                            row['BibTeX'], row.get('Crossref_BibTeX', row['BibTeX']),
                            row.get('Title_Similarity', 0), row.get('Journal_Abbreviation', ''),
                            row.get('Crossref_BibTeX_Abbrev', row['BibTeX']),
                            row.get('Crossref_BibTeX_Protected', row['BibTeX']),
                            row.get('Imported_Date', datetime.now().isoformat())
                        ))
                except sqlite3.IntegrityError as e:
                    print(f"⚠️ DB insert failed for {row['Key']}: {e}")
            
            conn.commit()
            db_id = session_id
            conn.close()

        response_df = df[[
            'Key', 'Type', 'Authors', 'Title', 'Journal/Booktitle', 'Year',
            'Crossref_BibTeX_Protected' if protect else 'Crossref_BibTeX_Abbrev' if abbreviate else 'BibTeX'
        ]].copy()
        
        response_df.columns = [
            'Key', 'Type', 'Authors', 'Title', 'Journal/Booktitle', 'Year', 'Final_BibTeX'
        ]

        return jsonify({
            'success': True,
            'count': len(df),
            'db_id': db_id,
            'data': response_df.to_dict(orient='records'),
            'full_data': df.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/entries', methods=['GET'])
def get_database_entries():
    """Get all entries from database"""
    if not check_api_key():
        return jsonify({'error': 'Unauthorized'}), 401
    
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

@app.route('/api/database/export', methods=['GET'])
def export_database():
    """Export database as CSV"""
    if not check_api_key():
        return jsonify({'error': 'Unauthorized'}), 401
    
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
    if not check_api_key():
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT key, crossref_bibtex_protected FROM bibliography ORDER BY key')
        
        bibtex_content = '\n\n'.join([row[1] for row in cursor.fetchall()])
        
        conn.close()
        
        return send_file(
            io.BytesIO(bibtex_content.encode()),
            mimetype='text/plain',
            as_attachment=True,
            download_name='references.bib'
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