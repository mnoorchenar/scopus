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
import io
import time
import random

app = Flask(__name__)
app.config['DATABASE'] = 'refs_management.db'

# Prepositions to keep lowercase in abbreviations
LOWERCASE_WORDS = {"and", "or", "in", "on", "of", "for", "to", "the", "a", "an"}

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
    """Initialize database"""
    conn = get_db_connection()
    conn.close()

def parse_bibtex_input(bibtex_content):
    """Parse BibTeX content from user input"""
    entries = ["@" + e for e in bibtex_content.split("@") if e.strip()]
    papers = []

    for entry in entries:
        match = re.match(r'@(\w+)\s*\{([^,]+),', entry)
        if not match:
            continue
        entry_type, entry_key = match.groups()
        fields = dict(re.findall(
            r'(\w+)\s*=\s*\{((?:[^{}]|\{[^}]*\})*)\}', 
            entry, flags=re.DOTALL
        ))

        papers.append({
            "Key": entry_key.strip(),
            "Type": entry_type,
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

def protect_acronyms_in_fields(bibtex):
    """Protect acronyms with braces"""
    def clean_field(field_name, text):
        text = text.strip()

        def wrap_token(token):
            if token.startswith("{") and token.endswith("}"):
                return token
            if sum(1 for c in token if c.isupper()) >= 2:
                return "{" + token + "}"
            return token

        tokens = re.split(r'(\s+)', text)
        fixed = "".join(wrap_token(tok) if tok.strip() else tok for tok in tokens)
        fixed = re.sub(r'\{\{([^{}]+)\}\}', r'{\1}', fixed)
        fixed = fixed.replace("}} ", "} ").replace("{{ ", "{ ")
        return f"{field_name}={{{fixed.strip()}}}"

    for field in ["title", "booktitle", "journal"]:
        bibtex = re.sub(
            rf'{field}\s*=\s*\{{([^}}]*)\}}',
            lambda m: clean_field(field, m.group(1)),
            bibtex,
            flags=re.IGNORECASE
        )

    return bibtex

def enrich_with_crossref(df):
    """Enrich references with Crossref data"""
    enriched_rows = []
    
    for idx, row in df.iterrows():
        enriched_data = dict(row)
        
        if not row['Title']:
            enriched_data['Crossref_BibTeX'] = row['BibTeX']
            enriched_data['Title_Similarity'] = 0
            enriched_rows.append(enriched_data)
            continue

        query = " ".join(filter(None, [
            row['Title'],
            row['Authors'].split(',')[0] if row['Authors'] else "",
            row['Journal/Booktitle'],
            row['Year'],
            row['Publisher']
        ]))

        try:
            url = f"https://api.crossref.org/works?query.bibliographic={requests.utils.quote(query)}&rows=3"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            items = response.json().get("message", {}).get("items", [])

            best_score = 0
            crossref_bibtex = row['BibTeX']

            for item in items:
                cr_title = item.get("title", [""])[0]
                score = SequenceMatcher(None, row['Title'].lower(), cr_title.lower()).ratio()
                
                if score > best_score:
                    best_score = score
                    if "DOI" in item:
                        try:
                            bibtex_response = requests.get(
                                f"https://doi.org/{item['DOI']}",
                                headers={"Accept": "application/x-bibtex"},
                                timeout=10
                            )
                            if bibtex_response.status_code == 200:
                                crossref_bibtex = bibtex_response.text.strip()
                        except:
                            pass

            enriched_data['Crossref_BibTeX'] = crossref_bibtex if best_score >= 0.95 else row['BibTeX']
            enriched_data['Title_Similarity'] = int(round(best_score * 100))
            
        except Exception as e:
            enriched_data['Crossref_BibTeX'] = row['BibTeX']
            enriched_data['Title_Similarity'] = 0

        time.sleep(0.1 + random.uniform(0, 0.2))
        enriched_rows.append(enriched_data)

    return pd.DataFrame(enriched_rows)

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
    """Process BibTeX input"""
    try:
        data = request.json
        bibtex_content = data.get('bibtex_content', '')
        enrich = data.get('enrich', False)
        abbreviate = data.get('abbreviate', False)
        protect = data.get('protect', False)
        save_to_db = data.get('save_to_db', False)
        
        if not bibtex_content.strip():
            return jsonify({'error': 'No BibTeX content provided'}), 400

        # Parse BibTeX
        df = parse_bibtex_input(bibtex_content)
        
        if df.empty:
            return jsonify({'error': 'No valid BibTeX entries found'}), 400

        # Add enrichment if requested
        if enrich:
            df = enrich_with_crossref(df)
        else:
            df['Crossref_BibTeX'] = df['BibTeX']
            df['Title_Similarity'] = 0

        # Add abbreviations if requested
        if abbreviate:
            df = add_journal_abbreviations(df)
        else:
            df['Journal_Abbreviation'] = ''
            df['Crossref_BibTeX_Abbrev'] = df['Crossref_BibTeX']
            df['Crossref_BibTeX_Protected'] = df['Crossref_BibTeX']

        # Protect acronyms if requested
        if protect:
            df['Crossref_BibTeX_Protected'] = df['Crossref_BibTeX_Abbrev'].apply(protect_acronyms_in_fields)

        # Save to database if requested
        db_id = None
        if save_to_db:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bibliography (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    key TEXT UNIQUE,
                    type TEXT,
                    authors TEXT,
                    title TEXT,
                    journal_booktitle TEXT,
                    year TEXT,
                    publisher TEXT,
                    volume TEXT,
                    pages TEXT,
                    doi TEXT,
                    bibtex TEXT,
                    crossref_bibtex TEXT,
                    title_similarity INTEGER,
                    journal_abbreviation TEXT,
                    crossref_bibtex_abbrev TEXT,
                    crossref_bibtex_protected TEXT,
                    imported_date TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            session_id = datetime.now().isoformat()
            
            for _, row in df.iterrows():
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO bibliography 
                        (session_id, key, type, authors, title, journal_booktitle, year, 
                         publisher, volume, pages, doi, bibtex, crossref_bibtex, 
                         title_similarity, journal_abbreviation, crossref_bibtex_abbrev, 
                         crossref_bibtex_protected, imported_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, row['Key'], row['Type'], row['Authors'],
                        row['Title'], row['Journal/Booktitle'], row['Year'],
                        row['Publisher'], row.get('Volume', ''), row.get('Pages', ''),
                        row.get('DOI', ''), row['BibTeX'], row.get('Crossref_BibTeX', row['BibTeX']),
                        row.get('Title_Similarity', 0), row.get('Journal_Abbreviation', ''),
                        row.get('Crossref_BibTeX_Abbrev', row['BibTeX']),
                        row.get('Crossref_BibTeX_Protected', row['BibTeX']),
                        row.get('Imported_Date', datetime.now().isoformat())
                    ))
                except sqlite3.IntegrityError:
                    cursor.execute('''
                        UPDATE bibliography SET
                        type=?, authors=?, title=?, journal_booktitle=?, year=?,
                        publisher=?, volume=?, pages=?, doi=?, bibtex=?, 
                        crossref_bibtex=?, title_similarity=?, journal_abbreviation=?,
                        crossref_bibtex_abbrev=?, crossref_bibtex_protected=?, imported_date=?
                        WHERE key=?
                    ''', (
                        row['Type'], row['Authors'], row['Title'], 
                        row['Journal/Booktitle'], row['Year'], row['Publisher'],
                        row.get('Volume', ''), row.get('Pages', ''), row.get('DOI', ''),
                        row['BibTeX'], row.get('Crossref_BibTeX', row['BibTeX']),
                        row.get('Title_Similarity', 0), row.get('Journal_Abbreviation', ''),
                        row.get('Crossref_BibTeX_Abbrev', row['BibTeX']),
                        row.get('Crossref_BibTeX_Protected', row['BibTeX']),
                        row.get('Imported_Date', datetime.now().isoformat()),
                        row['Key']
                    ))
            
            conn.commit()
            db_id = session_id
            conn.close()

        # Prepare response
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
        
        cursor.execute('SELECT COUNT(DISTINCT SUBSTR(year, 1, 4)) FROM bibliography WHERE year IS NOT NULL AND year != ""')
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