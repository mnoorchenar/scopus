"""
📚 Reference Management Pipeline (Final Updated Version)

Steps:
1. Parse main.tex citations
2. Parse Refs.bib entries
3. Merge citations with BibTeX
4. Save into SQLite DB (table: Refs)
5. Enrich with Crossref BibTeX
6. Add Journal abbreviations (custom capitalization rules)
7. Create Crossref_BibTeX_Abbrev (fixed keys + abbrev journals)
8. Create Crossref_BibTeX_Protected (acronyms preserved with braces)

Author: <you>
"""

import re
import time, random
import requests
import sqlite3
import pandas as pd
from difflib import SequenceMatcher

# Prepositions to keep lowercase in abbreviations
LOWERCASE_WORDS = {"and", "or", "in", "on", "of", "for", "to", "the", "a", "an"}


# ---------------------------------------------------------------------
# 1) Parse LaTeX citations
# ---------------------------------------------------------------------
def parse_citations_from_tex(tex_file: str) -> pd.DataFrame:
    print("📖 Parsing citations from", tex_file)
    with open(tex_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    clean_text = "\n".join(line for line in lines if not line.strip().startswith("%"))

    section_pattern = re.compile(r'\\section\{([^}]*)\}(?:\\label\{[^}]*\})?')
    cite_pattern = re.compile(r'\\cite\{([^}]*)\}')
    sections = section_pattern.split(clean_text)

    citations, ref_sections = [], {}
    for i in range(1, len(sections), 2):
        section_name = sections[i].strip()
        section_text = sections[i+1]
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


# ---------------------------------------------------------------------
# 2) Parse BibTeX
# ---------------------------------------------------------------------
def parse_bibtex_to_dataframe(bib_file: str) -> pd.DataFrame:
    print("📖 Parsing BibTeX from", bib_file)
    with open(bib_file, "r", encoding="utf-8") as f:
        content = f.read()

    entries = ["@" + e for e in content.split("@") if e.strip()]
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
            "Key": entry_key,
            "Type": entry_type,
            "Authors": fields.get("author", "").strip(),
            "Title": fields.get("title", "").strip(),
            "Journal/Booktitle": fields.get("journal", fields.get("booktitle", "")).strip(),
            "Year": fields.get("year", "").strip(),
            "Publisher": fields.get("publisher", fields.get("organization", "")).strip(),
            "BibTeX": entry.strip()
        })

    df = pd.DataFrame(papers).drop_duplicates(subset="Key", keep="first").reset_index(drop=True)
    print(f"✅ Parsed {len(df)} BibTeX records")
    return df


# ---------------------------------------------------------------------
# 3) Merge
# ---------------------------------------------------------------------
def merge_citations_with_bib(main_text_df: pd.DataFrame, references_bib_df: pd.DataFrame) -> pd.DataFrame:
    print("🔗 Merging citations with BibTeX")
    bib_lookup = references_bib_df.set_index("Key").to_dict(orient="index")
    merged_records = []
    for _, row in main_text_df.iterrows():
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
            "BibTeX": bib_info.get("BibTeX", "")
        })
    df = pd.DataFrame(merged_records)
    print(f"✅ Merged into {len(df)} rows")
    return df


# ---------------------------------------------------------------------
# 4) Save to DB
# ---------------------------------------------------------------------
def save_to_sqlite(df: pd.DataFrame, db_path="Refs.db", table="Refs"):
    print(f"💾 Saving DataFrame into SQLite DB: {db_path}, table={table}")
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print("✅ Data saved")


# ---------------------------------------------------------------------
# 5) Enrich with Crossref
# ---------------------------------------------------------------------
def enrich_references_with_crossref(db_path="Refs.db", table="Refs", topn=3):
    print("🌐 Enriching references with Crossref metadata")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(f'PRAGMA table_info("{table}")')
    existing_cols = [col[1] for col in cur.fetchall()]
    if "Crossref_BibTeX" not in existing_cols:
        cur.execute(f'ALTER TABLE "{table}" ADD COLUMN Crossref_BibTeX TEXT;')
    if "Title_Similarity" not in existing_cols:
        cur.execute(f'ALTER TABLE "{table}" ADD COLUMN Title_Similarity INTEGER;')
    conn.commit()

    cur.execute(f'SELECT Reference, Title, Authors, "Journal/Booktitle", Year, Publisher, BibTeX FROM "{table}"')
    rows = cur.fetchall()

    for i, (ref, title, authors, journal, year, publisher, local_bib) in enumerate(rows, start=1):
        print(f"\n[{i}/{len(rows)}] Processing Reference={ref}")

        if not title:
            print("⚠️ Skipping (no title)")
            continue

        query = " ".join(filter(None, [title, authors.split(',')[0] if authors else "", journal, year, publisher]))
        url = f"https://api.crossref.org/works?query.bibliographic={requests.utils.quote(query)}&rows={topn}"

        crossref_bibtex, best_score = "", 0
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])

            best = None
            for item in items:
                cr_title = item.get("title", [""])[0]
                score = SequenceMatcher(None, title.lower(), cr_title.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best = item

            best_score = int(round(best_score * 100, 0))
            if best and "DOI" in best:
                doi = best["DOI"]
                bibtex_r = requests.get(
                    f"https://doi.org/{doi}",
                    headers={"Accept": "application/x-bibtex"},
                    timeout=15
                )
                if bibtex_r.status_code == 200:
                    crossref_bibtex = bibtex_r.text.strip()

        except Exception as e:
            print(f"⚠️ Crossref fetch failed for {ref}: {e}")

        if best_score < 95:
            crossref_bibtex = local_bib
            print(f"ℹ️ Low similarity ({best_score}%), using local BibTeX")

        cur.execute(
            f'UPDATE "{table}" SET Crossref_BibTeX=?, Title_Similarity=? WHERE Reference=?',
            (crossref_bibtex, best_score, ref)
        )
        conn.commit()
        print(f"✅ Updated: Similarity={best_score}%, BibTeX length={len(crossref_bibtex)}")
        time.sleep(random.uniform(2, 5))

    conn.close()
    print("🎉 Crossref enrichment done")


# ---------------------------------------------------------------------
# 6) Abbreviate journals (new rules)
# ---------------------------------------------------------------------
def abbreviate_journal_custom(title: str) -> str:
    """Custom abbreviation: capitalize, no dots, prepositions lowercase."""
    if not title:
        return ""
    words = title.split()
    abbr = []
    for i, word in enumerate(words):
        if word.lower() in LOWERCASE_WORDS and i != 0:
            abbr.append(word.lower())
        else:
            abbr.append(word.capitalize() if len(word) <= 4 else word[:4].capitalize())
    return " ".join(abbr)


def add_journal_abbreviations(db_path="Refs.db", table="Refs"):
    print("🔤 Adding journal abbreviations (custom rules)")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(f'PRAGMA table_info("{table}")')
    col_names = [c[1] for c in cur.fetchall()]
    if "Journal_Abbrev" not in col_names:
        new_cols = []
        for name in col_names:
            new_cols.append(name)
            if name == "Journal/Booktitle":
                new_cols.append("Journal_Abbrev")
        col_defs = ", ".join(f'"{c}" TEXT' for c in new_cols)
        cur.execute(f'CREATE TABLE "{table}_new" ({col_defs});')
        select_expr = ", ".join([f'"{c}"' if c != "Journal_Abbrev" else "NULL" for c in new_cols])
        cur.execute(f'INSERT INTO "{table}_new" SELECT {select_expr} FROM "{table}";')
        cur.execute(f'DROP TABLE "{table}";')
        cur.execute(f'ALTER TABLE "{table}_new" RENAME TO "{table}";')
        conn.commit()

    cur.execute(f'SELECT Reference, "Journal/Booktitle" FROM "{table}"')
    for ref, journal in cur.fetchall():
        abbrev = abbreviate_journal_custom(journal)
        cur.execute(f'UPDATE "{table}" SET Journal_Abbrev=? WHERE Reference=?', (abbrev, ref))
    conn.commit()
    conn.close()
    print("✅ Journal abbreviations updated")


# ---------------------------------------------------------------------
# 7) Fix Crossref BibTeX with abbreviations
# ---------------------------------------------------------------------
def add_crossref_bibtex_with_abbrev(db_path="Refs.db", table="Refs"):
    print("🛠️ Creating Crossref_BibTeX_Abbrev")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(f'PRAGMA table_info("{table}")')
    existing_cols = [col[1] for col in cur.fetchall()]
    if "Crossref_BibTeX_Abbrev" not in existing_cols:
        cur.execute(f'ALTER TABLE "{table}" ADD COLUMN Crossref_BibTeX_Abbrev TEXT;')
        conn.commit()

    cur.execute(f'SELECT Reference, Journal_Abbrev, Crossref_BibTeX FROM "{table}"')
    for ref, journal_abbrev, crossref_bib in cur.fetchall():
        if not crossref_bib:
            continue

        new_bib = crossref_bib.strip()
        try:
            start_brace = new_bib.index("{")
            first_comma = new_bib.index(",", start_brace)
            entry_type = new_bib[:start_brace]
            new_start = f"{entry_type}{{{ref},"
            new_bib = new_start + new_bib[first_comma+1:]
        except ValueError:
            pass

        if journal_abbrev:
            new_bib = re.sub(
                r'(journal\s*=\s*\{)[^}]+(\})',
                rf'\1{journal_abbrev}\2',
                new_bib,
                flags=re.IGNORECASE
            )

        cur.execute(f'UPDATE "{table}" SET Crossref_BibTeX_Abbrev=? WHERE Reference=?', (new_bib, ref))
    conn.commit()
    conn.close()
    print("✅ Crossref_BibTeX_Abbrev created")


# ---------------------------------------------------------------------
# 8) Protect acronyms in fields (title, booktitle, journal)
# ---------------------------------------------------------------------
def protect_acronyms_in_fields(bibtex: str) -> str:
    """
    Clean BibTeX fields (title, booktitle, journal):
      - strip spaces before/after content
      - wrap tokens with >=2 uppercase letters (RNN, TinyML, Grad-CAM, SHAP, IEEE, ASHRAE)
      - avoid double {{ }}
      - normalize spaces after braces
    """

    def clean_field(field_name: str, text: str) -> str:
        text = text.strip()

        def wrap_token(token: str) -> str:
            # Already wrapped → leave it
            if token.startswith("{") and token.endswith("}"):
                return token
            # Wrap if token contains ≥2 uppercase letters
            if sum(1 for c in token if c.isupper()) >= 2:
                return "{" + token + "}"
            return token

        # Split by whitespace but keep spaces
        tokens = re.split(r'(\s+)', text)
        fixed = "".join(wrap_token(tok) if tok.strip() else tok for tok in tokens)

        # Remove accidental double braces {{...}} → {...}
        fixed = re.sub(r'\{\{([^{}]+)\}\}', r'{\1}', fixed)

        # Normalize spacing around braces
        fixed = fixed.replace("}} ", "} ").replace("{{ ", "{ ")

        return f"{field_name}={{{fixed.strip()}}}"

    # Apply to title, booktitle, journal
    for field in ["title", "booktitle", "journal"]:
        bibtex = re.sub(
            rf'{field}\s*=\s*\{{([^}}]*)\}}',
            lambda m: clean_field(field, m.group(1)),
            bibtex,
            flags=re.IGNORECASE
        )

    return bibtex

def add_crossref_bibtex_with_protected_titles(db_path="Refs.db", table="Refs"):
    print("🛡️ Creating Crossref_BibTeX_Protected with acronym-safe fields")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(f'PRAGMA table_info("{table}")')
    existing_cols = [col[1] for col in cur.fetchall()]
    if "Crossref_BibTeX_Protected" not in existing_cols:
        cur.execute(f'ALTER TABLE "{table}" ADD COLUMN Crossref_BibTeX_Protected TEXT;')
        conn.commit()

    cur.execute(f'SELECT Reference, Crossref_BibTeX_Abbrev FROM "{table}"')
    for ref, bib in cur.fetchall():
        if not bib:
            continue
        protected_bib = protect_acronyms_in_fields(bib)
        cur.execute(
            f'UPDATE "{table}" SET Crossref_BibTeX_Protected=? WHERE Reference=?',
            (protected_bib, ref)
        )
        print(f"✅ Updated {ref}")

    conn.commit()
    conn.close()
    print("✅ Crossref_BibTeX_Protected created (title, booktitle, journal cleaned)")


# ---------------------------------------------------------------------
# 9) Main entry point
# ---------------------------------------------------------------------
def main():
    answer = input("⚡ Do you have main.tex and Refs.bib ready? (y/n): ").strip().lower()
    if answer != "y":
        print("❌ Exiting. Please prepare main.tex and Refs.bib first.")
        return

    refs_bib = parse_bibtex_to_dataframe("Refs.bib")
    main_text = parse_citations_from_tex("main.tex")
    merged_df = merge_citations_with_bib(main_text, refs_bib)
    merged_df.insert(0, "Index", range(1, len(merged_df) + 1))
    save_to_sqlite(merged_df, "Refs.db", "Refs")

    enrich_references_with_crossref("Refs.db", "Refs")
    add_journal_abbreviations("Refs.db", "Refs")
    add_crossref_bibtex_with_abbrev("Refs.db", "Refs")
    add_crossref_bibtex_with_protected_titles("Refs.db", "Refs")

    print("🎉 All steps completed successfully")


if __name__ == "__main__":
    main()
