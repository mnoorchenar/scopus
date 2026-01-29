---
title: Reference Management System
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "latest"
app_file: app.py
pinned: false
---

# Reference Management System

A simple web app to process and organize BibTeX references.

## What It Does

**Input**: Paste your BibTeX entries  
**Process**: Enrich metadata, abbreviate journals, protect acronyms  
**Output**: View results and save to database  

No local files needed (main.tex, Refs.bib). Everything happens online.

## Quick Start

### Local Setup
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### Hugging Face Spaces
Already deployed at: `huggingface.co/spaces/mnoorchenar/scopus`

## How to Use

1. **Paste BibTeX** into the input box
2. **Choose options**:
   - 🌐 **Enrich**: Fetch latest data from Crossref API (~3-5 sec per reference)
   - 📖 **Abbreviate**: Shorten journal names (requires `ltwa.txt`)
   - 🛡️ **Protect**: Wrap acronyms in braces for LaTeX (`{RNN}`, `{LSTM}`)
   - 💾 **Save**: Store in database for later use
3. **Process** and view results in table, BibTeX, or JSON format
4. **Export** as CSV or .bib file

## Options Explained

| Option | What it does | Notes |
|--------|-------------|-------|
| Enrich with Crossref | Updates references from online database | Slower but most accurate |
| Journal Abbreviations | Converts "Nature" → "Nature" using LTWA standard | Needs `ltwa.txt` file |
| Protect Acronyms | Wraps multi-uppercase words in braces | Prevents LaTeX from lowercasing |
| Save to Database | Stores all versions locally | SQLite - portable & persistent |

## File Structure

```
app.py                    # Main application
templates/index.html      # Web interface
requirements.txt          # Dependencies
refs_management.db        # Database (auto-created)
ltwa.txt                  # Journal abbreviations (optional)
```

## Database

All processed references saved to `refs_management.db`:
- Original BibTeX
- Enriched metadata from Crossref
- Abbreviated journal names
- Acronym-protected versions
- Import timestamp

## Notes

- **No files to upload**: Paste content directly
- **Crossref API**: Free but rate-limited (~3-5 sec per entry)
- **Journal abbreviations**: Optional, but requires `ltwa.txt` with tab-separated format
- **Database**: SQLite - runs locally, no internet needed after processing
- **Export**: Download as CSV or BibTeX anytime

## Example Input

```bibtex
@article{Smith2024,
  author = {Smith, John},
  title = {Machine Learning for NLP},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2024}
}
```

## Troubleshooting

**App won't start**: Check `python app.py` runs on port 7860 (Hugging Face Spaces)

**"ltwa.txt not found"**: Optional file - app works without it (abbreviation feature just won't activate)

**Crossref slow**: Normal - API has rate limits. 10 references ≈ 30-50 seconds

**Results not saving**: Ensure "Save to Database" checkbox is enabled

## Learn More

- Crossref API: https://crossref.org
- Journal abbreviations: https://www.nlm.nih.gov/bsd/ltwa_mainpage.html