# Reference Management System - Flask Application

A web-based application for processing, enriching, and managing BibTeX references. Users paste their BibTeX entries, apply optional processing steps, view results, and save to a local database.

## Features

- **Paste & Process**: Directly input BibTeX entries without needing files
- **Crossref Enrichment**: Fetch updated metadata from Crossref API
- **Journal Abbreviations**: Apply LTWA abbreviations using local `ltwa.txt` file
- **Acronym Protection**: Wrap acronyms in braces to preserve LaTeX capitalization
- **Database Storage**: Save all versions to SQLite database
- **Multiple Export Formats**: Export as CSV or BibTeX file
- **Database Management**: View, delete, and manage stored references
- **Real-time Statistics**: Track total entries, types, and years

## Installation

### Requirements
- Python 3.8+
- pip

### Setup Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Journal Abbreviations (Optional)**
   - Place `ltwa.txt` in the same directory as `app.py`
   - Format: Tab-separated values
     ```
     Journal Full Name	Abbrev
     Nature Biotechnology	Nat Biotechnol
     ```

3. **Run the Application**
   ```bash
   python app.py
   ```
   - Open browser to `http://localhost:5000`

## Usage

### Basic Workflow

1. **Paste BibTeX**
   - Copy your BibTeX entries into the input area
   - Supports multiple entries

2. **Select Processing Options**
   - ✅ **Enrich with Crossref**: Query Crossref API (3-5 sec per entry)
   - ✅ **Apply Journal Abbreviations**: Requires ltwa.txt file
   - ✅ **Protect Acronyms**: Wraps multi-uppercase words in braces
   - ✅ **Save to Database**: Store for later access

3. **Process & View Results**
   - Click "Process References"
   - View in three formats: Table, BibTeX, JSON
   - Results include metadata, similarity scores, multiple BibTeX versions

4. **Manage & Export**
   - View database entries with statistics
   - Delete individual entries
   - Export all data as CSV or BibTeX

### Processing Options Explained

#### Enrich with Crossref
- Queries Crossref API for each reference
- Fetches complete metadata and BibTeX
- Shows title similarity percentage (95%+ uses Crossref data)
- Slower but most accurate for published papers

#### Journal Abbreviations
- **Requires**: `ltwa.txt` file in app directory
- Replaces full journal names with standard abbreviations
- Essential for formatting journal names correctly

#### Protect Acronyms
- Wraps acronyms/multi-uppercase words in braces: `{RNN}`, `{LSTM}`
- Prevents LaTeX from lowercasing acronyms in titles
- Applied to title, booktitle, and journal fields

#### Save to Database
- Stores original and all processed versions
- Preserves session/import date
- Enables future bulk exports

## Database Structure

The SQLite database (`refs_management.db`) includes:

| Column | Purpose |
|--------|---------|
| id | Primary key |
| key | BibTeX citation key |
| type | Entry type (article, book, etc.) |
| authors | Author names |
| title | Paper/book title |
| journal_booktitle | Journal or conference name |
| year | Publication year |
| bibtex | Original user-provided BibTeX |
| crossref_bibtex | Crossref API fetched version |
| crossref_bibtex_abbrev | With journal abbreviations |
| crossref_bibtex_protected | With protected acronyms |
| title_similarity | Crossref match percentage |
| imported_date | When imported |

## API Endpoints

### Process References
- **POST** `/api/process`
- **Body**: 
  ```json
  {
    "bibtex_content": "...",
    "enrich": true,
    "abbreviate": true,
    "protect": true,
    "save_to_db": true
  }
  ```

### Database Management
- **GET** `/api/database/entries` - List all entries
- **DELETE** `/api/database/delete/<key>` - Delete entry
- **GET** `/api/database/export` - Export CSV
- **GET** `/api/database/export-bibtex` - Export BibTeX

### Statistics
- **GET** `/api/stats` - Database statistics

## File Structure

```
.
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Web interface
├── requirements.txt      # Python dependencies
├── refs_management.db    # Database (created automatically)
└── ltwa.txt             # Journal abbreviations (optional)
```

## Example BibTeX Input

```bibtex
@article{Smith2024,
  author = {Smith, John and Doe, Jane},
  title = {Machine Learning for Natural Language Processing},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2024},
  volume = {46},
  pages = {1234-1250}
}

@book{Johnson2023,
  author = {Johnson, Michael},
  title = {Deep Learning Fundamentals},
  publisher = {Academic Press},
  year = {2023}
}
```

## Troubleshooting

### "No journal abbreviations found"
- Ensure `ltwa.txt` is in the same directory as `app.py`
- Verify tab-separated format

### Crossref enrichment is slow
- Normal behavior (API rate limiting)
- Average: 3-5 seconds per entry
- Results cached in database

### Database not updating
- Check `refs_management.db` permissions
- Ensure "Save to Database" option is checked

### Port 5000 already in use
- Change in app.py: `app.run(debug=True, port=5001)`

## Performance Notes

- **Crossref API**: ~3-5 seconds per reference (rate-limited)
- **Batch processing**: Process 10 references in ~30-50 seconds
- **Database**: Handles 1000+ entries efficiently

## Future Enhancements

- Multiple database tables for different projects
- BibTeX key normalization
- Duplicate detection
- Advanced search/filtering
- Batch upload capability

## Support

For issues with the Crossref API, visit: https://crossref.org
For journal abbreviation standards: https://www.nlm.nih.gov/bsd/ltwa_mainpage.html

## License

This application is provided as-is for reference management purposes.