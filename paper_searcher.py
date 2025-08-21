import streamlit as st
import os
import json
import re
import base64
import pymupdf  # PyMuPDF
import bibtexparser
from pathlib import Path
import pickle
import time
from datetime import datetime
import plistlib
import subprocess

st.set_page_config(page_title="Political Science Paper Search", layout="wide")

# Initialize session state variables
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
    
if 'has_searched' not in st.session_state:
    st.session_state.has_searched = False
    
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
    
# Track which expanders are open
if 'open_expanders' not in st.session_state:
    st.session_state.open_expanders = set()

# Function to toggle expander state
def toggle_expander(index):
    if index in st.session_state.open_expanders:
        st.session_state.open_expanders.remove(index)
    else:
        st.session_state.open_expanders.add(index)

# Function to open PDF without rerunning the search
def open_pdf(pdf_path, index):
    try:
        # Make sure this expander stays open
        if index not in st.session_state.open_expanders:
            st.session_state.open_expanders.add(index)
        
        subprocess.run(['open', pdf_path], check=True)
        st.success(f"Opening PDF {index+1}")
    except Exception as e:
        st.error(f"Error opening PDF: {str(e)}")


def decode_bibdesk_file_path(encoded_path):
    """Decode BibDesk encoded file path to reconstruct the actual file path."""
    try:
        # Decode the base64 data
        decoded_data = base64.b64decode(encoded_path)
        
        # Parse the binary plist data
        plist_data = plistlib.loads(decoded_data)
        
        # The relativePath key contains the file path
        if 'relativePath' in plist_data:
            return plist_data['relativePath']
        elif '$objects' in plist_data:
            # Sometimes the path is in a nested structure
            # Look through objects for strings that look like paths
            for obj in plist_data['$objects']:
                if isinstance(obj, str) and ('/' in obj or '.pdf' in obj.lower()):
                    # This is likely the path
                    return obj
                    
        return None
    except Exception as e:
        print(f"Error decoding bdsk-file-1: {e}")
        return None 
 
def get_pdf_path_from_entry(entry, base_dir=None):
    """Extract PDF path from a BibTeX entry with debug information."""
    # Look for the bdsk-file field(s) that contain the path to the PDF
    for field in entry.keys():
        if field.startswith('bdsk-file'):
            # Debug: Print the raw field content
            print(f"Raw field content: {entry[field]}")
            
            decoded_path = decode_bibdesk_file_path(entry[field])
            if base_dir:
                decoded_path = os.path.join(base_dir, decoded_path)
            print(f"Decoded path: {decoded_path}")
            
            if decoded_path and os.path.exists(decoded_path):
                print(f"File exists: {decoded_path}")
                return decoded_path
            elif decoded_path:
                print(f"File NOT found: {decoded_path}")
                
    return None

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""


def build_search_index(bib_file, update_only=False, progress_callback=None):
    """Process BibTeX entries and build a search index"""
    try:
        # Create storage directory if it doesn't exist
        storage_dir = Path("pdf_text_store")
        storage_dir.mkdir(exist_ok=True)
        
        # Create error log directory
        error_dir = storage_dir / "errors"
        error_dir.mkdir(exist_ok=True)
        
        # Create error log file
        error_log = error_dir / f"indexing_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        error_log_file = open(error_log, 'w', encoding='utf-8')
        
        # Load existing index if it exists
        index_file = storage_dir / "index.pkl"
        if index_file.exists():
            with open(index_file, 'rb') as f:
                index = pickle.load(f)
        else:
            index = {}
        
        # Parse BibTeX file
        with open(bib_file, 'r', encoding='utf-8') as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        # get the base dir of the bibtex file
        base_dir = os.path.dirname(bib_file)
        
        # Process each entry
        new_entries = 0
        skipped_entries = 0
        error_entries = 0
        
        total_entries = len(bib_database.entries)
        
        if progress_callback:
            progress_callback(0, total_entries)
        
        for i, entry in enumerate(bib_database.entries):
            if progress_callback:
                progress_callback(i, total_entries)
                
            entry_id = entry.get('ID', '')
            
            # Skip if already processed and only updating
            if update_only and entry_id in index and index[entry_id].get('processed', False):
                skipped_entries += 1
                continue
            
            # Get PDF path
            pdf_path = get_pdf_path_from_entry(entry, base_dir=base_dir)
            
            # Log error if no path found
            if not pdf_path:
                error_log_file.write(f"ERROR - No PDF path found for: {entry_id}\n")
                error_entries += 1
                continue
            
            # Check if file exists
            if not os.path.exists(pdf_path):
                error_log_file.write(f"ERROR - PDF file not found: {entry_id} -> {pdf_path}\n")
                error_entries += 1
                continue
            
            # Extract text
            try:
                text = extract_text_from_pdf(pdf_path)
                if not text:
                    error_log_file.write(f"ERROR - No text extracted from: {entry_id} -> {pdf_path}\n")
                    error_entries += 1
                    continue
            except Exception as e:
                error_log_file.write(f"ERROR - Text extraction failed: {entry_id} -> {pdf_path} -> {str(e)}\n")
                error_entries += 1
                continue
            
            # Store metadata and text
            index[entry_id] = {
                'title': entry.get('title', ''),
                'author': entry.get('author', ''),
                'year': entry.get('year', ''),
                'journal': entry.get('journal', ''),
                'keywords': entry.get('keywords', ''),
                'pdf_path': pdf_path,
                'text': text,
                'processed': True,
                'processed_time': datetime.now().isoformat()
            }
            
            new_entries += 1
            
            # Save text to individual file as backup
            text_file = storage_dir / f"{entry_id}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
        
        # Final progress update
        if progress_callback:
            progress_callback(total_entries, total_entries)
            
        # Close error log
        error_log_file.close()
        
        # Save index
        with open(index_file, 'wb') as f:
            pickle.dump(index, f)
        
        return {
            'total': len(bib_database.entries),
            'new': new_entries,
            'skipped': skipped_entries,
            'errors': error_entries,
            'error_log': str(error_log)
        }
    
    except Exception as e:
        st.error(f"Error building search index: {str(e)}")
        return None


def exclude_reference_matches(doc, query, page_num, search_results):
    """
    Filter out search results that appear in reference sections.
    Returns filtered search results.
    """
    # Skip all checks if this isn't a reference section
    if not is_reference_section(doc, page_num):
        return search_results
        
    # If we're in a reference section, remove all matches
    return []

def search_papers(query, index, exclude_references=True):
    """
    Search for papers matching the query with option to exclude reference matches.
    
    Parameters:
    - query: The search term to look for
    - index: Dictionary of indexed paper data
    - exclude_references: Whether to exclude matches found in reference sections
    
    Returns:
    - List of result dictionaries with matching papers and highlighted snippets
    """
    results = []
    
    # Step 1: Fast regex search to identify matching documents
    matching_entries = []
    for entry_id, entry_data in index.items():
        if query.lower() in entry_data.get('text', '').lower():
            matching_entries.append((entry_id, entry_data))

    st.write(f"Found {len(matching_entries)} initial keyword matches.")
    
    # Step 2: Use PyMuPDF for detailed searching in matching documents
    progress_bar = st.progress(0)
    total_entries = len(matching_entries)
    
    for i, (entry_id, entry_data) in enumerate(matching_entries):
        # Update progress bar
        progress_bar.progress((i + 1) / total_entries)
        pdf_path = entry_data.get('pdf_path', '')
        matches = []
        
        try:
            # Open the PDF only if we found matches in Step 1
            doc = pymupdf.open(pdf_path)
            
            # Initialize set to track reference section pages
            reference_pages = set()
            
            # Pre-identify reference sections if we're excluding them
            if exclude_references:
                for page_num in range(len(doc)):
                    if is_reference_section(doc, page_num):
                        reference_pages.add(page_num)
                        # Once we find a reference section, all subsequent pages
                        # are likely also references in academic papers
                        for p in range(page_num + 1, len(doc)):
                            reference_pages.add(p)
                            
                        # Optimization: stop scanning once we find the start of references
                        break
            
            # Search through each page
            match_pages = []
            for page_num in range(len(doc)):
                # Skip reference sections if we're excluding them
                if exclude_references and page_num in reference_pages:
                    continue
                    
                page = doc[page_num]
                # Use built-in search function to find matches
                search_results = page.search_for(query)
                
                if search_results:
                    match_pages.append((page_num, search_results))
            
            # Limit to top matches for display
            match_count = 0
            for page_num, search_results in match_pages:
                if match_count >= max_matches:
                    break
                    
                page = doc[page_num]
                
                for rect in search_results:
                    if match_count >= max_matches:
                        break
                        
                    # Store the original search rectangle
                    search_rect = rect
                    
                    # Expand rectangle for context
                    clip_rect = pymupdf.Rect(rect)
                    clip_rect.x0 = max(0, clip_rect.x0 - 400)
                    clip_rect.y0 = max(0, clip_rect.y0 - 100)
                    clip_rect.x1 = min(page.rect.width, clip_rect.x1 + 400)
                    clip_rect.y1 = min(page.rect.height, clip_rect.y1 + 100)
                    
                    # Set zoom factor
                    zoom = 4 
                    mat = pymupdf.Matrix(zoom, zoom)
                    
                    # Render the page section
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
                    
                    # Convert the search rectangle to image coordinates
                    search_rect_in_image = pymupdf.Rect(
                        (search_rect.x0 - clip_rect.x0) * zoom,
                        (search_rect.y0 - clip_rect.y0) * zoom,
                        (search_rect.x1 - clip_rect.x0) * zoom,
                        (search_rect.y1 - clip_rect.y0) * zoom
                    )
                    
                    # Draw the highlight on the image
                    # This requires PIL/Pillow
                    from PIL import Image, ImageDraw
                    import io
                    
                    # Convert PyMuPDF pixmap to PIL Image
                    img_data = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(img_data))
                    
                    # Create a drawing context
                    draw = ImageDraw.Draw(pil_img, "RGBA")
                    
                    # Draw a semi-transparent yellow rectangle for the highlight
                    draw.rectangle(
                        [
                            search_rect_in_image.x0, 
                            search_rect_in_image.y0,
                            search_rect_in_image.x1, 
                            search_rect_in_image.y1
                        ],
                        fill=(255, 255, 0, 128)  # Yellow with 50% transparency
                    )
                    
                    # Save to a temporary file
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                        temp_filename = temp.name
                        pil_img.save(temp_filename)
                    
                    # Get text context from this area
                    text_context = page.get_text("text", clip=clip_rect)
                    
                    # Highlight the search term in the text
                    if query.lower() in text_context.lower():
                        text_lower = text_context.lower()
                        start_pos = text_lower.find(query.lower())
                        if start_pos >= 0:
                            end_pos = start_pos + len(query)
                            text_context = (
                                text_context[:start_pos] + 
                                "**" + text_context[start_pos:end_pos] + "**" + 
                                text_context[end_pos:]
                            )
                    
                    matches.append({
                        'page': page_num + 1,
                        'image_path': temp_filename,
                        'text': text_context,
                        'rect': {
                            'x0': clip_rect.x0, 
                            'y0': clip_rect.y0,
                            'x1': clip_rect.x1,
                            'y1': clip_rect.y1
                        },
                        'is_reference': page_num in reference_pages  # Flag if this is from references
                    })
                    match_count += 1
            
            doc.close()
            
        except Exception as e:
            # Fallback to text matches if visual extraction fails
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            # TODO: figure out how to handle this better if it happens a lot?
        
        if matches:
            results.append({
                'id': entry_id,
                'title': entry_data.get('title', ''),
                'author': entry_data.get('author', ''),
                'year': entry_data.get('year', ''),
                'journal': entry_data.get('journal', ''),
                'keywords': entry_data.get('keywords', ''),
                'pdf_path': pdf_path,
                'matches': matches,
                'has_non_reference_matches': any(not m.get('is_reference', False) for m in matches)
            })
    
    # If excluding references, only return papers with at least one substantive match
    if exclude_references:
        results = [r for r in results if r.get('has_non_reference_matches', False)]
    
    return results

def is_reference_section(doc, page_num):
    """
    Use some simple rules to try to figure out what the references section is.
    (The reason we want this is that matches to references aren't very useful, especially
    if we end up keyword matching a widely-cited title.)
    
    Parameters:
    - doc: PyMuPDF document object
    - page_num: Page number to check
    
    Returns:
    - Boolean indicating if this page is likely a references section
    """
    # Get current page text
    page = doc[page_num]
    page_text = page.get_text("text")
    
    # 1. Check for reference section headers
    ref_headers = [
        r"(?i)^\s*references\s*$",
        r"(?i)^\s*bibliography\s*$",
        r"(?i)^\s*works\s+cited\s*$",
        r"(?i)^\s*literature\s+cited\s*$",
        r"(?i)^\s*references\s+and\s+notes\s*$",
        r"(?i)^\s*notes\s+and\s+references\s*$"
    ]
    
    header_match = any(re.search(pattern, page_text, re.MULTILINE) for pattern in ref_headers)
    
    # 2. Check for citation patterns commonly found in reference lists
    # Based on your examples
    citation_patterns = [
        r"\[\d+(?:,\s*\d+)*\]",               # [1] or [230,235,256]
        r"[A-Z][A-Za-z]+,\s+[A-Z]\.\s+\(\d{4}\)",  # Lastname, F. (YYYY)
        r"\b(?:19|20)\d{2}[a-z]?\b[,\.]",     # Years (1900-2099) followed by comma or period
        r"(?i)journal\s+of",                  # Journal of...
        r"Available at https?://",             # URL references
        r"Working Paper",                      # Working papers
        r"arXiv:",        # arXiv preprints
        r"et al\.,?\s+\d{4}",                 # et al., YYYY
        r"^\s*\d+\.\s+[A-Z][a-z]+",           # Numbered reference format: 1. Author
    ]
    
    pattern_matches = sum(len(re.findall(pattern, page_text)) for pattern in citation_patterns)
    text_length = len(page_text)
    
    # Calculate citation density (per 1000 characters)
    citation_density = pattern_matches / max(text_length, 1) * 1000
    
    # Higher threshold for citation density to avoid false positives
    very_high_citation_density = citation_density > 8
    high_citation_density = citation_density > 5
    
    # 3. Position in document
    total_pages = len(doc)
    is_in_last_third = page_num >= (total_pages * 2/3)
    
    # 4. Check for lines with year patterns (common in references)
    year_pattern = r"\b(?:19|20)\d{2}[a-z]?\b"
    lines = page_text.split('\n')
    lines_with_years = sum(1 for line in lines if re.search(year_pattern, line))
    year_line_ratio = lines_with_years / max(len(lines), 1)
    high_year_ratio = year_line_ratio > 0.4  # 40% of lines have years
    
    # Decision rules:
    # 1. If there's a reference header, it's definitely a reference section
    if header_match:
        return True
        
    # 2. If we're in the last third of the document AND there's a high citation density,
    #    it's likely a reference section
    if is_in_last_third and very_high_citation_density:
        return True
        
    # 3. If we're in the last third AND have both high year ratio and high citation density
    if is_in_last_third and high_year_ratio and high_citation_density:
        return True
        
    # 4. For pages after a confirmed reference section, check if citation density remains high
    if page_num > 0:
        prev_page = doc[page_num - 1]
        prev_text = prev_page.get_text("text")
        prev_header_match = any(re.search(pattern, prev_text, re.MULTILINE) for pattern in ref_headers)
        
        # If previous page had a reference header and this page has some citations,
        # consider this a continuation of references
        if prev_header_match and citation_density > 3:
            return True
    
    # By default, not a reference section
    return False

def perform_search(query, index, exclude_references=True):
    if query and (query != st.session_state.current_query or not st.session_state.has_searched):
        with st.spinner("Finding relevant passages..."):
            results = search_papers(query, index, exclude_references=exclude_references)
            
            # Process images and store them permanently for this session
            process_and_store_images(results)
            
            # Store in session state
            st.session_state.search_results = results
            st.session_state.has_searched = True
            st.session_state.current_query = query
            
            return results
    else:
        # Return cached results
        return st.session_state.search_results

# Function to process images and store them permanently for this session
def process_and_store_images(results):
    # Create a directory for storing images if it doesn't exist
    image_dir = Path("temp_images")
    image_dir.mkdir(exist_ok=True)
    
    # Process each result and its matches
    for result in results:
        for match in result['matches']:
            # Check if we already have a permanent path for this image
            if 'permanent_image_path' not in match and 'image_path' in match and match['image_path']:
                temp_path = match['image_path']
                if os.path.exists(temp_path):
                    # Create a permanent file path
                    permanent_path = image_dir / f"{hash(temp_path)}.png"
                    
                    # Copy the image to the permanent location if it doesn't exist
                    if not os.path.exists(permanent_path):
                        import shutil
                        shutil.copy2(temp_path, permanent_path)
                    
                    # Store the permanent path
                    match['permanent_image_path'] = str(permanent_path)
                    
                    # Now we can safely delete the temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

def get_image_cache_size():
    """
    Get the on-disk size of the temp_images directory (in MB)
    """
    image_dir = Path("temp_images")
    if image_dir.exists():
        total_size = sum(f.stat().st_size for f in image_dir.glob('**/*') if f.is_file())
        return total_size / (1024 * 1024)
    return 0


def clear_image_cache():
    """
    Delete the cached images
    """
    image_dir = Path("temp_images")
    if image_dir.exists():
        for f in image_dir.glob('**/*'):
            if f.is_file():
                try:
                    os.unlink(f)
                except Exception as e:
                    print(f"Error deleting file {f}: {e}")

# Streamlit UI
st.title("Political Science Paper Search")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # BibTeX file input
    bib_file = st.text_input("Path to BibTeX file", value="/Users/ahalterman/MIT/MIT.bib", help="Enter the full path to your BibTeX file (e.g., /Users/username/library.bib)")

    # Context window size
    context_window = st.slider("Context Window Size", min_value=0, max_value=500, value=100, help="Number of characters to show before and after the search term in the results") 
    max_matches = st.slider("Max Matches per Document", min_value=1, max_value=20, value=10, help="Maximum number of matches to display per document")
    show_raw_text = st.checkbox("Show Raw Text Context", value=False, help="Display the raw text context around the search term")

    exclude_references = st.checkbox(
        "Exclude matches in references", 
        value=True,
        help="Don't show matches in what appears to be bibliography/references sections"
    )

    cache_size = get_image_cache_size()
    st.info(f"Image cache size: {cache_size:.2f} MB")
    if st.button("Clear Image Cache"):
        clear_image_cache()
        st.success("Image cache cleared")

    # index building/updating
    st.subheader("Index Management")
    
    if st.button("Build/Update Index"):
        if bib_file and os.path.exists(bib_file):
            with st.spinner("Building search index..."):
                start_time = time.time()
                # Create progress bar
                progress_bar = st.progress(0)
            
                # Create a wrapper function to update progress
                def progress_callback(current, total):
                    if total > 0:
                        progress_bar.progress(current / total)
            
                # Pass the callback to build_search_index
                stats = build_search_index(bib_file, update_only=True, progress_callback=progress_callback)
            
                # Complete the progress bar
                progress_bar.progress(1.0)
                end_time = time.time()
                
                if stats:
                    st.success(f"Index built in {end_time - start_time:.2f} seconds")
                    st.write(f"Total entries: {stats['total']}")
                    st.write(f"New entries processed: {stats['new']}")
                    st.write(f"Skipped entries: {stats['skipped']}")
                    st.write(f"Entries with errors: {stats['errors']}")
        else:
            st.error("Please provide a valid BibTeX file path")
    
    if st.button("Rebuild Index (Process All)", key="rebuild_all"):
        if bib_file and os.path.exists(bib_file):
            with st.spinner("Rebuilding search index (processing all entries)..."):
                start_time = time.time()
                stats = build_search_index(bib_file, update_only=False)
                end_time = time.time()
                
                if stats:
                    st.success(f"Index rebuilt in {end_time - start_time:.2f} seconds")
                    st.write(f"Total entries: {stats['total']}")
                    st.write(f"Entries processed: {stats['new']}")
                    st.write(f"Entries with errors: {stats['errors']}")
        else:
            st.error("Please provide a valid BibTeX file path")


# Main content area
st.header("Paper Search")

# Check if index exists
index_file = Path("pdf_text_store/index.pkl")
if not index_file.exists():
    st.warning("No search index found. Please build the index using the sidebar options.")
else:
    # Load the index
    with open(index_file, 'rb') as f:
        index = pickle.load(f)
    
    st.info(f"Search index contains {len(index)} papers")
    
    # Search interface
    query = st.text_input("Search Query", help="Enter keywords to search for in the full text of papers")
    

    # In the main content area of your Streamlit app
    if st.button("Search", key="search"):
        if not query:
            st.warning("Please enter a search query")
        else:
            results = perform_search(query, index, exclude_references=exclude_references)
        if results:
            st.success(f"Found {len(results)} papers matching '{query}'")

            # order results by number of matches
            results.sort(key=lambda x: len(x['matches']), reverse=True)
            
            for i, result in enumerate(results):
                # Check if this expander should be open
                is_open = i in st.session_state.open_expanders
                
                # Using the default_expanded parameter to control expander state
                with st.expander(f"**{result['title']} ({result['year']})**", expanded=is_open):
                    # We'll use a different approach without the invisible toggle button
                    
                    st.write(f"Authors: **{result['author']}**")
                    st.write(f"Journal: *{result['journal']}*")
                    st.write(f"Keywords: {result['keywords']}")
                    
                    st.write("**Matches:**")
                    for j, match in enumerate(result['matches']):
                        st.write(f"**Page {match['page']}:**")
                        st.write()
                        
                        # Display visual snippet if available
                        if 'permanent_image_path' in match and os.path.exists(match['permanent_image_path']):
                            # Use the permanent image path if available
                            st.image(match['permanent_image_path'], caption=f"Match {j+1}", width=800)
                        elif match['image_path'] and os.path.exists(match['image_path']):
                            # Fallback to the original temporary path
                            st.image(match['image_path'], caption=f"Match {j+1}", width=800)
                            
                            # Clean up the temporary file after displaying
                            try:
                                os.unlink(match['image_path'])
                            except:
                                pass
                        
                        if show_raw_text:
                            # Always display text context
                            st.markdown(match['text'])
                        
                        st.markdown("---")
                    
                    st.write(f"**PDF Path:** {result['pdf_path']}")
                    
                    # Add an open PDF button that uses a callback
                    pdf_path = result['pdf_path']
                    st.button(
                        f"Open PDF", 
                        key=f"open_{i}_{hash(pdf_path)}", 
                        on_click=open_pdf, 
                        args=(pdf_path, i)
                    )
        else:
            st.info(f"No papers found matching '{query}'")

    if st.button("Show Sample Entry", key="show_sample"):
        if bib_file and os.path.exists(bib_file):
            with open(bib_file, 'r', encoding='utf-8') as bibtex_file:
                bib_database = bibtexparser.load(bibtex_file)

            if bib_database.entries:
                st.subheader("Sample BibTeX Entry")
                sample_entry = bib_database.entries[0]
                st.json(sample_entry)

                # Try to extract PDF path
                st.subheader("PDF Path Extraction Test")
                for field in sample_entry.keys():
                    if field.startswith('bdsk-file'):
                        st.write(f"Field: {field}")
                        st.write(f"Raw value: {sample_entry[field]}")
                        decoded = decode_bibdesk_file_path(sample_entry[field])
                        st.write(f"Decoded path: {decoded}")
                        if decoded:
                            st.write(f"File exists: {os.path.exists(decoded)}")