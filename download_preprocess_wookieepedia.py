import mwparserfromhell
import json
import re
from lxml import etree


## Load and Parse the XML File

# Download the XML dump from the Star Wars Wiki (Wookieepedia)
# 7z wiki dump downloaded from: https://starwars.fandom.com/wiki/Special:Statistics
# Unpacked using: py7zr, step 2 of https://bitbucket.org/robvanderg/fandom_download/src/master/

# Set the path to the XML file
origin = "/Users/max-peterschroder/Downloads/starwars_pages_current.xml"

# Define the namespace
NAMESPACE = {'ns': 'http://www.mediawiki.org/xml/export-0.11/'}

# Load and parse the XML file
try:
    with open(origin, "rb") as file:
        tree = etree.parse(file)
    root = tree.getroot()
except Exception as e:
    print("Error loading or parsing XML:", e)
    exit()

# Confirm the root and structure by printing the tag names
print(f"Root tag: {root.tag}")
print("Checking for <page> elements...")

# Update the XPath to include the namespace
pages = root.findall(".//ns:page", namespaces=NAMESPACE)
if not pages:
    print("No <page> elements found.")
else:
    print(f"Found {len(pages)} pages.")


## Extract and Process Data

# Define a list of prefixes to exclude
# Found from: https://starwars.fandom.com/wiki/Special:AllPages
# Intend to keep: 
# - Main (no prefix, just normal articles)
# - "Template:" (includes stuff like page infobox types, used to determine type of article)
# - "Category:" (includes stuff like page and template categories, used to determine common themes and topics)
EXCLUDE_PREFIXES = ("Talk:", "User:", "User talk:", "Wookieepedia:", "Wookieepedia talk:", "File:", "File talk:", "MediaWiki:", "MediaWiki talk:", "Template talk:", "Help:", "Help talk:", "Category talk:", "Forum:", "Forum talk:", "Wookieepedia Contests:", "Wookieepedia Contests talk:", "Bracket:", "Bracket talk:", "Instant Expert:", "Instant Expert talk:", "Wookieepedia Help:", "Wookieepedia Help talk:", "Index:", "Index talk:", "GeoJson:", "GeoJson talk:", "User blog:", "Module:", "Module talk:", "Message Wall:", "Gadget:", "Gadget talk:", "Gadget definition:", "Gadget definition talk:", "Map:", "Map talk:")

# Create a list to store the data for each page
pages_data = []

# Set to keep track of valid titles (titles that exist in the main namespace)
valid_titles = set()

# First, parse through all pages to extract the valid titles
for page in root.findall(".//ns:page", namespaces=NAMESPACE):
    title_element = page.find("ns:title", namespaces=NAMESPACE)
    
    # Extract the title if available
    title = title_element.text if title_element is not None else "No Title"

    # Skip pages with excluded prefixes
    if title.startswith(EXCLUDE_PREFIXES):
        continue  # Skip this page and move to the next one
    
    # Add the title to the valid_titles set
    valid_titles.add(title)




# Define attribute mapping for each subtype
astrographical_infoboxes = ["Moon", "Planet", "Star", "System", "Sector", "Region", "Trade_route", "Nebula", "Galaxy"] # Consider adding "Space_station"
group_infoboxes = ["Company", "Cultural group", "Family", "Fleet", "Government", "Military_unit", "Organization", "Religion"]
animate_infoboxes = ["Character", "Droid"]
demographic_infoboxes = ["Species", "Language"]
all_infoboxes = astrographical_infoboxes + group_infoboxes + animate_infoboxes + demographic_infoboxes


# Function to initialize the infobox dictionary for each page (sets page_type and page_subtype)
def initialize_infobox(page_subtype):
    # Determine the high-level type based on subtype category
    if page_subtype in astrographical_infoboxes:
        page_type = "Astrographical"
    elif page_subtype in group_infoboxes:
        page_type = "Group"
    elif page_subtype in animate_infoboxes:
        page_type = "Animate"
    elif page_subtype in demographic_infoboxes:
        page_type = "Demographic"
    else:
        page_type = "Not important"
    
    # Initialize infobox with type, subtype, and empty lists for each attribute
    infobox = {
        "page_type": page_type,
        "page_subtype": page_subtype
    }
    return infobox


# Function to process the infobox attributes
def process_infobox(infobox):
    def extract_linked_text(entry):
        """
        Extracts text of links in the form [[link]] or [[link|text]] using mwparserfromhell.
        If no links are found, returns a plain text version with Wikitext formatting removed.
        Excludes text in {{}} (like references) and <ref> tags from being saved.
        """
        # Remove Wikitext references enclosed in {{}} (like {{C|...}})
        entry = re.sub(r'\{\{.*?\}\}', '', entry)

        # Remove <ref> tags and their contents (like <ref name="TSL">''[[Star Wars: Knights of the Old Republic II: The Sith Lords]]''</ref>)
        entry = re.sub(r'<ref.*?>.*?</ref>', '', entry)

        # Parse the entry with mwparserfromhell
        wikicode = mwparserfromhell.parse(entry)
        
        # Find all links in the Wikitext
        links = wikicode.filter_wikilinks()
        
        if links:
            # If links are found, return them formatted as "link|text"
            return [f"{str(link.title)}|{str(link.text)}" if link.text else str(link.title) for link in links]
        else:
            # If no links are found, return the plain text with Wikitext removed
            return [str(wikicode.strip_code()).strip()]  # Convert Wikicode to string

    # Initialize processed_infobox with type and subtype
    processed_infobox = {
        "page_type": infobox.get("page_type", ""),
        "page_subtype": infobox.get("page_subtype", "")
    }

    for attribute, values in infobox.items():
        if attribute in ["page_type", "page_subtype"]:
             continue  # Skip type and subtype fields

        processed_values = []
        for entry in values:
            # Extract linked text or plain text if no links found
            linked_texts = extract_linked_text(entry)
            processed_values.extend(linked_texts)

        # Only add processed values if non-empty
        processed_infobox[attribute] = [val for val in processed_values if val]

    return processed_infobox


# Counter for debugging
i = 0

# Second, parse through all pages to extract the data
for page in root.findall(".//ns:page", namespaces=NAMESPACE):
    title_element = page.find("ns:title", namespaces=NAMESPACE)
    text_element = page.find(".//ns:text", namespaces=NAMESPACE)
    
    # Extract title and raw text if available
    title = title_element.text if title_element is not None else "No Title"
    raw_text = text_element.text if text_element is not None else "No Text"

    # Skip pages with excluded prefixes
    if title.startswith(EXCLUDE_PREFIXES):
        continue  # Skip this page and move to the next one

    # Process with mwparserfromhell if text is available
    if raw_text:
        # Parse the raw text using mwparserfromhell
        wikicode = mwparserfromhell.parse(raw_text)
        
        # Cleaned text without Wikitext formatting
        plain_text = wikicode.strip_code()
        
        # Extract internal links within the page
        links = [str(link.title) for link in wikicode.filter_wikilinks()]

        # Check for and filter out any categories (i.e., links like "Category:<...>")
        categories = [link for link in links if link.startswith("Category:")]
        # Remove categories and templates from the list of links
        links = [link for link in links if not link.startswith("Category:") and not link.startswith("Template:")]
        # Remove links that are not valid titles, and remove duplicates
        links = list(set([link for link in links if link in valid_titles and link != title]))

        # Remove duplicate categories and keep valid categories
        categories = list(set([category for category in categories if category in valid_titles]))


        # Extract templates (lines starting with {{) and initialize canon/legends status
        templates = []
        canon = False
        legends = False
        infobox = {}

        # Loop through templates once to build list and check canon/legends status
        for tpl in wikicode.filter_templates():
            template_name = str(tpl.name).strip()
            templates.append(f"Template:{template_name}")
            
            # Check for {{Top}} template and set canon/legends flags
            if template_name == "Top":
                
                # Get all parameter values in lowercase to simplify checks
                param_values = [param.value.strip().lower() for param in tpl.params]

                # Check if Canon should be marked
                # "can" is used for canon banner, BUT is for OOU (out-of-universe) articles, so it is noncanon
                # "leg" is used for legends banner
                # "legends" kinda wierd so we don't use it
                # "canon" also kinda wierd so we don't use it
                # "noncanon" is used for non-canon banner
                # "ncc" is used for non-canon banner (ncc = non-canon canon)
                # "ncl" is used for non-canon legends banner (ncl = non-canon legends)
                # "hide" is used for hiding the banner
                if not any(param in ["can", "leg", "legends", "canon", "noncanon", "ncc", "ncl", "hide"] for param in param_values) and \
                not title.endswith("/Legends") and \
                not any(param.startswith("notoc=") for param in param_values) and \
                not any(param.startswith("legends=") for param in param_values): # Additionally checking for "notoc=<...>" and "legends=<...>"
                    canon = True
                # Check if legends should be marked
                # "leg" is used for legends banner
                # "/Legends" as a suffix in the title is also used to mark legends
                elif any(param == "leg" for param in param_values) \
                or title.endswith("/Legends"):
                    legends = True
            
            # Check if template corresponds to a known infobox subtype
            if template_name in all_infoboxes:
                infobox = initialize_infobox(template_name)

                for param in tpl.params:
                    param_name = param.name.strip()
                    param_value = param.value.strip()

                    infobox[param_name] = []  # Ensure attributes are initialized as lists
                    if param_value: # Only add non-empty values
                        infobox[param_name].append(param_value)

                infobox = process_infobox(infobox)


        # Filter templates to only include valid ones and remove duplicates
        templates = list(set([template for template in templates if template in valid_titles]))

        # Save each page's data
        page_data = {
            "title": title,
            "plain_text": plain_text,
            "links": links,
            "categories": categories,
            "templates": templates,
            "Canon": canon,
            "Legends": legends,
            **infobox
        }
        
        # Add the page's data to our collection
        pages_data.append(page_data)
        
        i += 1
        print(f"Processed page {i}: {title}")


## Save to JSON at a specific location

# Set the save path for the cleaned data
output_path = "/Users/max-peterschroder/Library/Mobile Documents/com~apple~CloudDocs/Master Study/1. Semester/02805 Social Graphs/socialgraphs2024/final_project/cleaned_wookieepedia_data.json"

# Save the data to a JSON file
try:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=4)
    print(f"{i} pages processed successfully.")
    print(f"Data saved successfully to {output_path}")
except Exception as e:
    print("Error saving JSON:", e)






# SAVE CANON AND LEGENDS DATASETS

# Load the dataset (make sure the file path is correct)
input_path = output_path

try:
    with open(input_path, "r", encoding="utf-8") as f:
        pages_data = json.load(f)
    print(f"Dataset loaded successfully with {len(pages_data)} pages.")
except Exception as e:
    print(f"Error loading JSON file: {e}")
    exit()

# List to store canon and legends pages
canon_articles = []
legends_articles = []

# Iterate through each page to find canon and legends pages
for page in pages_data:
    # Check if the page is canon (Canon = True)
    if page.get("Canon") is True:
        canon_articles.append(page)
    elif page.get("Legends") is True:
        legends_articles.append(page)


print(f"Found {len(canon_articles)} canon article pages.")
print(f"Found {len(legends_articles)} legends article pages.")


# Save the filtered canon articles to a new JSON file
canon_dataset_path = "/Users/max-peterschroder/Library/Mobile Documents/com~apple~CloudDocs/Master Study/1. Semester/02805 Social Graphs/socialgraphs2024/final_project/canon_cleaned_wookieepedia_data.json"

legends_dataset_path = "/Users/max-peterschroder/Library/Mobile Documents/com~apple~CloudDocs/Master Study/1. Semester/02805 Social Graphs/socialgraphs2024/final_project/legends_cleaned_wookieepedia_data.json"

try:
    with open(canon_dataset_path, "w", encoding="utf-8") as f:
        json.dump(canon_articles, f, ensure_ascii=False, indent=4)
    print(f"Canon dataset saved successfully with {len(canon_articles)} pages.")
except Exception as e:
    print(f"Error saving JSON file: {e}")

try:
    with open(legends_dataset_path, "w", encoding="utf-8") as f:
        json.dump(legends_articles, f, ensure_ascii=False, indent=4)
    print(f"Legends dataset saved successfully with {len(legends_articles)} pages.")
except Exception as e:
    print(f"Error saving JSON file: {e}")

# Preprocessing:
# Found 645829 pages.
# 322218 pages processed successfully.

# Found 69272 canon article pages.
# Found 115030 legends article pages.
# Canon dataset saved successfully with 69272 pages.
# Legends dataset saved successfully with 115030 pages.

