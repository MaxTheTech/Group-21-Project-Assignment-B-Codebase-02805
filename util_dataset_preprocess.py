import re
from collections import Counter


### DATASET PREPROCESSING FUNCTIONS ###

# Extract the link text from a link|text string (if text exists) and remove /Legends (if it exists)
def extract_link(link):
    # Remove the text after the '|' (if it exists)
    link = re.sub(r'\|.*$', '', link)

    # Remove the '/Legends' part (if it exists)
    link = re.sub(r'/Legends$', '', link)

    return link

# Extract the link text from a link|text string
def extract_link_text(link_text):
    # If there's no pipe character, return an empty string since there's no text portion
    if '|' not in link_text:
        return None
    
    # Split the string at the pipe and take everything after it
    # The split() method returns a list, and [1] gets the second element (after the pipe)
    text = link_text.split('|')[1]
    return text


### CHARACTER ATTRIBUTE PROCESSING & ANALYSIS FUNCTIONS ###

## GENERAL ##

# Find the count and list of unique values for a given attribute
def unique_attribute_values(G, attribute):
    values = set()
    for node in G.nodes:
        values.update(G.nodes[node][attribute] if G.nodes[node].get(attribute) is not None else [])
    print(f"\nNumber of unique '{attribute}' attribute values: {len(values)}")
    print(f"Unique values of '{attribute}' attribute: {values}")


def most_common_n_attribute_values(G, attribute, top_n=10):
    print() # Acts as newline

    attribute_counter = Counter()
    for _, node_data in G.nodes(data=True):
        if attribute in node_data:
            attribute_counter.update(node_data[attribute])
    attribute_counter_top_n = attribute_counter.most_common(top_n)
    if len(attribute_counter_top_n) < top_n:
        top_n = len(attribute_counter_top_n)

        print(f"Most common '{attribute}' values (Only {top_n} unique values):")
    else:
        print(f"Most common '{attribute}' values (Top {top_n}):")

    for value, count in attribute_counter_top_n:
        print(f"  '{value}': {count} characters")

def find_character_with_attribute_value(G, attribute, value):
    characters_with_value = []
    for character, node_data in G.nodes(data=True):
        if attribute in node_data:
            if value in node_data[attribute]: # Check if the value is in the attribute values, if so, add the character to the list
                characters_with_value.append(character)
            elif not node_data[attribute] and value == None: # Check if the attribute is empty and we are looking for None, if so, add the character to the list
                characters_with_value.append(character)
    print(f"\nCharacters with '{attribute}' value '{value}': {characters_with_value}")


def replace_attribute_value(G, attribute, old_value, new_value, add_newline=True):
    i = 0
    for _, node_data in G.nodes(data=True):
        if attribute in node_data:
            new_attribute_values = []
            if not node_data[attribute] and old_value == None: # If the attribute is empty and we want to replace None, we replace the empty list with the new value
                new_attribute_values.append(new_value)
                i += 1
            else: 
                for v in node_data[attribute]: # In case an attribute is not emptry: Iterate over the attribute values and replace the old value with the new value
                    if v == old_value and v != new_value:
                        new_attribute_values.append(new_value)
                        i += 1
                    else:
                        new_attribute_values.append(v)
            node_data[attribute] = new_attribute_values
    if add_newline: print()
    if i > 0: print(f"Replaced {i} instances of '{old_value}' with '{new_value}' in '{attribute}'")

def remove_attribute_value(G, attribute, remove_value, add_newline=True):
    i = 0
    for _, node_data in G.nodes(data=True):
        if attribute in node_data:
            keep_attribute_values = []
            for v in node_data[attribute]:
                if v == remove_value:
                    i += 1
                else:
                    keep_attribute_values.append(v)
            node_data[attribute] = keep_attribute_values
    if add_newline: print()
    if i > 0: print(f"Removed {i} instances of '{remove_value}' from '{attribute}'")


## AFFILIATION REMAPPING FUNCTIONS ##
# Define criminal syndicate categories - these help identify Crime Syndicate affiliations
CRIMINAL_SYNDICATE_CATEGORIES = {
    "Category:Criminal syndicates and cartels",
    "Category:Criminal organizations",
    "Category:Smuggling organizations"
}

# Define the primary affiliations and their mapping rules
PRIMARY_AFFILIATIONS = {
    # Removed the following primary affiliations from "type" options:
    # Irrelevant: "Confederation", "Corporate", "Criminal", "Musician", "Dark Jedi", "Sith Empire"
    # Non-canon: "Eternal Empire", "GFFA", "Infinite Empire", "Je'daii", "Yuuzhan Vong"
    # Merged together into new primary affiliation: "Hutt Cartel" and "Criminal"

    # Renamed the following primary affiliations from "type" options:
    # "Bounty hunter" -> "Bounty Hunter"
    # "CIS" -> "Confederacy of Independent Systems"
    # "Criminal" & "Hutt Cartel" --> "Criminal Syndicates and Cartels"
    # "Dathomiri" -> "Nightsisters"
    # "Jedi" -> "Jedi Order"
    # "Mandalorian" -> "Mandalorians"
    # "Rebel" -> "Alliance to Restore the Republic"
    # "Sith" -> "Sith Order"
    
    # All other primary affiliations are kept as is

    # Each primary affiliation has a set of rules for mapping original affiliations to it.
    # These rules include exact matches (entire string), contains (substring), string_starts (starts with), word_prefix (word starts with), and categories (node categories).
    # If any rule matches, the original affiliation is mapped to the primary affiliation.

    "Bounty Hunter": {
        "exact_matches": ["bounty hunters' guild"],
        "contains": ["bounty hunter"],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    },
    "Chiss Ascendancy": {
        "exact_matches": ["chiss ascendancy"],
        "contains": ["chiss"],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    },
    "Confederacy of Independent Systems": {
        "exact_matches": ["confederacy of independent systems", "cis"],
        "contains": [],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    },
    "Crime Syndicates and Cartels": {
        "exact_matches": ["hutt cartel"],
        "contains": ["syndicate", "cartel", "hutt", "criminal", "crime"],
        "string_starts": [],
        "word_prefix": [],
        "categories": CRIMINAL_SYNDICATE_CATEGORIES
    },
    "Nightsisters": {
        "exact_matches": ["nightsisters", "dathomiri"],
        "contains": ["nightsister", "dathomir"],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    },
    "First Order": {
        "exact_matches": ["first order"],
        "contains": ["first order"],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    },
    "Galactic Empire": {
        "exact_matches": ["galactic empire"],
        "contains": ["galactic empire"],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    },
    "Galactic Republic": {
        "exact_matches": ["galactic republic", "grand army of the republic"],
        "contains": ["galactic republic"],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    },
    "Jedi Order": {
        "exact_matches": ["jedi order", "jedi"],
        "contains": ["jedi"],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    },
    "Mandalorians": {
        "exact_matches": ["mandalorian", "mandalorians"],
        "contains": [],
        "string_starts": [],
        "word_prefix": ["mandalorian"],
        "categories": []
    },
    "New Republic": {
        "exact_matches": ["new republic"],
        "contains": ["new republic"],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    },
    "Nihil": {
        "exact_matches": ["nihil"],
        "contains": ["nihil"],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    },
    "Alliance to Restore the Republic": {
        "exact_matches": ["rebel alliance", "alliance to restore the republic"],
        "contains": ["rebel alliance"],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    },
    "Resistance": {
        "exact_matches": ["resistance"],
        "contains": [],
        "string_starts": ["resistance"],
        "word_prefix": [],
        "categories": []
    },
    "Sith Order": {
        "exact_matches": ["sith", "sith order"],
        "contains": ["sith"],
        "string_starts": [],
        "word_prefix": [],
        "categories": []
    }
}

def clean_text(text):
    """Clean text by converting to lowercase and removing special characters."""
    return re.sub(r'[^\w\s]', '', text.lower())

def check_word_prefix(text, prefix):
    """
    Check if any word in the text starts with the given prefix.
    This is different from string_starts which checks the beginning of the entire string.
    """
    words = clean_text(text).split()
    return any(word.startswith(clean_text(prefix)) for word in words)

def map_affiliation(original_affiliation, node_categories, primary_affiliations=PRIMARY_AFFILIATIONS):
    """
    Map an original affiliation to a primary affiliation based on defined rules.
    Also considers the node's categories for certain mappings.
    Returns None if no mapping is found.
    """
    if not original_affiliation:
        return None
    
    # Clean the input affiliation
    cleaned_affiliation = clean_text(original_affiliation)
    
    # Check each primary affiliation's rules
    for primary, rules in primary_affiliations.items():
        # Check exact matches (case-insensitive)
        if cleaned_affiliation in [clean_text(match) for match in rules["exact_matches"]]:
            return primary
            
        # Check contains rules
        if any(clean_text(term) in cleaned_affiliation for term in rules["contains"]):
            return primary
            
        # Check string_starts rules (entire string must start with these)
        if any(cleaned_affiliation.startswith(clean_text(start)) for start in rules["string_starts"]):
            return primary
            
        # Check word_prefix rules (any word can start with these)
        if any(check_word_prefix(cleaned_affiliation, prefix) for prefix in rules["word_prefix"]):
            return primary
            
        # Check categories
        if rules["categories"] and any(cat in node_categories for cat in rules["categories"]):
            return primary
    
    return None

def remap_character_affiliations(G, primary_affiliations=PRIMARY_AFFILIATIONS):
    """
    Analyze affiliations in the character network and return detailed mapping statistics
    that can be used to update the graph's affiliation attributes.
    """
    # Statistics for reporting
    mapping_stats = {
        "total_characters": 0,
        "characters_with_affiliations": 0,
        "successful_mappings": 0,
        "unmapped_affiliations": set(),
        "affiliation_counts": {},
        "mapping_details": {},  # Original -> Primary mapping
        "node_mappings": {}     # Node -> List of mapped affiliations
    }
    
    for node in G.nodes():
        original_affiliations = G.nodes[node].get('affiliation', [])
        node_categories = G.nodes[node].get('categories', [])
        mapping_stats["total_characters"] += 1
        
        # Handle single string affiliations
        if isinstance(original_affiliations, str):
            original_affiliations = [original_affiliations]
            
        if original_affiliations:
            mapping_stats["characters_with_affiliations"] += 1
            
        # Map each affiliation
        cleaned_affiliations = []
        for aff in original_affiliations:
            mapped_aff = map_affiliation(aff, node_categories, primary_affiliations)
            if mapped_aff:
                cleaned_affiliations.append(mapped_aff)
                mapping_stats["successful_mappings"] += 1
                mapping_stats["affiliation_counts"][mapped_aff] = \
                    mapping_stats["affiliation_counts"].get(mapped_aff, 0) + 1
                mapping_stats["mapping_details"][aff] = mapped_aff
            else:
                mapping_stats["unmapped_affiliations"].add(aff)
        
        # Store the mapped affiliations for this node
        if cleaned_affiliations:
            mapping_stats["node_mappings"][node] = list(set(cleaned_affiliations))
    
    return mapping_stats

def print_mapping_statistics(stats):
    """Print detailed statistics about the affiliation mapping process, including lists of remappings."""
    print("\nAffiliation Mapping Statistics")
    print("=" * 30)
    print(f"Total characters processed: {stats['total_characters']}")
    print(f"Characters with affiliations: {stats['characters_with_affiliations']}")
    print(f"Successful mappings: {stats['successful_mappings']}")
    
    print("\nRemapping Lists by Primary Affiliation:")
    print("=" * 30)
    
    # Group mappings by primary affiliation
    primary_to_original = {}
    for original, primary in stats['mapping_details'].items():
        if primary not in primary_to_original:
            primary_to_original[primary] = []
        primary_to_original[primary].append(original)
    
    # Print each primary affiliation's remappings
    for primary in sorted(primary_to_original.keys()):
        originals = sorted(primary_to_original[primary])
        count = stats['affiliation_counts'].get(primary, 0)
        print(f"\n{primary}")
        print("-" * len(primary))
        print(f"Total characters: {count}")
        print("Maps from:")
        for orig in originals:
            print(f"- {orig}")
    
    print(f"\nUnmapped affiliations ({len(stats['unmapped_affiliations'])} total):")
    # Show first 10 unmapped affiliations as a sample
    sample = sorted(list(stats['unmapped_affiliations']))[:10]
    for aff in sample:
        print(f"- {aff}")

# Update the graph's affiliation attributes based on the mapping statistics
def update_graph_affiliations(G, mapping_stats):
    """
    Update the affiliation attribute of each node in the graph, replacing all original 
    affiliations with only their mapped primary affiliations. Nodes without mapped 
    affiliations will have an empty list as their affiliation value.
    """
    # Update every node in the graph
    for node in G.nodes():
        # Get the mapped affiliations for this node from stats, or empty list if none
        G.nodes[node]['affiliation'] = mapping_stats["node_mappings"].get(node, [])
    return G


def top_n_attribute_counter(G, attribute, top_n=5):
    attribute_counter = Counter()
    for _, node_data in G.nodes(data=True):
        if attribute in node_data:
            attribute_counter.update(node_data[attribute])
    attribute_counter_top_n = attribute_counter.most_common(top_n)
    return attribute_counter_top_n




