import nltk
import csv
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from util_dataset_preprocess import top_n_attribute_counter
from IPython.display import display, HTML

# Load the LabMT wordlist + corresponding average happiness scores
def load_labmt_wordlist(filepath):
    labmt_dict = {}
    with open(filepath, 'r') as file:
        reader = csv.reader(file, delimiter='\t')

        # Skip the headers (first four lines)
        for i in range(4):
            next(reader)
        
        # 
        for row in reader:
            # Retrieve word
            word = row[0].lower()
            # Retrieve corresponding average happiness score
            happiness_avg = float(row[2])
            labmt_dict[word] = happiness_avg
    return labmt_dict

# Function to clean and tokenize text
def tokenize_and_lemmatize(raw_text, lemmatizer, stop_words):
    # Clean text (remove extra wikipedia formatting)
    # clean_text = clean_wikipedia_text(raw_text)
    # Noticed formatting leftovers of type: thumb|left|220px|, Category:7th Sky Corps personnel
    
    # Tokenize
    tokens = nltk.tokenize.word_tokenize(raw_text)

    # Lowercase, remove punctuation, and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(token.lower()) for token in tokens
        if token.isalpha() and token.lower() not in stop_words
    ]

    return cleaned_tokens

def calculate_weighted_community_sentiment(characters, character_sentiments, character_tf):
    total_weight = 0
    weighted_sum = 0
    for character in characters:
        if character in character_sentiments and character_sentiments[character] is not None:
            # Use the total word count as the weight for this character
            weight = sum(character_tf[character].values())
            sentiment = character_sentiments[character]
            weighted_sum += weight * sentiment
            total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else None


def calculate_term_dispersion(community_groups, character_tf_idf):
    # Track which communities each term appears in
    term_community_presence = defaultdict(set)
    
    # For each community
    for comm_id, characters in community_groups.items():
        # Get all terms used in this community
        community_terms = set()
        for char in characters:
            if char in character_tf_idf:
                community_terms.update(character_tf_idf[char].keys())
        
        # Record that these terms appeared in this community
        for term in community_terms:
            term_community_presence[term].add(comm_id)
    
    # Calculate dispersion as proportion of communities where term appears
    dispersion_scores = {
        term: len(communities) / len(community_groups)
        for term, communities in term_community_presence.items()
    }
    return dispersion_scores


def create_mega_df(G, attributes, columns, document_frequency, character_tf, character_tf_idf, character_sentiments, top_n_characters, top_n_attribute, top_n_tf, top_n_tf_idf):
    df = pd.DataFrame()
    for column_id in ['Entire Network'] + list(columns):
        if column_id == 'Entire Network':
            characters = list(G.nodes())
            num_network_characters = len(characters)
        else:
            characters = [node for node, data in G.nodes(data=True) if 'affiliation' in data and column_id in data['affiliation']]

        G_column = G.subgraph(characters).copy()
        organics = [node for node in characters if G_column.nodes[node].get('page_subtype') == 'Character']
        droids = [node for node in characters if G_column.nodes[node].get('page_subtype') == 'Droid']
        num_characters = int(len(characters))
        num_organics = int(len(organics))
        num_droids = int(len(droids))

        df.at['Size', column_id] = f"{num_characters} ({num_characters / num_network_characters * 100:.1f}%)"
        df.at['Of these: Organics', column_id] = f"{num_organics} ({num_organics / num_characters * 100:.1f}%)" if num_characters > 0 else "0 (0.0%)"
        df.at['Of these: Droids', column_id] = f"{num_droids} ({num_droids / num_characters * 100:.1f}%)" if num_characters > 0 else "0 (0.0%)"

        if column_id != 'Entire Network':
            # Internal density
            possible_edges = len(characters) * (len(characters) - 1)
            internal_density = 0 if possible_edges == 0 else G_column.number_of_edges() / possible_edges
            
            # Conductance
            external_edges = sum(1 for n in characters for m in G.neighbors(n) if m not in characters)
            total_edges = external_edges + (2 * G_column.number_of_edges())
            conductance = 0 if total_edges == 0 else external_edges / total_edges
            
            # Average clustering
            avg_clustering = nx.average_clustering(G_column)

            df.at['Internal Density', column_id] = f"{internal_density:.5f} ({internal_density / network_density:.1f}x)"
            df.at['Conductance', column_id] = f"{conductance:.4f}"
            df.at['Average Clustering', column_id] = f"{avg_clustering:.4f}"
        else:
            network_density = nx.density(G)
            df.at['Internal Density', column_id] = network_density
            df.at['Conductance', column_id] = "NaN"
            df.at['Average Clustering', column_id] = f"{nx.average_clustering(G):.4f}"
        
        # Re-define the character TF and TF-IDF dictionaries for this community to only include the characters in this community
        community_character_tf = {character: character_tf[character] for character in characters}
        community_character_tf_idf = {character: character_tf_idf[character] for character in characters}

        # Aggregate all term frequencies across the community
        community_tf_count = Counter()

        for tf_count in community_character_tf.values():
            community_tf_count.update(tf_count)

        # Get the top 5 terms by count
        top_5_tf = community_tf_count.most_common(5)

        df.at[f'Top {top_n_tf} TF Counts', column_id] = '<br><br>'.join([f'{term} ({score:.1f})' for term, score in top_5_tf])


        # Create a community-wide summary of term importance
        community_tf_idf = {}
        for term in document_frequency.keys():
            # Calculate average TF-IDF score for each term across all characters
            term_scores = [char_tfidf[term] 
                            for char_tfidf in community_character_tf_idf.values() 
                            if term in char_tfidf]
            community_tf_idf[term] = sum(term_scores) / len(term_scores) if len(term_scores) > 0 else 0

        # Sort terms by importance and get the top 5
        top_5_tf_idf = sorted(community_tf_idf.items(), key=lambda x: x[1], reverse=True)[:5]

        df.at[f'Top {top_n_tf} TF-IDF Scores', column_id] = '<br><br>'.join([f'{term} ({score:.1f})' for term, score in top_5_tf_idf])


        # First, collect all valid sentiments and their corresponding text lengths for this community
        community_sentiment_data = []
        for character in characters:
            if character in character_sentiments and character_sentiments[character] is not None:
                # Get the sentiment score
                sentiment = character_sentiments[character]
                # Get the total word count for this character
                text_length = sum(character_tf[character].values())
                community_sentiment_data.append((sentiment, text_length))

        # Calculate weighted sentiment
        if community_sentiment_data:
            total_weight = sum(length for _, length in community_sentiment_data)
            weighted_sum = sum(sentiment * length for sentiment, length in community_sentiment_data)
            weighted_sentiment = weighted_sum / total_weight
            
            df.at['Sentiment Sample Size (Characters)', column_id] = f"{len(community_sentiment_data)}"
            df.at['(Weighted) Average Sentiment', column_id] = f"{weighted_sentiment:.4f}"

            # Calculate variance if we have more than one character
            if len(community_sentiment_data) > 1:
                # Extract just the sentiment values for variance calculation
                sentiment_values = [data[0] for data in community_sentiment_data]
                sentiment_variance = np.var(sentiment_values)
                df.at['Sentiment Variance', column_id] = f"{sentiment_variance:.4f}"
            else:
                df.at['Sentiment Variance', column_id] = "NaN"
            

        else:
            df.at['Sentiment Sample Size (Characters)', column_id] = "0"
            df.at['(Weighted) Average Sentiment', column_id] = "NaN"
            df.at['Sentiment Variance', column_id] = "NaN"
        
        
        k_in = dict(G_column.in_degree())
        k_out = dict(G_column.out_degree())

        # Format in-degree entries with HTML line breaks
        top_n_in_degree = sorted(k_in.items(), key=lambda x: x[1], reverse=True)[:top_n_characters]
        # Adding some spacing for better readability
        top_5_in_string = '<br><br>'.join([f'{character} ({degree})' for character, degree in top_n_in_degree])
        df.at[f'Top {top_n_characters} In-Degree', column_id] = top_5_in_string

        # Format out-degree entries similarly
        top_n_out_degree = sorted(k_out.items(), key=lambda x: x[1], reverse=True)[:top_n_characters]
        top_5_out_string = '<br><br>'.join([f'{character} ({degree})' for character, degree in top_n_out_degree])
        df.at[f'Top {top_n_characters} Out-Degree', column_id] = top_5_out_string
                
        # Get top N values for each attribute
        for attribute in attributes:
            top_n_values = top_n_attribute_counter(G_column, attribute, top_n_attribute)
            if attribute in ['affiliation', 'homeworld']:
                top_n_string = '<br><br>'.join([f'{value} ({count / num_characters * 100:.1f}%)' for value, count in top_n_values])
            elif attribute in ['species', 'gender']:
                top_n_string = '<br><br>'.join([f'{value} ({count / num_organics * 100:.1f}%)' for value, count in top_n_values])
            elif attribute in ['class', 'programmed_gender']:
                top_n_string = '<br><br>'.join([f'{value} ({count / num_droids * 100:.1f}%)' for value, count in top_n_values])

            if attribute in ['gender', 'programmed_gender']:
                df.at[f'{attribute} Distribution', column_id] = top_n_string
            else:
                df.at[f'Top {top_n_attribute} {attribute}', column_id] = top_n_string
    return df


def display_df_properties(df, rows_network_properties=None, inspect_largest_n_columns=None, transpose=False):
    if inspect_largest_n_columns is None:
        inspect_columns = ['Entire Network']
    elif inspect_largest_n_columns == 'All':
        inspect_columns = df.columns
    else:
        inspect_columns = ['Entire Network'] + [f"Community {i}" for i in range(1, inspect_largest_n_columns + 1)]

    if rows_network_properties is None:
        inspect_rows =  df.index
    else:
        inspect_rows = rows_network_properties
    
    df = df.loc[inspect_rows, inspect_columns]

    if transpose:
        df = df.transpose()

    display(HTML(df.to_html(escape=False)))
