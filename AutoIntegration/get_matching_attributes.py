# get_matching_attributes.py

import my_bert_score
from get_matching_entities import get_path_prods


# This function matches the attributes of two tables' schemas
def getMatchingAttributes(table1, table2):
    # Get schemas of each table
    cols1 = table1[0]
    cols2 = table2[0]

    # attributes with scores greater than this will match
    threshold = float(input("Input threshold for schema matching: "))

    # Create custom BERT scorer
    # Arguments:
    #     rescale_with_baseline: rescales scores to be more "human-readable"
    #     model_type: we choose best performing model (has 77% correlation score with human evaluation for matching sentences)
    getMatchingAttributes.scorer = my_bert_score.MyBERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")

    # Get similarity scores as a 2D list
    #   need to convert the arrays to strings to pass to scorer. Each attribute is separated by a comma
    scores = getMatchingAttributes.scorer.get_word_similarity(','.join(cols1), ','.join(cols2))
    # replace scores below the threshold with 0s so that they cannot match
    scores = [[val if val > threshold else 0 for val in row] for row in scores]

    # Map attributes in table 1 to table 2
    # Each column will map to itself if it has no match
    table1Map = {attr: attr for attr in cols1}
    # Get all possible schema maps and their scores
    schema_map_scores = get_path_prods(scores, threshold)
    # Get the map with the highest score
    vals = list(zip(*schema_map_scores))[0]  # values from the paths
    maps = list(zip(*schema_map_scores))[1]  # list of indices from the paths
    best_map_index = vals.index(max(vals))
    best_map = maps[best_map_index]
    # Create the schema map
    for i in range(len(best_map)):
        table1Map[cols1[i]] = cols2[best_map[i]] if best_map[i] != -1 else None

    return table1Map
