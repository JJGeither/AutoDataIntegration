# get_matching_entities.py

import my_bert_score
from categorization import levenshtein_distance
import numpy as np
import random
from log import log


def get_matching_entities(table1, table2, table1Map):
    # Create BERT scorer class as function property
    #   we use distilbert as it is significantly faster than deberta
    # get_matching_entities.scorer = my_bert_score.MyBERTScorer(lang="en", rescale_with_baseline=True,
    #                                                           model_type="microsoft/deberta-xlarge-mnli")
    get_matching_entities.scorer = my_bert_score.MyBERTScorer(lang="en", rescale_with_baseline=True,
                                                              model_type="distilbert-base-uncased")

    # Separate attributes and data for easy access
    cols1 = table1[0]
    cols2 = table2[0]
    data1 = table1[1:]
    data2 = table2[1:]

    # Map attributes to their indices.
    #   Key is the attribute and the value is its index in the table
    attr_to_index_1 = {cols1[i]: i for i in range(len(cols1))}
    attr_to_index_2 = {cols2[i]: i for i in range(len(cols2))}

    # Map each attribute in table 1 to the type of data it contains
    unique_threshold = 0.8  # ratio of unique values to values needed for an attribute to be classified as unique
    # attr_type_1: Key is the attribute name and value is the type, either "unique", "numerical", or "categorical"
    attr_type_1 = {}
    for i in range(len(cols1)):
        unique_vals = list(set([entity[i] for entity in data1]))  # list of all unique values in the attribute
        unique_val_ratio = len(unique_vals) / len(data1)
        if unique_val_ratio > unique_threshold:
            try:  # if the data can all be cast to float, it is numerical
                [float(data) for data in unique_vals]
            except ValueError:
                attr_type_1[cols1[i]] = "unique"
            else:
                attr_type_1[cols1[i]] = "numerical"
        else:
            attr_type_1[cols1[i]] = "categorical"
    log(f"Table 1 attribute types are: {attr_type_1}")

    # Maps each attribute in table 2 to the type of data it contains
    # attr_type_2: Key is the attribute name and value is the type, either "unique", "numerical", or "categorical"
    attr_type_2 = {}
    for i in range(len(cols2)):
        unique_vals = list(set([entity[i] for entity in data2]))  # list of all unique values in the attribute
        unique_val_ratio = len(unique_vals) / len(data2)
        if unique_val_ratio > unique_threshold:
            try:  # if the data can all be cast to float, it is numerical
                [float(data) for data in unique_vals]
            except ValueError:
                attr_type_2[cols2[i]] = "unique"
            else:
                attr_type_2[cols2[i]] = "numerical"
        else:
            attr_type_2[cols2[i]] = "categorical"
    log(f"Table 2 attribute types are: {attr_type_2}")

    # Map categories between tables for matching categorical attributes
    # Get the standard deviation for matching numerical unique attributes
    cat_threshold = 0.5  # score required for two categories of matching attributes to match
    # cat_map: The key is a category from an attribute in table 1. Categories only appear if they have a match.
    #   The value is its mapped category from its corresponding attribute in table 2.
    cat_map = {}
    # attr_stddev: The key is an attribute in table 1.
    #   The value is the standard deviation of all the data for that attribute and the data for its match in table 2.
    attr_stddev = {}
    for attr1 in table1Map.keys():
        attr2 = table1Map[attr1]
        if attr2 not in cols2:  # if the table 1 attribute maps to itself, i.e. doesn't map to an attribute in table 2
            continue
        # If the mapped attributes are both categorical, use BERT embeddings to map their categories.
        #   Note: ideally, both attributes should be the same type. If not, they shouldn't match (but this may happen).
        if attr_type_1[attr1] == "categorical" and attr_type_2[attr2] == "categorical":
            log(f"Calculating category scores for {attr1} and {attr2}...")
            # Lists of the categories in the corresponding attributes
            attr1_cats = list(set([entity[attr_to_index_1[attr1]] for entity in data1]))
            attr2_cats = list(set([entity[attr_to_index_2[attr2]] for entity in data2]))
            log(f"Categories for {attr1} are: {attr1_cats}")
            log(f"Categories for {attr2} are: {attr2_cats}")
            # Lists of all possible orders of these categories
            attr1_vals_perms = [random.sample(attr1_cats, len(attr1_cats)) for i in range(10)]
            attr2_vals_perms = [random.sample(attr2_cats, len(attr2_cats)) for i in range(10)]
            # Calculate the average score between each pair of categories over all permutation combinations.
            #   Necessary because BERT considers word order, which seem to greatly impact the category scores.
            # avg_scores: Key is a tuple of a category in the table 1 attribute and a category in the table 2 attribute.
            #   Value is the average score for those categories over all permutations.
            avg_scores = {}
            n = len(attr1_vals_perms) * len(attr2_vals_perms)  # number of values that will be averaged for each score
            # Calculate the category similarity scores for each combination of permutations
            for perm1 in attr1_vals_perms:
                for perm2 in attr2_vals_perms:
                    perm_scores = get_matching_entities.scorer.get_word_similarity(','.join(perm1), ','.join(perm2))
                    # Add the scores for each category pair to their averages
                    for i in range(len(perm_scores)):
                        for j in range(len(perm_scores[i])):
                            # Calculate average as a/n + b/n + c/n (instead of (a+b+c)/n) so we can do it in this loop
                            if (perm1[i], perm2[j]) not in avg_scores.keys():  # add to dict if not yet in
                                avg_scores[(perm1[i], perm2[j])] = perm_scores[i][j] / n
                            else:
                                avg_scores[(perm1[i], perm2[j])] += perm_scores[i][j] / n
            log(f"Average category scores for {attr1} and {attr2} are: {avg_scores}")
            # Put the category scores into a 2D array where the indices correspond to the lists of categories above
            cat_scores = [[0 for j in range(len(attr2_cats))] for i in range(len(attr1_cats))]
            for i in range(len(attr1_cats)):
                for j in range(len(attr2_cats)):
                    cat_scores[i][j] = avg_scores[(attr1_cats[i], attr2_cats[j])]
            cat_scores = [[val if val > cat_threshold else 0 for val in row] for row in cat_scores]  # set scores below threshold to 0
            # Create the map between categories based on their average BERT similarity scores
            # Get a score for each possible category map
            map_scores = get_path_prods(cat_scores, cat_threshold)
            if not map_scores:  # if the categories did not map to each other at all, do nothing
                continue
            # We want the path with the highest score
            vals = list(zip(*map_scores))[0]  # values from the paths
            paths = list(zip(*map_scores))[1]  # list of indices from the paths
            best_map_index = vals.index(max(vals))
            best_path = paths[best_map_index]
            # Finally add the categories from these attributes to the category map
            for i in range(len(best_path)):
                cat_map[attr1_cats[i]] = attr2_cats[best_path[i]] if best_path[i] != -1 else None

        # If the mapped attributes are both numerical, calculate the standard deviation
        elif attr_type_1[attr1] == "numerical" and attr_type_2[attr2] == "numerical":
            attr_stddev[attr1] = np.std([float(entity[attr_to_index_1[attr1]]) for entity in data1] +
                                        [float(entity[attr_to_index_2[attr2]]) for entity in data2])
    log(f"Category map for all attributes is: {cat_map}")

    # Calculate the similarity score for each pair of entities from table 1 and table 2
    entity_scores = [[0 for i in range(len(data2))] for j in range(len(data1))]
    for i in range(len(data1)):
        for j in range(len(data2)):
            log(f"Score for {data1[i][0]} (data1[{i}]) and {data2[j][0]} (data2[{j}]):")
            attribute_scores = []  # store the similarity scores of the values of each attribute for these entities
            # The entity score is based on the individual scores of all the shared attributes
            for table_1_attr in table1Map.keys():
                table_2_attr = table1Map[table_1_attr]
                if table_2_attr not in cols2:  # if the table 1 attribute does not match a table 2 attribute
                    continue
                entity1_val = data1[i][attr_to_index_1[table_1_attr]]
                entity2_val = data2[j][attr_to_index_2[table_2_attr]]
                # The attribute score will depend on the type of the attribute
                #   Use normalized levenshtein for names, identifiers, etc.
                #   Use score based on number of standard deviations away for numbers
                #   Use the BERT category scores from above if they are categorical
                if attr_type_1[table_1_attr] == "unique" and attr_type_2[table_2_attr] == "unique":
                    lev_dist = levenshtein_distance(entity1_val, entity2_val)
                    max_len = max(len(entity1_val), len(entity2_val))
                    score = 1 - lev_dist / max_len
                elif attr_type_1[table_1_attr] == "numerical" and attr_type_2[table_2_attr] == "numerical":
                    score = 1 / (abs(float(entity1_val) - float(entity2_val)) / attr_stddev[table_1_attr] + 1)
                elif attr_type_1[table_1_attr] == "categorical" and attr_type_2[table_2_attr] == "categorical":
                    if entity1_val not in cat_map.keys():  # if the category didn't match
                        score = 0
                    else:
                        score = 1 if cat_map[entity1_val] == entity2_val else 0
                else:
                    score = 0
                log(f"Score for {entity1_val} and {entity2_val} is {score}")
                attribute_scores.append(score)
            # The entity score is the average of the attribute scores
            entity_scores[i][j] = (sum(attribute_scores) / len(attribute_scores))
            log(f"Score for entity {data1[i][0]} with entity {data2[j][0]} is {entity_scores[i][j]}\n")

    # Minimum score required for two entities to be able to match
    entity_match_threshold = 0.6
    entity_scores = [[val if val > entity_match_threshold else 0 for val in row] for row in entity_scores]
    entity_map_scores = get_path_prods(entity_scores, entity_match_threshold)
    # entity_map: map where key is index of entity in table 1 and value is index of its match in table 2
    entity_map = {}
    # We want the path with the highest score
    vals = list(zip(*entity_map_scores))[0]  # values from the paths
    maps = list(zip(*entity_map_scores))[1]  # list of indices from the paths
    best_map_index = vals.index(max(vals))
    best_map = maps[best_map_index]
    log(f"Entity map is: {[(i, best_map[i]) for i in range(len(best_map))]}")
    # Create the entity map
    for i in range(len(best_map)):
        entity_map[i] = best_map[i] if best_map[i] != -1 else None

    return entity_map


# Calculates the scores (products) of all possible paths of a matrix.
#   The score is the result of multiplying the values of each step on the path together.
#   A value can only be a step on the path if its value is greater than 0.
#   Each row need not have a step on the path. Thresh is the value of excluding a row.
#   Each column can only be used once in the path.
# The result is a list of all the paths and their scores.
#   The path is represented by a tuple where the first value is the score and the second is the path taken.
#   The path taken is a list where list[i] represents the index of the column for the step of row i.
#   Rows that were skipped and are not on the path have a value of -1.
#   Ex. path: (0.5, [1, 2, -1, 4]). The score was 0.5. The steps on the path are m[0][1], m[1][2], and m[3][4].
def get_path_prods(matrix, thresh):
    # Base case (only one row)
    if len(matrix) == 1:
        path_prods = [((matrix[0][i]), [i]) for i in range(len(matrix[0]))]  # start paths with each value in this row
        temp = [(val, path) for (val, path) in path_prods if val != 0]  # remove paths with a score of 0
        temp.append((thresh, [-1]))  # start the path by skipping this row
        return temp

    # Recursive case
    paths = get_path_prods(matrix[1:], thresh)  # get the paths up to this row
    max_path_vals = (li := list(reversed(sorted([val for val, path in paths]))))[:min(len(li), 1000)]  # get best 1000
    paths = [(val, path) for val, path in paths if val in max_path_vals]
    # Calculate the score of extending each path with each value in this row
    new_paths = []
    for val, used_indices in paths:
        for i in range(len(matrix[0])):
            if i not in used_indices and matrix[0][i] != 0:
                new_paths.append((matrix[0][i] + val, [i] + used_indices))
        new_paths.append((thresh + val, [-1] + used_indices))  # extend the path by skipping this row
    return new_paths
