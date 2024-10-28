# get_matching_attributes.py
import my_bert_score


# This function matches the attributes of two tables' schemas
def getMatchingAttributes(table1, table2):
    # Get schemas of each table
    cols1 = table1[0]
    cols2 = table2[0]
    
    # Create custom BERT scorer
    # Arguments:
    #     rescale_with_baseline: rescales scores to be more "human-readable"
    #     model_type: choose best performing model (has 77% correlation score with human evaluation for matching sentences)
    scorer = my_bert_score.MyBERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")
    
    # Get similarity scores as a 2D list
    # need to convert the arrays to strings to pass to scorer. Each attribute is separated by a comma
    scores = scorer.get_word_similarity(','.join(cols1), ','.join(cols2))
    
    # attributes with scores greater than this will match
    threshold = float(input("Input threshold: "))
    
    # if attribute in cols2 has high score with attribute in cols1, it will match with that attribute
    table1Map = {attr:attr for attr in cols1}
    table2Map = {attr:attr for attr in cols2}
    for i in range(0, len(scores)):
        highestScore = -1
        for j in range(0, len(scores[0])):
            if scores[i][j] >= threshold and scores[i][j] > highestScore:
                highestScore = scores[i][j]
                # RHS is name in final schema. Replace later with user's choice (temporarily using table 1)
                table1Map[cols1[i]] = cols1[i]
                table2Map[cols2[j]] = cols1[i]
                break
    
    # return final_schema
    return table1Map, table2Map
