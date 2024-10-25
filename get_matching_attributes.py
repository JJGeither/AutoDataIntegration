import my_bert_score

# Example schemas for testing
cols1 = ["ISBN", "publisher", "pubCountry", "title", "review"]
cols2 = ["code", "name", "brand", "origin", "desc"]

# Create custom BERT scorer
# Arguments:
#     rescale_with_baseline: rescales scores to be more "human-readable"
#     model_type: choose best performing model (has 77% correlation score with human evaluation for matching sentences)
scorer = my_bert_score.MyBERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")

# Get similarity scores as a 2D list
# need to convert the arrays to strings to pass to scorer. Each attribute is separated by a comma
scores = scorer.get_word_similarity(','.join(cols1), ','.join(cols2))

# attributes with scores greater than this will match
threshold = 0.2

# if attribute in cols2 has high score with attribute in cols1, it will match with that attribute
used_cols2 = []
for i in range(0, len(scores)):
    for j in range(0, len(scores[0])):
        if scores[i][j] >= threshold:
            print(cols1[i] + " matches with " + cols2[j])
            used_cols2.append(cols2[j])

# if attribute in table 2 did not have a match in table 1, it must be added to the schema
final_schema = cols1.copy()
for attr in cols2:
    if attr not in used_cols2:
        final_schema.append(attr)

# return final_schema
print(final_schema)
