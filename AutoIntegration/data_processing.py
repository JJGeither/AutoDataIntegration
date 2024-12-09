# data_processing.py
from data_fields import DataFieldFactory
from categorization import assign_category, extract_column_values
import my_bert_score
from data_fields import get_categorical_subclasses, EMPTYVALUE
import re
from log import log


def standardize_table_units(dataTable):
    # Extract the first row as column headers
    headers = dataTable[0]

    # Transpose the table to group data by columns
    columns = list(zip(*dataTable))

    bert_friendly_columns = [
        " ".join(
            re.sub(r'[^a-zA-Z0-9:/]', '', str(value)).strip().replace(',', '')
            for value in sublist[:3] if value is not EMPTYVALUE
        ).lower()
        for sublist in columns
    ]

    # Retrieve predefined categories and their keys
    categories = get_categorical_subclasses()
    category_keys = list(categories.keys())

    # Initialize BERT scorer with appropriate settings
    scorer = my_bert_score.MyBERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")

    column_categories = []
    for bert_column, sublist in zip(bert_friendly_columns, columns):
        # Calculate similarity scores between the current category and each column header
        scores = [scorer.get_word_similarity(bert_column, categories[key][1])[0][0] for key in category_keys]
        # Find the column with the highest similarity score
        max_val = max(scores)
        idx_max = scores.index(max_val)

        log(f"Category: {category_keys[idx_max]} | Scores: {scores}")

        if max_val >= 0:
            # Create a data field and convert the column
            field = DataFieldFactory.create(category_keys[idx_max], sublist)
            field.convert()  # Converts the column values in-place
        else:
            field = DataFieldFactory.create("etc", sublist)

        # Extract the converted values into a list
        converted_values = list(field.value)

        # Store the converted column under the corresponding category
        column_categories.append(converted_values)

    # Transpose the list of columns back into rows
    transposed_result = list(zip(*column_categories))
    return transposed_result
