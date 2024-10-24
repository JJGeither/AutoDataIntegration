# data_processing.py
from data_fields import DataFieldFactory
from categorization import assign_category, extract_column_values

def categorize_table(data, categories):
    """
    Categorizes columns in the data based on predefined categories.
    """
    column_categories = {}

    for index, col_name in enumerate(data[0]):
        # Returns the string name of the assigned category
        category = assign_category(col_name, categories)

        # Retrieves all values within the column
        column_values = extract_column_values(data, index)

        # Creates polymorphic object based on category and fills it with column_values
        column_categories[category] = DataFieldFactory.create(category, column_values)

    return column_categories


def merge_tables(master, dependent, row_count_master, row_count_dependent, null_value='NULL'):
    """
    Merges two tables by appending missing fields and filling in NULL where needed.
    """
    merged_data = master.copy()
    all_columns = list(master.keys())

    all_columns.extend([key for key in dependent if key not in master])

    for column in all_columns:
        if column not in master:
            master[column] = DataFieldFactory.create(column, [null_value] * row_count_master)

        if column in dependent and dependent[column].value:
            master[column].value.extend(dependent[column].value)
        else:
            master[column].value.extend([null_value] * row_count_dependent)

    return master
