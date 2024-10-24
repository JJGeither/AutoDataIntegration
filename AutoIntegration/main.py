# main.py
from file_io import read_csv, read_category_csv
from data_processing import categorize_table, merge_tables
import csv

# Load CSV File
FILE_PATH_MASTER = 'data.csv'
FILE_PATH_DEPENDENT = 'data2.csv'
FILE_PATH_CATEGORIES = 'categories_data.csv'

master_table = read_csv(FILE_PATH_MASTER)
dependent_table = read_csv(FILE_PATH_DEPENDENT)

# Map where key is the category and value are the aliases used in tables
# i.e temperature: {temperature,temp,degrees,celsius,fahrenheit,thermometer}
category_aliases = read_category_csv(FILE_PATH_CATEGORIES)

# Categorize the Columns
master_categories = categorize_table(master_table, category_aliases)
dependent_categories = categorize_table(dependent_table, category_aliases)

# Merge the tables
merged_data = merge_tables(master_categories, dependent_categories, len(master_table) - 1, len(dependent_table) - 1)

# Write to CSV
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    column_keys = list(merged_data.keys())
    writer.writerow(column_keys)

    max_length = max(len(merged_data[key].value) for key in column_keys)

    for i in range(max_length):
        row = [merged_data[key].value[i] if i < len(merged_data[key].value) else 'NULL' for key in column_keys]
        writer.writerow(row)
    print("Successfully printed to csv file")