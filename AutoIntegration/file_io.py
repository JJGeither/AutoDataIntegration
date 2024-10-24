# file_io.py
import csv
import os

NULL_VALUE = 'NULL'


def read_csv(file_path):
    """
    Opens and reads a CSV file. Replaces empty values with a NULL placeholder.
    """
    data = []

    if os.path.exists(file_path):
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)
            col_count = len(header)
            file.seek(0)

            for row in reader:
                if len(row) < col_count:
                    row.append(NULL_VALUE)

                row = [NULL_VALUE if item == '' else item for item in row]
                data.append(row)
    else:
        print(f"The file '{file_path}' does not exist. Creating a new file.")
        with open(file_path, mode='w', newline=''):
            pass  # Create an empty file if it does not exist

    return data


def read_category_csv(file_path):
    category = read_csv(file_path)
    category_map = {}

    for row in category[1:]:
        if len(row) > 1:
            # First element as key, rest as the value
            category_map[row[0]] = row[1:]
        else:
            # If there's only one element in the row, map the key to an empty list
            category_map[row[0]] = []

    return category_map

