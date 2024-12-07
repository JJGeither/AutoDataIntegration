# main.py

from file_io import read_csv, read_category_csv
from data_processing import standardize_table_units, EMPTYVALUE
from merge_tables import merge_tables
import csv
from log import log, set_logging

set_logging(True)
log("Starting program...")

doLoop = True
while doLoop:
    # Load CSV File
    FILE_PATH_MASTER = input("Filename for table 1: ")
    FILE_PATH_DEPENDENT = input("Filename for table 2: ")
    FILE_PATH_CATEGORIES = 'categories_data.csv'
    
    master_table = read_csv(FILE_PATH_MASTER)
    dependent_table = read_csv(FILE_PATH_DEPENDENT)

    # Replace empty entries in the form of [''] with NULL
    master_table = [["NULL" if (len(value) == 2 and value[0] == "'" and value[-1] == "'") or value == " " else value for value in row] for row in master_table]
    dependent_table = [["NULL" if (len(value) == 2 and value[0] == "'" and value[-1] == "'") or value == " " else value for value in row] for row in dependent_table]

    # Remove quotes surrounding entries (e.g. 'user id' --> user id)
    master_table = [[value if value[0] != "'" and value[-1] != "'" else value[1:-1] for value in row] for row in master_table]
    dependent_table = [[value if value[0] != "'" and value[-1] != "'" else value[1:-1] for value in row] for row in dependent_table]

    # Standardizes the units for each element for homogeneity between tables
    master_table_standardized = standardize_table_units(master_table)
    dependent_table_standardized = standardize_table_units(dependent_table)

    # Merge the tables
    merged_table = merge_tables(master_table_standardized, dependent_table_standardized)
    
    # Write to CSV
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
    
        for row in merged_table:
            writer.writerow(row)
    
        print("Successfully printed to csv file")

    while True:
        doAgain = input("Would you like to merge more tables? (Y/N)")
        if doAgain == "N":
            doLoop = False
            break
        if doAgain == "Y":
            doLoop = True
            break
