# main.py
from file_io import read_csv, read_category_csv
from data_processing import standardize_table_units, EMPTYVALUE
from merge_tables import merge_tables
import csv



doLoop = True
while doLoop:
    # Load CSV File
    #FILE_PATH_MASTER = input("Filename for table 1: ") #'data.csv'
    FILE_PATH_MASTER = 'data2.csv'
    #FILE_PATH_DEPENDENT = input("Filename for table 2: ") #'data2.csv'
    FILE_PATH_DEPENDENT = 'data.csv'
    FILE_PATH_CATEGORIES = 'categories_data.csv'
    
    master_table = read_csv(FILE_PATH_MASTER)
    dependent_table = read_csv(FILE_PATH_DEPENDENT)

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
