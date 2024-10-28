# merge_tables.py
from get_matching_attributes import getMatchingAttributes


# This function merges two tables
def merge_tables(table1, table2):
    # Get mappings of each table's schema to the merged schema
    table1Map, table2Map = getMatchingAttributes(table1, table2)

    # Get merged schema, and add it to the final table 
    finalSchema = []

    for key in table1[0]:
        if table1Map[key] not in finalSchema:
            finalSchema.append(table1Map[key])

    for key in table2[0]:
        if table2Map[key] not in finalSchema:
                finalSchema.append(table2Map[key])

    finalTable = []
    finalTable.append(finalSchema)
    print(finalTable)

    # Add values from table 1 to the merged table
    for row in table1[1:]:
        finalTable.append([])
        for attribute in finalSchema:
            for index in range(len(table1[0])):
                if table1Map[table1[0][index]] == attribute:
                    finalTable[-1].append(row[index])
                    break
            else:
                finalTable[-1].append("NULL")

    # Add values from table 2 to the merged table
    for row in table2[1:]:
        finalTable.append([])
        for attribute in finalSchema:
            for index in range(len(table2[0])):
                if table2Map[table2[0][index]] == attribute:
                    finalTable[-1].append(row[index])
                    break
            else:
                finalTable[-1].append("NULL")

    return finalTable