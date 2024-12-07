# merge_tables.py

from get_matching_attributes import getMatchingAttributes
from get_matching_entities import get_matching_entities
from log import log
import csv


# This function merges two tables
def merge_tables(table1, table2):
    # Create the final table to be returned
    finalTable = []

    matchTable = []
    matchTable.append([])
    for attribute in table1[0]:
        matchTable[-1].append(attribute)
    for attribute in table2[0]:
        matchTable[-1].append(attribute)

    # Get mappings of each table's schema to the merged schema
    table1Map = getMatchingAttributes(table1, table2)

    # Get merged schema, and add it to the final table 
    finalSchema = []
    for attribute in table1[0]:
        finalSchema.append(attribute)
    for attribute in table2[0]:
        if attribute not in table1Map.values():
            finalSchema.append(attribute)
    finalTable.append(finalSchema)

    # Get entity map. Key is index of entity in table 1, value is matching entity in table 2, or -1 there is no match
    entity_map = get_matching_entities(table1, table2, table1Map)
    log(f"Number of matches is: {len(list(set(entity_map.values()))) - 1}")

    # Maps attributes to their indices. Used below to get the index of a final schema attribute in its original table.
    attr_to_index_1 = {table1[0][i]: i for i in range(len(table1[0]))}
    attr_to_index_2 = {table2[0][i]: i for i in range(len(table2[0]))}

    # Add entities from table1 to the merged table, with data from their matching entities in table2
    for i in range(len(table1) - 1):
        finalTable.append([])
        t1_entity = table1[i + 1]
        for attribute in finalSchema:
            if attribute in table1[0]:
                finalTable[-1].append(t1_entity[attr_to_index_1[attribute]])
            elif attribute in table2[0] and entity_map[i] is not None:
                t2_entity = table2[entity_map[i] + 1]
                finalTable[-1].append(t2_entity[attr_to_index_2[attribute]])
            else:
                finalTable[-1].append("NULL")

    # Add entities from table2 that were not matched with an entity in table1
    for i in range(len(table2) - 1):
        if i not in entity_map.values():
            finalTable.append([])
            t2_entity = table2[i + 1]
            for attribute in finalSchema:
                if attribute in table2[0]:
                    finalTable[-1].append(t2_entity[attr_to_index_2[attribute]])
                elif table1Map[attribute] in table2[0]:  # if attribute is a merged attribute with a name from table 1
                    finalTable[-1].append(t2_entity[attr_to_index_2[table1Map[attribute]]])
                else:
                    finalTable[-1].append("NULL")

    # Output file with both schemas and matching entities concatenated instead of merged
    for i in range(len(table1) - 1):
        matchTable.append([])
        t1_entity = table1[i + 1]
        for val in t1_entity:
            matchTable[-1].append(val)
        if entity_map[i] is not None:
            t2_entity = table2[entity_map[i] + 1]
            for val in t2_entity:
                matchTable[-1].append(val)
        else:
            for j in range(len(table2[0])):
                matchTable[-1].append("NULL")
    for i in range(len(table2) - 1):
        if i not in entity_map.values():
            matchTable.append([])
            t2_entity = table2[i + 1]
            for j in range(len(table1[0])):
                matchTable[-1].append("NULL")
            for val in t2_entity:
                matchTable[-1].append(val)
    # Write to CSV
    with open('data/matches.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        for row in matchTable:
            writer.writerow(row)

    return finalTable