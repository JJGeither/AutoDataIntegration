# categorization.py

CATEGORY_THRESHOLD = 10

def levenshtein_distance(str1, str2):
    """
    Calculates the Levenshtein distance between two strings.
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n]


def assign_category(column_name, categories_map):
    """
    Assigns a category to a column based on the closest match from known categories.
    """
    closest_category = "etc"
    min_distance = CATEGORY_THRESHOLD

    for category, aliases in categories_map.items():
        for alias in aliases:
            distance = levenshtein_distance(alias.lower(), column_name.lower())
            if distance < min_distance:
                min_distance = distance
                closest_category = category

    return closest_category


def extract_column_values(data_table, col_index):
    """
    Extracts all values from a specified column index.
    """
    return [row[col_index] for row in data_table[1:]]
