import streamlit as st
import math
from scipy.stats import fisher_exact, chi2, f, t

# Function to calculate binomial coefficient
def binomial_coefficient(n, k):
    return math.comb(n, k)

# Function to calculate factorial
def factorial(n):
    return math.factorial(n)

# Function to calculate the probability P for a 2x2 table (Fisher Exact Test)
def calculate_2x2_fisher(a, b, c, d):
    _, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return p_value

# Function to calculate the probability P for a 3x2 or 4x2 table (Fisher Exact Test approximation)
def calculate_fisher_for_larger_tables(table):
    row_totals = [sum(row) for row in table]
    col_totals = [sum(col) for col in zip(*table)]
    n = sum(row_totals)

    numerator = 1
    for i in range(len(row_totals)):
        numerator *= factorial(row_totals[i])
    for j in range(len(col_totals)):
        numerator *= factorial(col_totals[j])
    
    denominator = factorial(n)
    for i in range(len(table)):
        for j in range(len(table[i])):
            denominator *= factorial(table[i][j])
    
    p_value = numerator / denominator
    return p_value  # This is an approximation; for exact p-value use external libraries.

# Function to calculate Chi-square test
def calculate_chi_square(table):
    row_totals = [sum(row) for row in table]
    col_totals = [sum(col) for col in zip(*table)]
    n = sum(row_totals)

    expected = [[(row_total * col_total) / n for col_total in col_totals] for row_total in row_totals]
    chi2_stat = sum(((table[i][j] - expected[i][j]) ** 2) / expected[i][j] for i in range(len(table)) for j in range(len(table[0])))
    p_value = chi2.sf(chi2_stat, (len(row_totals) - 1) * (len(col_totals) - 1))
    return chi2_stat, p_value

# Function to calculate ANOVA
def calculate_anova_from_table(table):
    flattened_table = [item for row in table for item in row]
    overall_mean = sum(flattened_table) / len(flattened_table)

    group_means = [sum(row) / len(row) for row in table]
    between_group_sum_of_squares = sum(len(row) * (group_mean - overall_mean) ** 2 for row, group_mean in zip(table, group_means))
    within_group_sum_of_squares = sum(sum((x - group_mean) ** 2 for x in row) for row, group_mean in zip(table, group_means))

    df_between = len(table) - 1
    df_within = len(flattened_table) - len(table)
    
    mean_square_between = between_group_sum_of_squares / df_between
    mean_square_within = within_group_sum_of_squares / df_within

    f_stat = mean_square_between / mean_square_within
    p_value = f.sf(f_stat, df_between, df_within)
    return f_stat, p_value

# Function to calculate T-test
def calculate_t_test_from_table(table):
    group1 = table[0]
    group2 = table[1] if len(table) > 1 else []

    mean1 = sum(group1) / len(group1)
    mean2 = sum(group2) / len(group2)
    var1 = sum((x - mean1) ** 2 for x in group1) / (len(group1) - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (len(group2) - 1)
    n1, n2 = len(group1), len(group2)

    pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    t_stat = (mean1 - mean2) / math.sqrt(pooled_variance * (1/n1 + 1/n2))
    p_value = 2 * t.sf(abs(t_stat), df=n1 + n2 - 2)
    return t_stat, p_value

# Function to interpret results
def interpret_results(p_value, alpha=0.05):
    if p_value < alpha:
        st.success("Reject the null hypothesis. There is a significant difference between the groups.")
    else:
        st.info("Fail to reject the null hypothesis. There is no significant difference between the groups.")

# Streamlit app
test_type = st.sidebar.selectbox("Select the test to perform:", ["Fisher Exact Test", "Chi-square Test", "ANOVA", "T-test"])
st.title(f"{test_type} Calculator")

# Matrix size selection
matrix_size = st.selectbox("Select matrix size:", ["2x2", "3x2", "4x2"])

# Get the input values based on the matrix size
if matrix_size == "2x2":
    a = st.number_input("Enter value for a:", min_value=0, value=10)
    b = st.number_input("Enter value for b:", min_value=0, value=2)
    c = st.number_input("Enter value for c:", min_value=0, value=3)
    d = st.number_input("Enter value for d:", min_value=0, value=8)
    table = [[a, b], [c, d]]

elif matrix_size == "3x2":
    table = [[st.number_input(f"Group {i+1}, Outcome {j+1}:", min_value=0, value=1) for j in range(2)] for i in range(3)]

elif matrix_size == "4x2":
    table = [[st.number_input(f"Group {i+1}, Outcome {j+1}:", min_value=0, value=1) for j in range(2)] for i in range(4)]

# Perform the selected test
if test_type == "Fisher Exact Test":
    if matrix_size == "2x2":
        p_value = calculate_2x2_fisher(a, b, c, d)
    else:
        p_value = calculate_fisher_for_larger_tables(table)
    st.write(f"The calculated p-value for the {matrix_size} table is:", p_value)
    interpret_results(p_value)

elif test_type == "Chi-square Test":
    chi2_stat, p_value = calculate_chi_square(table)
    st.write(f"Chi-square statistic: {chi2_stat}")
    st.write(f"p-value: {p_value}")
    interpret_results(p_value)

elif test_type == "ANOVA":
    f_stat, p_value = calculate_anova_from_table(table)
    st.write(f"F-statistic: {f_stat}")
    st.write(f"p-value: {p_value}")
    interpret_results(p_value)

elif test_type == "T-test":
    if len(table) > 1:
        t_stat, p_value = calculate_t_test_from_table(table)
        st.write(f"T-test statistic: {t_stat}")
        st.write(f"p-value: {p_value}")
        interpret_results(p_value)
    else:
        st.write("T-test requires at least two groups.")