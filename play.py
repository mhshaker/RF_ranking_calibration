import scipy.stats as stats

observed_successes = 17  # The number of observed successes
expected_successes = 10   # The expected number of successes under the null hypothesis
n = 20                    # The total number of trials

# Calculate the probability of observing a value less than or equal to observed_successes
p_value_lower = stats.binom.cdf(observed_successes, n, 0.5)

# Calculate the probability of observing a value greater than or equal to observed_successes
p_value_upper = 1 - stats.binom.cdf(observed_successes - 1, n, 0.5)

# Combine the two-tailed p-values
p_value = 2 * min(p_value_lower, p_value_upper)

alpha = 0.05
print("p_value", p_value)

if p_value < alpha:
    print("Reject the null hypothesis. The observed frequency is significantly different from the expected frequency.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference.")


result = stats.binomtest(observed_successes, n=n, p=0.5, alternative='two-sided')
print("p_value", result.pvalue)
if p_value < alpha:
    print("Reject the null hypothesis. The observed frequency is significantly different from the expected frequency.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference.")
