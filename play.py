import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 40, 500]

# Create a plot
plt.plot(x, y)

# Set the maximum value on the y-axis
plt.ylim(0, 60)  # Adjust the values according to your needs

# Label the axes
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()
