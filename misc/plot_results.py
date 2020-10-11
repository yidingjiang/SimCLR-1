import matplotlib.pyplot as plt 
import seaborn as sns
# sns.set(style="whitegrid")

# lambda_ = [0, 5e-4, 5e-3, 5e-2, 0.5, 1, 2]
# top_1_test_acc = [74.507, 78.02, 75.79, 71.64, 10, 10, 10]
# # lambda_ = [0, 5e-4, 5e-3, 5e-2, 1]
# # top_1_test_acc = [74.507, 77.27, 10, 23.59, 21.18]
# lambda_plot = sns.barplot(x=lambda_, y=top_1_test_acc)
# lambda_plot.set(xlabel='lambda', ylabel='Top-1 Test Accuracy')
# lambda_plot.set_title("Test Accuracies against lambda for SimCLR + norm of Jacobian wrt color jitter params")
# fig = lambda_plot.get_figure()
# fig.savefig("Accuracy vs Lambda for color jitter")

data = [
[67.96,68.40,71.06,73.72,74.86,74.59],
[67.55,70.74,71.82,74.62,74.48,75.25],
[69.06,70.95,75.23,73.46,73.84,74.79],
[70.85,71.25,73.16,73.65,74.07,74.50],
[67.84,70.08,73.37,75.09,74.37,74.86],
[68.29,67.01,74.05,72.23,74.94,75.53],
]
xticklabels = ["0", "1e-4", "2.5e-4", "5e-4", "7.5e-4", "1e-3"]
yticklabels = ["0", "1e-4", "2.5e-4", "5e-4", "7.5e-4", "1e-3"]
ax = sns.heatmap(data, cmap="YlGnBu", xticklabels=xticklabels, yticklabels=yticklabels)
ax.set(xlabel='lambda1 (crop)', ylabel='lambda2 (color jitter)')
fig = ax.get_figure()
fig.savefig("Heatmap_top_1_accuracy")