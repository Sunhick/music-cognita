import numpy as np
import matplotlib.pyplot as plt

N = 6
Accuracy = (53.76,47.58,24.52,58.37,41.73,49.64)
Error = (100 - 53.76,100 - 47.58,100 - 24.52,100 - 58.37,100 - 41.73,100 - 49.64)
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, Accuracy, width, color='b')
rects2 = ax.bar(ind + width, Error, width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy')
ax.set_title('Classifier Vs Accuracy')
ax.set_xticks(ind+width)
ax.set_xticklabels(('LogReg', 'DecisionTree', 'NaiveBayes', 'RandForest', 'SVM', 'SVM+Boosting'))
ax.legend((rects1[0], rects2[0]), ('Accuracy', 'Error'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()