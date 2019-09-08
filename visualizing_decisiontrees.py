#https://towardsdatascience.com/a-guide-to-decision-trees-for-machine-learning-and-data-science-fe2607241956
import sklearn.datasets as datasets
import pandas as pd
iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target

from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(df,y)


from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = export_graphviz(dtree)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_pdf("tree.pdf")

#gini shows the impurity of the node. gini=0 means that the node is purely of one class, gini>0 means that it contains multiple classes.
#value list tells how many samples at a given node fall into each category.
#class refers to the class that is predicted to describe the node.