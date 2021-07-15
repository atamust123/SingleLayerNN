import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-0.005 * x))
def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)


df = pd.read_csv('./breast-cancer-wisconsin.csv').astype(str)
df= df.drop(["Code_number"], axis=1)
df.replace(to_replace="?",value="0",inplace=True)
df=df.replace(to_replace = np.nan, value =0).astype(int)
df_copy=df.copy()
train_set = df_copy.sample(frac=0.8, random_state=0)
test_set = df_copy.drop(train_set.index)
train_set_corr = train_set.iloc[:,:-1].corr("pearson")

heatmap = plt.pcolor(train_set_corr, cmap="inferno", )


plt.xticks(range(len(train_set_corr.columns)), train_set_corr.columns, rotation=45)
plt.yticks(range(len(train_set_corr.columns)), train_set_corr.columns, rotation=60)
plt.colorbar()
for y in range(train_set.shape[1]-1):
    for x in range(train_set.shape[1]-1):
        plt.text(x + 0.5, y + 0.5, '%.2f' % train_set_corr.iloc[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.title("Correlation of Attributes of Training Set")
plt.gca().invert_yaxis()


iteration_count = 1000
np.random.seed(1)
weights = 2 * np.random.random((9, 1)) - 1
accuracy_array = []
loss_array = []

training_outputs=train_set.iloc[:,-1].to_numpy()
training_outputs=np.transpose(training_outputs)
training_outputs=np.reshape(training_outputs,(-1,1))

label_matrix=test_set.iloc[:,-1].to_numpy()
label_matrix=np.transpose(label_matrix)
label_matrix=np.reshape(label_matrix,(-1,1))

for iteration in range(iteration_count):
    train_inputs = train_set.iloc[:,:-1]
    outputs=np.dot(train_inputs,weights)
    outputs=sigmoid(outputs)
    loss=training_outputs-outputs

    tuning=loss*sigmoid_derivative(outputs)
    weights+=np.dot(np.transpose(train_inputs),tuning)

    test_intputs=test_set.iloc[:,:-1]
    test_outputs=sigmoid(np.dot(test_intputs,weights))

    test_outputs=np.where(test_outputs<=0.5,0,test_outputs)
    test_outputs=np.where(test_outputs>0.5,1,test_outputs)
    tp_count = 0

    for i in range(len(test_outputs)):
        if test_outputs[i] == label_matrix[i]:
            tp_count += 1
    accuracy_array.append(tp_count/len(test_outputs))
    loss_array.append(loss.mean())
plt.show()
plt.plot(accuracy_array,label="Accuracy array")
plt.plot(loss_array,label="Loss array")
plt.legend()
plt.title("Train/Test accuracy and loss plots")
plt.show()
