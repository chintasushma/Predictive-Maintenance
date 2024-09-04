import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import seaborn as sn
import pandas as pd
from itertools import cycle
from sklearn import metrics
from sklearn.metrics import roc_curve

no_of_datasets = 5

def Plot_Results():
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    X = np.arange(5)

    ax.bar(X + 0.00, Eval[:, 2, 0, 4].astype('float'), color='orange', width=0.10, label="EOO-DWF-CResLSTM-AM")
    ax.bar(X + 0.10, Eval[:, 2, 1, 4].astype('float'), color='yellow', width=0.10, label="GSO-DWF-CResLSTM-AM")
    ax.bar(X + 0.20, Eval[:, 2, 2, 4].astype('float'), color='cyan', width=0.10, label="RSA-DWF-CResLSTM-AM")
    ax.bar(X + 0.30, Eval[:, 2, 3, 4].astype('float'), color='deeppink', width=0.10, label="AZO-DWF-CResLSTM-AM")
    ax.bar(X + 0.40, Eval[:, 2, 4, 4].astype('float'), color='crimson', width=0.10, label="EAZO-DWF-CResLSTM-AM")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
               ncol=3, fancybox=True, shadow=False)
    plt.xticks(X + 0.25, ('Dataset-1', 'Dataset-2', 'Dataset-3', 'Dataset-4', 'Dataset-5'))
    plt.xlabel('Datasets', fontsize=11)
    plt.ylabel('Accuracy')
    path = "./Results/Accuracy_alg-kfold.png"
    plt.savefig(path)
    plt.show()

    for i in range(no_of_datasets):
        Eval = np.load('Eval_all.npy', allow_pickle=True)[i]
        Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1-Score',
                 'MCC']
        Graph_Term = [0]
        Algorithm = ['TERMS', 'EOO', 'GSO', 'RSA', 'AZO', 'PROPOSED']
        Classifier = ['TERMS', 'MobileNet', 'DTCN', 'RNN', 'Without Opt', 'PROPOSED']

        value = Eval[2, :, 4:]
        value[:, :-1] = value[:, :-1] * 100

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('--------------------------------------------------Dataset-',str(i+1) ,' Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[j + 5, :])
        print('--------------------------------------------------Dataset-', str(i + 1), ' Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

        Eval = np.load('Eval_all.npy', allow_pickle=True)
        for j in range(len(Graph_Term)):
            Graph = np.zeros((Eval.shape[1], Eval.shape[2]))
            for k in range(Eval.shape[1]):
                for l in range(Eval.shape[2]):
                    if Graph_Term[j] == 10:
                        Graph[k, l] = Eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = Eval[i, k, l, Graph_Term[j] + 4] * 100

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)

            ax.bar(X + 0.00, Graph[:, 5], color='red', width=0.10, label="MobileNet")
            ax.bar(X + 0.10, Graph[:, 6], color='lime', width=0.10, label="DTCN")
            ax.bar(X + 0.20, Graph[:, 7], color='deeppink', width=0.10, label="RNN")
            ax.bar(X + 0.30, Graph[:, 8], color='blue', width=0.10,
                   label="LSTM")
            ax.bar(X + 0.40, Graph[:, 9], color='cyan', width=0.10,
                   label="CResLSTM-AM")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
                       ncol=3, fancybox=True, shadow=False)
            plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
            plt.xlabel('k-fold', fontsize=11)
            plt.ylabel(Terms[Graph_Term[j]], fontsize=11)
            path = "./Results/Dataset-%s-%s_cls.png" % (str(i + 1), Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()

def Plot_Confusion():
    # Confusion Matrix
    for i in range(no_of_datasets):
        Eval = np.load('Eval_all.npy', allow_pickle=True)[i]
        value = Eval[2, 4, :5]
        val = np.asarray([0, 1, 1])
        data = {'y_Actual': [val.ravel()],
                'y_Predicted': [np.asarray(val).ravel()]
                }
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'], colnames=['Predicted'])
        value = value.astype('int')

        confusion_matrix.values[0, 0] = value[1]
        confusion_matrix.values[0, 1] = value[3]
        confusion_matrix.values[1, 0] = value[2]
        confusion_matrix.values[1, 1] = value[0]

        sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = ' + str(Eval[2, 4, 4] * 100)[1:6] + '%')
        sn.plotting_context()
        path1 = './Results/Dataset-%s-Confusion.png' % str(i+1)
        plt.savefig(path1)
        plt.show()

def Plot_ROC():
        lw = 2
        cls = ['MobileNet', 'DTCN', 'RNN', 'LSTM', 'CResLSTM-AM']
        colors = cycle(["hotpink", "plum", "chocolate", "green", "magenta"])
        for i in range(no_of_datasets):
            Predicted = np.load('roc_score.npy', allow_pickle=True)[i]
            Actual = np.load('roc_act.npy', allow_pickle=True)[i]
            for j, color in zip(range(5), colors):  # For all classifiers

                false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[2, j+5][:, 0].astype('float'),
                                                                                  Predicted[2, j+5][:, 0].astype('float'))
                # auc = metrics.roc_auc_score(Actual[j, :], Predicted[j, :])
                auc = metrics.roc_auc_score(Actual[2, j+5][:, 0].astype('float'), Predicted[2, j+5][:, 0].astype('float'))
                plt.plot(
                    false_positive_rate1,
                    true_positive_rate1,
                    color=color,
                    lw=lw,
                    label="{0} (auc = {1:0.2f})".format(cls[j], auc),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            path = "./Results/Dataset-%s-Roc.png" % str(i+1)
            plt.savefig(path)
            plt.show()
def Plot_Convergence():
    convs = []
    for a in range(no_of_datasets):
        conv = np.load('Fitness.npy', allow_pickle=True)[a]
        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ['EOO', 'GSO', 'RSA', 'AZO', 'PROPOSED']

        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print('-------------------------------------------------- Dataset',str(a+1),'Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='red', linewidth=3, marker='d', markerfacecolor='deeppink', markersize=8,
                 label="EOO")
        plt.plot(iteration, conv[1, :], color='yellow', linewidth=3, marker='d', markerfacecolor='cyan', markersize=8,
                 label="GSO")
        plt.plot(iteration, conv[2, :], color='green', linewidth=3, marker='d', markerfacecolor='magenta', markersize=8,
                 label="RSA")
        plt.plot(iteration, conv[3, :], color='magenta', linewidth=3, marker='d', markerfacecolor='plum', markersize=8,
                 label="AZO")
        plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='d', markerfacecolor='blueviolet', markersize=8,
                 label="EAZO")
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        path1 = "./Results/Dataset-%s-convergence.png" % str(a+1)
        plt.savefig(path1)
        plt.show()

if __name__ == '__main__':
    Plot_Convergence()
    Plot_Results()
    Plot_Confusion()
    Plot_ROC()