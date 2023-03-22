from skseq.sequences.sequence_list import SequenceList
from skseq.sequences.label_dictionary import LabelDictionary
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

class Corpus(object):

    def __init__(self):

        # Word dictionary.
        self.word_dict = dict()

        # POS tag dictionary.
        # Initialize noun to be tag zero so that it the default tag.
        self.tag_dict = dict()
        

    def build_corpus(self, train):
        
        i = 0
        for tag in train.tags.unique():
            self.tag_dict[tag] = i
            i += 1
        i = 0
        for word in train.words.unique():
            self.word_dict[word] = i
            i += 1
        self.tag_dict = LabelDictionary(self.tag_dict)
        self.word_dict = LabelDictionary(self.word_dict)
        
    def read_sequence(self, train):
        seq_list = SequenceList(self.word_dict, self.tag_dict)
        for idx in tqdm(set(train.sentence_id)):
            seq_list.add_sequence(list(train.words[train.sentence_id == idx]), list(train.tags[train.sentence_id == idx]), self.word_dict, self.tag_dict)
        return seq_list


def evaluate_corpus(sequences, sequences_predictions, tag_dict, metric = 'all', exclude_O_acc = True):
    """Evaluate classification accuracy, confusion matrix and F1-score."""
    y_true = []
    y_pred = []
    for i, sequence in enumerate(sequences):
        pred = sequences_predictions[i]
        for j, y_hat in enumerate(pred.y):
            y_true.append(sequence.y[j])
            y_pred.append(y_hat)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    #Accuracy
    if (metric == 'all') | (metric == 'accuracy'):
        if exclude_O_acc:
            print('Note: Words with real tag "O" are not being used in the computation of accuracy.')
            y_true_aux = y_true[y_true != 0]
            y_pred_aux = y_pred[y_true != 0]
            acc = (y_true_aux == y_pred_aux).mean()
        else:
            acc = (y_true == y_pred).mean()
        print('Accuracy: ' + str(round(acc, 4)*100) + '%')
    
    if (metric == 'all') | (metric == 'precision'):
        print('Precisions by tag:')
        for i in range(len(tag_dict)):
            y_true_aux = y_true[y_true == i]
            y_pred_aux = y_pred[y_true == i]
            prec = (y_true_aux == y_pred_aux).mean()
            print('Tag ' + list(tag_dict.keys())[i] + ': ' + str(round(prec,4)*100) + '%')
        
    #Confusion matrix
    if (metric == 'all') | (metric == 'confusion_matrix'):
        colormap = sns.color_palette("Greens")
        print('Confusion matrix:')
        cm = confusion_matrix(y_true, y_pred)
        labels = list(tag_dict.values())
        class_names = list(tag_dict.keys())
        
        #Plot confusion matrix
        fig = plt.figure(figsize=(10, 8))
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax, fmt = 'g', norm = LogNorm(), cmap = colormap); #annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel('Predicted', fontsize=20)
        ax.xaxis.set_label_position('bottom')
        plt.xticks(rotation=90)
        ax.xaxis.set_ticklabels(class_names, fontsize = 10)
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True', fontsize=20)
        ax.yaxis.set_ticklabels(class_names, fontsize = 10)
        plt.yticks(rotation=0)

        plt.title('Confusion matrix', fontsize=20)
        plt.show()
        
    #F1-Score
    if (metric == 'all') | (metric == 'f1score'):
        f1 = f1_score(y_true, y_pred, average = 'weighted')
        print('F1-Score: ' + str(round(f1, 2)))
    
    return 'Metrics computed!'


