import numpy as np
import sys
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

# The functions for Euclidean, Manhattan and Cosine have been mainly generated using ChatGPT.
# Minor tweaks have been made for vectorization.
# Prompt: Given 2 np arrays find the cosine,euclidean and manhattan distance between them using numpy.

# The axis = -1 is there because we have vectorized the code so that we can directly find distances from all points in the training data set to avoid the for-loop


class kNearestNeighbours:
    def __init__(self, encoder, k, distance_metric, train_data):
        self.k = k;
        self.encoder = encoder
        self.distance_metric = distance_metric
        self.train_data = train_data
        
    def setParams(self, encoder, k, distance_metric):
        self.k = k;
        self.encoder = encoder
        self.distance_metric = distance_metric
        
    def Euclidean(self,a, b):
        return np.sqrt(np.sum((a-b)**2,axis=-1))

    def Manhattan(self,a, b):
        return np.sum(np.abs(a - b),axis=-1)
    
    def Cosine(self,a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cosine_similarity = dot_product / (norm_a * norm_b)
        cosine_distance = 1 - cosine_similarity
        return cosine_distance

    def getPrediction(self,train_data,tdata_point):
        d_list = []
        list_features = train_data[:,0]
        list_labels = train_data[:,1]
        t_data = np.vstack(list_features)
        
        if(self.distance_metric=='euclidean'):
            d_list = self.Euclidean(t_data,tdata_point[0])
        elif(self.distance_metric=='manhattan'):
            d_list = self.Manhattan(t_data,tdata_point[0])
        else:
            d_list = self.Cosine(t_data,tdata_point[0])
        
        datype = [('col1', float), ('col2', 'U20')]
        
        # Creating new empty n*2 numpy array so that tuple of distances and labels can be stored together
        dist_list = np.empty((len(d_list),2), dtype='object')    
        dist_list[:,0] = d_list
        dist_list[:,1] = list_labels
        
        # The next line has been generated using ChatGPT
        # Prompt: Write python code to extract minimum k elements in a 2d list by value of 1st element using numpy functions
        k_labels = dist_list[np.argsort(dist_list[:,0])[:self.k]]

        unique_elements, counts = np.unique(k_labels[:,1], return_counts=True)
        max_count_frequency = np.max(counts)
        
        # The next 2 lines have been generated using ChatGPT
        # Prompt: How to extract all labels whose frequency is maximum in a given array and the frequency is known.
        values_with_max_freq = unique_elements[counts == max_count_frequency]
        min_dist_el = k_labels[np.isin(k_labels[:,1],values_with_max_freq)]

        # The next line has been generated using ChatGPT
        # Prompt: How to find inverse of distances using numpy.
        values = 1 / (1 + min_dist_el[:, 0])
        
        keys,ind = np.unique(min_dist_el[:, 1],return_inverse=True)
        
        # The next 3 lines have been generated using ChatGPT
        # Prompt: How to get array element with maximum frequency in np array
        sums = np.bincount(ind, weights=values.astype(float))
        
        max_idx = np.argmax(sums)
        prediction = keys[max_idx]
        
        return prediction
        
                
        
    def predictClass(self, test_data):
        train_features = None
        test_features = None
        
        predicted_labels = []
        true_labels = test_data[:,3]
        
        if(self.encoder == 'resnet'):
            train_features = train_data[:, [1,3]]
            test_features = test_data[:, [1,3]]
        else:
            train_features = train_data[:, [2,3]]
            test_features = test_data[:, [2,3]]
            
        for i in test_features:
            label = self.getPrediction(train_features,i[0])
            predicted_labels.append(label)
            
        f1score = f1_score(true_labels, predicted_labels, average = "weighted", zero_division=1) 
        recall = recall_score(true_labels, predicted_labels, average = "weighted", zero_division=1)
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average = "weighted", zero_division=1)
        
        return f1score, accuracy, precision, recall

try:
    train_data = np.load("data.npy", allow_pickle=True)
    test_data = np.load(str(sys.argv[1]), allow_pickle=True)

    kNN = kNearestNeighbours('vit', 10, 'euclidean', train_data)

    f1score, accuracy, precision, recall = kNN.predictClass(test_data)

    print("F1-Score (weighted): " + str(f1score))
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
except Exception as e:
    print("Error!!! Exit message: ",e)