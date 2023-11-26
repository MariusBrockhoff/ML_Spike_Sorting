import tensorflow as tf
import keras.backend as K
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import KDTree
from sklearn import metrics
from scipy.optimize import minimize
from utils.pretrain_models import *
import wandb
import matplotlib.pyplot as plt



nmi = normalized_mutual_info_score
ari = adjusted_rand_score

#TODO: merge / incorporate DEC, IDEC, dynAE
#TODO: add weakly supervised finetuning
#TODO: add Pseudolabel finetuning

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    #y_pred = y_pred.argmax(1)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def acc_tf(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    y_pred = y_pred.argmax(1)
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def check_filepath_naming(filepath):
    if path.exists(filepath):
        numb = 1
        while True:
            newPath = "{0}_{2}{1}".format(*path.splitext(filepath) + (numb,))
            if path.exists(newPath):
                numb += 1
            else:
                return newPath
    return filepath

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        elif isinstance(item, (int, float)):  # Check for integers and floats
            flattened.append(item)
    return flattened

def sampling_weighted(label_ratio, rho_normed):

  bins = 100

  y_dist, x_dist ,_ = plt.hist(rho_normed, bins=bins)

  x_dist_ = x_dist + (1/(2*bins))
  x_dist_center = x_dist_[:-1]

  desired_fr = label_ratio

  def objective_function(p):

      fr = np.dot(y_dist,x_dist_center**p)/np.sum(y_dist)
      diff = fr - desired_fr

      return np.linalg.norm(diff)**2

  if label_ratio>0.35:
    p_guess = 1
  elif 0.2<label_ratio<0.35:
    p_guess = 2
  elif 0.1<label_ratio<0.2:
    p_guess = 3
  else:
    p_guess = 4

  result = minimize(objective_function, p_guess)
  optimal_p = result.x[0]

  ratios = x_dist_center**optimal_p

  conditions = [(i/100 <= rho_normed) & (rho_normed < (i+1)/100) for i in range(100)]

  selected_indices = []

  for condition, percentage in zip(conditions, ratios):
      # Find the indices that satisfy the condition
      indices = np.where(condition)[0]

      # Calculate the number of values to sample
      num_samples = int(percentage * len(indices))

      # Randomly sample values based on the condition
      sampled_indices = np.random.choice(indices, num_samples, replace=False)

      # Append the sampled values to the selected_values list
      selected_indices.extend(sampled_indices.tolist())

  selected_indices = np.array(flatten_list(selected_indices))

  np.random.shuffle(selected_indices)

  return selected_indices

def calculate_densities(data, k, density_function):
    n, d = data.shape
    if d <= 10:
        tree = KDTree(data)
        knn_dist, knn = tree.query(data, k=k)
    else:
        """#Faster but High RAM version
        dist = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
        knn = np.argsort(dist, axis=1)[:, 1:k+1]
        knn_dist = dist[np.arange(n)[:, None], knn]"""

        # Slower but less memory hungry
        dist = np.empty((n, n), dtype=np.float64)
        knn = np.empty((n, k), dtype=np.int64)
        fills = np.empty((n, d), dtype=np.float64)
        for i in range(n):
            # Compute squared Euclidean distances to all other points
            np.square(data[i] - data, out=fills)
            np.sum(fills, axis=1, out=dist[i])

            # Find indices of k+1 nearest neighbors
            neighbors = np.argpartition(dist[i], k + 1)[:k + 1]
            knn[i] = neighbors[1:]  # exclude the point itself

            # Replace distances to self with infinity
            dist[i, i] = np.inf
        # Compute distances to k nearest neighbors
        knn_dist = np.sqrt(dist[np.arange(n)[:, None], knn])
        for i, neighbors in enumerate(knn):
            knn_dist[i] = dist[i, neighbors]

    # calculate the knn-density value
    if density_function == 'default':
        rho = knn_dist[:, -1] ** -1
    elif density_function == 'mean':
        rho = np.mean(knn_dist, axis=1) ** -1

    rho_normed = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))

    return rho_normed
class ClusteringLayer(tf.keras.layers.Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = tf.keras.layers.InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def target_distribution(q):
  """
  computing an auxiliary target distribution
  """
  weight = q ** 2 / q.sum(0)
  return (weight.T / weight.sum(1)).T


class BaseModelDEC(tf.keras.Model):
    def __init__(self,

                 encoder,

                 n_clusters):
        super(BaseModelDEC, self).__init__()

        self.Encoder = encoder

        self.n_clusters = n_clusters

        self.clustering = ClusteringLayer(self.n_clusters, name='clustering')

    def call(self, inputs):

        logits = self.Encoder(inputs)

        out = self.clustering(logits)

        return out

class BaseModelIDEC(tf.keras.Model):
    def __init__(self,

                 encoder,

                 decoder,

                 n_clusters):
        super(BaseModelIDEC, self).__init__()

        self.Encoder = encoder

        self.Decoder = decoder

        self.n_clusters = n_clusters

        self.clustering = ClusteringLayer(self.n_clusters, name='clustering')

    def call(self, inputs):

        logits = self.Encoder(inputs)

        out = self.Decoder(logits)

        clus = self.clustering(logits)

        return clus, out

class IDEC(object):
    def __init__(self,
                 model,
                 input_dim,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):

        super(IDEC, self).__init__()

        self.autoencoder = model
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.model = None

    def initialize_model(self, ae_weights=None, gamma=0.1, optimizer='adam'):
        if ae_weights is not None:  # load pretrained weights of autoencoder
            dummy = tf.zeros(shape=[1, self.input_dim[0]], dtype=tf.dtypes.float32, name=None)
            self.autoencoder(dummy)
            self.autoencoder.load_weights(ae_weights)
        else:
            print('ae_weights, i.e. path to weights of a pretrained model must be given')
            exit()


        # prepare IDEC model
        self.model = BaseModelIDEC(self.autoencoder.Encoder, self.autoencoder.Decoder, self.n_clusters)
        dummy = tf.zeros(shape=[1, self.input_dim[0]], dtype=tf.dtypes.float32, name=None)
        y = self.model(dummy)
        print(self.model.summary())


        # prepare IDEC model
        self.model.compile(loss={'output_1': 'kld', 'output_2': 'mse'},
                           loss_weights=[gamma, 1],
                           optimizer=optimizer)



    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.model.Encoder.predict(x)


    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def clustering(self, x, y=None,
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   save_IDEC_dir='./results/idec'):

        print('Update interval', update_interval)
        save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
        print('Save interval', save_interval)

        # initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.model.Encoder.predict(x))
        y_pred_last = y_pred
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if y is not None:
                    acc_var = np.round(acc(y, y_pred), 5)
                    nmi_var = np.round(nmi(y, y_pred), 5)
                    ari_var = np.round(ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc_var, nmi_var, ari_var), ' ; loss=', loss)
                    wandb.log({"Iteration": ite, "Accuracy": acc_var, "NMI": nmi_var, "ARI": ari_var, "Loss 1": loss[0], "Loss 2": loss[1], "Loss 3": loss[2]})

                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            if (index + 1) * self.batch_size > x.shape[0] and index * self.batch_size < x.shape[0]:

                loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                 y=[p[index * self.batch_size::], x[index * self.batch_size::]])

                index = 0

            elif (index + 1) * self.batch_size > x.shape[0] and index * self.batch_size == x.shape[0]:
                index = 0


            else:
                loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                 y=[p[index * self.batch_size:(index + 1) * self.batch_size],
                                                    x[index * self.batch_size:(index + 1) * self.batch_size]])
                index += 1

            ite += 1

        save_IDEC_dir = check_filepath_naming(save_IDEC_dir)
        self.model.save_weights(save_IDEC_dir)
        return y_pred


class DEC(object):
    def __init__(self,
                 model,
                 input_dim,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):

        super(DEC, self).__init__()

        self.autoencoder = model
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.model = None

    def initialize_model(self, optimizer, ae_weights=None):
        if ae_weights is not None:  # load pretrained weights of autoencoder
            dummy = tf.zeros(shape=[512,self.input_dim[0]], dtype=tf.dtypes.float32, name=None)
            self.autoencoder(dummy)
            self.autoencoder.load_weights(ae_weights)
        else:
            print('ae_weights, i.e. path to weights of a pretrained model must be given')
            exit()


        # prepare DEC model
        self.model = BaseModelDEC(self.autoencoder.Encoder, self.n_clusters)
        self.model.compile(loss='kld', optimizer=optimizer)
        self.model.build((None, self.input_dim[0]))
        print(self.model.summary())


    def load_weights(self, weights_path):  # load weights of DEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.model.Encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def clustering(self, x, y=None,
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   save_DEC_dir='./results/dec'):

        print('Update interval', update_interval)

        # initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.model.Encoder.predict(x))
        y_pred_last = y_pred
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])


        loss = 0
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if y is not None:
                    acc_var = np.round(acc(y, y_pred), 5)
                    nmi_var = np.round(nmi(y, y_pred), 5)
                    ari_var = np.round(ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc_var, nmi_var, ari_var), ' ; loss=', loss)
                    wandb.log({"Iteration": ite, "Accuracy": acc_var, "NMI": nmi_var, "ARI": ari_var, "Loss": loss})

                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            if (index + 1) * self.batch_size > x.shape[0] and index * self.batch_size < x.shape[0]:

                loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                 y=p[index * self.batch_size::])
                index = 0

            elif (index + 1) * self.batch_size > x.shape[0] and index * self.batch_size == x.shape[0]:
                index = 0
                loss = 0

            else:
                loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                 y=p[index * self.batch_size:(index + 1) * self.batch_size])
                index += 1

            ite += 1

        save_DEC_dir = check_filepath_naming(save_DEC_dir)
        self.model.save_weights(save_DEC_dir)
        return y_pred


class PseudoLabel(object):
    def __init__(self,
                 model,
                 input_dim,
                 n_clusters,
                 sampling_method,
                 density_function,
                 k_nearest_neighbours,
                 batch_size,
                 epochs):

        super(PseudoLabel, self).__init__()

        self.autoencoder = model
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.sampling_method = sampling_method
        self.density_function = density_function
        self.k_nearest_neighbours = k_nearest_neighbours
        self.batch_size = batch_size
        self.epochs = epochs
        self.projection_head = None
        self.encoder = None
        self.pseudo = None

    def initialize_model(self, ae_weights=None):
        if ae_weights is not None:  # load pretrained weights of autoencoder
            dummy = tf.zeros(shape=[1,self.input_dim[0]], dtype=tf.dtypes.float32, name=None)
            self.autoencoder(dummy)
            self.autoencoder.load_weights(ae_weights)
        else:
            print('ae_weights, i.e. path to weights of a pretrained model must be given')
            exit()

        self.encoder = self.autoencoder.Encoder

    def initialize_model_NNCLR(self, ae_weights=None):
        if ae_weights is not None:  # load pretrained weights of autoencoder
            dummy = tf.zeros(shape=[1,self.input_dim[0]], dtype=tf.dtypes.float32, name=None)
            out_1, out_2 = self.autoencoder(dummy)
            self.projection_head = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(out_1.shape[1],)),
                    tf.keras.layers.Dense(out_1.shape[1], activation="relu"),
                    tf.keras.layers.Dense(out_1.shape[1]),
                ],
                name="projection_head",
            )
            self.encoder = self.autoencoder.Encoder
            self.pseudo = tf.keras.Sequential([self.encoder,
                                             self.projection_head])
                                             
            self.pseudo(dummy)

            self.pseudo.load_weights(ae_weights)


        else:
            print('ae_weights, i.e. path to weights of a pretrained model must be given')
            exit()

    def get_pseudo_labels(self, x, y, pseudo_label_ratio):

        data = self.autoencoder.Encoder.predict(x)
        rho_normed = calculate_densities(data=data, k=self.k_nearest_neighbours, density_function=self.density_function)
        OrdRho = np.argsort(-rho_normed)

        label_points = OrdRho[:int(data.shape[0] * pseudo_label_ratio)]
        unlabelled_points = OrdRho[int(data.shape[0] * pseudo_label_ratio):]

        y_label_points = y[label_points]
        x_label_points = x[label_points, :]

        y_unlabel_points = y[unlabelled_points]
        x_unlabel_points = x[unlabelled_points, :]

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred_labelled_points = kmeans.fit_predict(self.autoencoder.Encoder.predict(x_label_points))
        print("Accuracy on high density points:", acc(y_label_points, y_pred_labelled_points))
        wandb.log({"Accuracy on high density points": acc(y_label_points, y_pred_labelled_points)})


        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(data)
        print("vs. Accuracy on all points:", acc(y, y_pred))
        wandb.log({"Accuracy on all points": acc(y, y_pred)})

        return x_label_points, y_pred_labelled_points, x_unlabel_points, y_unlabel_points

    def get_pseudo_labels_NNCLR(self, x, y, pseudo_label_ratio):
        data = self.pseudo.predict(x)

        import time
        start_time = time.time()
        rho_normed = calculate_densities(data=data, k=int(self.k_nearest_neighbours*x.shape[0]), density_function=self.density_function)
        end_time = time.time()
        print("Time Density  Calculation: ", end_time - start_time)
        
        print("x.shape:", x.shape)
        print("K:", int(self.k_nearest_neighbours*x.shape[0]))
        
        
        if self.n_clusters is None:
            ks = []
            elbow_scores = []
            label_ratio = 0.2
            OrdRho = np.argsort(-rho_normed)
            label_points = OrdRho[:int(data.shape[0] * label_ratio)]
            y_train_label_points = y[label_points]
            x_train_label_points = data[label_points, :]
            for i in range(2, 15):
                n_clusters = i
                kmeans = KMeans(n_clusters=n_clusters, n_init=20)
                y_pred_labelled_points = kmeans.fit_predict(x_train_label_points)
                elbow_score = kmeans.inertia_
                ks.append(n_clusters)
                elbow_scores.append(elbow_score)
                print("Accuracy on high density points for ", n_clusters, "cluster:",
                      acc(y_train_label_points, y_pred_labelled_points), elbow_score)
                wandb.log({"K": n_clusters,
                           "Accuracy on Ks": acc(y_train_label_points, y_pred_labelled_points)})

            from kneed import KneeLocator
            kn = KneeLocator(ks, elbow_scores, curve='convex', direction='decreasing')
            print("Predicted number of clusters elbow method:", kn.knee)
            wandb.log({"Final number of clusters": kn.knee})


        print("Ratio of Pseudo Labelled points:", pseudo_label_ratio)
        if self.sampling_method == 'weighted':
            selected_indices = sampling_weighted(pseudo_label_ratio, rho_normed)
            np.random.shuffle(selected_indices)

        elif self.sampling_method == 'densest':
            OrdRho = np.argsort(-rho_normed)
            selected_indices = OrdRho[:int(OrdRho.shape[0] * pseudo_label_ratio)]
            np.random.shuffle(selected_indices)


        y_label_points = y[selected_indices]
        x_label_points = x[selected_indices, :]
        y_unlabel_points = np.delete(y, selected_indices)
        x_unlabel_points = np.delete(x, selected_indices, axis=0)

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred_labelled_points = kmeans.fit_predict(self.pseudo.predict(x_label_points))
        print("Accuracy on high density points:", acc(y_label_points, y_pred_labelled_points))
        wandb.log({"Accuracy on high density points": acc(y_label_points, y_pred_labelled_points)})

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(data)
        print("vs. Accuracy on all points:", acc(y, y_pred))
        wandb.log({"Accuracy on all points": acc(y, y_pred)})
        
        wandb.log({"Label Ratio": pseudo_label_ratio})

        return x_label_points, y_pred_labelled_points, x_unlabel_points, y_unlabel_points

    def finetune_on_pseudos(self, save_Pseudo_dir, x, y, x_label_points, y_pred_labelled_points, x_unlabel_points, y_unlabel_points):

        dataset_pseudolabeled = tf.data.Dataset.from_tensor_slices((x_label_points, y_pred_labelled_points)).batch(
            self.batch_size, drop_remainder=True)
        dataset_unlabellabed = tf.data.Dataset.from_tensor_slices((x_unlabel_points, y_unlabel_points)).batch(
            self.batch_size, drop_remainder=True)

        finetuning_model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=self.input_dim[0]),
                self.autoencoder.Encoder,
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(self.n_clusters, activation='softmax'),],
            name="finetuning_model",
        )
        finetuning_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[acc_tf], #tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            run_eagerly=True)

        from wandb.keras import WandbMetricsLogger
        finetuning_history = finetuning_model.fit(
            dataset_pseudolabeled, epochs=self.epochs, validation_data=dataset_unlabellabed, verbose=1, callbacks=[WandbMetricsLogger()])

        pred = finetuning_model.predict(x)
        y_pred = pred.argmax(1)
        save_Pseudo_dir = check_filepath_naming(save_Pseudo_dir)
        finetuning_model.save_weights(save_Pseudo_dir)
        return y_pred

    def finetune_on_pseudos_NNCLR(self, save_Pseudo_dir, input_dim, x, y, x_label_points, y_pred_labelled_points, x_unlabel_points, y_unlabel_points, classification_augmenter):

        dataset_pseudolabeled = tf.data.Dataset.from_tensor_slices((x_label_points, y_pred_labelled_points)).batch(
            self.batch_size, drop_remainder=True)
        dataset_unlabellabed = tf.data.Dataset.from_tensor_slices((x_unlabel_points, y_unlabel_points)).batch(
            self.batch_size, drop_remainder=True)


        finetuning_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_dim),
                ConditionalAugmentation(spikeaugmentation(**classification_augmenter)),
                self.pseudo,
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(self.n_clusters, activation='softmax'),
            ],
            name="finetuning_model",
        )
        finetuning_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[acc_tf], #tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            run_eagerly=True)

        from wandb.keras import WandbMetricsLogger
        finetuning_model.fit(
            dataset_pseudolabeled, epochs=self.epochs, validation_data=dataset_unlabellabed, verbose=0, callbacks=[WandbMetricsLogger()])

        pred = finetuning_model.predict(x)
        y_pred = pred.argmax(1)
        final_acc = acc(y, y_pred)
        print("final_acc:", final_acc)
        wandb.log({"Final Acc": final_acc})
        
        return y_pred, finetuning_model


def finetune_model(model, finetune_config, finetune_method, dataset, dataset_test):
    # data prep --> combined train and test
    for step, batch in enumerate(dataset):
        if step == 0:
            x = batch[0]
            y = batch[1]
        else:
            x = np.concatenate((x, batch[0]), axis=0)
            y = np.concatenate((y, batch[1]), axis=0)

    for stept, batcht in enumerate(dataset_test):
        x = np.concatenate((x, batcht[0]), axis=0)
        y = np.concatenate((y, batcht[1]), axis=0)


    if "NNCLR" in finetune_config.PRETRAINED_SAVE_DIR:

        if finetune_method == "DEC":

            dec = DEC(model=model, input_dim=x[0, :].shape, n_clusters=finetune_config.DEC_N_CLUSTERS,
                      batch_size=finetune_config.DEC_BATCH_SIZE)
            dec.initialize_model(
                optimizer=tf.keras.optimizers.SGD(learning_rate=finetune_config.DEC_LEARNING_RATE,
                                                  momentum=finetune_config.DEC_MOMENTUM),
                ae_weights=finetune_config.PRETRAINED_SAVE_DIR)

            y_pred_finetuned = dec.clustering(x, y=y, tol=finetune_config.DEC_TOL,
                                              maxiter=finetune_config.DEC_MAXITER,
                                              update_interval=finetune_config.DEC_UPDATE_INTERVAL,
                                              save_DEC_dir=finetune_config.DEC_SAVE_DIR)


        elif finetune_method == "IDEC":

            idec = IDEC(model=model, input_dim=x[0, :].shape, n_clusters=finetune_config.IDEC_N_CLUSTERS,
                        batch_size=finetune_config.IDEC_BATCH_SIZE)
            idec.initialize_model(
                optimizer=tf.keras.optimizers.SGD(learning_rate=finetune_config.IDEC_LEARNING_RATE,
                                                  momentum=finetune_config.IDEC_MOMENTUM),
                ae_weights=finetune_config.PRETRAINED_SAVE_DIR, gamma=finetune_config.IDEC_GAMMA)

            y_pred_finetuned = idec.clustering(x, y=y, tol=finetune_config.IDEC_TOL,
                                               maxiter=finetune_config.IDEC_MAXITER,
                                               update_interval=finetune_config.IDEC_UPDATE_INTERVAL,
                                               save_IDEC_dir=finetune_config.IDEC_SAVE_DIR)


        elif finetune_method == "PseudoLabel":

            pseudo_label = PseudoLabel(model=model,
                                       input_dim=x[0, :].shape,
                                       n_clusters=finetune_config.PSEUDO_N_CLUSTERS,
                                       sampling_method=finetune_config.SAMPLING_METHOD,
                                       density_function=finetune_config.DENSITY_FUNCTION,
                                       k_nearest_neighbours=finetune_config.K_NEAREST_NEIGHBOURS,
                                       batch_size=finetune_config.PSEUDO_BATCH_SIZE,
                                       epochs=finetune_config.PSEUDO_EPOCHS)

            pseudo_label.initialize_model_NNCLR(ae_weights=finetune_config.PRETRAINED_SAVE_DIR)

            if finetune_config.ITERATIVE_RATIOS is None:

                x_label_points, y_pred_labelled_points, x_unlabel_points, y_unlabel_points = pseudo_label.get_pseudo_labels_NNCLR(
                x=x, y=y, pseudo_label_ratio=finetune_config.PSEUDO_LABEL_RATIO)

                y_pred_finetuned, finetuning_model = pseudo_label.finetune_on_pseudos_NNCLR(save_Pseudo_dir=finetune_config.PSEUDO_SAVE_DIR,
                                                                x=x,
                                                                y=y,
                                                                input_dim=x[0, :].shape,
                                                                classification_augmenter=finetune_config.CLASSIFICATION_AUGMENTER,
                                                                x_label_points=x_label_points,
                                                                y_pred_labelled_points=y_pred_labelled_points,
                                                                x_unlabel_points=x_unlabel_points,
                                                                y_unlabel_points=y_unlabel_points)
                
                finetune_config.PSEUDO_SAVE_DIR = check_filepath_naming(finetune_config.PSEUDO_SAVE_DIR)
                finetuning_model.save_weights(finetune_config.PSEUDO_SAVE_DIR)

            else:
                for ratio in finetune_config.ITERATIVE_RATIOS:
                    x_label_points, y_pred_labelled_points, x_unlabel_points, y_unlabel_points = pseudo_label.get_pseudo_labels_NNCLR(
                        x=x, y=y, pseudo_label_ratio=ratio)

                    y_pred_finetuned, finetuning_model = pseudo_label.finetune_on_pseudos_NNCLR(
                        save_Pseudo_dir=finetune_config.PSEUDO_SAVE_DIR,
                        x=x,
                        y=y,
                        input_dim=x[0, :].shape,
                        classification_augmenter=finetune_config.CLASSIFICATION_AUGMENTER,
                        x_label_points=x_label_points,
                        y_pred_labelled_points=y_pred_labelled_points,
                        x_unlabel_points=x_unlabel_points,
                        y_unlabel_points=y_unlabel_points)
                        
                
                finetune_config.PSEUDO_SAVE_DIR = check_filepath_naming(finetune_config.PSEUDO_SAVE_DIR)
                finetuning_model.pseudo.save_weights(finetune_config.PSEUDO_SAVE_DIR)
                # save labels
                #ys = np.concatenate((y.reshape(len(y), 1), y_pred_finetuned.reshape(len(y_pred_finetuned), 1)), axis=1)
                #np.savetxt(finetune_config.PSEUDO_SAVE_DIR[:-3] + "_labels.txt",ys)

        return y_pred_finetuned, y

    else:
        if finetune_method == "DEC":

            dec = DEC(model=model, input_dim=x[0,:].shape, n_clusters=finetune_config.DEC_N_CLUSTERS, batch_size=finetune_config.DEC_BATCH_SIZE)
            dec.initialize_model(
                optimizer=tf.keras.optimizers.SGD(learning_rate=finetune_config.DEC_LEARNING_RATE, momentum=finetune_config.DEC_MOMENTUM),
                ae_weights=finetune_config.PRETRAINED_SAVE_DIR)

            y_pred_finetuned = dec.clustering(x, y=y, tol=finetune_config.DEC_TOL,
                                              maxiter=finetune_config.DEC_MAXITER, update_interval=finetune_config.DEC_UPDATE_INTERVAL,
                                              save_DEC_dir=finetune_config.DEC_SAVE_DIR)


        elif finetune_method == "IDEC":

            idec = IDEC(model=model, input_dim=x[0,:].shape, n_clusters=finetune_config.IDEC_N_CLUSTERS, batch_size=finetune_config.IDEC_BATCH_SIZE) # TODO: All models
            idec.initialize_model(
                optimizer=tf.keras.optimizers.SGD(learning_rate=finetune_config.IDEC_LEARNING_RATE, momentum=finetune_config.IDEC_MOMENTUM),
                ae_weights=finetune_config.PRETRAINED_SAVE_DIR, gamma=finetune_config.IDEC_GAMMA)

            y_pred_finetuned = idec.clustering(x, y=y, tol=finetune_config.IDEC_TOL,
                                                maxiter=finetune_config.IDEC_MAXITER, update_interval=finetune_config.IDEC_UPDATE_INTERVAL,
                                                save_IDEC_dir=finetune_config.IDEC_SAVE_DIR)


        elif finetune_method == "PseudoLabel":

            pseudo_label = PseudoLabel(model=model,
                                       input_dim=x[0,:].shape,
                                       n_clusters=finetune_config.PSEUDO_N_CLUSTERS,
                                       k_nearest_neighbours=finetune_config.K_NEAREST_NEIGHBOURS,
                                       batch_size=finetune_config.PSEUDO_BATCH_SIZE,
                                       epochs=finetune_config.PSEUDO_EPOCHS)

            pseudo_label.initialize_model(ae_weights=finetune_config.PRETRAINED_SAVE_DIR)

            x_label_points, y_pred_labelled_points, x_unlabel_points, y_unlabel_points = pseudo_label.get_pseudo_labels(x=x, y=y,
                                                                                                                        pseudo_label_ratio=finetune_config.PSEUDO_LABEL_RATIO)

            y_pred_finetuned = pseudo_label.finetune_on_pseudos(save_Pseudo_dir=finetune_config.PSEUDO_SAVE_DIR,
                                                                x=x,
                                                                y=y,
                                                                x_label_points=x_label_points,
                                                                y_pred_labelled_points=y_pred_labelled_points,
                                                                x_unlabel_points=x_unlabel_points,
                                                                y_unlabel_points=y_unlabel_points)

        return y_pred_finetuned, y

