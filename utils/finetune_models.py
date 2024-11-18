from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
import hnswlib
from kneed import KneeLocator

from utils.pretrain_models import *


nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.

    The function computes the accuracy of the clustering by finding an optimal match
    between the cluster labels and the true labels using the Hungarian algorithm.

    Arguments:
        y_true: True labels, numpy.array with shape `(n_samples,)`.
        y_pred: Predicted labels, numpy.array with shape `(n_samples,)`.

    Returns:
        accuracy: A float value between 0 and 1 indicating the clustering accuracy.
    """
    # y_pred = y_pred.argmax(1)
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
    Calculate clustering accuracy for TensorFlow tensors.

    Similar to `acc` but specifically designed for use with TensorFlow tensors.

    Arguments:
        y_true: True labels, numpy.array with shape `(n_samples,)`.
        y_pred: Predicted labels, numpy.array with shape `(n_samples,)`.

    Returns:
        accuracy: A float value between 0 and 1 indicating the clustering accuracy.
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

class ConditionalAugmentation(tf.keras.layers.Layer):
    def __init__(self, augmenter, **kwargs):
        super(ConditionalAugmentation, self).__init__(**kwargs)
        self.augmenter = augmenter

    def call(self, inputs, training=None):
        if training:
          return self.augmenter(inputs)
        return inputs

def check_filepath_naming(filepath):
    """
    Check and modify the file path to avoid overwriting existing files.

    If the specified file path exists, a number is appended to the file name to
    create a unique file name.

    Arguments:
        filepath: The file path to check.

    Returns:
        A modified file path if the original exists, otherwise the original file path.
    """
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
    """
    Flatten a nested list.

    Arguments:
        nested_list: A list possibly containing nested lists.

    Returns:
        A single flattened list.
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        elif isinstance(item, (int, float)):  # Check for integers and floats
            flattened.append(item)
    return flattened


def sampling_weighted(label_ratio, rho_normed):
    """
    Generates weighted sampling indices based on a given label ratio and a distribution of values.

    This function calculates the weighted indices for sampling from a distribution (rho_normed)
    to achieve a desired label ratio. It uses an optimization approach to find the best power
    coefficient (p) that, when applied to the distribution, achieves a label ratio closest to the desired one.

    Parameters:
    label_ratio (float): The desired ratio of labels in the final sample.
    rho_normed (numpy.ndarray): A normalized array representing the distribution of values to sample from.

    Returns:
    numpy.ndarray: An array of indices that have been selected based on the weighted sampling.
    """

    # Number of bins for histogram
    bins = 100

    # Calculate histogram of the rho_normed distribution
    y_dist, x_dist, _ = plt.hist(rho_normed, bins=bins)

    # Adjust x distribution for center of bins
    x_dist_ = x_dist + (1 / (2 * bins))
    x_dist_center = x_dist_[:-1]

    # Desired fraction from label ratio
    desired_fr = label_ratio

    def objective_function(p):
        """
        Objective function for optimization. It calculates the squared difference between the current
        fraction (based on the power p) and the desired fraction.

        Parameters:
        p (float): Power coefficient applied to x_dist_center.

        Returns:
        float: Squared difference between calculated fraction and desired fraction.
        """
        # Calculate the fraction based on power p
        fr = np.dot(y_dist, x_dist_center ** p) / np.sum(y_dist)

        # Difference from desired fraction
        diff = fr - desired_fr

        # Return squared difference
        return np.linalg.norm(diff) ** 2

    # Initial guess for power p based on label_ratio
    if label_ratio > 0.35:
        p_guess = 1
    elif 0.2 < label_ratio < 0.35:
        p_guess = 2
    elif 0.1 < label_ratio < 0.2:
        p_guess = 3
    else:
        p_guess = 4

    # Optimize to find the best power p
    result = minimize(objective_function, p_guess)
    optimal_p = result.x[0]

    # Ratios based on the optimal power p
    ratios = x_dist_center ** optimal_p

    # Generate conditions for each bin
    conditions = [(i / 100 <= rho_normed) & (rho_normed < (i + 1) / 100) for i in range(100)]

    selected_indices = []

    # Loop over each condition and bin ratio
    for condition, percentage in zip(conditions, ratios):
        # Indices that satisfy the current condition
        indices = np.where(condition)[0]

        # Number of samples to select from these indices
        num_samples = int(percentage * len(indices))

        # Randomly sample indices based on the calculated number
        sampled_indices = np.random.choice(indices, num_samples, replace=False)

        # Append sampled indices to the list
        selected_indices.extend(sampled_indices.tolist())

    # Flatten the list of selected indices
    selected_indices = np.array(selected_indices).flatten()

    # Shuffle the selected indices
    np.random.shuffle(selected_indices)

    return selected_indices


def calculate_densities(data, k, density_function):
    """
    Calculates density values for each data point in a given dataset.

    This function computes the density of each point in the dataset based on its k-nearest neighbors (k-NN).
    It supports two density functions: 'default' (inverse of the distance to the k-th nearest neighbor) and
    'mean' (inverse of the mean distance to k nearest neighbors). It also normalizes the density values.

    Parameters:
    data (numpy.ndarray): The dataset, an array of shape (n_samples, n_features).
    k (int): The number of nearest neighbors to consider for density calculation.
    density_function (str): The method to calculate density ('default' or 'mean').

    Returns:
    numpy.ndarray: An array of normalized density values for each point in the dataset.
    """

    # Number of points and dimensions in the dataset
    n, d = data.shape

    # Use KDTree for low-dimensional data
    if d <= 10:
        tree = KDTree(data)
        knn_dist, knn = tree.query(data, k=k)
    else:
        # Alternate approach for high-dimensional data (commented out due to high memory usage)
        # dist = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2))
        # knn = np.argsort(dist, axis=1)[:, 1:k+1]
        # knn_dist = dist[np.arange(n)[:, None], knn]

        # Slower but more memory efficient method for high-dimensional data
        dist = np.empty((n, n), dtype=np.float64)
        knn = np.empty((n, k), dtype=np.int64)
        fills = np.empty((n, d), dtype=np.float64)
        for i in range(n):
            # Compute squared Euclidean distances to all other points
            np.square(data[i] - data, out=fills)
            np.sum(fills, axis=1, out=dist[i])

            # Find indices of k+1 nearest neighbors (excluding the point itself)
            neighbors = np.argpartition(dist[i], k + 1)[:k + 1]
            knn[i] = neighbors[1:]  # Exclude the point itself

            # Set distance to self as infinity
            dist[i, i] = np.inf

        # Compute distances to k nearest neighbors
        knn_dist = np.sqrt(dist[np.arange(n)[:, None], knn])
        for i, neighbors in enumerate(knn):
            knn_dist[i] = dist[i, neighbors]

    # Calculate the k-NN density value
    if density_function == 'default':
        # Inverse of the distance to the k-th nearest neighbor
        rho = knn_dist[:, -1] ** -1
    elif density_function == 'mean':
        # Inverse of the mean distance to the k nearest neighbors
        rho = np.mean(knn_dist, axis=1) ** -1

    # Normalize the density values
    rho_normed = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))

    return rho_normed


class PseudoLabel(object):
    def __init__(self, model, input_dim, n_clusters, sampling_method, density_function, k_nearest_neighbours,
                 batch_size, epochs):
        """
        Initializes the PseudoLabel class for generating pseudo labels using various clustering and density-based
        methods.

        Parameters:
        model (tf.keras.Model): The neural network model to use.
        input_dim (tuple): Dimension of the input data.
        n_clusters (int): The number of clusters to use in KMeans clustering.
        sampling_method (str): The method to use for sampling ('weighted' or 'densest').
        density_function (str): The function to use for calculating density ('default' or 'mean').
        k_nearest_neighbours (float): The proportion of nearest neighbours to consider for density calculation.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        """

        super(PseudoLabel, self).__init__()

        # Initialization of attributes
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
        """
        Initializes the model with the given autoencoder weights.

        Parameters:
        ae_weights (str, optional): Path to the pre-trained weights of the autoencoder. If not provided, the function
         will exit.
        """
        if ae_weights is not None:
            dummy = tf.zeros(shape=[1, self.input_dim[0]], dtype=tf.dtypes.float32, name=None)
            self.autoencoder(dummy)
            self.autoencoder.load_weights(ae_weights)
        else:
            print('ae_weights, i.e. path to weights of a pretrained model must be given')
            exit()

        self.encoder = self.autoencoder.Encoder

    def initialize_model_NNCLR(self, ae_weights=None):
        """
        Initializes the NNCLR model with a projection head on top of the encoder.

        This method is used for scenarios where a non-linear projection head is required on top of the encoder.
        It initializes the NNCLR (Neural Network Contrastive Learning Representation) model, which is particularly
        useful in semi-supervised learning for tasks like clustering and pseudo-labeling. The method sets up the
        encoder, projection head, and the complete NNCLR model. If pre-trained weights are provided, it loads them
        into the model.

        Parameters:
        ae_weights (str, optional): The path to the pre-trained weights for the autoencoder. If not provided, the
                                    method will display a message and exit.

        The method first tests the autoencoder with a dummy input to initialize the layers. Then it sets up a
        projection head, a sequential model consisting of dense layers. The encoder part of the autoencoder is
        extracted and combined with the projection head to form the complete NNCLR model. If pre-trained weights
        are available, they are loaded with allowances for any mismatch in layer names and structures.
        """

        if ae_weights is not None:  # load pretrained weights of autoencoder
            # Creating a dummy input to initialize the model
            dummy = tf.zeros(shape=[1, self.input_dim[0]], dtype=tf.dtypes.float32, name=None)

            # Running the dummy input through the autoencoder to get output shapes
            out_1, out_2 = self.autoencoder(dummy)

            # Setting up the projection head with dense layers
            self.projection_head = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(out_1.shape[1],)),
                    tf.keras.layers.Dense(out_1.shape[1], activation="relu"),
                    tf.keras.layers.Dense(out_1.shape[1]),
                ],
                name="projection_head",
            )

            # Extracting the encoder part from the autoencoder
            self.encoder = self.autoencoder.Encoder

            # Creating the NNCLR model by combining the encoder and projection head
            self.pseudo = tf.keras.Sequential([self.encoder, self.projection_head])
            self.pseudo(dummy)  # Initializing the NNCLR model with dummy input

            # Loading pre-trained weights into the NNCLR model
            self.pseudo.load_weights(ae_weights, skip_mismatch=True, by_name=True)
        else:
            print('ae_weights, i.e. path to weights of a pretrained model must be given')
            exit()

    def get_pseudo_labels(self, x, y, pseudo_label_ratio):
        """
        Generates pseudo labels for a dataset using density-based sampling and KMeans clustering.

        This method is part of the semi-supervised learning approach where pseudo labels are generated for unlabeled
        data. It first computes the density of each data point using the encoded representation of the data. Then, it
        sorts the data pointsbased on their density and selects a subset based on the pseudo_label_ratio. This subset
        is assumed to have the most reliable pseudo labels. KMeans clustering is applied to generate these pseudo
        labels. The method also evaluates the accuracy of pseudo labels on high-density points compared to the actual
        labels and logs this information using wandb.

        Parameters:
        x (numpy.ndarray): The input features of the data.
        y (numpy.ndarray): The actual labels of the data.
        pseudo_label_ratio (float): The ratio of data points to be considered for generating pseudo labels based on
        density.

        Returns:
        tuple: A tuple containing:
               - x_label_points (numpy.ndarray): The features of data points selected for pseudo labeling.
               - y_pred_labelled_points (numpy.ndarray): The pseudo labels generated for the selected data points.
               - x_unlabel_points (numpy.ndarray): The features of the remaining data points.
               - y_unlabel_points (numpy.ndarray): The actual labels of the remaining data points.

        The method first encodes the data using the autoencoder's encoder, then calculates the densities and sorts the
        data points based on these densities. A subset of data points, determined by the pseudo_label_ratio, is
        selected for pseudo labeling. KMeans clustering is used to generatepseudo labels for these selected points. The
        method also calculates and logs the accuracy of the pseudo labels on high-density points for evaluation.
        """

        # Encode the data and calculate densities
        data = self.autoencoder.Encoder.predict(x)
        rho_normed = calculate_densities(data=data, k=self.k_nearest_neighbours, density_function=self.density_function)

        # Sort data points based on density
        OrdRho = np.argsort(-rho_normed)

        # Select a subset of data points for pseudo labeling
        label_points = OrdRho[:int(data.shape[0] * pseudo_label_ratio)]
        unlabelled_points = OrdRho[int(data.shape[0] * pseudo_label_ratio):]

        # Separate the selected points and their labels
        y_label_points = y[label_points]
        x_label_points = x[label_points, :]

        # Remaining data points and their labels
        y_unlabel_points = y[unlabelled_points]
        x_unlabel_points = x[unlabelled_points, :]

        # Apply KMeans clustering to generate pseudo labels
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred_labelled_points = kmeans.fit_predict(self.autoencoder.Encoder.predict(x_label_points))

        # Logging and printing the accuracy on high-density points
        print("Accuracy on high density points:", acc(y_label_points, y_pred_labelled_points))
        wandb.log({"Accuracy on high density points": acc(y_label_points, y_pred_labelled_points)})

        # Apply KMeans on all data and log the accuracy
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(data)
        print("vs. Accuracy on all points:", acc(y, y_pred))
        wandb.log({"Accuracy on all points": acc(y, y_pred)})

        return x_label_points, y_pred_labelled_points, x_unlabel_points, y_unlabel_points

    def get_pseudo_labels_NNCLR(self, x, y, pseudo_label_ratio):
        """
        Generates pseudo labels for a dataset using the NNCLR model, density-based sampling, and KMeans clustering.

        This method applies the NNCLR (Neural Network Contrastive Learning Representation) model to the dataset to
        generate embeddings. It then uses density-based sampling to select a subset of the data for pseudo labeling. The
        density is calculated using the HNSWliblibrary for efficient nearest neighbor search. Depending on the specified
        sampling method, it selects the appropriate data points and applies KMeans clustering to generate pseudo labels.
        The method also evaluates and logs the accuracy of pseudo labels on both high-density points and the entire
        dataset.

        Parameters:
        x (numpy.ndarray): The input features of the data.
        y (numpy.ndarray): The actual labels of the data.
        pseudo_label_ratio (float): The ratio of data points to be considered for generating pseudo labels based on
         density.

        Returns:
        tuple: A tuple containing:
               - x_label_points (numpy.ndarray): The features of data points selected for pseudo labeling.
               - y_pred_labelled_points (numpy.ndarray): The pseudo labels generated for the selected data points.
               - x_unlabel_points (numpy.ndarray): The features of the remaining data points.
               - y_unlabel_points (numpy.ndarray): The actual labels of the remaining data points.

        The method starts by predicting the embeddings of the data using the NNCLR model. It then performs an efficient
        nearest neighbor search to calculate the densities of the data points. Depending on the 'n_clusters' attribute,
        either the elbow method is used to determine the number of clusters, or the predefined number is used. The
        method selects data points for pseudo labeling based on the calculated densities and the pseudo_label_ratio. It
        applies KMeans clustering to generate pseudo labels and logs various performance metrics using wandb.
        """

        # Predicting data using the NNCLR model
        data = self.pseudo.predict(x)

        k = int(self.k_nearest_neighbours*x.shape[0])

        # Starting the density calculation
        start_time = time.time()

        num_elements, dim = data.shape

        p = hnswlib.Index(space='l2', dim=dim)

        # Initializing the index - the maximum number of elements should be known beforehand
        p.init_index(max_elements=num_elements, ef_construction=int(k*1.5), M=32)

        # Element insertion (can be called several times):
        p.add_items(data)

        # Controlling the recall by setting ef:
        p.set_ef(int(k*1.5))  # Setting ef to be higher than k to ensure good recall

        # Query the index:
        labels, knn_dist = p.knn_query(data, k=k)
        print("knn_dist.shape:", knn_dist.shape)
        rho = np.mean(knn_dist, axis=1) ** -1
        rho_normed = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))

        end_time = time.time()
        print("Time Density Calculation hnswlib: ", end_time - start_time)

        # Selecting data points for pseudo labeling
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

            kn = KneeLocator(ks, elbow_scores, curve='convex', direction='decreasing')
            print("Predicted number of clusters elbow method:", kn.knee)
            self.n_clusters = kn.knee
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

        # Applying KMeans clustering to generate pseudo labels
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred_labelled_points = kmeans.fit_predict(self.pseudo.predict(x_label_points))

        # Logging the accuracy on high-density points
        print("Accuracy on high density points:", acc(y_label_points, y_pred_labelled_points))
        wandb.log({"Accuracy on high density points": acc(y_label_points, y_pred_labelled_points)})

        # Apply KMeans on all data and log the overall accuracy
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(data)
        print("vs. Accuracy on all points:", acc(y, y_pred))
        wandb.log({"Accuracy on all points": acc(y, y_pred)})

        wandb.log({"Label Ratio": pseudo_label_ratio})

        return x_label_points, y_pred_labelled_points, x_unlabel_points, y_unlabel_points

    def finetune_on_pseudos(self, save_Pseudo_dir, x, y, x_label_points, y_pred_labelled_points, x_unlabel_points,
                            y_unlabel_points):
        """
        Fine-tunes the model on pseudo-labeled data and evaluates its performance on both pseudo-labeled and unlabeled
        data.

        This method is crucial in a semi-supervised learning setup, where the model is initially trained on a small set
        of labeled data and then fine-tuned on a larger set of pseudo-labeled data. It creates two datasets - one with
        pseudo-labeled data and another with the remaining unlabeled data. The model is then fine-tuned on this combined
        dataset. After fine-tuning, the method evaluates the model's performance and saves the fine-tuned model's
        weights.

        Parameters:
        save_Pseudo_dir (str): Directory to save the fine-tuned model weights.
        x (numpy.ndarray): The complete set of input features.
        y (numpy.ndarray): The complete set of actual labels.
        x_label_points (numpy.ndarray): Input features of the pseudo-labeled data.
        y_pred_labelled_points (numpy.ndarray): Pseudo labels for the pseudo-labeled data.
        x_unlabel_points (numpy.ndarray): Input features of the remaining unlabeled data.
        y_unlabel_points (numpy.ndarray): Actual labels of the remaining unlabeled data.

        The method sets up a fine-tuning model using the autoencoder's encoder followed by a dropout layer and a dense
        layer for classification. It compiles the model with the Adam optimizer and Sparse Categorical Crossentropy
        loss, and trains it on the pseudo-labeled data, using the unlabeled data for validation. The training process is
        logged with wandb. After training, the method predicts labels for the complete dataset and saves the model's
        weights.
        """

        # Create datasets for pseudo-labeled and unlabeled data
        dataset_pseudolabeled = tf.data.Dataset.from_tensor_slices((x_label_points, y_pred_labelled_points)).batch(
            self.batch_size, drop_remainder=True)
        dataset_unlabellabed = tf.data.Dataset.from_tensor_slices((x_unlabel_points, y_unlabel_points)).batch(
            self.batch_size, drop_remainder=True)

        # Constructing the fine-tuning model
        finetuning_model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=self.input_dim[0]),
             self.autoencoder.Encoder,
             tf.keras.layers.Dropout(0.1),
             tf.keras.layers.Dense(self.n_clusters, activation='softmax'),
             ],
            name="finetuning_model",
        )

        # Compiling the model
        finetuning_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[acc_tf],  # Custom accuracy metric function
            run_eagerly=True)

        # Training the model with logging
        from wandb.keras import WandbMetricsLogger
        finetuning_history = finetuning_model.fit(
            dataset_pseudolabeled, epochs=self.epochs, validation_data=dataset_unlabellabed, verbose=1,
            callbacks=[WandbMetricsLogger()])

        # Predicting and saving the fine-tuned model
        pred = finetuning_model.predict(x)
        y_pred = pred.argmax(1)
        save_Pseudo_dir = check_filepath_naming(save_Pseudo_dir)
        finetuning_model.save_weights(save_Pseudo_dir)

        return y_pred

    def finetune_on_pseudos_NNCLR(self, input_dim, x, y, x_label_points, y_pred_labelled_points,
                                  x_unlabel_points, y_unlabel_points, classification_augmenter):
        """
        Fine-tunes the NNCLR model on pseudo-labeled data using conditional augmentation and evaluates its performance.

        This method is an extension of the semi-supervised learning approach where the NNCLR model is fine-tuned on a
        dataset comprising both pseudo-labeled and unlabeled data. It incorporates an additional conditional
        augmentation step in the training process. The method creates two datasets: one with pseudo-labeled data and
        another with the remaining unlabeled data. The model, consisting of the NNCLR model and additional layers, is
        then fine-tuned on this combined dataset. After fine-tuning, the method evaluates the model's performance on the
        entire dataset and logs the final accuracy.

        Parameters:
        input_dim (tuple): Dimension of the input data.
        x (numpy.ndarray): The complete set of input features.
        y (numpy.ndarray): The complete set of actual labels.
        x_label_points (numpy.ndarray): Input features of the pseudo-labeled data.
        y_pred_labelled_points (numpy.ndarray): Pseudo labels for the pseudo-labeled data.
        x_unlabel_points (numpy.ndarray): Input features of the remaining unlabeled data.
        y_unlabel_points (numpy.ndarray): Actual labels of the remaining unlabeled data.
        classification_augmenter (dict): Parameters for the conditional augmentation process.

        The method sets up a fine-tuning model using the NNCLR model with a conditional augmentation layer, followed by
        dropout and dense layers for classification. The model is compiled with the Adam optimizer and Sparse
        Categorical Crossentropy loss, and trained on the pseudo-labeled data, using the unlabeled data for validation.
        The training process is logged with wandb. After training, the method predicts labels for the complete dataset,
        calculates the final accuracy, logs this information, and saves the model's weights.
        """

        # Create datasets for pseudo-labeled and unlabeled data
        dataset_pseudolabeled = tf.data.Dataset.from_tensor_slices((x_label_points, y_pred_labelled_points)).batch(
            self.batch_size, drop_remainder=True)
        dataset_unlabellabed = tf.data.Dataset.from_tensor_slices((x_unlabel_points, y_unlabel_points)).batch(
            self.batch_size, drop_remainder=True)

        # Constructing the fine-tuning model with conditional augmentation
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

        # Compiling the model
        finetuning_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[acc_tf],  # Custom accuracy metric function
            run_eagerly=True)

        # Training the model with logging
        from wandb.keras import WandbMetricsLogger
        finetuning_model.fit(
            dataset_pseudolabeled, epochs=self.epochs, validation_data=dataset_unlabellabed, verbose=0,
            callbacks=[WandbMetricsLogger()])

        # Predicting, evaluating, and saving the fine-tuned model
        pred = finetuning_model.predict(x)
        y_pred = pred.argmax(1)
        final_acc = acc(y, y_pred)
        print("final_acc:", final_acc)
        wandb.log({"Final Acc": final_acc})

        return y_pred, finetuning_model


def finetune_model(model, finetune_config, finetune_method, dataset, dataset_test, benchmark=False):
    """
    Fine-tunes a given model using a specified method and configuration on a combined dataset.

    This function handles the fine-tuning of a model, specifically for cases involving NNCLR (Neural Network Contrastive
    Learning Representation) and PseudoLabel methods. It combines the provided training and test datasets, prepares the
    data, and applies the fine-tuning process using either NNCLR or a standard pseudo-labeling approach, depending on
    the model's configuration. It supports both singular and iterative ratio-based pseudo labeling.

    Parameters:
    model (tf.keras.Model): The neural network model to be fine-tuned.
    finetune_config (Config_Finetuning): Configuration object containing parameters for fine-tuning.
    finetune_method (str): The method used for fine-tuning ('PseudoLabel', etc.). Here only PseudoLabel implemented
    dataset (tf.data.Dataset): The training dataset.
    dataset_test (tf.data.Dataset): The test dataset.

    The function first merges the training and test datasets. It then checks the model configuration to determine the
    fine-tuning method. For NNCLR-based models, it initializes a PseudoLabel object, performs pseudo-labeling, and
    fine-tunes the model iteratively or singularly based on the configuration. For other models, it follows a similar
    process but without the NNCLR-specific steps. The function saves the fine-tuned model's weights and optionally the
    generated labels.

    Returns:
    tuple: A tuple containing:
           - y_pred_finetuned (numpy.ndarray): The predicted labels by the fine-tuned model.
           - y (numpy.ndarray): The actual labels from the combined dataset.
    """

    y = None
    for step, batch in enumerate(dataset):
        if step == 0:
            x = batch[0]
            if benchmark:
                y = batch[1]
        else:
            x = np.concatenate((x, batch[0]), axis=0)
            if benchmark:
                y = np.concatenate((y, batch[1]), axis=0)

    for stept, batcht in enumerate(dataset_test):
        x = np.concatenate((x, batcht[0]), axis=0)
        if benchmark:
            y = np.concatenate((y, batcht[1]), axis=0)


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

        (x_label_points,
         y_pred_labelled_points,
         x_unlabel_points,
         y_unlabel_points) = pseudo_label.get_pseudo_labels_NNCLR(x=x,
                                                                  y=y,
                                                                  pseudo_label_ratio=finetune_config.
                                                                  PSEUDO_LABEL_RATIO)

        (y_pred_finetuned,
         finetuning_model) = (pseudo_label.
                              finetune_on_pseudos_NNCLR(x=x,
                                                        y=y,
                                                        input_dim=x[0, :].shape,
                                                        classification_augmenter=finetune_config.CLASSIFICATION_AUGMENTER,
                                                        x_label_points=x_label_points,
                                                        y_pred_labelled_points=y_pred_labelled_points,
                                                        x_unlabel_points=x_unlabel_points,
                                                        y_unlabel_points=y_unlabel_points))

        finetune_config.PSEUDO_SAVE_DIR = check_filepath_naming(finetune_config.PSEUDO_SAVE_DIR)
        finetuning_model.save_weights(finetune_config.PSEUDO_SAVE_DIR)

    else:
        for ratio in finetune_config.ITERATIVE_RATIOS:
            (x_label_points,
             y_pred_labelled_points,
             x_unlabel_points,
             y_unlabel_points) = pseudo_label.get_pseudo_labels_NNCLR(x=x, y=y, pseudo_label_ratio=ratio)

            y_pred_finetuned, finetuning_model = pseudo_label.finetune_on_pseudos_NNCLR(
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
        # save labels
        ys = np.concatenate((y.reshape(len(y), 1), y_pred_finetuned.reshape(len(y_pred_finetuned), 1)),
                            axis=1)
        np.savetxt(finetune_config.PSEUDO_SAVE_DIR[:-3] + "_labels_extra.txt", ys)

    return y_pred_finetuned, y
