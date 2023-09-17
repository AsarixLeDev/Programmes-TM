import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_blobs
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm


class NeuralNetwork:
	# Création des références des variables
	def __init__(self, learning_rate=0.1, epochs=50_000):
		# Vitesse d'apprentissage, souvent notée en mathématique par alpha.
		self.learning_rate = learning_rate
		# Nombre de fois que l'algorithme passera en revue toutes les
		# Données pour s'entraîner
		self.epochs = epochs
		# Paramètres : contient poids et biais
		self.parametres = {}

	# Initialisation des poids et des biais. Ici on utilise un dictionnaire,
	# c'est une solution plus visuelle.
	def initialisation(self, X, y, hidden_layers):
		# La liste devrait ressembler à [nx, nc1, ..., ncC, ny]
		# dans notre cas, ny vaut 1.
		dimensions = list(hidden_layers)
		dimensions.insert(0, X.shape[0])
		dimensions.append(y.shape[0])

		# Nombre de douches
		C = len(dimensions)

		# Détermination de la random seed
		np.random.seed(1)

		for c in range(1, C):
			self.parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
			self.parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

	# Processus de prédiction
	def forward_propagation(self, X):
		# A0 est X
		activations = {'A0': X}

		# Par couche il y a une matrice de poid et une matrice
		# de biais, il faut donc diviser le nombre de matrices
		# par 2 pour obtenir le nombre de couche
		C = len(self.parametres) // 2

		for c in range(1, C + 1):
			Z = self.parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + self.parametres['b' + str(c)]
			A = 1 / (1 + np.exp(-Z))
			# On stocke le résultat dans une matrice
			# pour l'utiliser plus tard dans la backpropagation
			activations['A' + str(c)] = A

		return activations

	# Correction des poids et des biais
	# grâce à la back propagation
	def back_propagation(self, y, activations):
		# Récupération du nombre de données
		# dans la dimensions de y : 1 x m
		m = y.shape[1]
		# Par couche il y a une matrice de poid et une matrice
		# de biais, il faut donc diviser le nombre de matrices
		# par 2 pour obtenir le nombre de couche
		C = len(self.parametres) // 2

		# Calcul de delta 1, l'erreur de la dernière couche
		dZ = activations['A' + str(C)] - y
		gradients = {}

		for c in reversed(range(1, C + 1)):
			# Récupération de Ac-1
			Acm1 = activations['A' + str(c - 1)]
			gradients['dW' + str(c)] = 1 / m * np.dot(dZ, Acm1.T)
			# On fait la somme de l'axe 1 pour obtenir la matrice b.
			# "keepdims" permet d'obtenir une matrice (n x 1) au lieu
			# d'un vecteur de dimension n
			gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
			# On de calcule pas delta 0 qui n'existe pas,
			# d'autant plus que W0 n'existe pas non plus
			if c > 1:
				Wc = self.parametres['W' + str(c)]
				dZ = np.dot(Wc.T, dZ) * Acm1 * (
							1 - Acm1)

		return gradients

	# Récupération d'une matrice qui contient True
	# ou False à l'index de l'élément donné en
	# fonction de la prédiction du modèle
	def predict(self, X):
		# Obtention de toutes les activations
		activations = self.forward_propagation(X)
		# Par couche il y a une matrice de poid et une matrice
		# de biais, il faut donc diviser le nombre de matrices
		# par 2 pour obtenir le nombre de couche
		C = len(self.parametres) // 2
		Af = activations['A' + str(C)]
		return Af >= 0.5

	# Entraînement du modèle
	def fit(self, X, y, hidden_layers=(16, 16, 16)):
		# initialisation des paramètres
		self.initialisation(X, y, hidden_layers)

		# tableau numpy contenant les futures accuracy et log_loss
		training_history = np.zeros((self.epochs, 2))

		# Par couche il y a une matrice de poid et une matrice
		# de biais, il faut donc diviser le nombre de matrices
		# par 2 pour obtenir le nombre de couche
		C = len(self.parametres) // 2

		# tqdm permet d'avoir une barre de progression
		# dans la console en fonction de l'avancement
		# de la boucle
		for i in tqdm(range(self.epochs)):
			# Récupération des activations
			# de chaque couche
			activations = self.forward_propagation(X)
			# Récupération des gradients
			gradients = self.back_propagation(y, activations)

			# Mise à jour des paramètres
			for c in range(1, C + 1):
				self.parametres['W' + str(c)] = self.parametres['W' + str(c)] \
												- self.learning_rate * gradients['dW' + str(c)]
				self.parametres['b' + str(c)] = self.parametres['b' + str(c)] \
												- self.learning_rate * gradients['db' + str(c)]

			# Récupération de la sortie du réseau de neurone
			Af = activations['A' + str(C)]

			# Calcul du log loss et de l'accuracy
			loss = log_loss(y.flatten(), Af.flatten())
			training_history[i, 0] = loss
			y_pred = self.predict(X)
			acc = accuracy_score(y.flatten(), y_pred.flatten())
			training_history[i, 1] = acc

		# Graphiques des calculs de la
		# performance du modèle
		plt.figure(figsize=(12, 4))
		plt.subplot(1, 2, 1)
		plt.plot(training_history[:, 0], label='train loss')
		plt.legend()
		plt.subplot(1, 2, 2)
		plt.plot(training_history[:, 1], label='train acc')
		plt.legend()
		plt.show()

		return training_history


# Retourne des ensembles de points différents
# pour visualiser la frontière de décision du modèle.
# Si t = 1 : Retourne des points séparés par une frontière
# circulaire
# Si t = 2 : Retourne des points séparés par deux frontières
# circulaires. les points d'un des types sont à l'intérieur
# des cercles, tandis que les autres sont à l'extérieur.
# Dans tous les autres cas : Retourne des points séparés par
# une frontière linéaire. C'est le même dataset qu'avec le
# perceptron.
def get_training_data(t: int):
	if t == 2:
		X_train0, y_train0 = make_circles(n_samples=50, noise=0.1, factor=0.4, random_state=0)
		X_train1, y_train1 = make_circles(n_samples=50, noise=0.1, factor=0.5, random_state=10_000)
		X_train1 += np.array([2, 2])
		X = np.concatenate((X_train0, X_train1))
		y = np.concatenate((y_train0, y_train1))
		y = y.reshape(y.shape[0], 1)
		return X, y
	if t == 1:
		X, y = make_circles(n_samples=100, noise=0.3, factor=0.6, random_state=0)
		y = y.reshape(y.shape[0], 1)
		return X, y
	else:
		X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
		y = y.reshape(y.shape[0], 1)
		return X, y


training_data_id = 0

# Récupération des données d'entraînement
X_train, y_train = get_training_data(training_data_id)

# Entraînement du modèle
neural_network = NeuralNetwork(epochs=100)
neural_network.fit(X_train.T, y_train.T, hidden_layers=[1000, 1000, 1000])

# Détermination de l'efficacité du modèle
predicted = neural_network.predict(X_train.T).T
print("Accuracy :", accuracy_score(predicted, y_train))

# --- Affichage de la frontière de décision ---

# Création d'une matrice qui contient toutes les coordonnées
# espacées régulièrement comprises dans un rectangle qui
# contient tous les points d'entraînement
x0_range = np.linspace(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 100)
x1_range = np.linspace(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5, 100)
xx0, xx1 = np.meshgrid(x0_range, x1_range)
X_grid = np.c_[xx0.ravel(), xx1.ravel()]

# Évaluer le modèle sur tous les points
predictions = neural_network.predict(X_grid.T).T

# Afficher la frontière de décision
plt.contourf(xx0, xx1, predictions.reshape(xx0.shape), alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
			label='Type 0', cmap='summer')
plt.show()