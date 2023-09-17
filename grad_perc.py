import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

class GradientPerceptron:
	# Initialisation des variables
	def __init__(self, learning_rate=0.01, epochs=100000):
		# Vitesse d'apprentissage
		self.learning_rate = learning_rate
		# Nombre d'epochs
		self.epochs = epochs
		# Facteurs de pondération
		self.weights = None
		# Biais
		self.bias = None

	# Processus de prédiction
	def model(self, X):
		Z = X.dot(self.weights) + self.bias
		y = 1 / (1 + np.exp(-Z))
		return y

	# Entraînement du modèle
	def fit(self, X, y_ref):
		# Initialisation des poids et du biais
		self.weights = np.random.randn(X.shape[1], 1)
		self.bias = np.random.randn(1)
		# Suivi de la fonction coût
		losses = []

		for i in range(self.epochs):
			y = self.model(X)
			# Calcul de la fonction coût
			loss = - 1 / len(y_ref) * np.sum(y_ref * np.log(y) + (1 - y_ref) * np.log(1 - y))
			losses.append(loss)
			# Calcul des gradients
			dLdW = 1 / len(y_ref) * np.dot(X.T, (y - y_ref))
			dLdb = 1 / len(y_ref) * np.sum(y - y_ref)
			# Mise à jour des poids et du biais
			self.weights = self.weights - self.learning_rate * dLdW
			self.bias = self.bias - self.learning_rate * dLdb

		# Affichage de l'évolution de la fonction coût
		plt.plot(losses, label='Log loss')
		plt.xlabel('Epoques')
		plt.ylabel('Valeur de la fonction coût')
		plt.show()

		# Calcul de la performance du modèle
		y_accuracy = self.predict(X)
		print("Accuracy :", accuracy_score(y_ref, y_accuracy))

	def predict(self, X):
		return self.model(X) >= 0.5

# Exemple d'utilisation
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Entrées
y_ref = np.array([0, 0, 0, 1])  # Sorties souhaitées
y_ref = y_ref.reshape((y_ref.shape[0]), 1)

perceptron = GradientPerceptron(learning_rate=0.1, epochs=10000)
perceptron.fit(X, y_ref)

# Affichage de la frontière de décision
x1 = np.linspace(-0.5, 2, 100)
x2 = (-perceptron.weights[0] * x1 - perceptron.bias) / perceptron.weights[1]
plt.plot(x1, x2, c='orange', lw=3)
plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_ref, cmap="summer")
plt.show()