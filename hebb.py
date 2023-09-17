import numpy as np
from matplotlib import pyplot as plt


class RosenblattPerceptron:
	# Initialisation des variables
	def __init__(self, learning_rate=0.01, epochs=1000):
		# Vitesse d'apprentissage, souvent notée en mathématique par alpha.
		self.learning_rate = learning_rate
		# Nombre de fois que l'algorithme passera en revue toutes les
		# Données pour s'entraîner
		self.epochs = epochs
		# Facteurs de pondération
		self.weights = None
		# Biais
		self.bias = None

	# Entraînement sur des données fournies
	def fit(self, X, y):
		# "Samples" représente le nombre de données
		# "Features" représente le nombre de variables
		num_samples, num_features = X.shape
		self.weights = np.zeros(num_features)
		self.bias = 0

		for epoch in range(self.epochs):
			for i in range(num_samples):
				# Dans nos notations mathématiques, linear_output est
				# représentée par z.
				linear_output = np.dot(X[i], self.weights) + self.bias
				# Dans nos notations mathématiques, predicted_y est
				# représentée par y.
				predicted_y = self.activation_function(linear_output)
				# Formule de Hebb modifiée
				self.weights += self.learning_rate * (y[i] - predicted_y) * X[i]
				self.bias += self.learning_rate * (y[i] - predicted_y)

	def predict(self, X):
		linear_output = np.dot(X, self.weights) + self.bias # Obtention de la variable z
		return self.activation_function(linear_output) # Obtention de la variable y

	def activation_function(self, x):
		# Retourne une matrice de la même dimension que x,
		# où chaque emplacement du tableau est remplacé
		# par 1 si la condition x >= 0 est respectée,
		# sinon 0. C'est le principe du "tout ou rien".
		return np.where(x >= 0, 1, 0)

# Exemple d'utilisation avec le problème booléen AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Entrées
y = np.array([0, 0, 0, 1])  # Sorties souhaitées

perceptron = RosenblattPerceptron(learning_rate=0.01, epochs=1000)
# Entraînement du modèle
perceptron.fit(X, y)

# Prédiction des données d'entraînement pour s'assurer de l'efficacité
# du modèle. Normalement ceci ne signifie rien puisque le modèle pourrait
# être tellement habitué à ses données d'entraînement que les nouvelles
# donnée lui parraîtraient étrangères, mais ici on couvre toutes les données
# de prédiction possibles.
# Affichage de la frontière de décision
x1 = np.linspace(-0.5, 2, 100)
x2 = (-perceptron.weights[0] * x1 - perceptron.bias) / perceptron.weights[1]
plt.plot(x1, x2, c='orange', lw=3)
plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="summer")
plt.show()