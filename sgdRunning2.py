
import numpy as np
import sqlite3
from sgdTutorial import ExplicitMF

def initialize():
	conn = sqlite3.connect( 'train_100k.db' )
	# collect an array of all users

	# list of items
	c = conn.cursor()
	c.execute( 'SELECT ItemID FROM example_table' )
	duplicate_items = c.fetchall()
	items = list(set(duplicate_items))
	items.sort()
	itemsList = [item[0] for item in items]



	c = conn.cursor()
	c.execute( 'SELECT UserID FROM example_table' )
	duplicate_items = c.fetchall()
	users = list(set(duplicate_items))
	users.sort()
	usersList = [user[0] for user in users]

	matrix = np.full((len(usersList),len(itemsList) ), 0.)

	print(matrix.shape)

	c = conn.cursor()
	c.execute( 'SELECT UserID, ItemId, Rating FROM example_table' )
	tupes = c.fetchall()
	for tup in tupes:
		matrix[usersList.index(tup[0])] [itemsList.index(tup[1])] = tup[2]

	return (matrix, usersList, itemsList)
ratings_matrix, usersList, itemsList = initialize()

sparsity = float(len(ratings_matrix.nonzero()[0]))
sparsity /= (ratings_matrix.shape[0] * ratings_matrix.shape[1])
sparsity *= 100
print ('Sparsity: {:4.2f}%'.format(sparsity))

def train_test_split(ratings):
	test = np.zeros(ratings.shape)
	train = ratings.copy()
	for user in range(ratings.shape[0]):
		test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
										size=10, 
										replace=False)
		train[user, test_ratings] = 0.
		test[user, test_ratings] = ratings[user, test_ratings]
		
	# Test and training are truly disjoint
	assert(np.all((train * test) == 0)) 
	return train, test

train, test = train_test_split(ratings_matrix)

def try_model(iter):
	MF_SGD = ExplicitMF(ratings_matrix, 100, learning='sgd', verbose=True)
	iter_array = [iter]#1, 2, 5, 10, 25, 50, 100, 200 
	predictions = MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.001)

	return predictions

def grid_search():
	iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
	learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
	latent_factors = [5, 10, 20, 40, 80, 100]

	best_params = {}
	best_params['learning_rate'] = None
	best_params['n_iter'] = 0
	best_params['latent_factors'] = None
	best_params['train_mse'] = np.inf
	best_params['test_mse'] = np.inf
	best_params['model'] = None


	for rate in learning_rates:
		for factor in latent_factors:
			print ('Rate: {}'.format(rate))
			MF_SGD = ExplicitMF(train, n_factors=factor, learning='sgd')
			MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=rate)
			min_idx = np.argmin(MF_SGD.test_mse)
			if MF_SGD.test_mse[min_idx] < best_params['test_mse']:
				best_params['n_iter'] = iter_array[min_idx]
				best_params['learning_rate'] = rate
				best_params['latent_factors'] = factor
				best_params['train_mse'] = MF_SGD.train_mse[min_idx]
				best_params['test_mse'] = MF_SGD.test_mse[min_idx]
				best_params['model'] = MF_SGD
				print ('New optimal hyperparameters')
				print (best_params)

matrix = try_model(200)
# grid_search()

userAvgMap = {}
def userMeanRating():
	conn2 = sqlite3.connect( 'train_100k.db')
	c = conn2.cursor()
	for user in usersList:
		# Gets the average ratings for all the items a user has rated
		c.execute('SELECT Avg(Rating) FROM example_table WHERE UserID = ?', (user,))
		# Stores the average rating for the user in the hashmap with the user ID as the key
		userAvgMap[user] = c.fetchone()[0]
# userMeanRating()

def compute_predictions():
	conn = sqlite3.connect('HANDIN_100k.db')
	cursor = conn.cursor()
	# for each user-item pair, predict the rating and update the database
	cursor.execute( 'SELECT UserID, ItemID FROM test_table' )
	user_items = cursor.fetchall()
	counter = 0

	for i in user_items:
		try:
			mat = matrix[usersList.index(i[0])][itemsList.index(i[1])]
		except:
			mat = userAvgMap[i[0]]#)"{:.0f}".format(
		cursor.execute( 'UPDATE test_table SET PredRating = ? WHERE UserID = ? AND ItemID = ?', (mat, i[0], i[1]) )
		conn.commit()
		counter +=1
		if counter % 100 == 0:
			print(counter)
		
# compute_predictions()


