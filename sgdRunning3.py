import numpy as np
import csv
from sgdTutorial import ExplicitMF

def initialize():

	userDict ={}
	itemDict = {}
	ratingsDict = {}
	with open('train_100k_withratings_new.csv', newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

		for row in spamreader:
			userId = int(row[0])
			itemId = int(row[1])
			if userId not in userDict:
				userDict[userId] = userId
			if itemId not in itemDict:
				itemDict[itemId] = itemId
			if userId not in ratingsDict:
				ratingsDict[userId] = [(itemId, float(row[2]))]
			else:
				ratingsDict[userId] = ratingsDict[userId] + [(itemId, float(row[2]))]
		
		print(len(userDict), len(itemDict))

	#need to sort the items in the dictionaries
	userList = [k for k, v in sorted(userDict.items(), key=lambda x: x[1])]
	itemList = [k for k, v in sorted(itemDict.items(), key=lambda x: x[1])]
	# itemDictSorted = itemDict.items().sort()


	print(ratingsDict[1])

	matrix = np.full((len(userList),len(itemList) ), 0., dtype=np.float16)

	for user in userList:
		for i in ratingsDict[user]:
			# print (userList.index(user), itemList.index(i[0]), i[1])
			matrix[userList.index(user)] [itemList.index(i[0])] = i[1]
			

	print(matrix[0][:100])

	return matrix, userList, itemList

ratings_matrix, usersList, itemsList = initialize()

print(ratings_matrix[0][:100])

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

matrix = try_model(10)
# grid_search()

import sqlite3

def extract_predictions():
	results = []
	with open('test_100k_withoutratings_new.csv', newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		counter = 0

		for row in spamreader:
			userId = int(row[0])
			itemId = int(row[1]) 
			try:
				mat = matrix[usersList.index(userId)][itemsList.index(itemId)]
			except:
				mat = 3.0
				# print ('AAAAAAAAAAAAAAAAAAAAA')
				print (userId, itemId, mat)

			results.append([userId, itemId, mat, row[2]])
			counter +=1
			if counter % 100 == 0:
				print(counter)
		
	with open('results.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		# for row in results:
		# 	writer.writerow(row)
		writer.writerow(['UserID', 'ItemID', 'PredRating', 'Timestamp'])
		writer.writerows(results)


extract_predictions()


