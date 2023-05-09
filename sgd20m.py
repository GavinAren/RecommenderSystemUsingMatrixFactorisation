import numpy as np
import csv
from sgdTutorial import ExplicitMF

def readFile(file):
	with open(file, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		return list(spamreader)

def initialize():

	userDict ={}
	itemDict = {}
	ratingsDict = {}
	rawFile = readFile('train_100k_withratings_new.csv')

	for row in rawFile:
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
	# userList = [k for k, v in sorted(userDict.items(), key=lambda x: x[1])]
	# itemList = [k for k, v in sorted(itemDict.items(), key=lambda x: x[1])]
	# itemDictSorted = itemDict.items().sort()


	sorted_User_dict = {key: idx for idx, key in enumerate(sorted(userDict))}
	sorted_Item_dict = {key: idx for idx, key in enumerate(sorted(itemDict))}

	print(len(sorted_Item_dict))
	# print(sorted_Item_dict)


	ratings_matrix = np.full((len(sorted_User_dict),len(sorted_Item_dict) ), 0., dtype=np.float16)
			
	for user in ratingsDict:
		for i in ratingsDict[user]:
			ratings_matrix[user-1] [sorted_Item_dict[i[0]]] = i[1]

	# print(matrix[0][:100])

	return (ratings_matrix, sorted_User_dict, sorted_Item_dict)
ratings_matrix, sorted_User_dict, sorted_Item_dict = initialize()

# sparsity = float(len(ratings_matrix.nonzero()[0]))
# sparsity /= (ratings_matrix.shape[0] * ratings_matrix.shape[1])
# sparsity *= 100
# print ('Sparsity: {:4.2f}%'.format(sparsity))
def sparsity(matrix):
    num_zeros = np.count_nonzero(matrix == 0)
    total_elements = matrix.size
    return num_zeros / total_elements

def train_test_split(ratings):
	test = np.zeros(ratings.shape, dtype=np.float16)
	train = ratings.copy()
	for user in range(ratings.shape[0]):
		test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
										size=10, 
										replace=False)
		train[user, test_ratings] = 0.
		test[user, test_ratings] = ratings[user, test_ratings]
		
	# Test and training are truly disjoint
	assert(np.all((train * test) == 0)) 

	print(sparsity(ratings)," ,", sparsity(train), " , ",sparsity(test))
	return train, test

train, test = train_test_split(ratings_matrix)

def try_model(iter):
	MF_SGD = ExplicitMF(train, 20, learning='sgd', verbose=True)
	iter_array = [iter]
	predictions = MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.001)

	return predictions
import time
start = time.time()
new_rating_matrix = try_model(5)
end = time.time()

print("Time taken = ", round(end-start,2),"seconds")


def extract_predictions():
	results = []
	outputFile = readFile('test_20m_withoutratings_new.csv')
	counter = 0

	for row in outputFile:
		userId = int(row[0])
		itemId = int(row[1]) 		
		try:
			rating_value = new_rating_matrix[sorted_User_dict[userId]][sorted_Item_dict[itemId]]
		except KeyError as e:
			rating_value = 3.0
			print('KeyError:', e)

		results.append([userId, itemId, rating_value, int(row[2])])
		counter +=1
		if counter % 100 == 0:
			print(counter)
		
	with open('results.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['UserID', 'ItemID', 'PredRating', 'Timestamp'])
		writer.writerows(results)


# extract_predictions()


