import sqlite3
import numpy as np
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

    return (matrix)
ratings = initialize()


# print (str(n_users) + ' users')
# print (str(n_items) + ' items')
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print ('Sparsity: {:4.2f}%'.format(sparsity))



def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        non_zero_indices = ratings[user, :].nonzero()[0]
        if non_zero_indices.size > 0:
            test_ratings = np.random.choice(non_zero_indices, 
                                             size=10, 
                                             replace=False)
            train[user, test_ratings] = 0.
            test[user, test_ratings] = ratings[user, test_ratings]

    assert(np.all((train * test) == 0)) 
    return train, test

train, test = train_test_split(ratings)



MF_SGD = ExplicitMF(train, 40, learning='sgd', verbose=True)
iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.00)