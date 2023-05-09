import numpy as np
import sqlite3
class getMatrix():
    # m using matrix factorization for 20 million ratings dataset
    def initialize(self):
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

        print(matrix[0][:100])
        return (matrix)

    def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
        '''
        R: rating matrix
        P: |U| * K (User features matrix)
        Q: |D| * K (Item features matrix)
        K: latent features
        steps: iterations
        alpha: learning rate
        beta: regularization parameter'''
        Q = Q.T

        for step in range(steps):
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        # calculate error
                        eij = R[i][j] - np.dot(P[i,:],Q[:,j])

                        for k in range(K):
                            # calculate gradient with a and beta parameter
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

            eR = np.dot(P,Q)

            e = 0

            for i in range(len(R)):

                for j in range(len(R[i])):

                    if R[i][j] > 0:

                        e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)

                        for k in range(K):

                            e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
            # 0.001: local minimum
            if e < 0.001:

                break

            if step%100 == 0:
                print (e)

        return P, Q.T, e

    def getRating(R, k):
        # R = [

        #     [5,3,0,1],

        #     [4,0,0,1],

        #     [1,1,0,5],

        #     [1,0,0,4],

        #     [0,1,5,4],
            
        #     [2,1,3,0],

        #     ]

        R = np.array(R)
        # N: num of User
        N = len(R)
        # M: num of Movie
        M = len(R[0])
        # Num of Features
        K = k

        
        P = np.random.rand(N,K)
        Q = np.random.rand(M,K)

        

        nP, nQ, e = matrix_factorization(R, P, Q, K)

        nR = np.dot(nP, nQ.T)
        print(nR[0], e)

    # getRating(matrix[:100,:100], 3)
