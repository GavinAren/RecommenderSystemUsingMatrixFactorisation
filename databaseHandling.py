import codecs, sqlite3, random
def createDBs():
	
	# connect to database (using sqlite3 lib built into python)
	conn = sqlite3.connect('HANDIN_100k.db')
	readHandle = codecs.open('test_100k_withoutratings_new.csv', 'r', 'utf-8', errors='replace')
	listLines = readHandle.readlines()
	readHandle.close()

	# creating table if it does not exist
	c = conn.cursor()
	c.execute('CREATE TABLE IF NOT EXISTS test_table (UserID INT, ItemID INT,  PredRating FLOAT, Timestamp FLOAT)')
	conn.commit()

	# clears the table of previous values
	c.execute('DELETE FROM test_table')
	conn.commit()

	# inserting data into the table from the listLines variable
	for strLine in listLines:
		if len(strLine.strip()) > 0:
			# userid, itemid, rating, timestamp
			listParts = strLine.strip().split(',')
			if len(listParts) == 3:
				# insert training set into table with a predicted rating column having its values initialised to 0.0 
				c.execute('INSERT INTO test_table VALUES (?,?,?,?)', (listParts[0], listParts[1], 0.0, listParts[2]))
			else:
				raise Exception('failed to parse csv : ' + repr(listParts))
	conn.commit()

	c.execute('CREATE INDEX IF NOT EXISTS test_table_index on test_table (UserID, ItemID)')
	conn.commit()

	# close the connection
	conn.close()
	
# createDBs()
import csv
def transfer_test_set_to_db():
		
	conn = sqlite3.connect('HANDIN_100k.db')
	cursor = conn.cursor()

	# Execute a SELECT statement to get all the data from the test_table of the test set
	cursor.execute('SELECT * FROM test_table')

	# Fetch all the results and store them in a list
	results = cursor.fetchall()

	# Geting the column name from the description
	headers = [i[0] for i in cursor.description]

	# Write the data to a CSV file, opening the file in write mode
	with open('results.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(headers)
		writer.writerows(results)

	# Close the database connection
	conn.close()
	
transfer_test_set_to_db()