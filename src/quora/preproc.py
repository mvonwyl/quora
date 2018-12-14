import os
import numpy as np
import pandas as pd
import threading
import queue


def __ifprint__(message,doit):
	if(doit): print(message)

def split_quora_csv(filename,train_prop=0.9,output_train="quora_training.csv",output_test="quora_test.csv",verbatim=True):
	"""
	Split quora data in training and test set of given proportion and keep same proportion of troll data between them
	input: filename original quora training file
	train_prop: proportion of training / test data in the outputf file. Default=0.9
	output_train: name of the generated training file. Default="quora_training.csv"
	output_test: name of the generated test file. Default="quora_test.csv"
	verbatim: if True, display debugging messages such as data size. Default=True
	"""

	data = pd.read_csv(filename)

	# split genuine and troll data, to keep same proportion in training and test sets

	troll_mask = np.array(data["target"] == 1)
	__ifprint__("Amount of troll data = {}".format(np.sum(troll_mask)),verbatim)
	troll_data = data[["qid","question_text","target"]][troll_mask]
	genu_mask = np.logical_not(troll_mask)
	__ifprint__("Amount of genuine data = {}".format(np.sum(genu_mask)),verbatim)
	genu_data = data[["qid","question_text","target"]][genu_mask]
	assert data.shape[0] == troll_data.shape[0] + genu_data.shape[0],"Kind data + troll data != total number"

	# Let's shuffle the data
	troll_shuff = troll_data.sample(n=troll_data.shape[0],random_state=1).reset_index(drop=True)
	genu_shuff = genu_data.sample(n=genu_data.shape[0],random_state=1).reset_index(drop=True)

	assert troll_shuff.shape[0] + genu_shuff.shape[0] == data.shape[0]

	# troll data
	train_part = int(troll_shuff.shape[0]*train_prop)
	train_data = troll_shuff[:train_part][:]
	test_data = troll_shuff[train_part:][:]
	assert troll_shuff.shape[0] == train_data.shape[0] + test_data.shape[0]

	# add the genuine data (note the index reseting)
	train_part = int(genu_shuff.shape[0]*train_prop)
	train_data = train_data.append(genu_shuff[:train_part][:]).reset_index(drop=True)
	test_data = test_data.append(genu_shuff[train_part:][:]).reset_index(drop=True)

	# Check out resulting size
	__ifprint__("Train data shape = {}".format(train_data.shape),verbatim)
	__ifprint__("Test data shape = {}".format(test_data.shape),verbatim)
	assert train_data.shape[0] + test_data.shape[0] == data.shape[0]

	# shuffle again and save
	train_data = train_data.sample(n=train_data.shape[0],random_state=1).reset_index(drop=True)
	test_data = test_data.sample(n=test_data.shape[0],random_state=1).reset_index(drop=True)
	train_data.to_csv(output_train,index=False)
	test_data.to_csv(output_test,index=False)
	__ifprint__("Saved in {} and {}".format(output_train,output_test),verbatim)

def load_glove(filename,verbatim=True):
	"""
	load the glove projection space given in filename 
	"""
	space = {}
	i = 1
	with open(filename,"rt") as f:
		for line in f:
			word,*coeff = line.split()
			try:
				space[word] = np.array(coeff,dtype=float)
			except ValueError as err:
				__ifprint__("Error \"{}\" at line {}".format(err,i),verbatim)
			i+=1
	# reshaping because I don't like (size,) format
	vsize = next(iter(space.values())).shape[0]
	for word in space.keys():
		space[word] = space[word].reshape(vsize,1)
	return space


def project_sentence(sent,space,lower=True):
	"""
	Project sentence in the given space. lower indicate if the sentence should be lower cased.
	"""
	sentence = sent.lower() if lower else sent
	vsize = next(iter(space.values())).shape[0]
	proj = np.zeros((vsize,1))
	count = 0
	for word in sentence.split():
		if word in space:
			proj += space[word]
			count += 1
	if count != 0:
		proj /= count
	return proj

def project_data(data,space):
	vsize = next(iter(space.values())).shape[0]

	# Project training data
	projections = np.zeros((data.shape[0],vsize,1))
	for i in range(data.shape[0]):
		sentence = data.iloc[i]["question_text"]
		projections[i] = project_sentence(sentence,space)
	
	return projections

def save_project(projections,index,filename):
	assert projections.shape[0] == index.shape[0]
	with open(filename,"wt") as f:
		f.write("{} {}".format(projections.shape[0],projections.shape[1]))
		for i,p_r in zip(index,projections):
			f.write("{},".format(i))
			for p_c in p_r:
				f.write("{} ".format(float(p_c)))

			

	# Save in a complete dataframe
	# train_complete = data.copy(deep=True)
	# train_complete["projections"] = None
	# for i in range(projections.shape[0]):
	#	train_complete.at[i,"projections"] = projections[i]


# DOESNT WORK and python multithreading doesn't spread on CPUs anyway
class _ImmaProjectingSentences(threading.Thread):
	"""
	Project its share of the data and put the projection in the queue
	"""
	def __init__(self,my_id,nb_threads,data,space,q,lock,column_name="question_text"):
		threading.Thread.__init__(self)
		self.my_id = my_id
		self.nb_threads = nb_threads
		self.data = data
		self.space = space
		self.q = q
		self.lock = lock
		self.column_name = column_name

	def run(self):
		my_id = self.my_id
		nb_threads = self.nb_threads
		data = self.data
		space = self.space
		q = self.q
		lock = self.lock
		column_name = self.column_name
		vsize = (next(iter(space.values())).shape[0])

		begin = int(data.shape[0] / nb_threads) * my_id
		end = int(data.shape[0] / nb_threads) * (my_id+1) if my_id != nb_threads-1 else data.shape[0]
		projections = np.zeros((end-begin,vsize,1))
		lock.acquire()
		print("Thread {}: starts at {}, ends at {}".format(my_id,begin,end))
		lock.release()
		j = 0
		for i in range(begin,end):
			sentence = data.iloc[i][column_name]
			projections[j] = project_sentence(sentence,space)
			j+=1
		lock.acquire()
		q.put(projections)
		lock.release()

# DOESNT WORK and python multithreading doesn't spread on CPUs anyway
def _project_data_thread(data,space,nb_threads=4):
	vsize = next(iter(space.values())).shape[0]

	# Project training data
	lock = threading.Lock()
	q = queue.Queue()
	threads = []
	for i in range(nb_threads):
		threads.append(_ImmaProjectingSentences(i,nb_threads,data,space,q,lock))
	for i in range(nb_threads):
		threads[i].start()
	for i in range(nb_threads):
		threads[i].join()

	projections = np.array([])
	while not q.empty():
		projections = np.append(projections,q.get())
	
	return projections
	
