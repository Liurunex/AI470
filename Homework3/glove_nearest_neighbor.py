import sys
from math import sqrt
from Queue import PriorityQueue as PQ


#IN_FILE = 'p2_test.txt'
IN_FILE = '/c/cs570/data/word_embedding/glove/glove.6B.200d.txt'

def cosine_distance(target, comp):
	sumOfxy = 0
	sumofx2 = 0
	sumofy2 = 0
	for i in range(len(target)):
		x = target[i]
		y = comp[i]
		sumOfxy += x*y
		sumofx2 += x*x
		sumofy2 += y*y
	return sumOfxy/sqrt(sumofx2*sumofy2)

def main_process(target):
	target_vector = []
	in_data = open(IN_FILE).readlines()
	# find the target and initial its vector
	for word in in_data:
		data_vector = word.split(" ")
		word_str    = data_vector[0]
		data_vector = data_vector[1:]
		if target == word_str:
			target_vector = data_vector
			break
	if len(target_vector) == 0:
		print "target not int dictionary"
		return
	else:
		target_vector = [float(num) for num in target_vector]

	pqForten = PQ()
	for compare in in_data:
		comp_vector = compare.split(" ")
		com_str     = comp_vector[0]
		if com_str == target:
			continue
		comp_vector = comp_vector[1:]
		comp_vector = map(float, comp_vector)
		diff        = cosine_distance(target_vector, comp_vector)
		if pqForten.qsize() == 10:
			curmin_p, pqword = pqForten.get()
			if curmin_p >= diff:
				pqForten.put((curmin_p, pqword))
				continue
		pqForten.put((diff, com_str))

	printlist = []
	while not pqForten.empty():
		thediff, theword = pqForten.get()
		printlist.append(theword)
	for item in reversed(printlist):
		print item
	
	return


if __name__ == '__main__':
	main_process(sys.argv[1])
