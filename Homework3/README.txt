1.	RNN is the simplest model, since it only contains one computation formula to calculating hidden layer output which will 	passed to next hidden layer. The training process work as follows: it go through each layer until getting the
	evaluation value at output layer, comparing it with sample result and then update weights thought the back-propagation through time. However it will failed facing the gradient varnishing problem. Our results has showed that RNN runs faster (only 7ms) than LSTM and GRU model but it has the worst accuraiueis  (less than 1% accuracy) of the training process.

	LSTM is variant of RNN to deal with gradient varnishing problem. The main feature of LSTM is its three unique gates: forget gate, input gate and output gate. Forget gate chooses information that will be removed from the cells state; input gate figures out which candidate hidden layer will be added to the cell state; and the output gate decides which information will be passed out. On basis of that, LSTM introduces a series of complicated formulas and mechanism, so that it run slower but will acquire higher accuracy.  Our results showed LSTM run about 18ms and have a higher accuracy around 40%~80%.

	GRU is simple version of LSTM, it only contains two gate: update gate and reset gate (actually its update gate is a combination of forget gate and input get from LSTM).  GRU can still deal with gradient varnishing problem and run faster than GRU as it only requires less parameters. The accuracy difference between GRU and LSTM should depend on the data set they are applied to, GRU is more effective when training data is large. Our result showed GRU run about 14ms and have a impressive accuracy about 70%~90%

2.	The attention mechanism enables copying from the input natural language query in addition to generating from the
 	vocabulary set. Our result showed that LSTM with copying attention runs much slower than LSTM without copying
 	attention, but it acquired a incredible accuracy around 80%~99%, which is much better than LSTM without copying attention.

3.	RNN model cannot answer complicated query, GRU and LSTM did better but still cannot generate complete correct query.
	my test case shows as following:
	natural language question: 
	how many rivers run through texas?

	correct Prolog Query:
	answer(A,count(B,(river(B),traverse(B,C),const(C,stateid(texas))),A)).
	result: Answer = [5]

	RNN generated Prolog Query
	answer(A,(population(B,A),const(B,stateid(texas)))).
	result: Answer = [14229000.0]

	LSTM generated Prolog Query
	answer(A,count(B,(river(B),loc(B,C),const(C,stateid(iowa))),A)).
	result: Answer = [2]

	GRU generated Prolog Query
	answer(A,count(B,(river(B),loc(B,C),const(C,countryid(usa))),A)).
	result: Answer = [46]

	we can se that RNN parsing is terrible since it cannot generate a reasonable query,
	while LSTM and GRU performed better

4.	The three ideas to improve performance:
	1. applying  more data for training process, since our model quality is highly constrained by those training data, therefore more data might help 
	2. changing the feature selections, removing some attributes with lower priority or relation to our parsing problem,  since differences features might cause better results
	3. change the initial weights value, namely change the initialization schema. The weights are the parameters we are trying to figures the best values, so that changing the initialization of weighs might improve our performance