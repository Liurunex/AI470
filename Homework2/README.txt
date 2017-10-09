/* Minimax Algorithm */

# Depth_level									depth = 3	depth = 4	depth = 5			
# Average number of nodes generated per turn	915			3608		77927
 
# Average number of duplicated nodes per turn	101			675			27442

# Average branching factor of a node 			11			6			12

# Average runtime per turn for 3+ depths		1042.50ms	3629.92ms	90936.78ms

==================================================================================
/* Alpha-Beta Algorithm */

# Depth_level									depth = 3	depth = 4	depth = 5			
# Average number of nodes generated per turn	234			581			3338
 
# Average number of duplicated nodes per turn	12			53			479

# Average branching factor of a node 			1			2			1

# Average runtime per turn for 3+ depths		257.35ms	538.45ms	3379.18ms


Discussion:
the experiment recorded four statistics for both vanilla minimax and alpha-beta, including average number of nodes generated per turn, averages number of duplicated nodes per turn, average branching factor per node and average runtime per turn. The results indicated alpha-beats pruning algorithm significantly improve the performance of the minimax.