from mpi4py import MPI
import numpy as np
import sys
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if rank == 0:
	num_nodes = int(input("Enter the number of nodes"))
	Edges =  num_nodes - 1

	print("Enter the edges:")
	adj_list = []
	for i in range(0,num_nodes):
		adj_list.append([])

	for i in range(0,Edges):
		u,v = input().split(' ')
		u, v = int(u), int(v)
		adj_list[u].append(v)
		adj_list[v].append(u)

	print(adj_list)
	for i in range(1,num_nodes):
		np_list = np.array(adj_list[i], dtype='d')
		# print(type(np_list), np_list, type(np_list[0]))
		num_adj = np.size(np_list)
		comm.send(num_adj, dest=i)
		comm.Send(np_list, dest=i) 
	neighbours = np.array(adj_list[0], dtype='d')
	numData = np.size(neighbours)
else:

	numData = comm.recv(source=0)
	neighbours = np.empty(numData, dtype = 'd')  
	comm.Recv(neighbours, source=0)

print("I am node %d and I have %d neighbours and my neighbours are:  "%(rank,numData), neighbours)
