from mpi4py import MPI
import numpy as np
import sys
from enum import Enum
from random import choice
from collections import deque
import time
from pprint import pprint

# https://www.programcreek.com/python/example/89111/mpi4py.MPI.ANY_SOURCE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) #

def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) # 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) #
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk)) #
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk)) #
def prCyan(ska,skk): print("\033[96m Rank : {} -> {}\033[00m" .format(ska,skk)) # 
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) #
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk)) 
def prBlue(skk): print("\033[94m {}\033[00m" .format(skk)) 

class Channel:
    def __init__(self, them):
        self.us = rank
        self.them = them
        self.is_coloured = False
        self.has_terminated = False


    def color_channel(self):
        self.is_coloured = True

    def __str__(self):
        return "Rank : {}-> Them -> {}: Color -> {} ".format(rank, self.them, self.is_coloured)

    def __repr__(self):
        return str(self)


class stateObject:
    def __init__(self):
        self.state = "NDT"
        self.channels = []
        self.children = []
        self.parent = None
        self.status = MPI.Status()
        self.num_basic_msgs = 2
        self.stack = deque()
        self.neighbours = None
        self.reqs = None
        self.warned = False
        self.has_terminated = False
        self.sent_terminated = False
        self.deterministic = True

    def fill_channels(self, data):
        self.channels = data

    def warn(self):
        for ele in self.neighbours:
            msg = {"msg_type": "warn", "src": rank}
            # reqs =
            comm.send(msg, dest=self.channels[ele].them)
            # reqs.wait()

    # send messgae to children
    def print_state(self):
        print("{} : {};{};{};{};{} {}:: STACK[{}]".format(rank, self.neighbours, self.state,
                                                        self.channels, self.children, self.parent, self.num_basic_msgs, str(self.stack)))

    def processMsg(self, msg_obj):
        if msg_obj["msg_type"] == "warn":
            # Color the channel
            prPurple(str(rank) + " GOT WARNING FROM " +str(msg_obj["src"]))
            self.channels[msg_obj["src"]].color_channel()
            # print(self.channels)
            if self.warned == False:
                self.state = "DT"
                self.warn()
                self.warned = True

        if msg_obj["msg_type"] == "basic_comp":
            prRed("{} GOT BASIC MSG FROM {}".format(rank,msg_obj["src"]))
            if self.channels[msg_obj["src"]].is_coloured == True:
                self.stack.append(["FROM", msg_obj["src"]])
            prCyan(rank,self.stack)

        if msg_obj["msg_type"] == "remove_entry":
            for element in reversed(self.stack):
                prGreen(str(rank)+" GOT REMOVE STACK ENTRY FROM "+ str(msg_obj["src"]))
                if element[0] == "TO" and element[1] == msg_obj["src"]:
                    self.stack.remove(element)
                    prCyan(rank,self.stack)
                    return

        if msg_obj["msg_type"] == "terminating":
            # print("GOT TERMINATE FROM ",msg_obj["src"])
            self.channels[msg_obj["src"]].has_terminated = True
        
        if msg_obj["msg_type"] == "terminate":
            self.has_terminated = True

    def fill_children(self, neighbours):
        self.neighbours = neighbours
        print(" Process {} received adjacency list {}".format(rank,neighbours))
        self.children = list(set(neighbours).difference({self.parent}))
        self.channels = {int(proc): Channel(int(proc))
                         for proc in self.neighbours}

    def send_basic_computation_msg(self):
        if self.deterministic:
            self.deterministic_basic_computation()
        else:
            self.non_deterministic_basic_computation()

    def deterministic_basic_computation(self):   
        if self.num_basic_msgs > 0:
            if len(self.neighbours) > 1:
                # print("REACHED")
                msg = {"msg_type": "basic_comp", "src": rank}
                dest = max(self.neighbours)
                comm.send(msg, dest=dest)
                prRed(str(rank) + " SENT BASIC MESSAGE TO " + str(dest))
                self.stack.append(["TO", dest])
                self.num_basic_msgs -= 1
                prCyan(rank,self.stack)
            else:
                self.num_basic_msgs -= 1
    
    def non_deterministic_basic_computation(self):   
        if self.num_basic_msgs > 0:
                # print("REACHED")
            msg = {"msg_type": "basic_comp", "src": rank}
            dest = choice(self.neighbours)
            comm.send(msg, dest=dest)
            prRed(str(rank) + " SENT BASIC MESSAGE TO " + str(dest))
            self.stack.append(["TO", dest])
            self.num_basic_msgs -= 1
            prCyan(rank,self.stack)

    def stack_clean_up(self):
        if not self.stack :
            # Return if empty stack
            return
        top_of_stack = self.stack[-1]
        if top_of_stack[0] == "FROM":
            self.stack.remove(top_of_stack)
            msg = {"msg_type": "remove_entry", "src": rank}
            comm.send(msg, dest=top_of_stack[1])
            prGreen(str(rank)+ " SENT A CLEANUP MESSAGE TO "+str(top_of_stack[1]))
        return
        
    def check_channels(self):
        answer = True
        for key in self.channels:
            answer = answer and self.channels[key].is_coloured
        return answer

    def children_have_terminated(self):
        if len(self.children) == 0:
            return True
        else:
            answer = True
            for ele in self.children:
                answer = answer and self.channels[ele].has_terminated 
            return answer

    def send_termination_msg(self):
        prLightPurple(str(rank)+ " SENDING TERMINATION  MESSAGE TO PARENT "+str(self.parent))
        self.sent_terminated = True
        msg = {"msg_type": "terminating","src":rank}
        comm.send(msg,dest=self.parent)

    def send_overall_termination(self):
        if rank == 0:
            for i in range(1,self.num_nodes):
                msg = {"msg_type":"terminate","src":rank}
                comm.send(msg,dest=i)

################################################################
processState = stateObject()
################################################################

if rank == 0:
    answer = input("Determinstic (y) or Non Determinstic simulation ? (n) :")
    if answer == "y":
        answer = True
    else:
        answer = False

###########
    if answer:
        print("Determinstic Simulation starting  ############### \n")
    else:
        print("Non Determinstic Simulation starting ############### \n")

    processState.deterministic = answer
#################################################################

if rank == 0:
    with open('inp','r') as f:
        num_nodes = int(f.readline().strip())
        processState.num_nodes = num_nodes
        Edges = num_nodes - 1
        parent_arr = [-1] * num_nodes
        # print("Enter the edges:")
        adj_list = []
        for i in range(0, num_nodes):
            adj_list.append([])

        for i in range(0, Edges):
            u, v = f.readline().strip().split(' ')
            u, v = int(u), int(v)
            parent_arr[v] = u
            adj_list[u].append(v)
            adj_list[v].append(u)
    # print(adj_list)

    for i in range(1, num_nodes):
        np_list = np.array(adj_list[i], dtype='d')
        # print(type(np_list), np_list, type(np_list[0]))
        num_adj = np.size(np_list)
        comm.send(num_adj, dest=i)
        comm.send(parent_arr[i], dest=i)
        comm.Send(np_list, dest=i)
        comm.send(processState.deterministic,dest=i)

    processState.children = [int(ele) for ele in np.array(adj_list[0], dtype='d')]
    processState.neighbours = processState.children
    print(" Process {} knows its adjacency list {}".format(rank,processState.neighbours))
    processState.channels = {int(proc): Channel(
        int(proc)) for proc in processState.children}

    # print(rank,": Channels:",[ele.them for ele in processState.channels])
    # print(processState.children)
    # print(rank, "Children :", processState.children)
    # print(rank, processState.neighbours)
    # processState.print_state()
else:

    numData = comm.recv(source=0)
    neighbours = np.empty(numData, dtype='d')
    processState.parent = int(comm.recv(source=0))
    comm.Recv(neighbours, source=0)
    processState.fill_children([int(ele) for ele in neighbours])
    processState.deterministic = comm.recv(source=0)
    # print(rank,": Channels:",[ele.them for ele in processState.channels])
    # print(rank, "Children :", processState.children)
    # print(rank, processState.neighbours)
    # processState.print_state()

# Done sending tree

time.sleep(2)

if rank == 0:
    processState.state = "DT"
    processState.warn()
    processState.warned = True
    # time.sleep(40)
    while not processState.has_terminated:
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=processState.status):
            node = processState.status.Get_source()
            data = comm.recv(source=node)
            processState.status = MPI.Status()
            processState.processMsg(data)
        else:
            # processState.clean_stack()
            if processState.check_channels() and processState.state == "DT" and processState.num_basic_msgs > 0:
                processState.send_basic_computation_msg()
                continue
                # processState.print_state()

        
            if processState.num_basic_msgs <= 0 and processState.stack:
                processState.stack_clean_up()            
                continue

            if not processState.stack and processState.num_basic_msgs <= 0:
                    # Stack cleaned up 
                    if processState.check_channels() and processState.children_have_terminated():
                        processState.has_terminated = True
                        processState.send_overall_termination()
                        # print(0,"DONE",processState.has_terminated)
                        
        time.sleep(1)
    time.sleep(4)
    prLightGray("Process 0 (Initiator) Terminating")
        
else:
    # print(rank,": Channels:",[ele.them for ele in processState.channels])
    # if processState.parent == 0:
    while not processState.has_terminated:
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=processState.status):
            node = processState.status.Get_source()
            data = comm.recv(source=node)
            processState.status = MPI.Status()
            processState.processMsg(data)

        else:
            # processState.clean_stack()
            if processState.check_channels() and processState.state == "DT" and processState.num_basic_msgs > 0:
                processState.send_basic_computation_msg()
                continue

            if processState.num_basic_msgs <= 0 and processState.stack:
                # non empty stack
                processState.stack_clean_up()
                continue

            if not processState.stack and processState.num_basic_msgs <= 0:
                # Stack cleaned up 
                if processState.check_channels() and processState.children_have_terminated() and not processState.sent_terminated:
                    processState.send_termination_msg()
        
        time.sleep(1)

    prLightGray("Process {} Terminating".format(rank))