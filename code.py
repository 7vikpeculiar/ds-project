from mpi4py import MPI
import numpy as np
import sys
from enum import Enum
from random import random
from collections import deque
import time

# https://www.programcreek.com/python/example/89111/mpi4py.MPI.ANY_SOURCE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


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
        self.num_basic_msgs = 1
        self.stack = deque()
        self.neighbours = None
        self.reqs = None
        self.warned = False
        self.has_terminated = False
        self.sent_terminated = False

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
        # print(rank, "Recieved message", msg_obj)
        if msg_obj["msg_type"] == "warn":
            # Color the channel
            print(rank, "GOT WARNING FROM", msg_obj["src"])
            # print(rank, self.channels)sent_terminated
            self.channels[msg_obj["src"]].color_channel()
            # print(self.channels)
            if self.warned == False:
                self.state = "DT"
                self.warn()
                self.warned = True

        if msg_obj["msg_type"] == "basic_comp":
            print("GOT BASIC MSG FROM {}".format(msg_obj["src"]))
            if self.channels[msg_obj["src"]].is_coloured == True:
                self.stack.append(["FROM", msg_obj["src"]])
            print(rank,self.stack)

        if msg_obj["msg_type"] == "remove_entry":
            for element in reversed(self.stack):
                print("GOT REMOVE ENTRY FROM ", msg_obj["src"])
                if element[0] == "TO" and element[1] == msg_obj["src"]:
                    self.stack.remove(element)
                    print("After removing",self.stack)
                    return

        if msg_obj["msg_type"] == "terminating":
            # print("GOT TERMINATE FROM ",msg_obj["src"])
            self.channels[msg_obj["src"]].has_terminated = True
        
        if msg_obj["msg_type"] == "terminate":
            self.has_terminated = True

    def fill_children(self, neighbours):
        self.neighbours = neighbours
        self.children = list(set(neighbours).difference({self.parent}))
        self.channels = {int(proc): Channel(int(proc))
                         for proc in self.neighbours}

    def send_basic_computation_msg(self):
        if self.num_basic_msgs > 0:
            # Always true for now
            # print(rank,"INREACHED")
            if len(self.neighbours) > 1:
                # print("REACHED")
                msg = {"msg_type": "basic_comp", "src": rank}
                dest = self.neighbours[-1]
                comm.send(msg, dest=dest)
                print(rank, " Sent message to ", dest)
                self.stack.append(["TO", dest])
            self.num_basic_msgs -= 1

    def stack_clean_up(self):
        for element in reversed(self.stack):
            if element[0] == "FROM":
                self.stack.remove(element)
                msg = {"msg_type": "remove_entry", "src": rank}
                comm.send(msg, dest=element[1])
                break

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
        print(rank, "TERMINATE")
        self.sent_terminated = True
        msg = {"msg_type": "terminating","src":rank}
        comm.send(msg,dest=self.parent)
    
    def send_overall_termination(self):
        if rank == 0:
            for i in range(1,self.num_nodes):
                msg = {"msg_type":"terminate","src":rank}
                comm.send(msg,dest=i)

processState = stateObject()
################################################################

if rank == 0:
    num_nodes = int(input("Enter the number of nodes"))
    processState.num_nodes = num_nodes
    Edges = num_nodes - 1
    parent_arr = [-1] * num_nodes
    # print("Enter the edges:")
    adj_list = []
    for i in range(0, num_nodes):
        adj_list.append([])

    for i in range(0, Edges):
        u, v = input().split(' ')
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

    processState.children = [int(ele) for ele in np.array(adj_list[0], dtype='d')]
    processState.neighbours = processState.children

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
    # print(rank,": Channels:",[ele.them for ele in processState.channels])
    # print(rank, "Children :", processState.children)
    # print(rank, processState.neighbours)
    # processState.print_state()
# Done sending tree


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

            print(rank,"HERE")
            if not processState.stack and processState.num_basic_msgs <= 0:
                    # Stack cleaned up 
                    if processState.check_channels() and processState.children_have_terminated():
                        # processState.send_termination_msg()
                        processState.has_terminated = True
                        processState.send_overall_termination()
                        print(0,"DONE",processState.has_terminated)
        
        time.sleep(1)
        
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

        # Stack is empty
            # if not processState.stack and processState.num_basic_msgs <= 0:
            #     print("STACK IS EMPTY")
            #     if TrueTrueTrueTrueTrue
        print(rank, processState.stack)
        time.sleep(1)