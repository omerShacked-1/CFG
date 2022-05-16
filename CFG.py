"""
In this exercise, we will convert a simple assembly language into a CFG (control flow graph).
CFGs (https://en.wikipedia.org/wiki/Control-flow_graph) are directed graphs (https://en.wikipedia.org/wiki/Directed_graph), 
where the nodes are usually instructions or basic blocks (https://en.wikipedia.org/wiki/Basic_block) and the edges represent 
possible flows between the nodes.
Usually, there are two kinds of edges - plain ones (meaning unconditional flow from one node to another) or conditional 
flows (meaning the flow will be decided from two options depending on some condition).

CFGs are useful for many kinds of analyses (liveness/dead code, stack recovery, pointer analysis,
folding, slicing, etc.)

Enough talk, let's get down to business!

Here's the definition for our assembly language:
"""
import enum
from dataclasses import dataclass
from typing import Optional, Union, List
from networkx.classes.multidigraph import MultiDiGraph


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class Operation(enum.Enum):
    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()


@dataclass
class Var:
    name: str

    def __str__(self):
        return self.name


Value = Union[Var, int]  # Variable (e.g. "foo", "x") or int for a const


@dataclass
class Expression:
    lhs: Value
    rhs: Value
    operation: Operation

    def __str__(self):
        return str(self.lhs) + " " + opstr(self.operation) + " " + str(self.rhs);


@dataclass
class Assignment:
    dst: Var
    src: Union[Value, Expression]

    def __str__(self):
        return str(self.dst) + " = " + str(self.src)


@dataclass
class Jump:
    target: int
    condition: Optional[Var]  # If a condition is provided, the branch is taken if the var is not zero


@dataclass
class Call:
    function_name: str
    args: List[Value]

    def __str__(self):
        sarr = [str(a) for a in self.args]
        return self.function_name + "(" + ", " . join(sarr) +")"


"""
Now let's write some code - e.g. exponentiation
"""

base = Var("base")
counter = Var("counter")
output = Var("output")

code = [
    Assignment(base, 2),  # 0
    Assignment(counter, 5),  # 1
    Assignment(output, 1),  # 2
    # Loop start
    Assignment(output, Expression(output, base, Operation.MUL)),  # 3
    Assignment(counter, Expression(counter, 1, Operation.SUB)),  # 4
    Jump(3, counter),  # 5
    Call("print", [output])  # 6
]


"""
Now, we want to build a control flow graph from this function.

Our nodes will be instructions, and our edges will contain conditions. For example, the code:

    a = 1
    jump(AFTER, a)
    b = 2
    jump(END)
    AFTER:
    b = 3
    END:
    print(b)
    
Would be represented as:

  a = 1
 ||    \\
 || !a  \\  a
 ||      \\
 b = 2    b = 3
 ||       //
 ||      //
 print(b)

You can use any library for basic graph functions - e.g. networkx. Don't forget to write tests!
"""


@dataclass
class Node():
    inedges : List[int]
    outedges : List[int]


#Constants
LABEL_STR='label'
CONNECTION_STYLE_DEF='arc3, rad = 0.1'
EMP_STR=""

#Method to beautify the str of an Operation Enum
def opstr(operation : Operation):
    match operation:
        case Operation.MUL:
            return "*"
        case Operation.SUB:
            return "-"
        case Operation.ADD:
            return "+"
        case Operation.DIV:
            return "/"
        case _:
            return ""
        

def build_G(instructions) -> MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node(0,label="start")
    #we assume "1" is the first node || "end" node
    G.add_edge(0,1)
    for index,instruction in enumerate(instructions):
        node_idx = index + 1
        G.add_node(node_idx,label=str(instruction))
        if isinstance(instruction,Jump):
            if instruction.condition is not None:
                G.add_edge(node_idx, instruction.target+1, label=str(instruction.condition))
                G.add_edge(node_idx, node_idx + 1,label="!"+str(instruction.condition))
            else:
                G.add_edge(node_idx, instruction.target + 1)
        else:
            G.add_edge(node_idx, node_idx + 1)
    return remove_jumps(G, instructions)

#Method to remove "Jump" nodes from G
#First we connect between the nodes points to "jump" node, and the nodes that "jump" node points to
#then we remove the current "jump" node
#Assume- instruction node index is the instruciton index in "instructions" array + 1
def remove_jumps(G, instructions) -> MultiDiGraph:
    for index,instruction in enumerate(instructions):
        if not isinstance(instruction,Jump): continue
        in_edges = G.in_edges(index + 1,data=True)
        out_edges = G.out_edges(index + 1,data=True)
        for i in list(in_edges):
            for j in list(out_edges):
                in_link_label,out_link_label = [x[2].get(LABEL_STR,EMP_STR) for x in [i,j]]
                and_sign = " and " if bool(in_link_label) and bool(out_link_label) else EMP_STR
                G.add_edge(i[0],j[1],label=in_link_label+ and_sign + out_link_label)
        G.remove_node(index + 1)
    return G

#Method to draw graph from G (argument) using matplotlib library
def draw_graph(G):
    pos = nx.circular_layout(G)
    nx.draw_circular(G,node_size=5000,connectionstyle=CONNECTION_STYLE_DEF)
    labels = {i : d.get(LABEL_STR,"end") for i,d in G.nodes(data=True)}
    nx.draw_networkx_labels(G,pos, labels=labels, font_size=10)
    edge_labels = dict([((n1, n2), n3.get(LABEL_STR,EMP_STR)) for n1, n2, n3 in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    plt.show()
    return

#Method to build a CFG from array of instructions
#Each insrtruction is of type Assignment | Jump | Call
def build_cfg_new_updated(instructions : List[Union[Assignment,Jump,Call]]):
    G = build_G(instructions)
    draw_graph(G)
    

def test1():
    code = [
    Assignment(base, 2),  # 0
    Assignment(counter, 5),  # 1
    Assignment(output, 1),  # 2
    # Loop start
    Assignment(output, Expression(output, base, Operation.MUL)),  # 3
    Assignment(counter, Expression(counter, 1, Operation.SUB)),  # 4
    Jump(3, counter),  # 5
    Call("print", [output])  # 6
    ]
    build_cfg_new_updated(code);
def test2():
    code = [
    Jump(4, counter),#0
    Assignment(base, 2),  # 1
    Assignment(counter, 5),  # 2
    Assignment(output, 1),  # 3
    # Loop start
    Assignment(output, Expression(output, base, Operation.MUL)),  # 4
    Assignment(counter, Expression(counter, 1, Operation.SUB)),  # 5
    Jump(3, counter),  # 6
    Call("print", [output])  # 7
    ]
    build_cfg_new_updated(code);
def test3():
    code = [
    Assignment(base, 2),  # 0
    Assignment(counter, 5),  # 1
    Jump(1, base), #2
    Assignment(output, 1),  # 3
    # Loop start
    Assignment(output, Expression(output, base, Operation.MUL)),  # 4
    Assignment(counter, Expression(counter, 1, Operation.SUB)),  # 5
    Jump(4, counter),  # 6
    Call("print", [output])  # 7
    ]
    build_cfg_new_updated(code);
def test4():
    code = [
    Assignment(base, 2),  # 0
    Assignment(counter, 5),  # 1
    Jump(7, counter), #2
    Assignment(output, 1),  # 3
    # Loop start
    Assignment(output, Expression(output, base, Operation.MUL)),  # 4
    Assignment(counter, Expression(counter, 1, Operation.SUB)),  # 5
    Jump(4, counter),  # 6
    Jump(3,base), #7
    Call("print", [output])  # 8
    ]
    build_cfg_new_updated(code);
def test5():
    code = [
    Assignment(base, 2),  # 0
    Assignment(counter, 5),  # 1
    Assignment(output, 1),  # 2
    Jump(2, counter),  # 5
    Jump(6, base),  # 5
    Jump(8, output),  # 5
    # Loop start
    Assignment(output, Expression(output, base, Operation.MUL)),  # 3
    Assignment(counter, Expression(counter, 1, Operation.SUB)),  # 4
    Jump(6, counter),  # 5
    Call("print", [output]),  # 6
    Jump(9, base),
    Call("print", [output]),  # 6
    ]
    build_cfg_new_updated(code);
def test6():
    code = []
    build_cfg_new_updated(code);
def test7():
    code = [
    Call("print", [output]),  # 6
    Call("print", [output]),  # 6
    Call("print", [output]),  # 6
    Call("print", [output]),  # 6
    Call("print", [output]),  # 6
    Call("print", [output]),  # 6
    Call("print", [output]),  # 6
    Call("print", [output]),  # 6
    Call("print", [output]),  # 6
    Call("print", [output]),  # 6

        ]
    build_cfg_new_updated(code);
def test8():
    code = [
    Jump(5, None),  # 0
    Call("print1", [output]),  # 1
    Call("print2", [output]),  # 2
    Jump(6, None),  # 3
    Call("print4", [output]),  # 4
    Jump(2, output),  # 5
    Jump(2, base),
    Call("print5", [output]),  # 6
    Call("print6", [output]),  # 7
    Call("print7", [output]),  # 8
    Call("print8", [output]),  # 9

    ]
    build_cfg_new_updated(code);

# test1()
# test2()
# test3()
# test4()
# test5()
# test6()
# test7()
# test8()

