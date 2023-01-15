# distutils: language=c++

from mcts cimport CNode
from libcpp.vector cimport vector

cdef create_py_node(CNode* node): 
    _node = Node(None, 0, 0) # tmp
    _node.node = node # overwrite
    return _node

cdef class Node: 
    cdef CNode *node

    def __cinit__(self, Node parent, float p_value, int action):
        if parent is None: 
            self.node = new CNode(NULL, p_value, action)
        else:
            self.node = new CNode(parent.node, p_value, action)

    def expand(self, vector[int] actions, vector[float] priors):
        assert(len(actions) == len(priors))
        self.node.expand(actions, priors)
    
    def n_children(self):
        return self.node.n_children()

    @property
    def children(self):
        n = self.n_children()
        _children = []
        for i in range(n):
            _children.append(self.get_child(i))
        return _children

    @property
    def n_visits(self):
        return self.node.n_visits
    
    @property
    def action(self):
        return self.node.action

    def select(self, float c_puct):
        child = create_py_node(self.node.select(c_puct))
        return (child, child.action)

    def get_child(self, int idx):
        _node = create_py_node(self.node.get_child(idx))
        return _node
    
    def get_child_by_action(self, int action):
        for child in self.children:
            if child.action == action:
                return child
        return None

    def delete_tree(self):
        for child in self.children:
            child.delete_tree()

        if self.node != NULL: 
            del self.node

    def update(self, float value):
        self.node.update(value)

    def update_recursive(self, float value):
        self.node.update_recursive(value)

    def get_value(self, float c_puct):
        return self.node.get_value(c_puct)
    
    def is_leaf(self):
        return self.node.is_leaf()

    def is_root(self):
        return self.node.is_root()

    def __dealloc__(self):
        pass # we need to handle this manually (see test.py) -- see delete_tree()