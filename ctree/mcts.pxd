# distutils: language=c++

from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "mcts.cpp" namespace "water": 

    cdef cppclass CNode: 
        CNode(CNode* parent, float p_value, int action)
        CNode* parent
        vector[CNode*] children
        int n_visits 
        float q_value
        float u_value
        float p_value
        int action

        void expand(vector[int] actions, vector[float] priors)
        void update(float leaf_value)
        CNode* select(float c_puct)
        void update_recursive(float leaf_value)
        float get_value(float c_puct)
        bool is_leaf()
        bool is_root()
        int n_children()
        CNode* get_child(int idx)
