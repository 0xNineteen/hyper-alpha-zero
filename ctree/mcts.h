#ifndef MCTS_H // need this!
#define MCTS_H

#include <vector>

namespace water { 
    class CNode { 
        public: 
            CNode(CNode* parent, float p_value, int action);
            CNode* parent;
            std::vector<CNode*>* children;
            std::vector<int>* actions;
            int n_visits; 
            float q_value;
            float u_value;
            float p_value;
            int action;

            void expand(std::vector<int> actions, std::vector<float> priors);
            void update(float leaf_value);
            void update_recursive(float leaf_value);
            int select(float c_puct);
            float get_value(float c_puct);
            CNode* get_child(int idx);
            bool is_leaf();
            bool is_root();
            int n_children();
    };
}

#endif