#include "mcts.h"
#include <cstddef>
#include <math.h>

namespace water { 

    CNode::CNode(CNode* parent, float p_value, int action) { 
        this->parent = parent; 
        this->p_value = p_value; 
        this->children = new std::vector<CNode*>(); 
        this->n_visits = 0; 
        this->q_value = 0; 
        this->u_value = 0; 
        this->action = action;
    }

    void CNode::expand(std::vector<int> actions, std::vector<float> priors) { 
        for (int i = 0; i < (int)priors.size(); i++) { 
            this->children->push_back(new CNode(this, priors[i], actions[i])); 
        }
    }

    void CNode::update(float leaf_value) { 
        this->n_visits += 1; 
        this->q_value += (leaf_value - this->q_value) / this->n_visits;
    }

    void CNode::update_recursive(float leaf_value) { 
        if (this->parent != NULL) { 
            this->parent->update_recursive(-leaf_value);
        }
        this->update(leaf_value); 
    }

    int CNode::n_children() { 
        return (int)this->children->size();
    }

    int CNode::select(float c_puct) { 
        float best_value = -1000000000; 
        int best_idx = 0; 

        for (int i = 0; i < (int)this->children->size(); i++) { 
            float value = (*(this->children->at(i))).get_value(c_puct); 
            if (value > best_value) { 
                best_value = value; 
                best_idx = i;
            }
        }

        return best_idx;
    }

    CNode* CNode::get_child(int idx) { 
        return this->children->at(idx);
    }

    float CNode::get_value(float c_puct) { 
        if (this->parent != NULL) { 
            this->u_value = c_puct * this->p_value * sqrt(this->parent->n_visits) / (1 + this->n_visits);
        } else { 
            this->u_value = 0;
        }
        return this->q_value + this->u_value; 
    }

    bool CNode::is_leaf() { 
        return this->children->size() == 0;
    }

    bool CNode::is_root() { 
        return this->parent == NULL;
    }

}