# hyper-alpha-zero

hyper-optimized alpha-zero implementation with ray + cython for speed

train an agent that beats random actions and pure MCTS in 2 minutes

### file structure 

- `train.py`: distributed training with ray 
- `ctree/`: mcts nodes in cython (node.py = pure python)
- `mcts.py`: mcts playouts 
- `network.py`: neural net stuff 
- `board.py`: gomoku board 

## system design 
- ray distributed parts (`train.py`):
  - one distributed replay buffer 
  - N actors with the 'best model' weights which self-play games and store data in replay buffer 
  - M 'candidate models' which pull from the replay buffer and train 
    - each iteration they play against the 'best model' and if they win the 'best model' weights is updated 
    - include write/evaluation locks on 'best weights'
  - 1 best model weights store 
    - stores the best weights which are retrived by self-play and updated when candidates win 

![](imgs/2023-01-15-09-18-19.png)

- cython impl
  - `ctree/`: c++/cython mcts 
  - `node.py`: pure python mcts

-- todos -- 

- jax network impl 
- tpu + gpu support 
- saved model weights

### references 
- based off: https://github.com/junxiaosong/AlphaZero_Gomoku
- distributed rl: http://rail.eecs.berkeley.edu/deeprlcourse-fa18/static/slides/lec-21.pdf 