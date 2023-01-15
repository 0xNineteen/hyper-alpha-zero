## cython MCTS

build: `bash make.sh`

## speed comparison
100_000 mcts playouts on gomeku board
```
  mcts = MCTS(policy_value_fn)
  for _ in range(100_000):
      board_copy = copy.deepcopy(board)
      mcts._playout(board_copy)
```

- `time python mcts.py` (**cython = 8x speedup**)
  - pure python: 
    - 32.89s user 2.38s system 107% cpu 32.850 total
  - cython: 
    - 4.80s user 1.84s system 158% cpu 4.185 total

## file layout
- `mcts.cpp`: cpp impl which we want to wrap 
  - also note all classes have the 'C' appended to them to denote they are the cpp version of the class 
  - `mcts.h`: corresponding header file (notice the macro to stop double imports)
- `mcts.pxd`: re-defining the types of mcts.cpp for cython (copy pasta of cpp header)
- `cmcts.pyx`: python exposed wrapper (we dont want our code (`mcts.cpp`) to be overwritten -- so we append 'c' at the start of this file name)
- `setup.py`: build script 
- `test.py`: simple test to make sure no segfaults

## random notes 
- cython doesnt support pair/tuples very well 
- when using ptrs (as we do bc tree structures) we should handle deallocing nodes ourselves
  - thus we pass on `__dealloc__`