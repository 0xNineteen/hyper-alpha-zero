from cmcts import Node

if __name__ == "__main__":
    # test expand node
    node = Node(None, 1, 0)
    node.expand([1, 2], [1, 2])

    # test other functions
    node.update(1)
    node.update_recursive(1)
    value = node.get_value(1)
    print("value", value)

    child = Node(node, 2, 1)
    child.update(1)
    child.update_recursive(1)
    value = child.get_value(1)
    print("child value", value)

    assert not node.is_leaf()
    assert child.is_leaf()

    assert node.is_root()
    assert not child.is_root()

    a, _child = node.select(1)
    print(a, _child)

    print(node.n_children())
    print(node.children)
    print(node.n_visits)
    print(node.action)

    print(node.get_child_by_action(1))
    print(node.get_child_by_action(19))

    node.delete_tree()

    # del node
    # del child # we drop child here but it still exists in _child
    # # del _child -- this will cause segfault (aka why we need to handle dealloc manually)

    print("done")
