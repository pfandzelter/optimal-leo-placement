import typing
import numpy as np
import itertools
import tqdm

from config import DEBUG

def weighted_lee(a: typing.Tuple[int, int], b: typing.Tuple[int, int], k1: int, k2: int, d1: float, d2: float) -> float:
        return min((a[0] - b[0]) % k1, (b[0] - a[0]) % k1)*d1 + min((a[1] - b[1]) % k2, (b[1] - a[1]) % k2)*d2

def __placement(k1: int, k2: int, d1: float, d2: float, t: float, debug: bool = False) -> typing.Tuple[typing.List[typing.Tuple[int, int]], float]:

    # calculate initial values

    # t' is the hops on the modified torus
    t_ = np.floor(t/d1)

    if debug:
        print("t':", t_)

    # k2' is the vertical size of the modified torus
    # k2'r is k2/k2'
    k2_ = int(np.ceil( (d2 / d1) * k2))
    k2_r = k2 / k2_

    if debug:
        print("k2':", k2_)
        print("k2'r:", k2_r)

    # now we have a torus of size k1*k2_ and perform a t_-hop placement on that.
    # to convert y coordinates for resource nodes, transform ky_ back into y.
    # build a torus of k*k so that k = 2(t_**2) + 2t_ + 1
    k = int(2*(t_**2) + 2*t_ + 1)
    if debug:
        print("k:", k)

    # construct a perfect t_-hop placement on the k*k torus
    p = np.array(
        [y for y in
            [
                [
                    int(i % k), int((2 * (t_**2) * i ) % k)
                ]
                for i in np.arange(0, k)
            ]
        ]
    )

    # if debug:
    #     print("p:", p)

    # use that to tile the x*ky_ torus
    # we need np.ceil(x/k) times np.ceil(ky_/k) tiles, then we cut off some stuff
    n = []

    for i in range(int(np.ceil(k1/k))):
        for j in range(int(np.ceil(k2_/k))):
            for pl in p:
                # convert ky_ back into y
                # only if this is still in range, otherwise just discard
                cx  = (i * k) + pl[0]
                cy = (j * k) + pl[1]

                if cx < k1 and cy < k2_:
                    #print("add {} {} ({} {}) (i: {} j: {} k: {} pl:{})".format(cx, cy, cx, (j * k) + pl[1], i, j, k, pl))
                    n.append((cx, cy))

    if debug:
        print("n:", n)


    # 6. because we cut off some resource nodes, we have some "uncovered" nodes at the edges.
    # go through all the nodes
    # one by one, add the nodes with the largest nodes until all are covered
    # takes O(n^3) time (?) but should be fine

    uncovered = set(itertools.product(range(k1), range(k2_)))

    while len(uncovered) > 0:
        max_dist = 0
        max_node = None

        for node in tqdm.tqdm(list(uncovered), disable=not debug):
            min_dist = np.inf

            for rn in n:
                dist = weighted_lee(node, rn, k1, k2_, 1.0, 1.0)

                if dist < min_dist:
                    min_dist = dist

                if dist <= t_:
                    uncovered.remove(node)
                    break

            if min_dist > t_ and min_dist > max_dist:
                max_dist = min_dist
                max_node = node

        if max_node == None:
            break

        n.append(max_node)
        uncovered.remove(max_node)

    # convert the y coordinates back into the old k2
    n = [(x, int(k2_r * y)) for x, y in n]
    if debug:
        print("n:", n)

    #cy = int(np.floor(k2_r * ((j * k) + pl[1])))
    # if debug:
    #     print("n:", n)
    # figure out epsilon
    eps = -np.inf

    for node in itertools.product(range(k1), range(k2)):
    # for node in itertools.product(range(k1), range(k2)):
        min_dist = np.inf

        for rn in n:
            dist = weighted_lee(node, rn, k1, k2, d1, d2)

            if dist < min_dist:
                min_dist = dist

        if min_dist-t > eps:
            eps = min_dist-t

        if min_dist > t:
            if debug:
                print("{}".format(min_dist - t))

    if debug:
        print("epsilon:", eps)

    if debug:
        print("n:", n)

    if debug:
        for node in itertools.product(range(k1), range(k2)):
        # for node in itertools.product(range(k1), range(k2)):
            min_dist = np.inf

            for rn in n:
                dist = weighted_lee(node, rn, k1, k2, d1, d2)

                if dist < min_dist:
                    min_dist = dist

            if min_dist > t + eps:
                print("{}".format(min_dist - t))
                raise ValueError("there is a min_dist > t!")

    return n, eps

def placement(k1: int, k2: int, d1: float, d2: float, t: float, debug: bool = False) -> typing.Tuple[typing.List[typing.Tuple[int, int]], float]:
    """
    Calculates a distance-t placement on a k1 x k2 2D torus with horizontal
    weights d1 and vertical weights d2.

    Parameters:
    k1, k2: The dimensions of the torus.
    d1, d2: The weights of thee torus.
    t: The target distance limit

    Returns:
    A list of tuples, where each tuple contains the x and y coordinates of a
    resource node.
    """

    switch = d1 > d2

    if switch:
        k1, k2 = k2, k1
        d1, d2 = d2, d1

    if debug:
        print("k1: {}, k2: {}, d1: {}, d2: {}, t: {}".format(k1, k2, d1, d2, t))

    if d1 <= t and d2 <= t:
        n, eps = __placement(k1, k2, d1, d2, t, debug)
        if switch:
            n = [(y, x) for x, y in n]

        return n, eps

    elif d1 <= t and d2 > t:
        # tile with the k1 x 1 torus

        dist = int(np.floor(t/d1)) * 2 + 1
        n = [(x, 0) for x in range(0, k1, dist)]
        if debug:
            print("n:", n)

        n_ = []
        for i in range(k2):
            for rn in n:
                n_.append((rn[0], i))
        n = n_

        if switch:
            n = [(y, x) for x, y in n]

        return n, 0

    else:
        # every node is a resource node
        n = [(x, y) for x, y in itertools.product(range(k1), range(k2))]
        if switch:
            n = [(y, x) for x, y in n]

        return n, 0

if "__main__" == __name__:
    np.random.seed(0)

    for i in range(0, 20):
        # make random x, y, dy, dx, t
        x = np.random.randint(1, 50)
        y = np.random.randint(1, 50)
        dx = np.random.random() * 100 + 0.01
        dy = np.random.random() * 100 + 0.01
        t = np.random.random() * 100 + 0.01

        # get a placement
        n, eps = placement(x, y, dx, dy, t)

        # make sure it is within t + epsilon
        with tqdm.tqdm(total=x*y*len(n), desc="Checking error") as pbar:
            for bx in range(0, x):
                for by in range(0, y):
                    best_dist = np.inf
                    best_p = (np.inf, np.inf)
                    for i in range(0, len(n)):
                        p = n[i]
                        dist = weighted_lee(p, (bx, by), x, y, dx, dy)
                        if dist < best_dist:
                            best_dist = dist
                            best_p = p
                        pbar.update(1)

                    if best_dist > t + eps:
                        print(x, y, dx, dy, t, bx, by, best_p[0], best_p[1], best_dist)
                        exit(1)