import numpy as np
import time

# Helper function to exchange group members
def exchange(stallions):
    for i in range(len(stallions)):
        groups = stallions[i].group
        stallions[i].group = np.random.choice(groups, len(groups), replace=False)
    return stallions

def PROPOSED(Positions, fobj, LB, UB, Max_iter):

    N, dim = Positions.shape
    lb = LB[0, :]
    ub = UB[0, :]
    SP = 0.5  # Stallions Percentage
    PC = 0.1  # Crossover Percentage
    NStallion = int(np.ceil(SP * N))  # Number of Stallions
    Nfoal = N - NStallion

    Convergence_curve = np.zeros(Max_iter)
    gBest = np.zeros(dim)
    gBestScore = np.inf

    # Create initial population
    empty = {"pos": [], "cost": []}
    group = [empty.copy() for _ in range(Nfoal)]

    for i in range(Nfoal):
        group[i]["pos"] = Positions[i]
        group[i]["cost"] = fobj(group[i]["pos"])

    Stallion = [empty.copy() for _ in range(NStallion)]

    for i in range(NStallion):
        Stallion[i]["pos"] = Positions[i]
        Stallion[i]["cost"] = fobj(Stallion[i]["pos"])

    ngroup = len(group)
    a = np.random.permutation(ngroup)
    group = [group[i] for i in a]

    i = 0
    k = 1
    for j in range(ngroup):
        i = i + 1
        Stallion[i - 1]["group"].append(group[j])
        if i == NStallion:
            i = 0
            k = k + 1

    Stallion = exchange(Stallion)
    value, index = min((stallion["cost"], i) for i, stallion in enumerate(Stallion))
    WR = Stallion[index]  # Global
    gBest = WR["pos"]
    gBestScore = WR["cost"]

    Convergence_curve[0] = WR["cost"]
    l = 2  # Loop counter
    ct = time.time()
    while l < Max_iter + 1:
        TDR = 1 - l * (1 / Max_iter)

        for i in range(NStallion):
            ngroup = len(Stallion[i]["group"])
            sorted_group = sorted(Stallion[i]["group"], key=lambda x: x["cost"])

            for j in range(ngroup):
                # Feeding Activity Phase
                z = np.random.rand(dim) < TDR
                r1 = gBestScore / (100 * np.mean(group[j]["cost"]))
                r2 = np.random.rand(dim)
                idx = z == 0
                r3 = r1 * idx + r2 * ~idx
                rr = 1 - l * (1 / Max_iter)

                if np.random.rand() < 0.5:
                    Stallion[i]["group"][j]["pos"] = (
                        2 * r3 * np.sin(2 * np.pi * rr * np.random.rand()) * (Stallion[i]["pos"] - Stallion[i]["group"][j]["pos"])
                        + Stallion[i]["pos"]
                    )
                else:
                    Stallion[i]["group"][j]["pos"] = (
                        2 * r3 * np.cos(2 * np.pi * rr * np.random.rand()) * (Stallion[i]["pos"] - Stallion[i]["group"][j]["pos"])
                        + Stallion[i]["pos"]
                    )

                Stallion[i]["group"][j]["pos"] = np.minimum(Stallion[i]["group"][j]["pos"], ub)
                Stallion[i]["group"][j]["pos"] = np.maximum(Stallion[i]["group"][j]["pos"], lb)

                Stallion[i]["group"][j]["cost"] = fobj(Stallion[i]["group"][j]["pos"])

                if Stallion[i]["group"][j]["cost"] <= Stallion[i]["cost"]:
                    Stallion[i]["pos"] = Stallion[i]["group"][j]["pos"]
                    Stallion[i]["cost"] = Stallion[i]["group"][j]["cost"]

                # Breeding Activity Phase
                if np.random.rand() > PC:
                    A = np.random.permutation(NStallion)
                    A = A[A != i]
                    a = A[0]
                    c = A[1]
                    x1 = Stallion[c]["group"][-1]["pos"]
                    x2 = Stallion[a]["group"][-1]["pos"]
                    y1 = (x1 + x2) / 2  # Crossover
                    Stallion[i]["group"][j]["pos"] = y1
                else:
                    A = np.random.permutation(NStallion)
                    A = A[A != i]
                    a = A[0]
                    d = A[2]
                    x2 = Stallion[a]["group"][-1]["pos"]
                    x3 = Stallion[d]["group"][-1]["pos"]
                    y2 = (x2 + x3) / 2
                    Stallion[i]["group"][j]["pos"] = y2

                Stallion[i]["group"][j]["pos"] = np.minimum(Stallion[i]["group"][j]["pos"], ub)
                Stallion[i]["group"][j]["pos"] = np.maximum(Stallion[i]["group"][j]["pos"], lb)
                Stallion[i]["group"][j]["cost"] = fobj(Stallion[i]["group"][j]["pos"])

            R = np.random.rand()
            # Group Leadership Phase
            if R < 0.5:
                k = (
                    2
                    * r3
                    * np.sin(2 * np.pi * rr * np.random.rand())
                    * (WR["pos"] - Stallion[i]["pos"])
                    + WR["pos"]
                )
            else:
                k = (
                    2
                    * r3
                    * np.cos(2 * np.pi * rr * np.random.rand())
                    * (WR["pos"] - Stallion[i]["pos"])
                    - WR["pos"]
                )

            k = np.minimum(k, ub)
            k = np.maximum(k, lb)
            fk = fobj(k)

            if fk < Stallion[i]["cost"]:
                Stallion[i]["pos"] = k
                Stallion[i]["cost"] = fk

        Stallion = exchange(Stallion)
        value, index = min((stallion["cost"], i) for i, stallion in enumerate(Stallion))

        if value < WR["cost"]:
            WR = Stallion[index]

        gBest = WR["pos"]
        gBestScore = WR["cost"]
        Convergence_curve[l - 1] = WR["cost"]
        l = l + 1
    ct = time.time() - ct
    return gBestScore, Convergence_curve, gBest, ct
