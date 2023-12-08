import numpy as np
import pandas as pd
import plotnine as pn
from rdot import ba_ib_torch

def main():
    # Make sure we test when |Y| != |X|, e.g. |X| = 100, |Y| = 10
    # Gaussian-like p(y|x)
    py_x = np.array([[np.exp(-(i - j)**2) for j in range(0, 100, 10)] for i in range(100)])
    py_x /= py_x.sum(axis=1)[:, None]
    # get joint by multiplying by p(x)
    px = np.full(py_x.shape[0], 1/100)
    pxy = py_x * px[:, None]

    # Test many values of beta to sweep out a curve. 
    betas = np.linspace(1e-5, 1e10, num=100)

    results = ba_ib_torch.ib_method(pxy, betas, num_restarts=1, num_processes=6)
    rd_values = [result[-3:] for result in results]
    breakpoint()    

    data = pd.DataFrame(rd_values, columns=["rate", "distortion", "accuracy"])
    print(data)

    (
        pn.ggplot(data, pn.aes(x="rate", y="distortion"))
        + pn.geom_point()
        + pn.geom_line()
    ).save("curve_distortion.png")
    (
        pn.ggplot(data, pn.aes(x="rate", y="accuracy"))
        + pn.geom_point()
        + pn.geom_line()
    ).save("curve_accuracy.png")

if __name__ == "__main__":
    main()