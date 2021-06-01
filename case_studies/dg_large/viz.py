#!/usr/bin/python
import sys
import hdviz
import numpy as np
import os
from sklearn.decomposition import PCA


# Get data
def run_viz(idx: int):

    # Read data
    n_latent = idx + 1
    folder_name = "latent_" + str(n_latent)
    latent = np.loadtxt(os.path.join(folder_name, "latent.txt"))
    colors = np.loadtxt(
        os.path.join(folder_name, "colors.txt"), dtype=np.str, comments=None
    )
    types = np.loadtxt(
        os.path.join(folder_name, "types.txt"), dtype=np.str, comments=None
    )

    # Plot
    fig_name = "latent_" + str(n_latent) + ".png"
    hdviz.visualize(
        points=latent,
        labels=types,
        colors=colors,
        save_name=fig_name,
        save_dir="figures",
        scatter_kwargs=dict(s=2),
    )

    # Plot PCA
    if n_latent > 2:
        pca = PCA(n_components=2)
        low = pca.fit_transform(X=latent)
        print(low.shape)
        fig_name = "latent_" + str(n_latent) + "_pca.png"
        hdviz.visualize(
            points=low,
            labels=types,
            colors=colors,
            save_name=fig_name,
            save_dir="figures",
            scatter_kwargs=dict(s=2),
        )


if __name__ == "__main__":
    print("Commandline arguments", str(sys.argv))
    idx = int(sys.argv[1])
    print("idx = ", idx)
    run_viz(idx)
