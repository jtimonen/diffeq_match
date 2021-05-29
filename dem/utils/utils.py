import torch
import torch.nn as nn
import imageio
import os
import glob
import pkg_resources
from dem.utils.settings import session_info


def tensor_to_numpy(x: torch.Tensor):
    return x.cpu().detach().numpy()


def add_noise(x: torch.Tensor, sigma):
    """Add normally distributed noise to Tensor."""
    if sigma > 0:
        x = x + sigma * torch.randn_like(x)
    return x


def num_trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def animate(path, prefix, ext=".png", frame_duration=0.15, outfile=None):
    if outfile is None:
        fn = os.path.join(path, prefix) + ".gif"
    else:
        fn = outfile
    pattern = os.path.join(path, prefix + "*" + ext)
    print("Creating " + fn + " from files that match pattern '" + pattern + "'...")
    filenames = glob.glob(pattern)
    n_images = len(filenames)
    if n_images == 0:
        print("No matches found!")
        return None
    print("Found %d image files, start writing..." % len(filenames))
    fps = 1.0 / frame_duration
    _animate(filenames, fps, fn)
    print("Animation ready  (%1.1f fps)." % fps)


def _animate(filenames, fps, outfile):
    images = []
    for file_name in sorted(filenames):
        images.append(imageio.imread(file_name))
    imageio.mimsave(outfile, images, fps=fps)


def html_viewer(outfile="output.html", description_txt="No description"):
    info_txt = session_info(skip_cuda=False, quiet=True)
    code = pkg_resources.resource_string(__name__, "viewer.html").decode("utf-8")
    code = code.replace("__MODEL_DESCRIPTION__", description_txt)
    code = code.replace("__PKG_INFO__", info_txt)
    with open(outfile, "w") as f:
        f.write(code)
