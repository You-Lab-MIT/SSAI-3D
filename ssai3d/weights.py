"""Released model weights: download and verify."""
import hashlib
import logging
import os

logger = logging.getLogger('ssai3d')

# name: (Google Drive file id, sha256)
WEIGHTS = {
    'NAFNet-REDS-width64.pth': (
        '14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X',
        '175fe8b3cdf3abedfbc87769779c3d9f491e05bb2e73ea9d627883f90a4b2df3'),
    'generalizable_model.pth': (
        '1mhpZ00h3UvXvTfsA_feYd1B2sWD-06uY',
        '477a1ead4e7c9cc98c37bd73d9e121405130a5e5139b4250826e74f9aef25e19'),
    'denoising_model.pth': (
        '1Lkg5a8xtjze7cKitdMl8bIY38cLAIojT',
        'db25b2b7fcb38a970a92c65c11e59e06dc1a3ce0a90861b961ced753172b78d0'),
    'net_g_neurons.pth': (
        '1Q3d7y96dQsd3Xk4l8c05M2VnUlu9q9CL',
        '7358ef7b3930738d76e6f0f5e3536e73b710640b77ef4516909b2c93f32d6428'),
}

# Mouse brain neurons demo stack (Z, Y, X) = (194, 1153, 893), uint16.
DEMO_DATA = ('raw_data.tif', '1p3CUWhaSJXAA_9k8p4nRrhjBmbegQ-vJ',
             '4958b4fab51b0d6b684a9e2130c2915015ec55ac82cba8ed98d2b2b202fbfc33')


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def fetch(name, file_id, checksum, dest):
    """Download a Google Drive file to ``dest/name`` unless a verified copy exists."""
    import gdown

    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, name)
    if os.path.exists(path) and sha256(path) == checksum:
        logger.info(f'{name}: already present.')
        return path
    gdown.download(id=file_id, output=path, quiet=False)
    if sha256(path) != checksum:
        raise RuntimeError(f'{name}: checksum mismatch after download.')
    return path


def download_weights(dest, names=None):
    """Download the released weights (all, or those in ``names``) into ``dest``."""
    return [fetch(n, *WEIGHTS[n], dest) for n in (names or WEIGHTS)]


def download_demo_data(dest):
    return fetch(*DEMO_DATA, dest)
