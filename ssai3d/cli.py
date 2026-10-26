"""Command line interface: ``ssai3d run|restore|download``."""
import argparse
import logging
import os
import sys


def _setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format='%(asctime)s %(levelname)s %(message)s')


def cmd_run(args):
    from ssai3d.pipeline import load_config, run
    cfg = load_config(args.config)
    _setup_logging(os.path.join(cfg['out_dir'], 'run.log'))
    steps = args.steps.split(',') if args.steps else None
    run(args.config, steps=steps, force=args.force)


def cmd_restore(args):
    _setup_logging()
    import numpy as np
    import tifffile
    from ssai3d import data
    from ssai3d.gpu import setup_device
    from ssai3d.restore import restore

    device = setup_device(args.gpu, args.max_mem_gb)
    work = os.path.join(args.out, 'slices')
    data.generate_raw_data(args.raw, work, args.downsample)
    for view in ('xz', 'yz'):
        restore(os.path.join(work, f'test_{view}'), os.path.join(args.out, 'restored', view),
                args.model, tile=args.tile, device=device,
                norm=None if args.norm == 'auto' else args.norm)
    vol = data.fuse_views(os.path.join(args.out, 'restored', 'xz'),
                          os.path.join(args.out, 'restored', 'yz'))
    path = os.path.join(args.out, 'restored.tif')
    tifffile.imwrite(path, vol.astype(np.float32), imagej=True, metadata={'axes': 'ZYX'})
    logging.getLogger('ssai3d').info(f'Wrote {path} with shape {vol.shape} (Z, Y, X).')


def cmd_download(args):
    _setup_logging()
    from ssai3d.weights import download_demo_data, download_weights
    download_weights(args.dest, args.names)
    if args.demo_data:
        download_demo_data(args.demo_dest)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='ssai3d', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('run', help='run the full pipeline from a YAML config')
    p.add_argument('config', help='run config (see ssai3d/configs/example_run.yml)')
    p.add_argument('--steps', help='comma-separated subset, e.g. "restore,fuse"')
    p.add_argument('--force', action='store_true', help='redo steps already marked done')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('restore', help='restore a stack with an existing network')
    p.add_argument('raw', help='3D TIFF, shape (Z, Y, X)')
    p.add_argument('--model', required=True, help='network weights (.pth)')
    p.add_argument('--out', required=True, help='output folder')
    p.add_argument('--downsample', type=int, default=5, help='axial/lateral sampling ratio')
    p.add_argument('--tile', type=int, default=None, help='tile size in pixels')
    p.add_argument('--gpu', type=int, default=0, help='CUDA index, -1 for CPU')
    p.add_argument('--max-mem-gb', type=float, default=None, help='GPU memory budget')
    p.add_argument('--norm', choices=('auto', 'slice', 'volume'), default='auto',
                   help='input scaling; auto = as recorded in the model')
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser('download', help='download the released weights')
    p.add_argument('--dest', default='weights', help='destination folder')
    p.add_argument('--names', nargs='*', help='subset of weight files')
    p.add_argument('--demo-data', action='store_true', help='also fetch the demo stack')
    p.add_argument('--demo-dest', default='demo', help='destination of the demo stack')
    p.set_defaults(func=cmd_download)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
