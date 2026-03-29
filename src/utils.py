import os, sys
import time
import random
import math
import pickle
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import shutil
import logging

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class TimeEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def occumpy_mem(args):
    """
        Occupy GPU memory in advance.
    """
    torch.cuda.set_device(args.gpu_id)
    total, used = check_mem(args.gpu_id)
    total, used = int(total), int(used)
    block_mem = int((total-used) * args.occ_gpu)
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x


def seed_everything(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_env(problem):
    from envs import TSPTWEnv_SPIP, STSPTWEnv, STSPTWEnv_v2
    all_problems = {
        'TSPTW_SPIP': TSPTWEnv_SPIP.TSPTWEnv_SPIP,
        'STSPTW': STSPTWEnv.STSPTWEnv,
        'STSPTW_v2': STSPTWEnv_v2.STSPTWEnv_v2,
    }
    if problem == "ALL":
        return list(all_problems.values())
    else:
        return [all_problems[problem]]


def get_opt_sol_path(dir, problem, size, hardness):
    if problem == "TSPTW_SPIP":
        return os.path.join(dir, f"lkh_tsptw{size}_{hardness}.pkl")
    elif problem in ("STSPTW", "STSPTW_v2"):
        return os.path.join(dir, f"lkh_tsptw{size}_{hardness}.pkl")
    else:
        all_opt_sol = {
            'CVRP': {50: 'hgs_cvrp50_uniform.pkl', 100: 'hgs_cvrp100_uniform.pkl'},
            'OVRP': {50: 'or_tools_200s_ovrp50_uniform.pkl', 100: 'lkh_ovrp100_uniform.pkl'},
            'VRPB': {50: 'or_tools_200s_vrpb50_uniform.pkl', 100: 'or_tools_400s_vrpb100_uniform.pkl'},
            'VRPL': {50: 'or_tools_200s_vrpl50_uniform.pkl', 100: 'lkh_vrpl100_uniform.pkl'},
            'VRPTW': {50: 'hgs_vrptw50_uniform.pkl', 100: 'hgs_vrptw100_uniform.pkl'},
            'OVRPTW': {50: 'or_tools_200s_ovrptw50_uniform.pkl', 100: 'or_tools_400s_ovrptw100_uniform.pkl'},
            'OVRPB': {50: 'or_tools_200s_ovrpb50_uniform.pkl', 100: 'or_tools_400s_ovrpb100_uniform.pkl'},
            'OVRPL': {50: 'or_tools_200s_ovrpl50_uniform.pkl', 100: 'or_tools_400s_ovrpl100_uniform.pkl'},
            'VRPBL': {50: 'or_tools_200s_vrpbl50_uniform.pkl', 100: 'or_tools_400s_vrpbl100_uniform.pkl'},
            'VRPBTW': {50: 'or_tools_200s_vrpbtw50_uniform.pkl', 100: 'or_tools_400s_vrpbtw100_uniform.pkl'},
            'VRPLTW': {50: 'or_tools_200s_vrpltw50_uniform.pkl', 100: 'or_tools_400s_vrpltw100_uniform.pkl'},
            'OVRPBL': {50: 'or_tools_200s_ovrpbl50_uniform.pkl', 100: 'or_tools_400s_ovrpbl100_uniform.pkl'},
            'OVRPBTW': {50: 'or_tools_200s_ovrpbtw50_uniform.pkl', 100: 'or_tools_400s_ovrpbtw100_uniform.pkl'},
            'OVRPLTW': {50: 'or_tools_200s_ovrpltw50_uniform.pkl', 100: 'or_tools_400s_ovrpltw100_uniform.pkl'},
            'VRPBLTW': {50: 'or_tools_200s_vrpbltw50_uniform.pkl', 100: 'or_tools_400s_vrpbltw100_uniform.pkl'},
            'OVRPBLTW': {50: 'or_tools_200s_ovrpbltw50_uniform.pkl', 100: 'or_tools_400s_ovrpbltw100_uniform.pkl'},
        }
        return os.path.join(dir, all_opt_sol[problem][size])


def num_param(model):
    nb_param = 0
    for param in model.parameters():
        nb_param += param.numel()
    print('Number of Parameters: {}'.format(nb_param))


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename, disable_print=False):
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    if not disable_print:
        print(">> Save dataset to {}".format(filename))


def load_dataset(filename, disable_print=False):
    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
    if not disable_print:
        print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
    return data


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass  # This flush method is required for file-like object.


def create_logger(filename, log_path=None):
    if log_path and not os.path.exists(log_path):
        os.makedirs(log_path)

    file_mode = 'a' if os.path.isfile(os.path.join(log_path, filename)) else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    # Clear existing handlers
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # Write to file
    fileout = logging.FileHandler(os.path.join(log_path,filename), mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # Write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Redirect print to logging
    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True, disable_tqdm=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    os.makedirs(directory, exist_ok=True)
    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval, disable=disable_tqdm))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    # assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    if len(failed) != 0:
        "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def copy_all_src(dst_root):

    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            # skip non-existent files
            if not os.path.exists(src_abspath):
                print(f"[copy_all_src] Warning: {src_abspath} not found. Skipping.")
                continue

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                shutil.copy(src_abspath, dst_filepath)


def read_pkl_file(file_path, N):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = (torch.empty(0, N, 2), torch.empty(0, N), torch.empty(0, N), torch.empty(0, N), torch.empty(0, N))
    return data


def write_pkl_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def add_data_to_pkl(file_path, new_data, N):

    data = read_pkl_file(file_path, N)

    updated_data = tuple(torch.cat((data[i], new_data[i]), dim=0) for i in range(len(data)))

    write_pkl_file(file_path, updated_data)
