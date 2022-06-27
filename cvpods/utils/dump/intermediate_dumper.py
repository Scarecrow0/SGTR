import copy
import os
import pickle
import sys
from collections import defaultdict

import numpy
import torch

from cvpods.structures.instances import Instances
from cvpods.structures.relationship import Relationships


from cvpods.utils.distributed import get_world_size, get_rank


def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]

    return inner



@singleton
class _GlobalBuffer():
    """a singleton buffer for store data in anywhere of program
    """

    def __init__(self):
        self.multi_proc = (get_world_size() > 1)
        self.data = defaultdict(list)
        self.buffer_id = 0
        self.curr_root_id = None
        self.hist_root_id = set()

        self.sample_idx = None

    def add_data(self, key, vals: list):

        def recursive_to_cpu(val):
            if isinstance(val, dict):
                for key in val.keys():
                    val[key] = recursive_to_cpu(val[key])
            if isinstance(val, list):
                val = [recursive_to_cpu(each) for each in val]
            if isinstance(val, tuple):
                val = [recursive_to_cpu(each) for each in val]

            # move to cpu
            if hasattr(val, "to"):
                val = val.to("cpu")
            # compress tensor
            if isinstance(val, torch.Tensor):
                if val.dtype == torch.float:
                    val = val.half()
                if val.dtype == torch.long:
                    val = val.int()
            if isinstance(val, Instances) or isinstance(val, Relationships):
                val = val.compress_tensor()

            return val

        for idx, (root_id, val) in enumerate(zip(self.curr_root_id, vals)):
            # image的输入每次基本都是一样的 所以只在第一次加进来的时候append, 后面就直接跳过了
            if self.sample_idx[idx]:
                if key == 'batched_inputs' and root_id in self.hist_root_id:
                    continue

                self.hist_root_id.add(root_id)

                val = recursive_to_cpu(val)
                # take the session last actived buffer for saving data
                self.data[root_id][-1][key] = val

    def __str__(self):
        root_ids = list(self.data.keys())
        ret_str = f'key size: {len(root_ids)} \n'
        # ret_str += f'data size: {sys.getsizeof(self.data)}\n'

        # ret_str += f"Buffer contains data: (key, value type) \n"
        # for k, v in self.data[root_ids[-1]][-1].items():
        #     ret_str += f"    {k}, {type(v).__name__} \n"
        # ret_str += f"id {id(self)}"

        return ret_str


# creat the root for logging thing at the start of each iteration
# 因为一切的输入输出都是围绕着 imamge,
# 因此将image 作为 root 代表写入buffer的唯一id
# 将该iteration中出现的image 作为当前活跃的image name, 之后一切写入的数据都会
# 往这root上追加
def create_save_root(root_ids, sampled=False, sample_rate=0.02):
    rank_id = get_rank()
    if rank_id != 0:
        return

    buffer = _GlobalBuffer()
    buffer.curr_root_id = root_ids

    if sampled:
        buffer.sample_idx = numpy.random.rand(len(root_ids)) < sample_rate
    else:
        buffer.sample_idx = numpy.random.rand(len(root_ids)) > -1

    for idx, each_root in enumerate(root_ids):
        if buffer.sample_idx[idx]:
            buffer.data[each_root].append({})


# 往当前活跃的root上追加 key val数据
def store_data(k, v):
    rank_id = get_rank()
    if rank_id != 0:
        return

    buffer = _GlobalBuffer()
    if buffer.curr_root_id is None:
        return

    assert buffer.curr_root_id is not None
    buffer.add_data(k, copy.deepcopy(v))


def add_dataset_metadata(meta_data):
    rank_id = get_rank()
    if rank_id != 0:
        return

    buffer = _GlobalBuffer()
    buffer.data["dataset_metadata"] = meta_data


def is_empty():
    rank_id = get_rank()
    # only main process save the buffer.
    if rank_id != 0:
        return False

    buffer = _GlobalBuffer()
    return not numpy.any(buffer.sample_idx)


def save_buffer(output_dir, curr_iter):
    rank_id = get_rank()
    # only main process save the buffer.
    if rank_id != 0:
        return

    if is_empty():
        return

    buffer = _GlobalBuffer()
    store_data("curr_iter", [curr_iter for _ in range(len(buffer.curr_root_id))])
    buffer.curr_root_id = None

    if sys.getsizeof(buffer.data) >  2**9:
        save_dir = os.path.join(output_dir, f"inter_data_buffer-rank{rank_id}")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("save pth", save_dir)
        print("save buffer:", str(buffer))
        buffer_data = buffer.data
        for k in buffer_data.keys():
            if k == 'dataset_metadata':
                continue

            for i in range(len(buffer_data[k])):
                for k_in in buffer_data[k][i].keys():
                    if isinstance(buffer_data[k][i][k_in], torch.Tensor):
                        buffer_data[k][i][k_in] = buffer_data[k][i][k_in].detach().cpu()

                    if hasattr(buffer_data[k][i][k_in], "to"):
                        buffer_data[k][i][k_in] = buffer_data[k][i][k_in].to("cpu")

        with open(os.path.join(save_dir, f'{str(buffer.buffer_id)}.pkl'), 'wb') as f:
            pickle.dump(buffer_data, f)
            buffer.data = defaultdict(list)
            buffer.buffer_id += 1
