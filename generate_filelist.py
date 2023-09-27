import os
import random
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str)
parser.add_argument("--name", type=str, default="lrs2")
parser.add_argument("--ratio", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--step", type=int, default=1)
args = parser.parse_args()

random.seed(args.seed)


def is_valid_target(target: str):
    basename = os.path.basename(target)
    return os.path.isdir(target) is True and basename[0] != "."


lstrip_path_len = len(args.dir)

clip_list = []

for video_dir in glob.glob(os.path.join(args.dir, "*")):
    if is_valid_target(video_dir) is False:
        continue

    for clip_dir in glob.glob(os.path.join(video_dir, "*")):
        if is_valid_target(clip_dir) is False:
            continue
        clip_list.append(clip_dir[lstrip_path_len:].strip("/"))

random.shuffle(clip_list)
if args.step > 1:
    clip_list = clip_list[::args.step]

clip_count = len(clip_list)
test_count = int(args.ratio * clip_count)
print("{} clips found, {} in test set".format(clip_count, test_count))


train_set = sorted(clip_list[:-test_count])
test_set = sorted(clip_list[clip_count - test_count:])
assert len(train_set) + len(test_set) == clip_count


def save(list2save: list, set_name: str):
    prefix = ""
    base_dir = os.path.join(os.path.dirname(__file__), "filelists", args.name)
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, "{}.txt".format(set_name)), "w") as f:
        for i in list2save:
            f.write("{}{}".format(prefix, i))
            prefix = "\n"


save(train_set, "train")
save(test_set, "test")
