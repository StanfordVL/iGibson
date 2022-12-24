import argparse
import random
import itertools

task = ["catch", "navigate", "place", "slice", "throw", "wipe"]
vi_choices = ["cataract", "amd", "glaucoma", "presbyopia", "myopia"]
level=[1, 2, 3]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate demo collection sequence")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        nargs="?",
        help="Name of the experiment subject",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    all_choices = list(itertools.product(task, vi_choices, level))
    random.shuffle(all_choices)
    with open(f"igibson/examples/vr/visual_disease_demo_mtls/{args.name}.txt", "w") as f:
        for choice in all_choices:
            f.write(f"python igibson/examples/vr/impairment_task.py --name {args.name} --task {choice[0]} --vi {choice[1]} --level {choice[2]}\n")

    print(f"Task sequence for {args.name} is generated. Please run the command in {args.name}.txt to start the experiment.")

if __name__ == "__main__":
    main() 