import itertools

vi_seq = [
    [1,2,3,4,5],
    [4,2,5,3,1],
    [4,1,5,3,2],
    [5,4,1,2,3],
    [5,3,2,1,4],
    [3,2,4,1,5],
    [4,2,1,5,3],
    [5,4,3,2,1],
    [3,1,4,5,2],
    [4,2,1,3,5],
    [5,3,4,1,2],
    [2,5,4,1,3],
    [1,3,2,5,4],
    [2,1,4,3,5],
    [2,5,4,3,1],
    [2,3,5,1,4],
    [1,4,5,2,3],
    [1,5,4,3,2],
    [3,5,2,4,1],
    [3,5,1,2,4],
]

tasks = ["catch", "navigate", "place", "slice", "throw", "wipe"]
vi_choices = {
    1: "cataract", 
    2: "amd", 
    3: "glaucoma", 
    4: "presbyopia", 
    5: "myopia"
}
level=[1, 2, 3]


def main():
    for i in range(20):
        all_choices = [["normal", 1]] + list(itertools.product([vi_choices[j] for j in vi_seq[i]], level)) + [["normal", 1]]
        with open(f"igibson/data/seq/{i + 1}.txt", "w") as f:
            for task in tasks:
                f.write(f"python igibson/examples/vr/impairment_task.py --id {i + 1} --task {task} --training\n")
                for choice in all_choices:
                    f.write(f"python igibson/examples/vr/impairment_task.py --id {i + 1} --task {task} --vi {choice[0]} --level {choice[1]}\n")

if __name__ == "__main__":
    main() 