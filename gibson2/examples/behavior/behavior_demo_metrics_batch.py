import argparse

from gibson2.examples.behavior.behavior_demo_batch import behavior_demo_batch
from gibson2.metrics.agent import AgentMetric
from gibson2.metrics.disarrangement import KinematicDisarrangement, LogicalDisarrangement
from gibson2.metrics.gaze import GazeMetric
from gibson2.metrics.task import TaskMetric

def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect metrics from BEHAVIOR demos in manifest.')
    parser.add_argument('demo_root', type=str,
                        help="Directory containing demos listed in the manifest.")
    parser.add_argument('log_manifest', type=str,
                        help='Plain text file consisting of list of demos to replay.')
    parser.add_argument('out_dir', type=str,
                        help="Directory to store results in.")
    return parser.parse_args()


def main():
    args = parse_args()

    def get_metrics_callbacks():
        metrics = [
            KinematicDisarrangement(),
            LogicalDisarrangement(),
            AgentMetric(),
            GazeMetric(),
            TaskMetric()
        ]

        return (
            [metric.start_callback for metric in metrics],
            [metric.step_callback for metric in metrics],
            [metric.end_callback for metric in metrics],
            [metric.gather_results for metric in metrics]
        )

    behavior_demo_batch(args.demo_root, args.log_manifest, args.out_dir, get_metrics_callbacks)


if __name__ == "__main__":
    main()