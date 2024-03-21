import csv
import os
import os.path as osp
import re
import shutil
import warnings

from cube_dl.callback import CubeCallback
from cube_dl.core import CUBE_CONTEXT
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class MetricsCSVCallback(CubeCallback):
    """Specific callback for `pytorch_lightning.loggers.CSVLogger`."""

    @rank_zero_only
    def on_run_start(self):
        # Suppress the non-empty log directory warning
        warnings.filterwarnings(
            action="ignore",
            message="Experiment logs directory .* exists and is not empty. Previous log files in this directory will "
            "be deleted when the new ones are saved!",
            category=UserWarning,
            module="lightning_fabric",
        )

        # Since `pytorch_lightning.loggers.CSVLogger` will override previous "metrics.csv",
        # special process is needed when resuming fit to avoid "metrics.csv" being overridden.
        if CUBE_CONTEXT["run"].is_resuming:
            run_dir = CUBE_CONTEXT["run"].run_dir
            metrics_csv_path = osp.join(run_dir, "metrics.csv")
            if osp.exists(metrics_csv_path):
                # Rename the original "metrics.csv" to "metrics_<max_cnt>.csv",
                # to reserve the filename "metrics.csv" when resuming fit.
                indices = []
                for filename in os.listdir(run_dir):
                    match = re.match(r"metrics_(\d+).csv", filename)
                    if match:
                        indices.append(int(match.group(1)))
                max_cnt = 1 if len(indices) == 0 else max(indices) + 1
                shutil.move(metrics_csv_path, metrics_csv_path[:-4] + f"_{max_cnt}.csv")

    @rank_zero_only
    def on_run_end(self):
        run_dir = CUBE_CONTEXT["run"].run_dir

        # Remove empty "hparams.yaml" generated by PyTorch-Lightning
        hparams_yaml_path = osp.join(run_dir, "hparams.yaml")
        needs_removed = False
        if osp.exists(hparams_yaml_path):
            with open(hparams_yaml_path) as f:
                if f.read().strip() == "{}":
                    needs_removed = True
        if needs_removed:
            os.remove(hparams_yaml_path)

        # Merge multiple metrics_csv (if any) into "merged_metrics.csv"
        # The correct order is "metrics_1.csv" -> "metrics_2.csv" -> ... -> "metrics.csv".
        metrics_csv_files = [
            filename
            for filename in os.listdir(run_dir)
            if filename.startswith("metrics_") and filename.endswith(".csv")
        ]
        metrics_csv_files.sort(key=lambda x: int(osp.splitext(x)[0].split("_")[1]))

        # Append the latest one to the last.
        if osp.exists(osp.join(run_dir, "metrics.csv")):
            metrics_csv_files.append("metrics.csv")

        # Merge
        if len(metrics_csv_files) > 1:
            # Get the field names from the 1st csv
            with open(osp.join(run_dir, metrics_csv_files[0])) as metrics_csv:
                reader = csv.DictReader(metrics_csv)
                fieldnames = list(next(reader).keys())

            with open(osp.join(run_dir, "merged_metrics.csv"), "w") as merged_metrics_csv:
                writer = csv.DictWriter(merged_metrics_csv, fieldnames=fieldnames)
                writer.writeheader()
                for csv_fn in metrics_csv_files:
                    with open(osp.join(run_dir, csv_fn)) as metrics_csv:
                        reader = csv.DictReader(metrics_csv)
                        writer.writerows(reader)
