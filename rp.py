import argparse
import os
import logging
from scipy.stats import pearsonr, spearmanr
from util_io import get_scores
logger = logging.getLogger(__name__)

def eval_rp(gold_dir, corrupted_dir):
    goldfile = os.path.join(gold_dir, "train.txt")
    corruptedfile = os.path.join(corrupted_dir, "train.txt")
    gold_labels = get_scores(goldfile)
    noisy_labels = get_scores(corruptedfile)

    pearson_corr = pearsonr(gold_labels, noisy_labels)[0]
    spearman_corr = spearmanr(gold_labels, noisy_labels)[0]
    print(pearson_corr, spearman_corr)
    logger.info("%s pearson_corr: %s, and spearman_corr: %s ", corrupted_dir, str(pearson_corr), str(spearman_corr))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                            default=None,
                            type=str,
                            required=True,
                            help="The input data dir.")
    parser.add_argument("--corruption_dir",
                            default=None,
                            type=str,
                            required=True,
                            help="corruption data directory.")
    parser.add_argument("--logfile_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the training log")
    args = parser.parse_args()
    logging.basicConfig(filename = args.logfile_dir,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    eval_rp(args.data_dir, args.corruption_dir)
