import options
import utils
from trainer import Validator

if __name__ == "__main__":

    print("=======================================================")
    print("Evaluate distance of 3D Point Cloud generation model.")
    print("=======================================================")

    cfg = options.get_arguments()
    cfg.batchSize = cfg.inputViewN
    # cfg.chunkSize = 50

    RESULTS_PATH = f"results/{cfg.model}_{cfg.experiment}"

    dataloaders = utils.make_data_fixed(cfg)
    test_dataset = dataloaders[1].dataset


    validator = Validator(cfg, test_dataset) 

    hist = validator.eval_dist().dropna()

    # find average distance accross all samples, for "pred->GT" and "GT->pred", ignoring "cad"
    avg_dist = hist[["pred->GT", "GT->pred"]].mean()
    print(f"Average distance (pred->GT): {avg_dist['pred->GT']}")
    print(f"Average distance (GT->pred): {avg_dist['GT->pred']}")

    hist.to_csv(f"{RESULTS_PATH}_testerror.csv", index=False)
