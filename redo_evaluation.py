# %%
import wandb
import os
from argparse import ArgumentParser
from evaluation import evaluate_model
import torch
from torch.utils.data import DataLoader
from monai.data import list_data_collate
import yaml
from utils import train_utils


parser = ArgumentParser()
parser.add_argument('--dataset',
                    default=None,)
parser.add_argument('--sweep_identifier', default=None)

# %%
def prepare_reeval(dataset, sweep_identifier):
    arguments = []
    entity = "topo_pitfalls"

    api = wandb.Api()
    project = api.project(f"{entity}")
    all_sweeps = project.sweeps()

    # filter all sweeps out that do not contain "drive"
    for sweep in all_sweeps:
        if not dataset in sweep.name or not f"_{sweep_identifier}" in sweep.name:
            continue
        
        best_run = sweep.best_run()
        print(sweep.name)
        if "topograph" in sweep.name:
            path_suffix = f"_Topograph_alpha_{best_run.config['alpha']}"
        elif "bm" in sweep.name and "wrong" in sweep.name:
            path_suffix = f"_FastMulticlassDiceBettiMatching_eight_connectivityTrue_alpha_{best_run.config['alpha']}"
        elif "bm" in sweep.name and not "wrong" in sweep.name:
            path_suffix = f"_FastMulticlassDiceBettiMatching_eight_connectivityFalse_alpha_{best_run.config['alpha']}"
        elif "cldice" in sweep.name:
            path_suffix = "_ClDice"
        elif "dice" in sweep.name and not "cldice" in sweep_identifier:
            path_suffix = "_Dice"
        elif "hutopo" in sweep.name:
            path_suffix = "_HuTopo"
        elif "mosin" in sweep.name:
            path_suffix = "_MOSIN"
        run_path = f"./models/cremi/sweep_{best_run.name}_{best_run.id}{path_suffix}"
        config_path = f"{run_path}/config.yaml"
        model_path = f"{run_path}/best_model_dict.pth"

        print("Path:", run_path)

        # get num_res_units from config
        num_res_units = best_run.config["num_res_units"]
        channels = best_run.config["channels"]

        #now write it to the config file overwriting the old line while keeping the rest
        with open(config_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "NUM_RES_UNITS" in line:
                    lines[i] = f"  NUM_RES_UNITS: {num_res_units}\n"
                if "CHANNELS" in line and not "IN_CHANNELS" in line and not "OUT_CHANNELS" in line:
                    lines[i] = f"  CHANNELS: {channels}\n"

        with open(config_path, "w") as f:
            f.writelines(lines)

        arguments.append((sweep.name, config_path, model_path, best_run.name, best_run.id))
    
    return arguments

def redo_evaluation(args, eval_arguments):
    csv_text = "sweep,dice,cldice,bm,bm0,bm1,b0,b1,topograph,voi,ari,run_name,run_id\n"

    for sweep_name, config_path, model_path, run_name, run_id in eval_arguments:

        # Load the config files
        with open(config_path) as f:
            print('\n*** Config file')
            print(config_path)
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = train_utils.dict2obj(config)

        # create UNet, DiceLoss and Adam optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on", device)

        args.integrate_test = True

        train_ds, val_ds, test_dataset = train_utils.binary_dataset_selection(config, args)

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=config.TRAIN.NUM_WORKERS,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
            sampler=None,
            drop_last=False
        ) 

        # Create model
        model = train_utils.select_model(config, device)

        # Start from pretrained model
        dic = torch.load(model_path, map_location=device)
        model.load_state_dict(dic['model'], strict=True)
        
        dice_score, clDice_score, b0, b1, bm0, bm1, bm, norm_bm, topograph_score, voi_score, adapted_rand_score = evaluate_model(
            model, 
            test_loader, 
            device,
            config.DATA.INCLUDE_BACKGROUND,
            config.LOSS.EIGHT_CONNECTIVITY,
            config=config,
            logging=False,
            mask_background=False,
            reproducible=True,
            limited_eval=True
        )

        print("testing model at step", dic['scheduler']['last_epoch'])

        csv_text += f"{sweep_name},{dice_score},{clDice_score},{bm},{bm0},{bm1},{b0},{b1},{topograph_score},{voi_score},{adapted_rand_score},{run_name},{run_id}\n"

    # Ensure the directory exists
    output_dir = os.path.join("reevaluated", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # save csv as text
    with open(os.path.join("reevaluated", args.dataset, f"{args.sweep_identifier}.csv"), "w") as f:
        f.write(csv_text)
# %%

if __name__ == "__main__":
    args = parser.parse_args()

    arguments = prepare_reeval(args.dataset, args.sweep_identifier)

    # reverse order of list
    arguments = arguments[::-1]

    redo_evaluation(args, arguments)

