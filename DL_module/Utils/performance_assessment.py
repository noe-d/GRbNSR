import sys
sys.path.append('./')

from Utils.tasks import *
import argparse
from Configs.configs_parser import ConfigParser


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%        %%%%        %%        %%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%  %%%%%%  %%     %%%%%      %%%%%%%%
# %%  %%%%%  %%%  %%%%%%%%  %%%%%%%%%%%%
# %%        %%%%        %%  %%%%%%%%  %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
_LOGGER_NAME = "train"
_DEFAULT_CONFIG = "Configs/config_files/config_graphmae_repro.json"
_SELECTED_TUDATA = [
    "REDDIT-BINARY",
    "COLLAB",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "PROTEINS",
    "MUTAG"
]
_DEFAULT_RUN_IDS = [
    0,
    1,
    2,
    3,
    4
]
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




def main(
    save_dir:str,
    run_ids:list,
    assess_untrained:bool=True,
    datasets:list=_SELECTED_TUDATA,
    **kwargs,
):
    if save_dir.endswith(".json"):
        for data in datasets:
            model_test = testmodel_dataset(
                model=save_dir,
                dataset=data,
                **kwargs
            )
    
    else:
        for run_id in run_ids:
            path_to_dir = save_dir+str(run_id)+"/"

            # assess performance from model_best.pth
            for data in datasets:
                model_test = testmodel_dataset(
                    model=path_to_dir+"model_best.pth",
                    dataset=data,
                    **kwargs
                )

                model_test.write_results_csv(
                    path_to_csv = save_dir+"best_results.csv",
                    data_name = data,
                    run_id=run_id
                )

                # assess performance of model_untrained.pth ?
                if assess_untrained:
                    model_test = testmodel_dataset(
                        model=path_to_dir+"model_untrained.pth",
                        dataset=data,
                        **kwargs
                    )

                    model_test.write_results_csv(
                        path_to_csv = save_dir+"untrained_results.csv",
                        data_name = data,
                        run_id=run_id
                    )
    


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='')
    args.add_argument('-s'
                      , '--save_dir'
                      , default=_DEFAULT_CONFIG
                      , type=str
                      , help='path to saved models directory (default: {})'.format(_DEFAULT_CONFIG)
                     )
    args.add_argument('-r'
                      , '--run_ids'
                      , nargs='+'
                      , default=_DEFAULT_RUN_IDS
                      , help='run ids (default: {})'.format(_DEFAULT_RUN_IDS)
                     )
    args.add_argument('-d'
                      , '--datasets'
                      , nargs='+'
                      , default=_SELECTED_TUDATA
                      , help='datasets to evaluate (default: {})'.format(_SELECTED_TUDATA)
                     )
    args.add_argument('-u'
                      , '--assess_untrained'
                      , action='store_false'
                      , help="assess untrained model ? (default: {})".format(True)
                     )
    args.add_argument('-j'
                      , '--from_json'
                      , action='store_true'
                      , help="Load model from json (default: {})".format(True)
                     )
    args = args.parse_args()
    
    if args.save_dir.endswith(".json") and not args.from_json:
        config = ConfigParser.from_json(
            json_path = args.save_dir,
            write_config=False,
            #run_id=""
        )
        save_dir = "/".join(str(config._save_dir).split("/")[:-1])+"/"
    else:
        save_dir = args.save_dir

    main(
        save_dir=save_dir,
        run_ids = args.run_ids,
        datasets = args.datasets,
        assess_untrained = args.assess_untrained
    )