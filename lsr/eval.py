from pprint import pprint
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pprint import pprint
import logging
import wandb
import os
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
logger = logging.getLogger(__name__)

LSR_OUTPUT_KEY = "LSR_OUTPUT_PATH"
ANSERINI_OUTPUT_KEY = "ANSERINI_OUTPUT_PATH"


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def eval(conf: DictConfig):
    hydra_cfg = HydraConfig.get()
    experiment_name = hydra_cfg.runtime.choices["experiment"]
    run_name = f"{experiment_name}"
    output_root = os.getenv(LSR_OUTPUT_KEY)
    if output_root is not None:
        output_dir = Path(output_root)/run_name
    else:
        output_dir = Path("./outputs")/run_name
    if not output_dir.is_dir():
        output_dir.mkdir()
    anserini_root = os.getenv(ANSERINI_OUTPUT_KEY)
    if anserini_root is not None:
        anserini_output_dir = Path(anserini_root)/experiment_name
    else:
        anserini_output_dir = Path("./anserini-outputs")/experiment_name
    if not anserini_output_dir.is_dir():
        anserini_output_dir.mkdir()

    with open_dict(conf):
        conf.training_arguments.output_dir = str(output_dir)
        conf.training_arguments.run_name = run_name
        conf.index.anserini_output_path = str(anserini_output_dir)
        del conf.trainer.train_dataset
        del conf.trainer.eval_dataset

    resolved_conf = OmegaConf.to_container(conf, resolve=True)
    resolved_conf = OmegaConf.to_container(conf, resolve=True)
    pprint(resolved_conf)
    os.environ["WANDB_PROJECT"] = conf.wandb.setup.project
    wandb.init(
        group=run_name,
        job_type="eval",
        config=resolved_conf,
        resume=conf.wandb.resume,
        settings=wandb.Settings(start_method="fork"),
    )
    logger.info(f"Working directiory: {os.getcwd()}")
    trainer = instantiate(conf.trainer)
    test_dataset = instantiate(conf.test_dataset)
    model_dir = output_dir/"model"
    trainer._load_from_checkpoint(model_dir)
    trainer.predict(test_dataset)
    wandb.finish()


if __name__ == "__main__":
    eval()
