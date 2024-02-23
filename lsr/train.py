from pprint import pprint
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pprint import pprint
import logging
import wandb
import os
from hydra.core.hydra_config import HydraConfig
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def train(conf: DictConfig):
    hydra_cfg = HydraConfig.get()
    experiment_name = hydra_cfg.runtime.choices["experiment"]
    run_name = f"{experiment_name}"
    output_dir = f"./outputs/{run_name}"
    with open_dict(conf):
        conf.training_arguments.output_dir = output_dir
        conf.training_arguments.run_name = run_name
    resolved_conf = OmegaConf.to_container(conf, resolve=True)
    resolved_conf = OmegaConf.to_container(conf, resolve=True)
    pprint(resolved_conf)
    os.environ["WANDB_PROJECT"] = conf.wandb.setup.project
    wandb.init(
        group=conf.exp_name,
        job_type="train",
        config=resolved_conf,
        resume=conf.wandb.resume,
        settings=wandb.Settings(start_method="fork"),
    )
    logger.info(f"Working directiory: {os.getcwd()}")
    trainer = instantiate(conf.trainer)
    trainer.train(conf.resume_from_checkpoint)
    trainer.save_model()
    wandb.finish()


if __name__ == "__main__":
    train()
