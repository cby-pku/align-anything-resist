# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trainer for supervised training."""


import argparse
import os
import sys
from typing import Any, List

import deepspeed
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_to_text.supervised import SupervisedBatch, SupervisedDataset
from align_anything.evaluation.paloma_collapse import PalomaCollapseConfig, PalomaCollapseEvaluator
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.trainers.base import SupervisedTrainerBase
from align_anything.utils.device_utils import get_current_device, torch_gc, torch_set_device
from align_anything.utils.multi_process import get_current_device, is_main_process
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    namedtuple_to_dict,
    prepare_ds_train_cfgs,
    read_cfgs,
    seed_everything,
    update_dict,
)


class SupervisedTrainer(SupervisedTrainerBase):

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize the SFT trainer."""
        self.cfgs = cfgs
        self.ds_train_cfgs = prepare_ds_train_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.global_step = 0
        self.infer_batch = lambda batch: {k: v for k, v in batch.items() if k != 'meta_info'}

        self.init_check()
        dist.barrier()
        self.init_models()
        if hasattr(self.model, 'infer_batch'):
            self.infer_batch = self.model.infer_batch
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()
        self._init_collapse_eval()

    def init_check(self) -> None:
        """Initial configuration checking."""
        super().init_check()

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs is not None and self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_train_cfgs)
        self.bnb_cfgs = self.cfgs.bnb_cfgs
        self.lora_cfgs = self.cfgs.lora_cfgs
        self.model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=True,
            bnb_cfgs=self.bnb_cfgs,
            lora_cfgs=self.lora_cfgs,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            SupervisedDataset, SupervisedDataset
        )

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.init_deepspeed_engines()

    def _init_collapse_eval(self) -> None:
        """Prepare PALOMA collapse evaluation."""
        self.collapse_eval_cfg: PalomaCollapseConfig | None = None
        self.collapse_eval_runner: PalomaCollapseEvaluator | None = None
        self.collapse_eval_steps: List[int] = []
        self.collapse_eval_at_start: bool = False

        raw_cfg = getattr(self.cfgs, 'collapse_eval_cfgs', None)
        if raw_cfg is None:
            return

        cfg = PalomaCollapseConfig.from_dict(namedtuple_to_dict(raw_cfg))
        self.collapse_eval_cfg = cfg
        if not cfg.enabled:
            return
        if not self.cfgs.logger_cfgs.output_dir:
            raise ValueError(
                'collapse_eval_cfgs.enabled is True but logger_cfgs.output_dir is not set.'
            )

        total_steps = max(0, self.cfgs.train_cfgs.epochs * len(self.train_dataloader))
        scheduled_steps = set()
        for raw_step in cfg.steps or []:
            if raw_step == 0:
                self.collapse_eval_at_start = True
            elif raw_step > 0:
                scheduled_steps.add(int(raw_step))

        num_passes = int(cfg.num_passes or 0)
        if num_passes > 0 and total_steps > 0:
            interval = max(1, total_steps // num_passes)
            for idx in range(1, num_passes + 1):
                step_val = interval * idx
                if step_val <= total_steps:
                    scheduled_steps.add(step_val)

        self.collapse_eval_steps = sorted(scheduled_steps)

        if is_main_process():
            self.collapse_eval_runner = PalomaCollapseEvaluator(
                cfg=cfg,
                output_root=self.cfgs.logger_cfgs.output_dir,
                logger=None,
            )
            if self.collapse_eval_steps:
                self.logger.print(
                    f'PALOMA collapse evaluation scheduled at steps: {self.collapse_eval_steps}'
                )
            if self.collapse_eval_at_start:
                self.logger.print('PALOMA collapse evaluation will run before training starts.')

    def loss(self, sft_batch: SupervisedBatch) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs = self.model(**self.infer_batch(sft_batch))
        return {'loss': outputs.loss}

    def train_step(self, sft_batch: SupervisedBatch) -> dict[str, Any]:
        """Performs a single training step."""
        loss = self.loss(sft_batch)['loss']
        self.model.backward(loss)
        self.model.step()

        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }

    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.cfgs.train_cfgs.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.cfgs.train_cfgs.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )
        progress_bar.update(self.global_step)

        if self.cfgs.data_cfgs.eval_datasets:
            self.logger.log(self.eval(), step=0)
        if self.collapse_eval_at_start and self.collapse_eval_cfg and self.collapse_eval_cfg.enabled:
            self._run_collapse_eval(step=0)

        remain_epoch = self.cfgs.train_cfgs.epochs - (
            self.global_step // len(self.train_dataloader)
        )

        start_batch_idx = self.global_step % len(self.train_dataloader)

        for epoch in range(int(remain_epoch)):
            self.model.train()
            progress_bar.set_description(
                f'Resuming from checkpoint {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
            )

            for batch_idx, batch in enumerate(self.train_dataloader):
                if epoch == 0 and batch_idx < start_batch_idx:
                    continue

                info = self.train_step(batch)
                torch_gc()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)

                save_interval = (
                    self.cfgs.train_cfgs.epochs
                    * len(self.train_dataloader)
                    // self.cfgs.logger_cfgs.save_total_limit
                )
                if self.global_step % save_interval == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    self.save(tag=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.cfgs.data_cfgs.eval_datasets
                    and self.cfgs.train_cfgs.eval_strategy == 'steps'
                    and self.global_step % self.cfgs.train_cfgs.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)

                if self._should_run_collapse_eval(self.global_step):
                    self._run_collapse_eval(step=self.global_step)

            if self.cfgs.data_cfgs.eval_datasets and self.cfgs.train_cfgs.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.cfgs.train_cfgs.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)
            self.model.tput_timer.update_epoch_count()

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return {}
        if not isinstance(self.eval_dataloader, dict):
            named_eval_dataloader = {self.cfgs.data_cfgs.eval_template: self.eval_dataloader}
        else:
            named_eval_dataloader = self.eval_dataloader

        self.model.eval()
        if self.cfgs.train_cfgs.gradient_checkpointing and not self.lora_enabled:
            self.model.gradient_checkpointing_disable()

        loss_logger = {}

        for template, raw_eval_dataloader in named_eval_dataloader.items():
            eval_dataloader = tqdm(
                raw_eval_dataloader,
                desc=f'Evaluating {template}',
                disable=not is_main_process(),
                position=1,
                leave=False,
            )
            batch = None
            eval_loss = []
            for batch in eval_dataloader:
                loss = self.loss(batch)['loss']
                eval_loss.append(loss.item())

                if batch is None:
                    self.logger.print(f'WARNING: `{template}` eval_dataloader is empty.')
                    return {}
            if len(eval_loss) > 0:
                loss_logger[f'eval/loss/{template}'] = sum(eval_loss) / len(eval_loss)

        self.model.train()
        if self.cfgs.train_cfgs.gradient_checkpointing and not self.lora_enabled:
            self.model.gradient_checkpointing_enable()
        return loss_logger

    def _should_run_collapse_eval(self, step: int) -> bool:
        return (
            self.collapse_eval_cfg is not None
            and self.collapse_eval_cfg.enabled
            and step in self.collapse_eval_steps
        )

    def _run_collapse_eval(self, step: int) -> None:
        if not self.collapse_eval_cfg or not self.collapse_eval_cfg.enabled:
            return

        tag = f'collapse_tmp_{step}'
        if is_main_process():
            self.logger.print(
                f'\n***** Running PALOMA collapse evaluation at step {step} *****'
            )

        self.save_transformers(model=self.model, tag=tag)
        checkpoint_dir = os.path.join(
            self.cfgs.logger_cfgs.output_dir,
            f'slice_{tag}',
        )

        cleanup_flag = False
        if is_main_process() and self.collapse_eval_runner is not None:
            try:
                self.collapse_eval_runner.evaluate(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                )
            finally:
                cleanup_flag = True

        dist.barrier()

        if cleanup_flag:
            PalomaCollapseEvaluator.cleanup_checkpoint(checkpoint_dir)

        dist.barrier()

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        self.save_transformers(model=model, tag=tag)


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch_set_device(current_device)

    # get custom configs from command line (需要先解析以获取 config_name)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config_name',
        type=str,
        default='sft',
        help='Name of the config file to use (e.g., sft, sft_resist, sft_main_process)',
    )
    args, unknown_args = parser.parse_known_args()

    # read default configs from the yaml file
    task = os.path.join('text_to_text', args.config_name)
    dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)

    # process remaining custom configs from command line
    override_args = {}
    idx = 0
    while idx < len(unknown_args):
        token = unknown_args[idx]
        if token == '--':
            idx += 1
            continue
        if not token.startswith('--'):
            idx += 1
            continue
        key = token[2:]
        value = 'True'
        if idx + 1 < len(unknown_args) and not unknown_args[idx + 1].startswith('--'):
            value = unknown_args[idx + 1]
            idx += 1
        override_args[key] = value
        idx += 1

    for k, v in override_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    # finetune the model
    trainer = SupervisedTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
