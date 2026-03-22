# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# openpi model configs

import os

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    actor_train_config = get_openpi_config(config_name, model_path=cfg.model_path)
    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_config_kwargs = cfg.openpi
    if override_config_kwargs is not None:
        for key, val in override_config_kwargs.items():
            actor_model_config.__dict__[key] = val
    # load model
    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        for weight_path in weight_paths:
            safetensors.torch.load_model(model, weight_path, strict=False)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats = None
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
# Copyright 2025 The RLinf Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# openpi model configs

# import os
# import glob
# import torch
# import safetensors.torch
# from omegaconf import DictConfig

# def get_model(cfg: DictConfig, torch_dtype=None):
#     import openpi.shared.download as download
#     import openpi.transforms as transforms
#     from openpi.training import checkpoints as _checkpoints

#     from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
#     from rlinf.models.embodiment.openpi.openpi_action_model import (
#         OpenPi0Config,
#         OpenPi0ForRLActionPrediction,
#     )

#     # 1. 解析配置
#     config_name = getattr(cfg.openpi, "config_name", None)
#     actor_train_config = get_openpi_config(config_name, model_path=cfg.model_path)
#     actor_model_config = actor_train_config.model
#     actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    
#     override_config_kwargs = cfg.openpi
#     if override_config_kwargs is not None:
#         for key, val in override_config_kwargs.items():
#             actor_model_config.__dict__[key] = val

#     # 2. 初始化模型结构
#     model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
#         actor_model_config
#     )

#     if actor_model_config.train_expert_only:
#         model.freeze_vlm()

#     # 3. 准备权重路径
#     checkpoint_dir = download.maybe_download(str(cfg.model_path))
#     full_weights_path = os.path.join(checkpoint_dir, "model_state_dict", "full_weights.pt")
#     actor_full_weights_path = os.path.join(checkpoint_dir, "actor", "model_state_dict", "full_weights.pt")

#     # 定义安全加载函数：核心修复点
#     def safe_load_state_dict(target_model, state_dict):
#         model_dict = target_model.state_dict()
#         filtered_dict = {}
#         for k, v in state_dict.items():
#             if k in model_dict:
#                 if v.shape == model_dict[k].shape:
#                     filtered_dict[k] = v
#                 else:
#                     print(f"[RLinf Warning] 维度不匹配，跳过层: {k} "
#                           f"(权重文件: {list(v.shape)} -> 当前模型: {list(model_dict[k].shape)})")
#             else:
#                 # 忽略权重文件中多余的键
#                 pass
        
#         # 加载匹配的部分，不匹配的部分（如 ValueHead）将保持随机初始化
#         target_model.load_state_dict(filtered_dict, strict=False)
#         print(f"[RLinf Info] 成功从 SFT 权重加载了 {len(filtered_dict)} 个参数层。")

#     # 4. 执行权重加载逻辑
#     if os.path.exists(full_weights_path):
#         print(f"正在从 PT 文件加载: {full_weights_path}")
#         sd = torch.load(full_weights_path, map_location="cpu")
#         safe_load_state_dict(model, sd)
        
#     elif os.path.exists(actor_full_weights_path):
#         print(f"正在从 Actor PT 文件加载: {actor_full_weights_path}")
#         sd = torch.load(actor_full_weights_path, map_location="cpu")
#         safe_load_state_dict(model, sd)
        
#     else:
#         print(f"正在从 Safetensors 文件夹加载: {checkpoint_dir}")
#         weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
#         if not weight_paths:
#             weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        
#         combined_sd = {}
#         for wp in weight_paths:
#             combined_sd.update(safetensors.torch.load_file(wp))
#         safe_load_state_dict(model, combined_sd)

#     # 5. 后置处理与数据转换
#     model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    
#     data_config = actor_train_config.data.create(
#         actor_train_config.assets_dirs, actor_model_config
#     )
    
#     norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

#     # 6. 设置 Wrappers
#     repack_transforms = transforms.Group()
#     model.setup_wrappers(
#         transforms=[
#             *repack_transforms.inputs,
#             transforms.InjectDefaultPrompt(None),
#             *data_config.data_transforms.inputs,
#             transforms.Normalize(
#                 norm_stats, use_quantiles=data_config.use_quantile_norm
#             ),
#             *data_config.model_transforms.inputs,
#         ],
#         output_transforms=[
#             *data_config.model_transforms.outputs,
#             transforms.Unnormalize(
#                 norm_stats, use_quantiles=data_config.use_quantile_norm
#             ),
#             *data_config.data_transforms.outputs,
#             *repack_transforms.outputs,
#         ],
#     )

#     return model
