import os
import re
import argparse
import shutil
from omegaconf import OmegaConf

# from .model_cfg import default_decoder_cfg_v1, default_decoder_cfg_v3, mvq_args, default_encoder_cfg
import pdb


def is_empty_dir(path):
    for _, _, files in os.walk(path):  
        if files:
            return False  
    return True 

def copy_files_from_dir_to_dir(src_dir, dst_dir):  
    """  
    复制源文件夹内的所有文件到目标文件夹，不包括子文件夹和它们的文件。   
    """  
    for filename in os.listdir(src_dir):  
        src_file = os.path.join(src_dir, filename)  
        dst_file = os.path.join(dst_dir, filename)  
  
        if os.path.isfile(src_file):  
            shutil.copy2(src_file, dst_file)

def is_numeric(str):
    """
    print(is_numeric("123"))  # True
    print(is_numeric("3.14"))  # True
    print(is_numeric("-1.23e-4"))  # True
    print(is_numeric("abc"))  # False
    """
    if str is None or len(str) == 0:
        return False
    matchObj = re.fullmatch(r"[+-]?\d*(\.\d+)?([eE][+-]?\d+)?", str)
    return True if matchObj else False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Train the ImageBind model with PyTorch Lightning and LoRA.")
    parser.add_argument('--cfg', type=str, required=True, help="Path to the config file")
    parser.add_argument('--cfg_model', type=str, default='config/cfg_model.yaml')
    parser.add_argument('--train', type=str2bool, default=True, help='Initiate the training process')

    # parser.add_argument("--device", type=str, nargs='+', default="cpu", help="Device to use for training ('cpu' or list of 'cuda')")
    
    # Add any other argument to replace the yaml settings.
    # 
    """
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility")
    parser.add_argument("--datasets_dir", type=str, default="./.datasets", help="Directory containing the datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["dreambooth"], choices=["dreambooth", "nyu"],
                        help="Datasets to use for training and validation")
    parser.add_argument("--full_model_checkpoint_dir", type=str, default="./.checkpoints/full",
                        help="Directory to save the full model checkpoints")
    parser.add_argument("--full_model_checkpointing", action="store_true", help="Save full model checkpoints")
    parser.add_argument("--loggers", type=str, nargs="+", choices=["tensorboard", "wandb", "comet", "mlflow"],
                        help="Loggers to use for logging")
    parser.add_argument("--loggers_dir", type=str, default="./.logs", help="Directory to save the logs")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (Don't plot samples on start)")

    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--momentum_betas", nargs=2, type=float, default=[0.9, 0.95],
                        help="Momentum beta 1 and 2 for Adam optimizer")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for InfoNCE loss")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--self_contrast", action="store_true", help="Use self-contrast on the image modality")

    parser.add_argument("--only_eval", action="store_true", help="only_eval")
    parser.add_argument("--lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LoRA layers")
    parser.add_argument("--lora_checkpoint_dir", type=str, default="./.checkpoints/lora",
                        help="Directory to save LoRA checkpoint")
    
    parser.add_argument("--modality_eval", type=str,
                        choices=["vision", "text", "audio", "thermal", "depth", "imu"],
                        help="Modality to eval")
    
    parser.add_argument("--modality_pair", nargs="+", type=str, default=["vision", "text"],
                        choices=["vision", "text", "audio", "thermal", "depth", "imu"],
                        help="Modality names to apply LoRA")

    parser.add_argument("--lora_modality_names", nargs="+", type=str, default=["vision", "text"],
                        choices=["vision", "text", "audio", "thermal", "depth", "imu"],
                        help="Modality names to apply LoRA")
    
    parser.add_argument("--lora_freeze_modality_names", nargs="+", type=str, default=[],
                        choices=["vision", "text", "audio", "thermal", "depth", "imu"])
                        
    parser.add_argument("--lora_layer_idxs", nargs="+", type=int,
                        help="Layer indices to apply LoRA")
    parser.add_argument("--lora_layer_idxs_vision", nargs="+", type=int,
                        help="Layer indices to apply LoRA for vision modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_text", nargs="+", type=int,
                        help="Layer indices to apply LoRA for text modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_audio", nargs="+", type=int,
                        help="Layer indices to apply LoRA for audio modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_thermal", nargs="+", type=int,
                        help="Layer indices to apply LoRA for thermal modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_depth", nargs="+", type=int,
                        help="Layer indices to apply LoRA for depth modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_imu", nargs="+", type=int,
                        help="Layer indices to apply LoRA for imu modality. Overrides lora_layer_idxs if specified")

    parser.add_argument("--linear_probing", action="store_true",
                        help="Freeze model and train the last layers of the head for each modality.")
    """
    
    
    # return parser.parse_args()
    return parser.parse_known_args()


def parse_command_line_args_del(uk_cli):
    d_keys = uk_cli[:-1:2]
    d_keys = [i.replace('--', '') for i in d_keys]
    d_vals = uk_cli[1::2]
    d_vals = [int(s) if s.isnumeric() else s for s in d_vals]  # to int 
    d_vals = [None if s == 'None' else s for s in d_vals]      # 'None' to None    

    # uk_dict = dict(zip(d_keys, d_vals))
    uk_dict = dict()
    for ks, v in zip(d_keys, d_vals):
        if ks.split('.'):
            kv = v
            ik = ks.split('.')
            ik.reverse()
            for k in ik:
                kv = {k:kv}
        else:
            kv = {ks:v}
        uk_dict.update(kv)

    return uk_dict


def parse_command_line_args(uk_cli):
    print("uk_cli:", uk_cli)
    k_list = [i for i in  uk_cli if i.startswith('--')]
    # print(k_list)
    pk = None
    pv = None
    uk_dict = {}
    for i in uk_cli:
        if i.startswith('--'):
            nk = i.replace('--', '')
            if (pk is not None) and (pv is not None):
                pv = None if pv == 'None' else pv
                uk_dict.update({pk: pv})
            pk = nk
            pv = None
        else:
            if is_numeric(i):
                # i = int(i) if i.isnumeric() else i  # 整数
                # i = float(i) if 'e' in i else i  # 科学计数
                # i = float(i) if '.' in i else i  # 浮点数
                if ('.' in i) or ('e' in i):
                    i = float(i)
                elif i.isnumeric():
                    i = int(i)
            elif isinstance(i, str):
                if i.lower() == 'true':
                    i = True 
                elif i.lower() == 'false':
                    i = False
                # else i = i
                    
            if pv == None:
                pv = i
            else:
                if isinstance(pv, list):
                    pv.append(i)
                else:
                    pv = [pv, i]
    if (pk is not None) and (pv is not None):
        uk_dict.update({pk: pv})
    return uk_dict


# def get_model_cfg_file(uk_dict, cfg_yaml):
#     if uk_dict.get('cfg_model'):
#         return uk_dict.get('cfg_model')

#     if cfg_yaml.get('cfg_model'):
#         return cfg_yaml.get('cfg_model')
    
#     return 'config/cfg_model.yaml'
    

def build_model_cfg(parser_args, cfg_file='config/cfg_model.yaml'):
    # --- load model cfg from 'cfg_model.yaml'
    # abs_dir, _ = os.path.split(os.path.abspath(__file__))
    # cfg_model = OmegaConf.load(os.path.join(abs_dir, cfg_file))
    cfg_model = OmegaConf.load(cfg_file)

    cfg_encoder = cfg_model.get('encoder')
    cfg_decoder = cfg_model.get('decoder')
    cfg_vq = cfg_model.get('vector_quantise')
    
    modality_list = ['vision', 'audio', 'depth', 'thermal', 'imu', 'tactile']

    # update cfg_model based on parser_args
    for key, value in parser_args.items():
        if cfg_encoder is not None and key in cfg_encoder.keys():
            cfg_encoder.update({key: value})
            parser_args.pop(key)
        elif (cfg_decoder is not None) and (key in cfg_decoder.keys()):
            cfg_decoder.update({key: value})
            parser_args.pop(key)
        elif (cfg_vq is not None) and (key in cfg_vq.keys()):
            if key == 'codevector_dim':
                cfg_vq.get('codevector_dim').update({modality_name: value for modality_name in cfg_vq.get('codevector_dim').keys() if modality_name != 'text'})
                cfg_vq.update({'single_vq_embed_dim': value})
                parser_args.pop('codevector_dim')
            elif key == 'codevector_num':
                cfg_vq.get('codevector_num').update({'common': value[0]})
                cfg_vq.get('codevector_num').update({modality_name: value[1] for modality_name in cfg_vq.get('codevector_num').keys() if modality_name != 'common'})
                cfg_vq.update({'single_vq_embed_num': value[0]})
                parser_args.pop('codevector_num')
            else:
                cfg_vq.update({key: value})
                parser_args.pop(key)

    if not cfg_encoder.get('add_new_output_head'):
        # update output_embed_dim based on whether to use new head to distinguish common and specific info
        output_embed_dim_cfg = cfg_encoder.get('output_embed_dim')
        for d_k in modality_list:
            output_embed_dim_cfg.update({d_k: 0})
    
        # use single codebook when vq is avaiable without using new output head to distinguish common and specific info
        # if cfg_vq is not None:
        #     cfg_vq.update({'use_mvq': False})


    if cfg_decoder.get("use_decoder") and (cfg_decoder.get('input_embed_dim') is None):
        # 根据 encoder.output_embed_dim 推算出来decoder.input_embed_dim
        input_embed_dim_cfg = {}

        # if cfg_decoder.get('input_type') == 'encoder_trunk_feature':
        #     _k = 'trunk_embed_dim'
        #     common_dim = 0  #  该结构下，不适用dim_common + dim_specific
        # elif cfg_decoder.get('input_type') == 'encoder_output_feature':
        #     _k = 'output_embed_dim'
        #     common_dim = cfg_encoder.get(_k).get('common')
        _k = 'output_embed_dim'
        common_dim = cfg_encoder.get(_k).get('common')

        _enocder_embed_dim = cfg_encoder.get(_k)
        
        for d_k in modality_list:
            d_v = common_dim + _enocder_embed_dim.get(d_k)
            input_embed_dim_cfg.update({d_k: d_v})
        
        cfg_decoder.update({'input_embed_dim': input_embed_dim_cfg})

        # add trunk_embed_dim in encoder into decoder
        cfg_decoder.update({'encoder_trunk_embed_dim': cfg_encoder.get('trunk_embed_dim')})
    
    # 根据 encoder.output_embed_dim 来设定vector_quantise.input_vector_dim和vector_quantise.codebook_num
    if cfg_vq and cfg_vq.get('use_mvq') and (cfg_vq.get('input_vector_dim') is None):
        common_dim = cfg_encoder.get('output_embed_dim').get('common')
        codevector_dim = cfg_vq.get('codevector_dim')
        input_vector_dim_cfg = {}
        for k, v in cfg_encoder.get('output_embed_dim').items():
            if cfg_vq.get('seg_type') == 2:
                assert common_dim / codevector_dim.get('common') == v / codevector_dim.get(k)
            
            input_vector_dim_cfg.update({k: v})
        cfg_vq.update({'input_vector_dim': input_vector_dim_cfg})

        # codebook number
        codebook_num = 0
        for embed_dim in cfg_encoder.get('output_embed_dim').values():
            if embed_dim != 0:
                codebook_num += 1
        cfg_vq.update({'codebook_num': codebook_num})

    # 当decoder.input_type == 'encoder_output_feature'时，vq_all_token=True or False 对decoder的输入才有影响
    # 当decoder.input_type == 'encoder_trunk_feature'时，只需vq_all_token=False 
    # if cfg_vq and cfg_vq.get('vq_all_token') and cfg_decoder.get('input_type') == 'encoder_trunk_feature':
    #     cfg_vq.update({'vq_all_token': False})

    return parser_args, cfg_model
        
def build_dir_cfg(cfg_all):
    # pdb.set_trace()

    if cfg_all.get('loggers_dir') is None:
        cfg_all.update({"loggers_dir": f"{cfg_all.get('expname')}/log"})
    print(f"loggers_dir: {cfg_all.get('loggers_dir')}")
    os.makedirs(cfg_all.get('loggers_dir'), exist_ok=True)

    if cfg_all.get('log_img_dir') is None:
        cfg_all.update({"log_img_dir": f"{cfg_all.get('expname')}/log_img"})
    print(f"log_img_dir: {cfg_all.get('log_img_dir')}")
    os.makedirs(cfg_all.get('log_img_dir'), exist_ok=True)

    if cfg_all.get('checkpoint_dir') is None:
        cfg_all.update({'checkpoint_dir': f"{cfg_all.get('expname')}/checkpoints"})
    print(f"checkpoint_dir: {cfg_all.get('checkpoint_dir')}")
    os.makedirs(cfg_all.get('checkpoint_dir'), exist_ok=True)
      
    # if cfg_all.train_mode == "lora":
    #     if cfg_all.get(f"lora_checkpoint_dir") is None:
    #         cfg_all.update({"lora_checkpoint_dir": f"{cfg_all.get('checkpoint_dir')}/lora"})
    #     print(f"lora_checkpoint_dir: {cfg_all.get('lora_checkpoint_dir')}")
    #     os.makedirs(cfg_all.get('lora_checkpoint_dir'), exist_ok=True)
    #     cur_checkpoint_dir = cfg_all.get('lora_checkpoint_dir')
    # else:
    #     if cfg_all.get(f"full_checkpoint_dir") is None:
    #         cfg_all.update({"full_checkpoint_dir": f"{cfg_all.get('checkpoint_dir')}/full"})
    #     print(f"full_checkpoint_dir: {cfg_all.get('full_checkpoint_dir')}")
    #     os.makedirs(cfg_all.get('full_checkpoint_dir'), exist_ok=True)
    #     cur_checkpoint_dir = cfg_all.get('full_checkpoint_dir')
    # print(f"cur_checkpoint_dir: {cur_checkpoint_dir}")

    if cfg_all.get('load_checkpoint_dir') is not None:
        print(f"load checkpoint from {cfg_all.get('load_checkpoint_dir')}")
        cur_checkpoint_dir = cfg_all.get('checkpoint_dir')
        # copy existing checkpoint to current checkpoint dir
        if is_empty_dir(cur_checkpoint_dir):
            copy_files_from_dir_to_dir(cfg_all.get('load_checkpoint_dir'), cur_checkpoint_dir)

    if cfg_all.get('save_cfg_filepath') is None:
        cfg_all.update({'save_cfg_filepath': f"{cfg_all.get('expname')}/cfg_all_train.yaml"})
    
    # update expname to a full path
    cfg_all.update({'expname': os.path.dirname(cfg_all.get('save_cfg_filepath'))})


    return cfg_all 


def load_cfg():
    raw_args, uk_cli = parse_args()
    uk_dict = parse_command_line_args(uk_cli)

    # Create config from multiple sources.
    cfg_args = OmegaConf.create(vars(raw_args))     # From parser
    cfg_clin = OmegaConf.create(uk_dict)            # From command line arguments
    cfg_args = OmegaConf.merge(cfg_args, cfg_clin)  # merge command line and parser arguments
    cfg_yaml = OmegaConf.load(raw_args.cfg)         # From a YAML file
    if cfg_args.get('train'):
        if cfg_args.get('resume_expname') is not None:
            cfg_args.update({'load_checkpoint_dir': os.path.join(cfg_args.get('resume_expname'), 'checkpoints')}),  
        cfg_args = build_dir_cfg(cfg_args)
        cfg_args, cfg_model= build_model_cfg(cfg_args, raw_args.cfg_model) # override cfg_model with corresponding command line arguments
        cfg_all = OmegaConf.merge(cfg_yaml, cfg_args, cfg_model)  # Later arguments override earlier ones
        OmegaConf.save(cfg_all, cfg_all.get('save_cfg_filepath'))
    
    else:
        if cfg_args.get('resume_expname') is None:
            print("use imagebind pretrain in evaluation mode")
            cfg_args = build_dir_cfg(cfg_args)
            cfg_args, cfg_model= build_model_cfg(cfg_args, 'config/cfg_ablation/cfg_model_ablation_base.yaml')
            cfg_all = OmegaConf.merge(cfg_yaml, cfg_args, cfg_model)
            OmegaConf.save(cfg_all, cfg_all.get('save_cfg_filepath').replace('cfg_all_train', 'cfg_all_test'))

        else:
            resume_expname = cfg_args.get('resume_expname')
            resume_cfg_path = f"{resume_expname}/cfg_all_train.yaml"
            print(f"Resume cfg from {resume_cfg_path}")
            cfg_resume = OmegaConf.load(resume_cfg_path)
            cfg_all = OmegaConf.merge(cfg_resume, cfg_yaml, cfg_args)  # 需要cfg_yaml中的数据信息

            # 零时补丁：早期部分代码，lora保存在checkpoints，而不是checkpoints/lora
            if os.path.exists(os.path.join(resume_expname, 'checkpoints/lora')):
                lora_checkpoint_dir = os.path.join(resume_expname, 'checkpoints/lora')
            else:
                lora_checkpoint_dir = os.path.join(resume_expname, 'checkpoints')
                print(f"Set lora_checkpoint_dir = {lora_checkpoint_dir}")
                                                    
            cfg_all.update({'loggers': None, 
                            'train_mode': cfg_resume.get('train_mode'),  # cfg_all = OmegaConf.merge(cfg_resume, cfg_yaml, cfg_args)中会覆盖，故此处
                            'load_checkpoint_dir': os.path.join(resume_expname, 'checkpoints'),       # head tune
                            'lora_checkpoint_dir': lora_checkpoint_dir, # os.path.join(resume_expname, 'checkpoints/lora'),  # lora
                            }
                            )
            # save the test config file
            if cfg_all.get('expname') in resume_expname:
                save_cfg_filepath = f"{resume_expname}/cfg_all_test.yaml"  # train.yaml --> test.yaml
            else:
                save_cfg_filepath = f"{cfg_all.get('expname')}/cfg_all_test.yaml"
            OmegaConf.save(cfg_all, save_cfg_filepath)

        
    print(OmegaConf.to_yaml(cfg_all))
    print("# " * 20)
    return cfg_all