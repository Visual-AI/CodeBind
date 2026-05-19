from typing import Optional, List, Dict
import torch
from tqdm import tqdm
from models.codebind_model import ModalityType, CodeBindModel
from datasets import data
import pdb


def get_retrieval_embeds(retrive_data, retrive_modality, retrieval_model, vector_quantiser=None, device=None):

    if device is None:
        device = next(retrieval_model.parameters()).device

    with torch.no_grad():
        retrive_bs = 20 # retrive data batch size
        retrive_feats = []  
        for i in tqdm(range(0, len(retrive_data), retrive_bs)):
            retrive_batch_data = retrive_data[i: min(len(retrive_data), i+retrive_bs)]
            if retrive_modality in ['video', 'vision']:
                retrive_batch_data = data.load_and_transform_video_data(retrive_batch_data, device)
            elif retrive_modality == 'audio':
                retrive_batch_data = data.load_and_transform_audio_data(retrive_batch_data, device)
            else:
                raise NotImplementedError
            
            feats_r = retrieval_model.model({retrive_modality: retrive_batch_data})
            feats_r_tensor = feats_r.get(retrive_modality)  # [retrive_bs, 1024]
            if vector_quantiser is not None:
                if vector_quantiser.__class__.__name__ == "VectorQuantizer":
                    vq_dict = vector_quantiser(feats_r_tensor)
                    feats_r_tensor = vq_dict.get('q')
                else:
                    vq_dict = vector_quantiser(feats_r_tensor, retrive_modality)
                    feats_r_tensor = vq_dict.get('common')
            
            if retrieval_model.model_postprocessors is not None:
                feats_r_tensor = retrieval_model.model_postprocessors[retrive_modality](feats_r_tensor)
            
            feats_r_tensor /= feats_r_tensor.norm(dim=-1, keepdim=True)
            retrive_feats.append(feats_r_tensor)
        retrive_feats = torch.cat(retrive_feats, dim=0) # len(retrive_data), 1024

    return retrive_feats

class CrossModalityRetrieval:
    def __init__(self,
                 classifier_model: CodeBindModel,
                 retrieval_modality_type: str,
                 retrieval_data: Optional[List[str]] = None,
                 device: Optional[torch.device] = None, 
                 ): 
        """ 
        """

        self.classifier_model = classifier_model
        self.device = next(classifier_model.parameters()).device if device is None else device

        self.vector_quantiser = classifier_model.modality_vq
        if self.vector_quantiser is not None:
            print("Set vector_quantiser for CrossModalityRetrieval")
        else:
            print("Set vector_quantiser=None for CrossModalityRetrieval")

        # get other modality embedding in t2o retrieval
        if retrieval_data is None:
            if retrieval_modality_type in ['vision', 'audio']:
                self.retrieval_data = classifier_model.trainer.val_dataloaders.dataset.data_paths
            else:
                raise NotImplementedError(f"retrieval evaluation not Implemented for {retrieval_modality_type} ")
        else:
            self.retrieval_data = retrieval_data
        print(f"Get {len(self.retrieval_data)} retrieval_data.")

        self.retrive_feats = get_retrieval_embeds(self.retrieval_data, retrieval_modality_type, classifier_model, 
                                                  self.vector_quantiser)

        self.set_modality_type(retrieval_modality_type)
        

    def set_modality_type(self, modality_type):
        self.retrieval_modality_type = modality_type
        print("Init CrossModalityRetrieval for ModalityType:", self.retrieval_modality_type)


    def get_batch_recall(self, batch_data, batch_data_class, gt_name, topk=[1, 5, 10], vector_quantiser=None):
        # implement text to other modality retrieval (t2o)
        # get other modality data from datasets and get text data from batch data

        assert batch_data_class != self.retrieval_modality_type
        # get batch data embedding (i.e. text embedding in t2o retrieval)
        feats_b = self.classifier_model.model({batch_data_class: batch_data})
        feats_b_tensor = feats_b.get(batch_data_class)  # bs, 1024
        
        if vector_quantiser is not None:
            if self.classifier_model.cfg_vq.get('use_mvq'):
                vq_dict_b = vector_quantiser(feats_b_tensor, batch_data_class)
                feats_b_tensor = vq_dict_b.get('common')
            else:
                vq_dict_b = vector_quantiser(feats_b_tensor)
                feats_b_tensor = vq_dict_b.get('q')
        elif self.classifier_model.cfg_encoder.get('add_new_output_head'):
            common_dim = self.classifier_model.cfg_encoder.get('output_embed_dim').get('common')
            feats_b_tensor = feats_b_tensor[:, :common_dim]
        
        if self.classifier_model.model_postprocessors is not None:
            feats_b_tensor = self.classifier_model.model_postprocessors[batch_data_class](feats_b_tensor)
        
        batch_feats = feats_b_tensor / feats_b_tensor.norm(dim=-1, keepdim=True)

        # --- retrive the topk closeset retrieval data as the retrieval result of the input batch data
        sim_score = 100. * batch_feats @ self.retrive_feats.t()

        # prepare the matching groundtruth
        gt_idx = torch.tensor(gt_name).to(self.device)

        # get recall@k within a batch
        _, pred_idx = sim_score.topk(max(topk), dim=1, largest=True, sorted=True)  # 'torch.return_types.topk', namedtuple of (values, indices) 
        pred_idx = pred_idx.t()
        correct = pred_idx.eq(gt_idx.reshape(1, -1).expand_as(pred_idx))
        res = []
        for k in topk:
            # correct_k = correct[:k].reshape(-1).float().sum(0)
            correct_k = correct[:k].any(dim=0).float().sum(0)
            res.append(correct_k)
        res = torch.stack(res)
        
        # recall = 0
        # for i in range(len(gt_idx)):
        #     if int(gt_idx[i]) in pred_idx[i, :]:
        #         recall += 1

        return res
    
    def get_retrieval_recall(self, input_dataloader, top_k: list = [1, 5, 10]):

        recall = torch.zeros(len(top_k)).to(self.device)
        number_of_examples = len(input_dataloader.dataset)

        for input_dict in tqdm(input_dataloader):
            data_in = input_dict.get('text')
            data_in = data_in.to(self.device)
            class_in = 'text'
            assert 'retrieval_id' in input_dict.keys(), 'retrieval_id must be provided in dataset for retrieval task'
            retrieval_id = input_dict.get('retrieval_id')
            recall += self.get_batch_recall(data_in, class_in, retrieval_id, topk=top_k, vector_quantiser=self.vector_quantiser)

        recall = recall / number_of_examples * 100
        [print(f"Recall {i} {acc:.3f}%", end="  |") for i, acc in enumerate(recall)]
        # print(f'recall = {recall:.2f}%')