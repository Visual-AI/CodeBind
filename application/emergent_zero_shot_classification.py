"""
# Emergent zero-shot classification

# 80 templates from CLIP:
# https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
"""
from typing import Optional, List
import torch
from tqdm import tqdm
from models.codebind_model import ModalityType, CodeBindModel
from datasets import data
import numpy as np
import pdb
from sklearn import metrics

from datasets.nyu import nyu_other_18_semantic_class


def get_zeroshot_classifier(classnames, templates, text_model, vector_quantiser=None, device=None):

    if device is None:
        device = next(text_model.parameters()).device
    print('get_zeroshot_classifier: device =', device, "Num of classnames", len(classnames))

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class

            # TODO Processing 80 templates in a single batch requires 8G of GPU memory. 
            # To minimize memory usage, it is recommended to process templates in smaller batches 
            # rather than processing all 80 at once.
            # 
            # for t in texts: 
            #     data.load_and_transform_text([t], device)
            #     ...

            texts = data.load_and_transform_text(texts, device)

            input_dict = {ModalityType.TEXT: texts}
            class_embedding = text_model.model(input_dict).get(ModalityType.TEXT)
            # pdb.set_trace()
            use_vq = True
            if use_vq and vector_quantiser is not None:
                if vector_quantiser.__class__.__name__ == "VectorQuantizer":
                    vq_dict = vector_quantiser(class_embedding)
                    class_embedding = vq_dict.get('q')
                else:
                    vq_dict = vector_quantiser(class_embedding, ModalityType.TEXT)
                    class_embedding = vq_dict.get('common')
            
            if text_model.model_postprocessors is not None:
                class_embedding = text_model.model_postprocessors[ModalityType.TEXT](class_embedding)

            class_embedding = class_embedding.mean(dim=0)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)

            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)  # shape = [ebd_dims, class_nums]
    return zeroshot_weights

def map_calculation(outputs, targets):
    """
    mAP calculation using sklearn package
    Args:
        outputs: network outputs, predicted probability for each class  (n_samples, n_classes)
        targets: onehot matrix, groundtruth label for each class (n_samples, n_classes)

    Returns: mAP score in percentage (%)

    """
    # Class-wise statistics
    ap_class = []
    # print("map_calculation: targets.shape =", targets.shape)
    for k in range(outputs.size(1)):
        # Average precision
        avg_precision = metrics.average_precision_score(
            targets[:, k], outputs[:, k], average=None)
        ap_class.append(avg_precision)

    map = np.mean(ap_class) * 100
    return map

class EmergentZeroShotClassifier:
    def __init__(self,
                 classifier_model: CodeBindModel,
                 classnames: Optional[List[str]] = None,
                 dataset_name: Optional[str] = None,
                 modality_type: Optional[str] = None,
                 device: Optional[torch.device] = None, 
                 text_template = None,
                 ): # 
        """ classnames: list of class names
            classifier_model: model used to get text embedding of each class name, 
                              and get embedding of other modality
        """

        # index_to_class_name_dict = {}
        # pdb.set_trace()
        if classnames is None:
            self.classnames = classifier_model.trainer.val_dataloaders.dataset.class_names  # 因数据集而不同
        else:
            self.classnames = classnames
        self.classifier_model = classifier_model
        if dataset_name is None:
            self.dataset_name = classifier_model.trainer.val_dataloaders.dataset.dataset_name
        else:
            self.dataset_name = dataset_name
        self.device = next(classifier_model.parameters()).device if device is None else device
        self.vector_quantiser = classifier_model.modality_vq
        if self.vector_quantiser is not None:
            print("Set vector_quantiser for EmergentZeroShotClassifier")
        else:
            print("Set vector_quantiser=None for EmergentZeroShotClassifier")

        # get text embedding for zeroshot text classification
        text_template_name = text_template if text_template is not None else 'imagenet'
        print(f"Use {text_template_name} text_template")
        self.text_template = data.text_template_dict.get(text_template_name)
        self.classifier = get_zeroshot_classifier(self.classnames, self.text_template, classifier_model, self.vector_quantiser, device)
        self.intramodal_vector = None
        self.set_modality_type(modality_type)

        # adjust topk so that it is smaller than the number of classes
        self.top_k = [1, 5, 10]
        while self.top_k and max(self.top_k) >= len(self.classnames):  
            self.top_k.pop()  

    def set_classifier(self, input_dataloader, intramodal_vector='common'):
        self.intramodal_vector = intramodal_vector # 'common' or 'specific' or 'common_and_specific'
        self.classifier = get_intramodal_classifier(self.classnames, input_dataloader, self.classifier_model, self.vector_quantiser, self.device,
                                                    intramodal_vector=intramodal_vector)

    def set_modality_type(self, modality_type):
        self.modality_type = modality_type
        print("Init EmergentZeroShotClassifier for ModalityType:", self.modality_type)

    def get_batch_classification_acc(self, data_b, class_b, gt_name, vector_quantiser=None):
 

        feats_b = self.classifier_model.model({class_b: data_b})
        feats_b_tensor = feats_b.get(self.modality_type)  # bs, 1024
        # implement for self contrast: data_b: bs n_views C H W for vision, depth; bs 1 77 for text
        # feats_b = [model({class_b: data_b_i}) for data_b_i in data_b]
        # feats_b_tensor = torch.cat([feats_b_i.get(class_b) for feats_b_i in feats_b], dim=0) # bs, 1024
        if vector_quantiser is not None:
            if self.classifier_model.cfg_vq.get('use_mvq'):
                vq_dict_b = vector_quantiser(feats_b_tensor, class_b)
                feats_b_tensor = vq_dict_b.get('common')
                # debug 
                if self.intramodal_vector:
                    if self.intramodal_vector == 'common':
                        feats_b_tensor = vq_dict_b.get('common')
                    elif self.intramodal_vector == 'specific':
                        feats_b_tensor = vq_dict_b.get('specific')
                    elif self.intramodal_vector == 'common_and_specific':
                        feats_b_tensor = vq_dict_b.get('concat')
                    else:
                        raise ValueError()
                
            else:
                vq_dict_b = vector_quantiser(feats_b_tensor)
                feats_b_tensor = vq_dict_b.get('q')
        elif self.classifier_model.cfg_encoder.get('add_new_output_head'):
            common_dim = self.classifier_model.cfg_encoder.get('output_embed_dim').get('common')
            feats_b_tensor = feats_b_tensor[:, :common_dim]
        
        if self.classifier_model.model_postprocessors is not None:
            feats_b_tensor = self.classifier_model.model_postprocessors[self.modality_type](feats_b_tensor)


        batch_feats = feats_b_tensor / feats_b_tensor.norm(dim=-1, keepdim=True)
        lan_feats = self.classifier

        # --- retrive the closeset class name as the classification result of the input image/depth/audio
        logits = 100. * batch_feats @ lan_feats
    
        _, pred_idx = logits.topk(max(self.top_k), dim=1, largest=True, sorted=True)

        if gt_name is not None:
            target = torch.tensor([self.classnames.index(classname_i) for classname_i in gt_name]).to(self.device)
        # target and predict idx replacement based on dataset
        if self.dataset_name in ["nyu", "sun_evalnyu"]:
            target[target >= 9] = 10
            pred_idx[pred_idx >= 9] = 10
        # elif self.dataset_name == 'llvip':
        #     # target = torch.tensor([0 if i % 2 == 0 else 1 for i in range(pred_idx.size(0))]).to(self.device)
        #     # target = torch.zeros(pred_idx.size(0)).to(self.device)
        #     # target[pred_idx.size(0)//2:] = 1
        #     target[target < 4] = 0
        #     target[target >= 4] = 1
        #     pred_idx[pred_idx < 4] = 0
        #     pred_idx[pred_idx >= 4] = 1
        pred_idx = pred_idx.t()
        correct = pred_idx.eq(target.reshape(1, -1).expand_as(pred_idx))

        res = []
        for k in self.top_k:
            # correct_k = correct[:k].reshape(-1).float().sum(0)
            correct_k = correct[:k].any(dim=0).float().sum(0)
            # if self.dataset_name == 'llvip':
            #     correct_k /= 2
            res.append(correct_k)
        res = torch.stack(res)

        return res


    def get_classification_acc(self, input_dataloader):
        
        accuracy = torch.zeros(len(self.top_k)).to(self.device)
        number_of_examples = 0

        for batch_idx, input_dict in enumerate(tqdm(input_dataloader)):

            if self.modality_type is None:
                # infer the modality from the data
                _m_type = list(input_dict.keys())
                _m_type.remove('label')
                print("_m_type = ",  _m_type)
                if len(_m_type) > 1 and ModalityType.TEXT in _m_type:
                    _m_type.remove(ModalityType.TEXT)
                if len(_m_type) == 1:
                    self.modality_type = _m_type[0]
                    print("EmergentZeroShotClassifier for ModalityType:", self.modality_type)
                else:
                    raise ValueError(f"Get {len(_m_type)} ModalityType: {_m_type}")
            
            class_in = self.modality_type
            data_in = input_dict.get(class_in)
            gt_name = input_dict.get('label', None)
            data_in = data_in.to(self.device)
            accuracy += self.get_batch_classification_acc(data_in, class_in, gt_name, 
                                                          vector_quantiser=self.vector_quantiser)
            number_of_examples += data_in.size(0)

        accuracy  = accuracy / number_of_examples * 100
        [print(f"Accuracy {i} {acc:.3f}%", end="  |") for i, acc in enumerate(accuracy)]

    def get_batch_classification_map(self, data_b, class_b, gt_name, vector_quantiser=None):
        feats_b = self.classifier_model.model({self.modality_type: data_b})
        feats_b_tensor = feats_b.get(self.modality_type)  # bs, 1024

        # implement for self contrast: data_b: bs n_views C H W for vision, depth; bs 1 77 for text
        # feats_b = [model({class_b: data_b_i}) for data_b_i in data_b]
        # feats_b_tensor = torch.cat([feats_b_i.get(class_b) for feats_b_i in feats_b], dim=0) # bs, 1024

        if vector_quantiser is not None:
            # feats_b_tensor = vector_quantiser(feats_b_tensor, class_b, output_type='common')
            if self.classifier_model.cfg_vq.get('use_mvq'):
                vq_dict_b = vector_quantiser(feats_b_tensor, class_b)
                feats_b_tensor = vq_dict_b.get('common')
            else:
                vq_dict_b = vector_quantiser(feats_b_tensor)
                feats_b_tensor = vq_dict_b.get('q')
        elif self.classifier_model.cfg_encoder.get('add_new_output_head'):
            common_dim = self.classifier_model.cfg_encoder.get('output_embed_dim').get('common')
            feats_b_tensor = feats_b_tensor[:, :common_dim]

        if self.classifier_model.model_postprocessors is not None:
            feats_b_tensor = self.classifier_model.model_postprocessors[self.modality_type](feats_b_tensor)

        batch_feats = feats_b_tensor / feats_b_tensor.norm(dim=-1, keepdim=True)
        lan_feats = self.classifier
        # cosine similarity between eval modality and text size: [bs of eval modality, classification numbers]
        cos_sim = 100. * batch_feats @ lan_feats
        # map (-1, 1) to (0, 1)
        sim_score = (cos_sim + 1) / 2

        # create onehot label for evaluation
        label_indices = torch.zeros_like(sim_score)
        for i, gt_name_i in enumerate(gt_name):
            for label_str in gt_name_i.split(';'):  # for ASA, all labels are considered positive 
                label_indices[i, self.classnames.index(label_str)] = 1.0
        return sim_score, label_indices

    def get_classification_map(self, input_dataloader):
        # input data: audio

        all_preds = []   # model prediction for all data
        all_labels = []  # ground true onehot label for all data
        assert self.dataset_name == 'asa', f'mAP evaluation is available only for Audioset, not for {self.dataset_name}'

        for batch_idx, input_dict in enumerate(tqdm(input_dataloader)):

            gt_name = input_dict.get('label')
            data_in = input_dict.get(self.modality_type)
            data_in = data_in.to(self.device)

            sim_score, label_indices = self.get_batch_classification_map(data_in, self.modality_type, gt_name, vector_quantiser=self.vector_quantiser)
            all_preds.append(sim_score)
            all_labels.append(label_indices)

        audio_output = torch.cat(all_preds).to('cpu').detach()  # (samples_num, classes_num)
        target = torch.cat(all_labels).to('cpu').detach()

        # calculate map
        map = map_calculation(audio_output, target)
        print(f'map = {map:.2f}%')


def get_intramodal_classifier(classnames, input_dataloader, imagebind_model, vector_quantiser=None, device=None, intramodal_vector='common'):
    # 模态内的分类：使用训练集上数据的embedding中心，作为classifier。验证使用common vector和增加specific vector，这两种的分类精度
    # get_intramodal_classifier(classnames, templates, image_model, vector_quantiser=None, device=None)
    if device is None:
        device = next(imagebind_model.parameters()).device
    print('get_intramodal_classifier: intramodal_vector =', intramodal_vector)

    # pdb.set_trace()
    with torch.no_grad():
        zeroshot_weights = []
        class_embedding_dict = {}
        class_count_dict = {}
        for batch_idx, input_dict in enumerate(tqdm(input_dataloader)):
            # pdb.set_trace()
            modality_type = 'depth'
            data_in = input_dict.get(modality_type).to(device)
            gt_name = input_dict.get('label', None)
            # if len(class_embedding_dict.get(gt_name[0], [])) >= 3: continue  # only for debug: batch_size must = 1

            # if gt_name in nyu_other_18_semantic_class:
            #     gt_name = 'other'  
            # gt_name = ['other' if i_gt in nyu_other_18_semantic_class else i_gt for i_gt in gt_name]
            feats_b = imagebind_model({modality_type: data_in}, modality_list=['depth'])
            feats_b_tensor = feats_b.get(modality_type) 
            embed_dim = feats_b_tensor.shape[-1]

            # 此处省略了VQ
            for idx, i_gt in enumerate(gt_name):  # batchsize = len(gt_name)
                class_embedding = class_embedding_dict.get(i_gt, [])
                i_embedding = feats_b_tensor[idx]
                if len(class_embedding) == 0:
                    class_embedding_dict.update({i_gt: [i_embedding]})
                else:
                    class_embedding_dict.update({i_gt: class_embedding + [i_embedding]})
                class_count = class_count_dict.get(i_gt, 0)
                class_count_dict.update({i_gt: class_count + 1})

        # pdb.set_trace()
        for classname in classnames:
            print(classname, class_count_dict.get(classname))
            if class_count_dict.get(classname):  # is not None, and > 0
                # class_embedding = class_embedding_dict.get(classname) / class_count_dict.get(classname)
                class_embedding_list = class_embedding_dict.get(classname)
                class_embedding_mean = torch.mean(torch.stack(class_embedding_list, dim=0), dim=0)
                # 
                l2_dist_matrix = torch.cdist(torch.stack(class_embedding_list, dim=0), class_embedding_mean.unsqueeze(0), p=2)  # N, 1
                l2_dist_arr = l2_dist_matrix.squeeze(1)
                _k = max(int(len(class_embedding_list) * 0.9), 1)
                print(classname, class_count_dict.get(classname), f'top{_k}')
                values, indices = l2_dist_arr.topk(_k)
                topk_embedding_list = [class_embedding_list[i] for i in indices]
                class_embedding_center =  torch.mean(torch.stack(topk_embedding_list, dim=0), dim=0)
            else:
                class_embedding_center = torch.zeros(embed_dim).to(device)
                # raise ValueError
        
            # debug 
            if intramodal_vector == 'common':
                # feats_b_tensor = vq_dict_b.get('common')
                class_embedding_center = class_embedding_center[0:1024]
            elif intramodal_vector == 'specific':
                class_embedding_center = class_embedding_center[1024:]
            elif intramodal_vector == 'common_and_specific':
                pass
            else:
                raise ValueError()
            zeroshot_weights.append(class_embedding_center)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)  # shape = [ebd_dims, class_nums]
    return zeroshot_weights

