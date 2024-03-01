## author @Yash Sharma, B20241
## imporying all the neccesary modules
## loading important libraries
import numpy as np
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from transformers import Wav2Vec2Processor, EvalPrediction
import numpy as np
from typing import Any, Dict, Union, Tuple
import torch
from torch import nn
from tdnn import TDNN
import torch.nn.functional as F

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)


################### X_vector Class ####################
#### Source: https://github.com/cvqluu/TDNN
class X_vector(nn.Module):
    def __init__(self, input_dim = 39, num_classes=11):
        super(X_vector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3,dropout_p=0.5)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        tdnn1_out = F.relu(self.tdnn1(inputs))
        # print(f'shape of tdnn1 is {tdnn1_out.shape}')
        tdnn2_out = self.tdnn2(tdnn1_out)
        # print(f'shape of tdnn2 is {tdnn2_out.shape}')
        tdnn3_out = self.tdnn3(tdnn2_out)
        # print(f'shape of tdnn3 is {tdnn3_out.shape}')
        tdnn4_out = self.tdnn4(tdnn3_out)
        # print(f'shape of tdnn4 is {tdnn4_out.shape}')
        tdnn5_out = self.tdnn5(tdnn4_out)
        # print(f'shape of tdnn5 is {tdnn5_out.shape}')
        
        ### Stat Pooling        
        mean = torch.mean(tdnn4_out,1)
        # print(f'shape of mean is {mean.shape}')
        std = torch.var(tdnn4_out,1,)
        # print(f'shape of std is {std.shape}')
        stat_pooling = torch.cat((mean,std),1)
        # print(f'shape of stat_pooling is {stat_pooling.shape}')
        segment6_out = self.segment6(stat_pooling)
        # print(f'shape of segment6 is {segment6_out.shape}')
        # segment6_out1 = segment6_out[-1]

        # print(f'shape of segment6 is {segment6_out1.shape}')
        #ht = torch.unsqueeze(ht, 0)
        # segment6_out1 = torch.unsqueeze(segment6_out1, 0)
        # print(f'shape of segment6 is {segment6_out.shape}')
        x_vec = self.segment7(segment6_out)
        #print(x_vec)
        # print(f'shape of x_vec is {x_vec.shape}')
        predictions = self.output(x_vec)
        return predictions
    
    def extract_x_vec(self, inputs):
        tdnn1_out = F.relu(self.tdnn1(inputs))
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        ### Stat Pooling        
        mean = torch.mean(tdnn4_out,1)
        std = torch.var(tdnn4_out,1,)
        stat_pooling = torch.cat((mean,std),1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        return x_vec


####### Wave2Vec2 code ##############################################################################################################################
##########################################################################################################################################################

is_regression = False
torch.cuda.empty_cache()
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
#         self.pooling_mode = config.pooling_mode
        self.pooling_mode = 'mean'
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy( self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward( self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None,labels=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def extract_hidden_states(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Extract the hidden states from the Wave2Vec2 model
        hidden_states = outputs.last_hidden_state
        return hidden_states


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
    

##########################################################################################################
##########################################################################################################
    # U-vector architecture code
##########################################################################################################
##########################################################################################################

class LSTMNet(torch.nn.Module):
    def __init__(self, e_dim: int = 128*2):
        super(LSTMNet, self).__init__()
        self.e_dim = e_dim
        self.lstm1 = nn.LSTM(1024, 256,bidirectional=True)
        self.lstm2 = nn.LSTM(2*256, 128,bidirectional=True)
        #self.linear = nn.Linear(2*64,e_dim)
               
        self.fc_ha=nn.Linear(self.e_dim,100) 
        self.fc_1= nn.Linear(100,1)           
        self.sftmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x) 
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht = torch.unsqueeze(ht, 0) 
        #ht = torch.tanh(self.linear(ht))      
        ha = torch.tanh(self.fc_ha(ht))
        alp = self.fc_1(ha)
        al = self.sftmax(alp) 
        
       
        T = list(ht.shape)[1]  
        batch_size = list(ht.shape)[0]
        D = list(ht.shape)[2]
        c = torch.bmm(al.view(batch_size, 1, T),ht.view(batch_size,T,D))        
        c = torch.squeeze(c,0)        
        return (c)


class CCSL_Net(nn.Module):
    def __init__(self, model1,model2, nc: int, e_dim: int = 128*2):
        super(CCSL_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.e_dim = e_dim
        self.nc = nc

        self.att_fc = nn.Linear(e_dim,e_dim)
        #self.cla_fc = nn.Linear(e_dim,e_dim)
        
        self.sftmx = torch.nn.Softmax(dim=1)

        self.lang_classifier = nn.Linear(self.e_dim, self.nc, bias = True)
        self.adv_classifier = nn.Linear(self.e_dim, self.nc, bias = True) 
        
        
    def attention(self, att, cla):
        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)
        norm_att = att / torch.sum(att, dim=1)[:, None, :]

        u_LID = torch.sum(norm_att * cla, dim=1)  # Disentagle LID-specific and channel-specific u-vectors
        u_ch = torch.sum(1-norm_att * cla, dim=1)
        
        return u_LID, u_ch   
        
        
    def forward(self, x1,x2):
        e1 = self.model1(x1)
        e2 = self.model2(x2) 
        
        att_input = torch.cat((e1,e2), dim=0)
        att_input = torch.unsqueeze(att_input, 0)
        
        att = torch.sigmoid(self.att_fc(att_input))
        cla = att_input # No additional layer 
        u_lid, u_ch = self.attention(att, cla) # Get LID-specific and channel-specific u-vectors.

        lang_output = self.lang_classifier(u_lid)      # Restitute the u_lid  
        lang_output = self.sftmx(lang_output) # Langue prediction from language classifier
        
        return (lang_output)




    # if is_apex_available():
#     from apex import amp

# if version.parse(torch.__version__) >= version.parse("1.6"):
#     _is_native_amp_available = True
#     from torch.cuda.amp import autocast, GradScaler

# class CTCTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         if self.use_cuda_amp:
#             self.scaler = GradScaler()

#     def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
#         """
#         Perform a training step on a batch of inputs.

#         Subclass and override to inject custom behavior.

#         Args:
#             model (:obj:`nn.Module`):
#                 The model to train.
#             inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
#                 The inputs and targets of the model.

#                 The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
#                 argument :obj:`labels`. Check your model's documentation for all accepted arguments.

#         Return:
#             :obj:`torch.Tensor`: The tensor with training loss on this batch.
#         """

#         model.train()
#         inputs = self._prepare_inputs(inputs)

#         if self.use_cuda_amp:
#             with autocast():
#                 loss = self.compute_loss(model, inputs)
#         else:
#             loss = self.compute_loss(model, inputs)

#         if self.args.gradient_accumulation_steps > 1:
#             loss = loss / self.args.gradient_accumulation_steps

#         if self.use_cuda_amp:
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
#         elif self.use_apex:
#             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
#                 scaled_loss.backward()
#                 self.optimizer.step()
#         elif self.deepspeed:
#             self.deepspeed.backward(loss)
#             self.optimizer.step()
#         else:
#             loss.backward()
#             self.optimizer.step()

#         return loss.detach()
