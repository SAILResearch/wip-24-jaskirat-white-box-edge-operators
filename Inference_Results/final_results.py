import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.DataFrame(columns=['Inference', 'Device', 'Model'])

with open('mobile_bert_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Bert'}, ignore_index=True)

with open('mobile_bert_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Bert Pruned'}, ignore_index=True)

with open('mobile_bert_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Bert Distilled'}, ignore_index=True)

with open('mobile_bert_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Bert Distilled SPTQ'}, ignore_index=True)

with open('mobile_bert_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Bert SPTQ'}, ignore_index=True)

with open('mobile_bert_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Bert QAT'}, ignore_index=True)

with open('mobile_roberta_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Roberta'}, ignore_index=True)

with open('mobile_roberta_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Roberta Pruned'}, ignore_index=True)

with open('mobile_roberta_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Roberta Distilled'}, ignore_index=True)

with open('mobile_roberta_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Roberta Distilled SPTQ'}, ignore_index=True)


with open('mobile_roberta_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Roberta SPTQ'}, ignore_index=True)

with open('mobile_roberta_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'Roberta QAT'}, ignore_index=True)

with open('mobile_resnet_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNet'}, ignore_index=True)

with open('mobile_resnet_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNet Pruned'}, ignore_index=True)

with open('mobile_resnet_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNet Distilled'}, ignore_index=True)

with open('mobile_resnet_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNet Distilled SPTQ'}, ignore_index=True)

with open('mobile_resnet_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNet SPTQ'}, ignore_index=True)

with open('mobile_resnet_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNet QAT'}, ignore_index=True)

with open('mobile_resnext_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNext'}, ignore_index=True)

with open('mobile_resnext_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNext Pruned'}, ignore_index=True)

with open('mobile_resnext_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNext Distilled'}, ignore_index=True)

with open('mobile_resnext_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNext Distilled SPTQ'}, ignore_index=True)

with open('mobile_resnext_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNext QAT'}, ignore_index=True)

with open('mobile_resnext_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile',
                        'Model': 'ResNext SPTQ'}, ignore_index=True)

with open('edge_roberta_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Roberta'}, ignore_index=True)

with open('edge_roberta_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Roberta Pruned'}, ignore_index=True)

with open('edge_roberta_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Roberta Distilled'}, ignore_index=True)

with open('edge_roberta_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Roberta Distilled SPTQ'}, ignore_index=True)

with open('edge_roberta_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Roberta SPTQ'}, ignore_index=True)

with open('edge_roberta_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Roberta QAT'}, ignore_index=True)

with open('edge_bert_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Bert'}, ignore_index=True)

with open('edge_bert_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Bert Pruned'}, ignore_index=True)

with open('edge_bert_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Bert Distilled'}, ignore_index=True)

with open('edge_bert_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Bert Distilled SPTQ'}, ignore_index=True)

with open('edge_bert_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Bert SPTQ'}, ignore_index=True)

with open('edge_bert_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'Bert QAT'}, ignore_index=True)

with open('edge_resnet_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNet'}, ignore_index=True)

with open('edge_resnet_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNet Pruned'}, ignore_index=True)

with open('edge_resnet_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNet Distilled'}, ignore_index=True)

with open('edge_resnet_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNet Distilled SPTQ'}, ignore_index=True)

with open('edge_resnet_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNet QAT'}, ignore_index=True)

with open('edge_resnet_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNet SPTQ'}, ignore_index=True)

with open('edge_resnext_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNext'}, ignore_index=True)

with open('edge_resnext_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNext Pruned'}, ignore_index=True)

with open('edge_resnext_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNext Distilled'}, ignore_index=True)

with open('edge_resnext_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNext Distilled SPTQ'}, ignore_index=True)

with open('edge_resnext_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNext QAT'}, ignore_index=True)

with open('edge_resnext_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Edge',
                        'Model': 'ResNext SPTQ'}, ignore_index=True)

with open('cloud_bert_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Bert'}, ignore_index=True)

with open('cloud_bert_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Bert Pruned'}, ignore_index=True)

with open('cloud_bert_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Bert Distilled'}, ignore_index=True)

with open('cloud_bert_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Bert Distilled SPTQ'}, ignore_index=True)

with open('cloud_bert_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Bert SPTQ'}, ignore_index=True)

with open('cloud_bert_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Bert QAT'}, ignore_index=True)
        
with open('cloud_roberta_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Roberta'}, ignore_index=True)

with open('cloud_roberta_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Roberta Pruned'}, ignore_index=True)

with open('cloud_roberta_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Roberta Distilled'}, ignore_index=True)

with open('cloud_roberta_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Roberta Distilled SPTQ'}, ignore_index=True)

with open('cloud_roberta_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Roberta SPTQ'}, ignore_index=True)

with open('cloud_roberta_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'Roberta QAT'}, ignore_index=True)
        
with open('cloud_resnet_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNet'}, ignore_index=True)
        
with open('cloud_resnet_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNet Pruned'}, ignore_index=True)

with open('cloud_resnet_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNet Distilled'}, ignore_index=True)

with open('cloud_resnet_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNet Distilled SPTQ'}, ignore_index=True)

with open('cloud_resnet_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNet QAT'}, ignore_index=True)

with open('cloud_resnet_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNet SPTQ'}, ignore_index=True)

with open('cloud_resnext_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNext'}, ignore_index=True)
        
with open('cloud_resnext_pruned_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNext Pruned'}, ignore_index=True)

with open('cloud_resnext_distilled_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNext Distilled'}, ignore_index=True)

with open('cloud_resnext_distilled_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNext Distilled SPTQ'}, ignore_index=True)

with open('cloud_resnext_int8_qat_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNext QAT'}, ignore_index=True)

with open('cloud_resnext_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Cloud',
                        'Model': 'ResNext SPTQ'}, ignore_index=True)

with open('mobile_edge_roberta_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile-Edge',
                        'Model': 'Roberta'}, ignore_index=True)

with open('mobile_edge_roberta_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile-Edge',
                        'Model': 'Roberta SPTQ'}, ignore_index=True)

with open('mobile_edge_bert_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile-Edge',
                        'Model': 'Bert'}, ignore_index=True)

with open('mobile_edge_bert_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile-Edge',
                        'Model': 'Bert SPTQ'}, ignore_index=True)

with open('mobile_edge_resnet_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile-Edge',
                        'Model': 'ResNet'}, ignore_index=True)
        
with open('mobile_edge_resnet_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile-Edge',
                        'Model': 'ResNet SPTQ'}, ignore_index=True)

with open('mobile_edge_resnext_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile-Edge',
                        'Model': 'ResNext'}, ignore_index=True)
        
with open('mobile_edge_resnext_int8_sptq_results.txt', 'r') as f:
    lines = f.readlines()[1:][-500:]
    for value in lines:
        df = df._append({'Inference': value, 'Device': 'Mobile-Edge',
                        'Model': 'ResNext SPTQ'}, ignore_index=True)
        
df.to_csv('Inference_Results.csv', index=False)
