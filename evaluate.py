# from transformers import AutoTokenizer, AutoModel, pipeline
# import torch
# import numpy as np

# from BAM.eval_utils import PasskeyEvaluator, PromptGenerator, load_model

# torch.set_float32_matmul_precision('high')
# DEVICE = 'cuda:0'
# DEVICE = 'cuda:3'
# # DEVICE = 'cpu'


# seq_lens = [0, 1024, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
# # seq_lens = [0, 1024, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 11000]
# # seq_lens = [0, 1024, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000]
# # seq_lens = [0, 1024, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000]
# seq_lens = [0, 1024, 1250, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
# seq_lens = [0, 1024, 1250, 1500, 1750, 2000, 3000, 4000, 5000, 6000]
# seq_lens = [0, 1024, 1050, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
# seq_lens = [0, 128,256,512,640,768,896,1000,1024, 1050, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
# # seq_lens = [0,1024, 2048,2100,2150,2200,2250,2300,2350,2400,2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
# # seq_lens = [0, 1024, 1050, 1100, 1250, 1350, 1500, 1750, 2000, 4000, 6000]
# seq_lens = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 20000, 24000, 28000, 32000, 36000, 40000]
# seq_lens = [0, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 45_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000]
# seq_lens = [0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000]
# seq_lens = [0, 2_000, 4_000, 6_000, 8_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000]
# # seq_lens = [0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000, 220_000, 240_000, 260_000, 280_000, 300_000]
# # seq_lens = [0, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000]
# # seq_lens = [0, 100_000, 200_000, 300_000, 400_000, 500_000]
# # seq_lens = [0, 300_000, 400_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000]
# # seq_lens = [0, 25_000, 50_000, 75_000, 90_000]
# seq_lens = [0, 4_000, 500_000]
# seq_lens = [0, 4_000, 750_000]
# seq_lens = [0, 4_000, 1_000_000]
# seq_lens = [0, 4_000, 100_000, 1_000_000]

# ctx = torch.autocast('cuda', enabled=True, dtype=torch.bfloat16)

# sample_size = 4	
# sample_size = 10
# # sample_size = 20
# # sample_size = 100
# evaluator = PasskeyEvaluator(seq_lens, device=DEVICE, sampling='beginning', preffix_digits=0, patience=2)
# # evaluator = PasskeyEvaluator(seq_lens, device=DEVICE, sampling='beginning', pred_digits=2, patience=2)
# # evaluator = PasskeyEvaluator(seq_lens, device=DEVICE, sampling='beginning', preffix_digits=1, patience=2)
# # evaluator = PasskeyEvaluator(seq_lens, device=DEVICE, preffix_digits=0)
# # evaluator = PasskeyEvaluator(seq_lens, device=DEVICE, preffix_digits=0)


# # model = load_model('./logs/l12/bam_ssmax/version_00/')
# # model = load_model('./logs/l12/bam_ssmax/version_01/')
# # model = load_model('./logs/l12/bam_ssmax/version_02/')
# model = load_model('./logs/l12/bam_ssmax/version_03/')
# # bam_ssmax_lens, bam_ssmax_accs = evaluator.evaluate(model, sample_size=sample_size)
# with ctx:
#     bam_ssmax_lens, bam_ssmax_accs = evaluator.evaluate(model, sample_size=sample_size)



from eval_utils import Evaluator
DEVICE = 'cuda:0'

evaluator = Evaluator(device=DEVICE)

nope_1024           = 'logs_paper/1024/nope/version_00/'
sin_1024            = 'logs_paper/1024/sinusoidal/version_00/'
rotary_1024         = 'logs_paper/1024/rotary/version_00/'
alibi_1024          = 'logs_paper/1024/alibi/version_00/'
bam_1024            = 'logs_paper/1024/bam/version_00/'
rotary_ssmax_1024   = 'logs_paper/1024/rotary_ssmax/version_00/'
bam_ssmax_1024      = 'logs_paper/1024/bam_ssmax/version_00/'

for model_path in [nope_1024, sin_1024, rotary_1024, alibi_1024, bam_1024, bam_ssmax_1024]:
    print(f"Evaluating {model_path}")
    evaluator.evaluate(model_path)
    print('\n')

