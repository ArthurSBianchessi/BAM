from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np

from eval import PasskeyEvaluator, PromptGenerator, load_model

torch.set_float32_matmul_precision('high')
DEVICE = 'cuda:2'
# DEVICE = 'cpu'


seq_lens = [0, 1024, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
# seq_lens = [0, 1024, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 11000]
# seq_lens = [0, 1024, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000]
# seq_lens = [0, 1024, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000]
seq_lens = [0, 1024, 1250, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
seq_lens = [0, 1024, 1250, 1500, 1750, 2000, 3000, 4000, 5000, 6000]
seq_lens = [0, 1024, 1050, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
seq_lens = [0, 128,256,512,640,768,896,1000,1024, 1050, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
# seq_lens = [0,1024, 2048,2100,2150,2200,2250,2300,2350,2400,2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
# seq_lens = [0, 1024, 1050, 1100, 1250, 1350, 1500, 1750, 2000, 4000, 6000]
seq_lens = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 20000, 24000, 28000, 32000, 36000, 40000]
seq_lens = [0, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 45_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000]
seq_lens = [0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000]
# seq_lens = [0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000, 220_000, 240_000, 260_000, 280_000, 300_000]
seq_lens = [0, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000]
seq_lens = [0, 100_000, 200_000, 300_000, 400_000, 500_000]
# seq_lens = [0, 300_000, 400_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000]
# seq_lens = [0, 25_000, 50_000, 75_000, 90_000]
# seq_lens = [0, 1_000_000]

sample_size = 4	
sample_size = 10
# sample_size = 20
# sample_size = 100
evaluator = PasskeyEvaluator(seq_lens, device=DEVICE, sampling='beginning', preffix_digits=0, patience=2)
# evaluator = PasskeyEvaluator(seq_lens, device=DEVICE, sampling='beginning', preffix_digits=1, patience=2)
# evaluator = PasskeyEvaluator(seq_lens, device=DEVICE, preffix_digits=0)
# evaluator = PasskeyEvaluator(seq_lens, device=DEVICE, preffix_digits=0)


# model = load_model('./logs/l12/bam_ssmax/version_02/')
model = load_model('./logs/l12/bam_ssmax/version_03/')
bam_ssmax_lens, bam_ssmax_accs = evaluator.evaluate(model, sample_size=sample_size)
